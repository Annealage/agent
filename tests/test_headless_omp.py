"""Tests for ``headless.run_prompt`` through a real ``OmpSession``.

Permission is pinned here because the gate is the session's own host-tool path
asking the broker ``run_prompt`` built: a scripted session would only show the
test's own script calling a tool. The omp client is a fake (``FakeRpcClient``,
the seam ``tests/test_omp_session.py`` uses), so no ``omp`` binary is needed, but
``session/omp.py`` imports ``omp_rpc`` at the top, so this file needs that
package installed (it requires Python 3.11) and CI skips it where it is not.
The rest of ``run_prompt``'s tests, which need neither, are in
``tests/test_headless.py``.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import tool
from test_headless import make_args, make_toolset

from annealage_agent import headless
from annealage_agent.headless import ToolSet, run_prompt
from annealage_agent.session import omp as omp_module
from annealage_agent.tools import Grading, ok

pytestmark = pytest.mark.asyncio


class FakeRpcClient:
    """The ``omp_rpc.RpcClient`` surface ``OmpSession`` uses. ``prompt`` starts
    ``script(client)`` on a thread of its own and returns, as omp does, so the
    script may call a host tool's ``execute`` (which blocks on the event loop,
    and so must not run on it) and may outlive the run."""

    instances = []
    script = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.custom_tools = kwargs.get("custom_tools") or ()
        self.stopped = False
        self.cost = 0.0
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._listeners = {}
        self._lock = threading.Lock()
        FakeRpcClient.instances.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def get_state(self):
        return SimpleNamespace(session_id="s", session_file=None, context_usage=None)

    def get_session_stats(self):
        return SimpleNamespace(cost=self.cost, tokens=SimpleNamespace(**self.tokens))

    def get_available_models(self):
        return ()

    def prompt(self, message, *, images=None, streaming_behavior=None):
        self.prompt_message = message
        self.thread = threading.Thread(target=type(self).script, args=(self,), daemon=True)
        self.thread.start()

    def abort(self):
        pass

    def set_custom_tools(self, tools):
        return ()

    def send_ui_confirmation(self, request_id, confirmed):
        pass

    def cancel_ui_request(self, request_id, *, timed_out=False):
        pass

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda listener: self._listeners.__setitem__(name[3:], listener)
        raise AttributeError(name)

    # -- what a script does ----------------------------------------------------

    def say(self, text):
        self._listeners["message_update"](
            SimpleNamespace(assistant_message_event={"type": "text_delta", "delta": text})
        )

    def call(self, name, params, call_id):
        """One host-tool call as omp makes it; returns (result or None, error text)."""
        self._listeners["tool_execution_start"](
            SimpleNamespace(tool_call_id=call_id, tool_name=name, args=params)
        )
        tool_def = next(t for t in self.custom_tools if t.name == name)
        try:
            result = tool_def.execute(params, SimpleNamespace(tool_call_id=call_id))
            error = None
        except Exception as exc:
            result, error = {"content": [{"type": "text", "text": str(exc)}]}, str(exc)
        self._listeners["tool_execution_end"](
            SimpleNamespace(
                tool_call_id=call_id, tool_name=name, result=result, is_error=error is not None
            )
        )
        return result, error

    def end(self):
        self.cost = 0.0125
        self.tokens = {"input": 5, "output": 3, "cache_read": 0, "cache_write": 0}
        self._listeners["agent_end"](SimpleNamespace(is_terminal=True))


@pytest.fixture
def omp_client(monkeypatch):
    FakeRpcClient.instances = []
    monkeypatch.setattr(omp_module, "RpcClient", FakeRpcClient)
    return FakeRpcClient


async def test_omp_run_end_to_end_with_a_read_view_and_ungranted_write_tool(omp_client, tmp_path):
    saved = []
    refusals = []

    def script(client):
        client.say("checking")
        refusals.append(client.call("lookup", {"q": "x"}, "c1"))
        refusals.append(client.call("pan", {}, "c2"))
        refusals.append(client.call("save", {"v": "nope"}, "c3"))
        client.say(" done")
        client.end()

    omp_client.script = script
    result = await run_prompt(
        "do it",
        **make_args(tmp_path, tools=make_toolset(saved=saved), model="prov/model", system="ctx"),
    )

    # Read and view run unasked; the write-grade tool is refused with a reason the
    # model reads, and its handler never ran.
    assert refusals[0][1] is None and refusals[1][1] is None
    assert "unattended run" in refusals[2][1] and "save" in refusals[2][1]
    assert saved == []
    assert result.calls == (("lookup", True), ("pan", True), ("save", False))
    # What was said after the last tool result: the narration before is dropped.
    assert (result.text, result.stop_reason, result.error) == (" done", "end", None)
    assert result.cost_usd == pytest.approx(0.0125)
    assert result.tokens == {"input": 5, "output": 3, "cache_read": 0, "cache_write": 0}

    (client,) = omp_client.instances
    assert client.kwargs["model"] == "prov/model"
    assert client.kwargs["tools"] == ()
    assert client.kwargs["extra_args"] == ("--auto-approve", "--no-extensions")
    assert "--no-extensions" in client.kwargs["extra_args"]
    assert "ctx" in client.kwargs["append_system_prompt"]
    assert Path(client.kwargs["cwd"]) == tmp_path
    assert client.stopped
    # The conversation file went under the run's own state, which is gone.
    assert not Path(client.kwargs["session_dir"]).exists()
    assert tmp_path not in Path(client.kwargs["session_dir"]).parents


async def test_thinking_is_passed_to_omp(omp_client, tmp_path):
    omp_client.script = lambda client: client.end()
    result = await run_prompt("do it", **make_args(tmp_path, thinking="high"))
    assert result.error is None
    (client,) = omp_client.instances
    assert client.kwargs["extra_args"] == (
        "--auto-approve",
        "--no-extensions",
        "--thinking",
        "high",
    )


async def test_a_granted_write_tool_runs_and_nothing_is_remembered(omp_client, tmp_path):
    saved = []
    outcome = []

    def script(client):
        outcome.append(client.call("save", {"v": "kept"}, "c1"))
        client.end()

    omp_client.script = script
    first = await run_prompt(
        "do it", **make_args(tmp_path, tools=make_toolset(grant=("save",), saved=saved))
    )
    assert saved == ["kept"] and outcome[0][1] is None
    assert first.calls == (("save", True),)

    # The next run, with no grant, is refused: the grant was for the first run.
    outcome.clear()
    second = await run_prompt("do it", **make_args(tmp_path, tools=make_toolset(saved=saved)))
    assert saved == ["kept"]
    assert second.calls == (("save", False),)

    # Nothing was written anywhere a grant could be remembered.
    assert not list(tmp_path.rglob("permissions.toml"))


async def test_omp_overrides_reach_the_session(omp_client, tmp_path):
    omp_client.script = lambda client: client.end()
    binary = tmp_path / "omp-bin"
    await run_prompt(
        "go",
        **make_args(tmp_path, omp_binary=str(binary), omp_agent_dir=str(tmp_path / "agent")),
    )

    (client,) = omp_client.instances
    assert client.kwargs["executable"] == str(binary)
    assert client.kwargs["env"]["PI_CODING_AGENT_DIR"] == str(tmp_path / "agent")


async def test_run_prompt_omp_request_timeout_controls_the_rpc_ack_deadline(tmp_path):
    binary = tmp_path / "fake-omp-rpc"
    binary.write_text(
        "#!" + sys.executable + "\n"
        "import json\n"
        "import sys\n"
        "import time\n"
        "print(json.dumps({'type': 'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    command = request['type']\n"
        "    if command == 'prompt':\n"
        "        time.sleep(1.2)\n"
        "    data = {}\n"
        "    if command == 'set_host_tools':\n"
        "        data = {'toolNames': [tool['name'] for tool in request.get('tools', [])]}\n"
        "    elif command == 'get_state':\n"
        "        data = {'sessionId': 'timeout-test'}\n"
        "    elif command == 'get_session_stats':\n"
        "        data = {'tokens': {}, 'cost': 0.0}\n"
        "    elif command == 'get_available_models':\n"
        "        data = {'models': []}\n"
        "    print(json.dumps({'type': 'response', 'id': request['id'], "
        "'success': True, 'data': data}), flush=True)\n"
        "    if command == 'prompt':\n"
        "        print(json.dumps({'type': 'agent_end', 'isTerminal': True}), flush=True)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)

    rejected = await run_prompt(
        "go",
        **make_args(
            tmp_path,
            timeout=4,
            omp_binary=str(binary),
            omp_request_timeout=0.8,
        ),
    )
    assert rejected.stop_reason == "rejected"
    assert "Timed out waiting for response to prompt" in rejected.error

    accepted = await run_prompt(
        "go",
        **make_args(
            tmp_path,
            timeout=4,
            omp_binary=str(binary),
            omp_request_timeout=180,
        ),
    )
    assert accepted.stop_reason == "end"
    assert accepted.error is None

    defaulted = await run_prompt(
        "go",
        **make_args(tmp_path, timeout=4, omp_binary=str(binary)),
    )
    assert defaulted.stop_reason == "end"
    assert defaulted.error is None


async def test_a_tool_call_after_a_timeout_is_refused_and_a_running_one_is_cancelled(
    omp_client, tmp_path
):
    ran = []
    cancelled = threading.Event()
    outcome = {}

    @tool("slow", "a slow granted write", {})
    async def slow(args):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        ran.append("slow finished")
        return ok(text="slow")

    @tool("save", "save a value", {"v": str})
    async def save(args):
        ran.append(args["v"])
        return ok(text="saved")

    @tool("peek", "a read tool", {})
    async def peek(args):
        ran.append("peek")
        return ok(text="peeked")

    def script(client):
        # Blocks until the run ends and the handler is cancelled under it.
        outcome["slow"] = client.call("slow", {}, "c1")
        # Calls arriving after the run is over, from a thread still alive: a
        # granted write (the broker is shut down) and a read (no broker at all,
        # only the handler's own refusal).
        outcome["late"] = client.call("save", {"v": "late"}, "c2")
        outcome["late_read"] = client.call("peek", {}, "c3")

    omp_client.script = script
    tools = ToolSet(
        tools=[slow, save, peek],
        grading=Grading(read=("peek",), view=(), write=("slow", "save")),
        grant=("slow", "save"),
    )
    result = await run_prompt("go", **make_args(tmp_path, tools=tools, timeout=0.3))
    (client,) = omp_client.instances
    await asyncio.get_running_loop().run_in_executor(None, client.thread.join, 5)

    assert (result.stop_reason, result.error) == ("timeout", "timeout")
    assert cancelled.is_set()
    assert outcome["slow"][1] is not None
    assert "shutting down" in outcome["late"][1]
    assert "run has ended" in outcome["late_read"][1]
    assert ran == []


async def test_a_timeout_during_omp_start_does_not_wait_for_the_start(monkeypatch, tmp_path):
    release = threading.Event()

    class StuckStart(FakeRpcClient):
        def start(self):
            # omp not answering its ready signal; ending the process fails the call.
            if release.wait(20):
                raise RuntimeError("omp process stopped before ready")
            return self

        def stop(self):
            self.stopped = True
            release.set()

    FakeRpcClient.instances = []
    monkeypatch.setattr(omp_module, "RpcClient", StuckStart)
    began = time.monotonic()
    result = await run_prompt("go", **make_args(tmp_path, timeout=0.3))
    elapsed = time.monotonic() - began

    assert (result.stop_reason, result.error) == ("timeout", "timeout")
    assert elapsed < 5
    (client,) = FakeRpcClient.instances
    assert client.stopped


async def test_handler_cleanup_after_cancellation_finishes_before_run_prompt_returns(
    omp_client, tmp_path
):
    cleaned = []

    @tool("slow", "a slow granted write", {})
    async def slow(args):
        try:
            await asyncio.sleep(3600)
        finally:
            # Awaited cleanup: runs after the cancellation is delivered, and
            # touches the caller's state.
            await asyncio.sleep(0.2)
            cleaned.append("cleaned")

    omp_client.script = lambda client: client.call("slow", {}, "c1")
    tools = ToolSet(
        tools=[slow], grading=Grading(read=(), view=(), write=("slow",)), grant=("slow",)
    )
    result = await run_prompt("go", **make_args(tmp_path, tools=tools, timeout=0.3))

    assert result.error == "timeout"
    # Already done when run_prompt returned, not on a later loop iteration.
    assert cleaned == ["cleaned"]


async def test_a_handler_that_will_not_stop_is_abandoned_after_the_drain(
    omp_client, monkeypatch, tmp_path
):
    monkeypatch.setattr(headless, "_DRAIN_TIMEOUT", 0.2)

    release = asyncio.Event()

    @tool("stubborn", "ignores cancellation until released", {})
    async def stubborn(args):
        while not release.is_set():
            try:
                await asyncio.wait_for(release.wait(), 3600)
            except asyncio.CancelledError:
                pass

    omp_client.script = lambda client: client.call("stubborn", {}, "c1")
    tools = ToolSet(
        tools=[stubborn],
        grading=Grading(read=(), view=(), write=("stubborn",)),
        grant=("stubborn",),
    )
    began = time.monotonic()
    result = await run_prompt("go", **make_args(tmp_path, tools=tools, timeout=0.3))

    release.set()
    assert result.error == "timeout"
    assert time.monotonic() - began < 3
