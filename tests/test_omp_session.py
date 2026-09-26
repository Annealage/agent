"""Tests for ``session/omp.py``, driving ``OmpSession`` through a fake
``omp_rpc.RpcClient`` (plan: Annealage Mesh's ``phase4_omp-session.md``).

``RpcClient`` has no injectable transport the way ``ClaudeSDKClient`` does
(``tests/test_sdk_session.py``'s ``FakeTransport``): it hardcodes a real
subprocess and two real reader threads internally. ``OmpSession``'s own
injection seam is one level up instead -- ``client_factory``, the callable
that would otherwise be ``RpcClient`` itself -- and ``FakeRpcClient`` below
stands in for it, implementing the same public method surface (``start``,
``stop``, ``get_state``, ``prompt``, ``abort``, the ``on_*`` listener
registrations, ``send_ui_confirmation``, ``cancel_ui_request``) plus the
real ``omp_rpc.host_tool``-built ``HostTool`` objects ``OmpSession`` passes
it as ``custom_tools``.

Every test that exercises a write-class tool's ``execute`` callback or the
``confirm`` UI-request handler calls it via ``loop.run_in_executor``, never
awaited directly: both block on ``asyncio.run_coroutine_threadsafe(...)
.result()``, and calling either from the loop thread itself would deadlock
the one loop they need to schedule the broker's coroutine on -- the same
cross-thread behaviour production code exercises against ``omp_rpc``'s
genuine per-call and reader threads (see ``session/omp.py``'s module
docstring). ``tests/test_codex_session.py``'s identical pattern for
``CodexClient``'s own single reader thread is the direct precedent.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import SdkMcpTool
from omp_rpc import RpcClient

from annealage_agent import launch, sessions
from annealage_agent.session import omp as omp_module
from annealage_agent.session.base import (
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentModelChanged,
    AgentStatus,
    PermissionRequest,
    PermissionResolved,
    SessionReset,
    TextDelta,
    ToolResult,
    ToolUse,
    TurnEnd,
)
from annealage_agent.session.omp import (
    OmpSession,
    _api_key_env_name,
    _build_custom_provider,
    _provider_id,
    _write_agent_dir,
)
from annealage_agent.session.permissions import PermissionBroker
from annealage_agent.tools import ToolSpec, _wrap, ok
from annealage_agent.viewers import ViewerBus


class FakeRpcClient:
    """Stands in for ``omp_rpc.RpcClient``. See module docstring for why
    this, not a hand-rolled stdio transport, is the fake seam."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.custom_tools = kwargs.get("custom_tools") or ()
        self.started = False
        self.stopped = False
        self.prompt_calls = []
        self.abort_calls = 0
        self.confirmations = []
        self.cancellations = []
        self.set_model_calls = []
        self.switch_calls = []
        self.set_custom_tools_calls = []
        self._listeners = {}
        self.session_id = "omp-sess-1"
        self.session_file = "/sessions/omp-sess-1.jsonl"
        # What the next prompt raises, if anything (a refusal or a dead process).
        self.prompt_error = None
        # The cumulative figures get_session_stats reports.
        self.cost = 0.0
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def get_state(self):
        return SimpleNamespace(session_id=self.session_id, session_file=self.session_file)

    def get_session_stats(self):
        return SimpleNamespace(cost=self.cost, tokens=SimpleNamespace(**self.tokens))

    def switch_session(self, session_path):
        self.switch_calls.append(session_path)
        self.session_file = str(session_path)
        self.session_id = "omp-resumed"
        return SimpleNamespace(cancelled=False)

    def set_custom_tools(self, tools):
        self.set_custom_tools_calls.append(tuple(tools))
        self.custom_tools = tuple(tools)
        return tuple(t.name for t in tools)

    def prompt(self, message, *, images=None, streaming_behavior=None):
        if self.prompt_error is not None:
            raise self.prompt_error
        self.prompt_calls.append(
            SimpleNamespace(message=message, images=images, streaming_behavior=streaming_behavior)
        )

    def abort(self):
        self.abort_calls += 1

    def set_model(self, provider, model_id):
        self.set_model_calls.append((provider, model_id))
        return SimpleNamespace(provider=provider, model_id=model_id)

    def on_message_update(self, listener):
        self._listeners["message_update"] = listener

    def on_tool_execution_start(self, listener):
        self._listeners["tool_execution_start"] = listener

    def on_tool_execution_end(self, listener):
        self._listeners["tool_execution_end"] = listener

    def on_agent_end(self, listener):
        self._listeners["agent_end"] = listener

    def on_ui_request(self, listener):
        self._listeners["ui_request"] = listener

    def on_protocol_error(self, listener):
        self._listeners["protocol_error"] = listener

    def send_ui_confirmation(self, request_id, confirmed):
        self.confirmations.append((request_id, confirmed))

    def cancel_ui_request(self, request_id, *, timed_out=False):
        self.cancellations.append(request_id)

    # -- test helpers --------------------------------------------------------

    def tool(self, name):
        return next(t for t in self.custom_tools if t.name == name)

    def push_message_update(self, assistant_message_event):
        self._listeners["message_update"](
            SimpleNamespace(assistant_message_event=assistant_message_event)
        )

    def push_tool_execution_start(self, tool_call_id, tool_name, args):
        self._listeners["tool_execution_start"](
            SimpleNamespace(tool_call_id=tool_call_id, tool_name=tool_name, args=args)
        )

    def push_tool_execution_end(self, tool_call_id, tool_name, result, is_error=False):
        self._listeners["tool_execution_end"](
            SimpleNamespace(
                tool_call_id=tool_call_id, tool_name=tool_name, result=result, is_error=is_error
            )
        )

    def push_ui_request(self, request):
        """Invoke ``OmpSession._on_ui_request`` exactly as ``omp_rpc``'s
        reader thread would. Call from a worker thread via
        ``loop.run_in_executor``, never awaited directly -- see this
        module's docstring."""
        self._listeners["ui_request"](request)

    def push_agent_end(self, is_terminal=True):
        self._listeners["agent_end"](SimpleNamespace(is_terminal=is_terminal))

    def push_protocol_error(self, command, remote_error):
        self._listeners["protocol_error"](
            SimpleNamespace(command=command, remote_error=remote_error)
        )


class FakeUiRequest:
    """Stands in for ``omp_rpc.protocol.ExtensionUiRequest``: the one
    method ``OmpSession`` reads plus the ``id``/``method`` fields."""

    def __init__(self, id, method):
        self.id = id
        self.method = method

    def requires_response(self):
        return self.method in {"select", "input", "editor"}


async def _read_handler(args):
    return {"content": [{"type": "text", "text": "read-ok:%s" % args.get("q", "")}]}


async def _write_handler(args):
    return {"content": [{"type": "text", "text": "wrote:%s" % args.get("path", "")}]}


async def _failing_write_handler(args):
    return {"content": [{"type": "text", "text": "could not write"}], "is_error": True}


def _tool_table(write_handler=_write_handler):
    return {
        "get_view": ToolSpec(
            schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": []},
            description="read the current view",
            handler=_read_handler,
            write=False,
        ),
        "add_note": ToolSpec(
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            description="save a note",
            handler=write_handler,
            write=True,
        ),
    }


class EventRecorder:
    """Collects every ``AgentEvent`` an ``OmpSession`` emits, and lets a
    test await the next one deterministically. Mirrors
    ``tests/test_codex_session.py``'s recorder of the same name."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self.all = []

    def __call__(self, event) -> None:
        self.all.append(event)
        self._queue.put_nowait(event)

    async def next(self, timeout: float = 2.0, include_status: bool = False):
        while True:
            event = await asyncio.wait_for(self._queue.get(), timeout)
            if include_status or not isinstance(event, AgentStatus):
                return event


async def _started_session(*, viewer_count=1, tool_table=None, **kwargs):
    """A ready ``OmpSession`` over a fresh ``FakeRpcClient``, plus the
    recorder and the fake client itself (for assertions and for driving
    events/confirms)."""
    holder = {}

    def _client_factory(**client_kwargs):
        fake = FakeRpcClient(**client_kwargs)
        holder["fake"] = fake
        return fake

    recorder = EventRecorder()
    broker = kwargs.pop("broker", "__default__")
    if broker == "__default__":
        broker = PermissionBroker(recorder, timeout=2.0, no_viewer_grace=0.05)
    kwargs.setdefault("base_url", "http://127.0.0.1:11434/v1")
    session = OmpSession(
        recorder,
        cwd="/proj/root",
        session_id="toy-sess-1",
        broker=broker,
        tool_table=tool_table if tool_table is not None else _tool_table(),
        client_factory=_client_factory,
        **kwargs,
    )
    await session.start()
    assert session.agent_status() == AGENT_READY
    session.on_viewer_presence(viewer_count)
    return session, holder["fake"], recorder, broker


# ---------------------------------------------------------------------------
# startup: custom-provider config generation and the no-builtin-tools launch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_launches_omp_scoped_to_a_custom_provider_with_no_builtin_tools():
    session, fake, recorder, broker = await _started_session(
        model="llama3.1:8b", base_url="http://127.0.0.1:11434/v1", api_key="secret-key"
    )
    try:
        assert fake.kwargs["model"] == "toy-local/llama3.1:8b"
        # No builtin tools: the model's only capabilities are the host tools
        # this session registers (see session/omp.py's module docstring on
        # why the write-class gate lives in host_tool_call, not confirm).
        assert fake.kwargs["tools"] == ()
        assert fake.kwargs["no_session"] is True
        assert "--auto-approve" in fake.kwargs["extra_args"]
        # Ambient .omp/.pi extension discovery is disabled too: without
        # this flag, a project-local extension would load as trusted code
        # and could register native tools entirely outside the broker gate
        # (see session/omp.py's module docstring, Finding 1).
        assert "--no-extensions" in fake.kwargs["extra_args"]
        assert {t.name for t in fake.custom_tools} == {"get_view", "add_note"}

        agent_dir = Path(fake.kwargs["env"]["PI_CODING_AGENT_DIR"])
        assert agent_dir.is_dir()
        models_doc = json.loads((agent_dir / "models.yml").read_text())
        provider = models_doc["providers"]["toy-local"]
        assert provider["baseUrl"] == "http://127.0.0.1:11434/v1"
        # The literal secret is never written into models.yml: apiKey
        # carries the name of an env var set on the launched subprocess
        # instead, so omp's own env-var-name-first apiKey resolution does
        # the substitution (see session/omp.py's module docstring,
        # Finding 3).
        env_var_name = _api_key_env_name("toy-sess-1")
        assert provider["apiKey"] == env_var_name
        assert "secret-key" not in json.dumps(models_doc)
        assert fake.kwargs["env"][env_var_name] == "secret-key"
        assert provider["models"] == [{"id": "llama3.1:8b", "name": "llama3.1:8b"}]
        # Registering only the startup model_id, with no discovery, is
        # exactly what made a live switch to any other model rejected by
        # omp's real RpcClient.set_model with "Model not found" (Finding 1):
        # discovery: {type: proxy} (omp://providers.md's "Discovery-enabled
        # provider" shape) makes every model the endpoint reports live-
        # switchable, not only this one.
        assert provider["discovery"] == {"type": "proxy"}

        assert session.sdk_session_id == "omp-sess-1"
    finally:
        await session.close()
    assert not agent_dir.exists()


@pytest.mark.asyncio
async def test_set_model_to_a_model_other_than_the_startup_one_is_not_rejected_by_the_config():
    """The real ``omp`` binary's ``RpcClient.set_model`` rejects any
    ``modelId`` its session does not already know about -- registered in
    ``models.yml`` or discovered at runtime (``omp://providers.md``'s
    registry-assembly order). ``FakeRpcClient.set_model`` below never
    rejects anything, so it cannot by itself prove a live switch to a
    second model actually works against the real binary; what it *can*
    prove is that this session no longer generates the config that made
    the real binary reject it -- registering only the one startup
    ``model_id`` with discovery disabled. This asserts the generated
    ``models.yml`` now enables discovery (``discovery: {type: "proxy"}``),
    which is what makes any model the endpoint actually reports selectable,
    not only ``llama3.1:8b``, before also exercising the switch itself
    through the fake to confirm the call still goes through end to end."""
    session, fake, recorder, broker = await _started_session(
        model="llama3.1:8b", base_url="http://127.0.0.1:11434/v1"
    )
    try:
        agent_dir = Path(fake.kwargs["env"]["PI_CODING_AGENT_DIR"])
        models_doc = json.loads((agent_dir / "models.yml").read_text())
        provider = models_doc["providers"]["toy-local"]
        # The seam the real bug lived in: without this, omp would only ever
        # know about "llama3.1:8b" and reject a switch to anything else.
        assert provider["discovery"] == {"type": "proxy"}

        await session.set_model("mixtral-8x7b")
        assert fake.set_model_calls == [(_provider_id(), "mixtral-8x7b")]
        event = await recorder.next()
        assert isinstance(event, AgentModelChanged)
        assert event.model == "mixtral-8x7b"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_no_base_url_uses_omp_own_configured_providers_directly():
    """With no ``omp_base_url``, this session must not synthesize a
    custom provider at all: ``model`` goes straight to `omp` as the
    ``"provider/model"`` reference a human would type at the CLI (e.g.
    ``"titan/qwen3.8-27b"``), against whatever providers `omp` is already
    configured with on its own, and ``PI_CODING_AGENT_DIR`` is left unset
    so `omp`'s own default agent dir and credential resolution are
    untouched."""
    session, fake, recorder, broker = await _started_session(
        model="titan/qwen3.8-27b", base_url=None
    )
    try:
        assert fake.kwargs["model"] == "titan/qwen3.8-27b"
        assert "PI_CODING_AGENT_DIR" not in fake.kwargs["env"]
        assert fake.kwargs["env"] == {}
        assert session.agent_status() == AGENT_READY
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_no_base_url_with_no_model_passes_none_through_to_omp():
    """No ``model`` either: `omp` falls back to its own default the same
    way a bare ``omp`` CLI invocation would, rather than this session
    inventing a placeholder model string."""
    session, fake, recorder, broker = await _started_session(model=None, base_url=None)
    try:
        assert fake.kwargs["model"] is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_api_key_without_base_url_fails_without_launching_omp():
    """``omp_api_key`` only means something alongside a synthesized
    custom provider; without ``omp_base_url`` there is no such provider
    for it to authenticate, so this is a misconfiguration this session
    catches itself rather than silently ignoring the key."""
    recorder = EventRecorder()
    broker = PermissionBroker(recorder, timeout=2.0, no_viewer_grace=0.05)
    factory_calls = []

    def _client_factory(**kwargs):
        factory_calls.append(kwargs)
        raise AssertionError("must not construct a client with an inconsistent config")

    session = OmpSession(
        recorder,
        cwd="/proj/root",
        session_id="toy-sess-1",
        broker=broker,
        base_url=None,
        api_key="secret-key",
        tool_table=_tool_table(),
        client_factory=_client_factory,
    )
    await session.start()
    assert session.agent_status() == AGENT_UNAVAILABLE
    assert factory_calls == []
    error = await recorder.next()
    assert isinstance(error, AgentError)
    assert "omp_api_key" in error.remediation
    assert "omp_base_url" in error.remediation


@pytest.mark.asyncio
async def test_set_model_without_base_url_splits_the_provider_model_reference():
    session, fake, recorder, broker = await _started_session(
        model="titan/qwen3.8-27b", base_url=None
    )
    try:
        await session.set_model("titan/qwen3.9-70b")
        assert fake.set_model_calls == [("titan", "qwen3.9-70b")]
        event = await recorder.next()
        assert isinstance(event, AgentModelChanged)
        assert event.model == "titan/qwen3.9-70b"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_model_without_base_url_passes_a_bare_model_id_straight_through():
    """No synthesized provider exists to assume for a bare id the way
    ``base_url``-mode's own ``_provider_id()`` can, and this method makes no
    local guess about what `omp` will accept: ``model.partition("/")``
    finding no ``"/"`` puts the whole string in ``provider`` and an empty
    string in ``model_id``, sent to the RPC exactly as split -- `omp`'s own
    ``set_model`` response is the real validation, not a local pre-check."""
    session, fake, recorder, broker = await _started_session(
        model="titan/qwen3.8-27b", base_url=None
    )
    try:
        await session.set_model("qwen3.9-70b")
        assert fake.set_model_calls == [("qwen3.9-70b", "")]
        event = await recorder.next()
        assert isinstance(event, AgentModelChanged)
        assert event.model == "qwen3.9-70b"
    finally:
        await session.close()


def test_build_custom_provider_writes_the_given_env_var_name_as_apikey():
    provider = _build_custom_provider("http://host/v1", "TOY_OMP_API_KEY_abc123")
    assert provider == {
        "baseUrl": "http://host/v1",
        "api": "openai-completions",
        "discovery": {"type": "proxy"},
        "apiKey": "TOY_OMP_API_KEY_abc123",
    }


def test_build_custom_provider_without_api_key_is_auth_none_not_empty_header():
    provider = _build_custom_provider("http://host/v1", None)
    assert provider == {
        "baseUrl": "http://host/v1",
        "api": "openai-completions",
        "discovery": {"type": "proxy"},
        "auth": "none",
    }
    assert "apiKey" not in provider


def test_api_key_env_name_is_deterministic_and_collision_resistant_across_sessions():
    name_a = _api_key_env_name("session-a")
    name_b = _api_key_env_name("session-b")
    assert name_a == _api_key_env_name("session-a")
    assert name_a != name_b
    assert name_a.isidentifier()


def test_write_agent_dir_never_writes_the_literal_secret_and_scopes_it_to_an_env_var():
    agent_dir, extra_env = _write_agent_dir(
        "http://host/v1", "sk-literal-secret", "model-x", "sess-1"
    )
    try:
        models_doc = json.loads((agent_dir / "models.yml").read_text())
        provider = models_doc["providers"]["toy-local"]
        env_var_name = _api_key_env_name("sess-1")
        assert provider["apiKey"] == env_var_name
        assert "sk-literal-secret" not in json.dumps(models_doc)
        assert extra_env == {env_var_name: "sk-literal-secret"}
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)


def test_write_agent_dir_without_api_key_returns_no_extra_env():
    agent_dir, extra_env = _write_agent_dir("http://host/v1", None, "model-x", "sess-1")
    try:
        assert extra_env == {}
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)


def test_write_agent_dir_cleans_up_the_created_directory_when_the_write_fails(monkeypatch):
    """``mkdtemp()`` can succeed and create the directory before a later
    write fails; the directory must not leak just because the caller never
    got a path back to remember it by (session/omp.py Finding 4)."""
    created = {}
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created["path"] = Path(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

    def _boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)

    with pytest.raises(OSError, match="disk full"):
        _write_agent_dir("http://host/v1", "secret", "model-x", "sess-1")

    assert "path" in created
    assert not created["path"].exists()


# ---------------------------------------------------------------------------
# set_host_tools -> host_tool_call -> host_tool_result: read-class tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_tool_call_round_trip_for_a_read_class_tool():
    session, fake, recorder, broker = await _started_session()
    try:
        tool = fake.tool("get_view")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, tool.execute, {"q": "hello"}, None)
        assert result == {"content": [{"type": "text", "text": "read-ok:hello"}], "details": {}}
        # A read-class tool never touches the broker: no PermissionRequest
        # of any kind should appear.
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_host_tool_call_reports_a_handler_failure_as_a_wire_level_error():
    """``is_error`` from the handler must become an ``isError``-shaped
    exception, not a "successful" result that happens to contain the word
    "failed" -- the same invariant ``tools.py``'s ``fail()``
    documents for the Claude backend."""
    session, fake, recorder, broker = await _started_session(
        tool_table=_tool_table(write_handler=_failing_write_handler)
    )
    try:
        tool = fake.tool("add_note")
        loop = asyncio.get_running_loop()
        execute_future = loop.run_in_executor(None, tool.execute, {"path": "a"}, None)
        request = await recorder.next()
        await session.decide_permission(request.request_id, "allow")
        with pytest.raises(RuntimeError, match="could not write"):
            await execute_future
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# extension_ui_request{confirm}: the defensive second gate, keyed off the
# in-flight tool tracked from tool_execution_start/_end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_round_trip_allows_a_write_class_tool():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.push_tool_execution_start("call-1", "add_note", {"path": "x"})
        tool_use = await recorder.next()
        assert isinstance(tool_use, ToolUse)
        assert tool_use.name == "add_note"

        loop = asyncio.get_running_loop()
        confirm_future = loop.run_in_executor(
            None, fake.push_ui_request, FakeUiRequest("ui-1", "confirm")
        )

        request = await recorder.next()
        assert isinstance(request, PermissionRequest)
        assert request.tool == "add_note"

        await session.decide_permission(request.request_id, "allow")
        await confirm_future

        resolved = await recorder.next()
        assert isinstance(resolved, PermissionResolved)
        assert resolved.outcome == "allow"
        assert fake.confirmations == [("ui-1", True)]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_confirm_round_trip_denies_a_write_class_tool():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.push_tool_execution_start("call-1", "add_note", {"path": "x"})
        await recorder.next()  # ToolUse

        loop = asyncio.get_running_loop()
        confirm_future = loop.run_in_executor(
            None, fake.push_ui_request, FakeUiRequest("ui-1", "confirm")
        )

        request = await recorder.next()
        assert isinstance(request, PermissionRequest)

        await session.decide_permission(request.request_id, "deny", "not now")
        await confirm_future

        resolved = await recorder.next()
        assert isinstance(resolved, PermissionResolved)
        assert resolved.outcome == "deny"
        assert fake.confirmations == [("ui-1", False)]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_confirm_with_no_tool_in_flight_declines_rather_than_guessing():
    session, fake, recorder, broker = await _started_session()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fake.push_ui_request, FakeUiRequest("ui-1", "confirm"))
        assert fake.confirmations == [("ui-1", False)]
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_confirm_with_two_tools_in_flight_declines_rather_than_guessing():
    """A ``confirm`` frame carries no tool-call id on the wire; with two
    concurrent host tool calls pending, this handler must not pick either
    one -- session/omp.py's Finding 2 regression case."""
    session, fake, recorder, broker = await _started_session()
    try:
        fake.push_tool_execution_start("call-1", "add_note", {"path": "x"})
        await recorder.next()  # ToolUse for call-1
        fake.push_tool_execution_start("call-2", "add_note", {"path": "y"})
        await recorder.next()  # ToolUse for call-2

        events_before = len(recorder.all)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fake.push_ui_request, FakeUiRequest("ui-1", "confirm"))

        assert fake.confirmations == [("ui-1", False)]
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all[events_before:])
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_non_confirm_ui_requests_are_answered_without_hanging():
    session, fake, recorder, broker = await _started_session()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fake.push_ui_request, FakeUiRequest("ui-2", "select"))
        assert fake.cancellations == ["ui-2"]
        # A passive notification (no response ever required) is left alone.
        await loop.run_in_executor(None, fake.push_ui_request, FakeUiRequest("ui-3", "notify"))
        assert fake.cancellations == ["ui-2"]
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# the ticket's central adversarial case: an already-granted write-class tool
# needs no extension_ui_request at all -- the broker's own granted-tools
# check runs, and stops there, before anything human-facing exists.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_already_granted_write_class_tool_needs_no_extension_ui_request_at_all():
    session, fake, recorder, broker = await _started_session()
    try:
        tool = fake.tool("add_note")
        loop = asyncio.get_running_loop()

        # First call: an ordinary ask, answered allow_always.
        execute_future = loop.run_in_executor(None, tool.execute, {"path": "a"}, None)
        request = await recorder.next()
        assert isinstance(request, PermissionRequest)
        await session.decide_permission(request.request_id, "allow_always")
        result = await execute_future
        assert result["content"][0]["text"] == "wrote:a"

        events_before = len(recorder.all)

        # Second call: same tool, now broker-granted. No new
        # PermissionRequest may appear, and in particular no
        # extension_ui_request{confirm} round trip happens either -- the
        # gate lives entirely inside broker.ask(), which this second call
        # never even lets create a request in the first place.
        result2 = await loop.run_in_executor(None, tool.execute, {"path": "b"}, None)
        assert result2["content"][0]["text"] == "wrote:b"
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all[events_before:])
        assert fake.confirmations == []
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# resolving one of the human's review comments: asked once per call through
# the omp host-tool path, whatever the product graded it, and never
# remembered (review/tools.py).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("grade", ["read", "view", "write"])
async def test_resolving_a_human_comment_asks_once_per_call_and_is_never_remembered(
    tmp_path, grade
):
    from toy_product import TOY_PAUSED_MESSAGE, toy_review_store

    from annealage_agent.review.tools import review_tools
    from annealage_agent.tools import Grading, ToolServer

    store = toy_review_store(tmp_path)
    for text in ("one", "two"):
        store.add_comment(anchor={"card": "back", "x": 1, "y": 1}, text=text, author="human")
    store.add_comment(anchor={"card": "back", "x": 2, "y": 2}, text="mine", author="model")

    names = {"read": ["list_comments"], "view": ["add_callout"], "write": ["delete_callout"]}
    names[grade].append("resolve_comment")
    bus = SimpleNamespace(paused=False, review_store=store, broker=None)
    server = ToolServer(
        review_tools(store, bus=bus),
        grading=Grading(*(tuple(names[g]) for g in ("read", "view", "write"))),
        bus=bus,
        paused_message=TOY_PAUSED_MESSAGE,
    )
    # The broker's own recorder: its events are the cards.
    recorder = EventRecorder()
    broker = PermissionBroker(
        recorder, timeout=2.0, no_viewer_grace=0.05, never_remembered=server.never_remembered
    )
    bus.broker = broker
    session, fake, _session_events, _broker = await _started_session(
        tool_table=server.tool_table(), broker=broker
    )
    try:
        execute = fake.tool("resolve_comment").execute
        loop = asyncio.get_running_loop()

        first = loop.run_in_executor(None, execute, {"id": 1}, None)
        request = await recorder.next()
        assert isinstance(request, PermissionRequest)
        assert (request.tool, request.rememberable) == ("mcp__toy__resolve_comment", False)
        await session.decide_permission(request.request_id, "allow_always")
        await first
        assert isinstance(await recorder.next(), PermissionResolved)

        # The next human comment is asked about again, exactly once.
        second = loop.run_in_executor(None, execute, {"id": 2}, None)
        again = await recorder.next()
        assert isinstance(again, PermissionRequest)
        await session.decide_permission(again.request_id, "allow")
        await second
        assert isinstance(await recorder.next(), PermissionResolved)

        # The model's own callout resolves with no card.
        before = len(recorder.all)
        await loop.run_in_executor(None, execute, {"id": 3}, None)
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all[before:])
        assert store.list_comments(status="open").comments == ()
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# no-viewer grace path: nobody is left to answer, so a write-class call must
# deny immediately, without ever creating a card.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_viewer_denies_a_write_class_tool_call_without_a_card():
    session, fake, recorder, broker = await _started_session(viewer_count=0)
    try:
        tool = fake.tool("add_note")
        loop = asyncio.get_running_loop()
        with pytest.raises(RuntimeError):
            await loop.run_in_executor(None, tool.execute, {"path": "a"}, None)
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_viewer_that_connects_and_then_disconnects_still_denies_confirm():
    session, fake, recorder, broker = await _started_session(viewer_count=1)
    try:
        session.on_viewer_presence(0)
        fake.push_tool_execution_start("call-1", "add_note", {"path": "x"})
        await recorder.next()  # ToolUse

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fake.push_ui_request, FakeUiRequest("ui-1", "confirm"))
        assert fake.confirmations == [("ui-1", False)]
        assert not any(isinstance(e, PermissionRequest) for e in recorder.all)
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# event mapping: message_update/tool_execution_* -> the agent layer's own AgentEvents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_delta_becomes_one_text_delta_event():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.push_message_update({"type": "text_delta", "delta": "hello"})
        event = await recorder.next()
        assert isinstance(event, TextDelta)
        assert event.text == "hello"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_tool_execution_events_become_tool_use_and_tool_result():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.push_tool_execution_start("call-1", "get_view", {"q": "x"})
        tool_use = await recorder.next()
        assert isinstance(tool_use, ToolUse)
        assert tool_use.name == "get_view"
        assert tool_use.input == {"q": "x"}

        fake.push_tool_execution_end(
            "call-1", "get_view", {"content": [{"type": "text", "text": "ok"}]}, is_error=False
        )
        tool_result = await recorder.next()
        assert isinstance(tool_result, ToolResult)
        assert tool_result.text == "ok"
        assert tool_result.is_error is False
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_stops_the_client_and_removes_the_agent_dir():
    session, fake, recorder, broker = await _started_session()
    agent_dir = Path(fake.kwargs["env"]["PI_CODING_AGENT_DIR"])
    assert agent_dir.is_dir()
    await session.close()
    assert fake.stopped is True
    assert session.agent_status() == AGENT_UNAVAILABLE
    assert not agent_dir.exists()


@pytest.mark.asyncio
async def test_submit_turn_sends_the_prompt_and_returns_without_waiting_for_agent_end():
    session, fake, recorder, broker = await _started_session()
    try:
        await session.submit_turn([{"type": "text", "text": "hi there"}])
        assert len(fake.prompt_calls) == 1
        assert fake.prompt_calls[0].message == "hi there"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_interrupt_denies_every_pending_request_before_aborting():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.push_tool_execution_start("call-1", "add_note", {"path": "x"})
        await recorder.next()  # ToolUse

        loop = asyncio.get_running_loop()
        confirm_future = loop.run_in_executor(
            None, fake.push_ui_request, FakeUiRequest("ui-1", "confirm")
        )
        await recorder.next()  # PermissionRequest

        await session.interrupt()
        assert fake.abort_calls == 1
        await confirm_future
        assert fake.confirmations == [("ui-1", False)]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_model_calls_the_rpc_set_model_with_the_provider_and_model():
    """``RpcClient.set_model(provider, model_id)`` (``omp://rpc.md``'s
    ``{type: "set_model", provider, modelId}`` wire shape), run through the
    same ``_run_blocking`` executor bridge every other one-shot
    ``RpcClient`` call in this file uses."""
    session, fake, recorder, broker = await _started_session()
    try:
        await session.set_model("llama-70b")
        assert fake.set_model_calls == [(_provider_id(), "llama-70b")]
        event = await recorder.next()
        assert isinstance(event, AgentModelChanged)
        assert event.model == "llama-70b"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_model_to_the_current_model_is_a_no_op():
    session, fake, recorder, broker = await _started_session(model="llama-70b")
    try:
        await session.set_model("llama-70b")
        assert fake.set_model_calls == []
        assert not any(isinstance(e, AgentModelChanged) for e in recorder.all)
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# steering, refused messages, and what a turn end reports
# ---------------------------------------------------------------------------


class RpcCommandError(Exception):
    """Named like ``omp_rpc``'s: a command omp answered with an error."""


class RpcProcessExitError(Exception):
    """Named like ``omp_rpc``'s: the omp process is gone."""


async def _next_of(recorder, kind):
    while True:
        event = await recorder.next()
        if isinstance(event, kind):
            return event


def _text(text):
    return [{"type": "text", "text": text}]


@pytest.mark.asyncio
async def test_a_message_sent_while_a_turn_runs_steers_it_and_the_session_stays_ready():
    session, fake, recorder, broker = await _started_session()
    try:
        await session.submit_turn(_text("draw the regulator"))
        fake.push_message_update({"type": "text_delta", "delta": "Working"})
        assert (await _next_of(recorder, TextDelta)).turn == 1

        await session.submit_turn(_text("use the LDO instead"))
        # Both as a steer-capable prompt: omp starts a turn with the first and
        # redirects the running one with the second, never refusing either.
        assert [(c.message, c.streaming_behavior) for c in fake.prompt_calls] == [
            ("draw the regulator", "steer"),
            ("use the LDO instead", "steer"),
        ]
        steered = await _next_of(recorder, TurnEnd)
        assert (steered.turn, steered.stop_reason) == (1, "steered")
        assert session.agent_status() == AGENT_READY

        # The agent's further output belongs to the steering message's turn,
        # which omp's one agent_end for the whole run ends.
        fake.push_message_update({"type": "text_delta", "delta": "Switching"})
        assert (await _next_of(recorder, TextDelta)).turn == 2
        fake.push_agent_end()
        ended = await _next_of(recorder, TurnEnd)
        assert (ended.turn, ended.stop_reason) == (2, "end")

        # Idle again: the next message ends nothing before it starts.
        await session.submit_turn(_text("thanks"))
        assert [e.turn for e in recorder.all if isinstance(e, TurnEnd)] == [1, 2]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_refused_message_is_reported_and_leaves_the_session_ready():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.prompt_error = RpcCommandError("No model selected")
        await session.submit_turn(_text("hello"))
        error = await _next_of(recorder, AgentError)
        assert error.stderr == "No model selected"
        assert "still ready" in error.remediation
        # The turn the message would have started is closed, so the pane is
        # not left waiting on it.
        ended = await _next_of(recorder, TurnEnd)
        assert (ended.turn, ended.stop_reason) == (1, "rejected")
        assert session.agent_status() == AGENT_READY

        fake.prompt_error = None
        await session.submit_turn(_text("hello again"))
        assert [c.message for c in fake.prompt_calls] == ["hello again"]
        assert not any(isinstance(e, TurnEnd) and e.stop_reason == "steered" for e in recorder.all)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_message_refused_after_omp_acknowledged_it_is_reported_the_same_way():
    """``omp://rpc.md``: a prompt's scheduling can fail after its immediate
    success response, as an error response nothing is waiting on."""
    session, fake, recorder, broker = await _started_session()
    try:
        await session.submit_turn(_text("hello"))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fake.push_protocol_error, "set_model", "unrelated")
        await loop.run_in_executor(None, fake.push_protocol_error, "prompt", "quota exhausted")
        error = await _next_of(recorder, AgentError)
        assert error.stderr == "quota exhausted"
        ended = await _next_of(recorder, TurnEnd)
        assert (ended.turn, ended.stop_reason) == (1, "rejected")
        assert session.agent_status() == AGENT_READY
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_only_a_dead_omp_process_makes_a_sent_message_mark_the_session_unavailable():
    session, fake, recorder, broker = await _started_session()
    try:
        fake.prompt_error = RpcProcessExitError("RPC process stopped")
        await session.submit_turn(_text("hello"))
        assert session.agent_status() == AGENT_UNAVAILABLE
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_turn_end_reports_that_turn_s_cost_and_tokens_from_the_session_stats():
    session, fake, recorder, broker = await _started_session()
    try:
        await session.submit_turn(_text("one"))
        fake.cost = 0.25
        fake.tokens = {"input": 1000, "output": 200, "cache_read": 50, "cache_write": 10}
        fake.push_agent_end()
        first = await _next_of(recorder, TurnEnd)
        assert first.cost_usd == pytest.approx(0.25)
        assert first.tokens == {"input": 1000, "output": 200, "cache_read": 50, "cache_write": 10}

        await session.submit_turn(_text("two"))
        fake.cost = 0.40
        fake.tokens = {"input": 1600, "output": 260, "cache_read": 950, "cache_write": 10}
        fake.push_agent_end()
        second = await _next_of(recorder, TurnEnd)
        # This turn's share, not the session's running total.
        assert second.cost_usd == pytest.approx(0.15)
        assert second.tokens == {"input": 600, "output": 60, "cache_read": 900, "cache_write": 0}
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# the conversation file: recorded, and resumed by switch_session
# ---------------------------------------------------------------------------


def _omp_launch(tmp_path, recorder, *, resumed, base_url):
    """``launch.build_session``'s omp session for ``tmp_path``'s session."""
    sid = sessions.resolve_continue(tmp_path) if resumed else sessions.create_session(tmp_path)
    sessions.record_last_session(tmp_path, sid)
    tools = SimpleNamespace(
        host_tool_table=_tool_table, never_remembered=(), remote_instructions=None
    )
    return launch.build_session(
        "omp",
        recorder,
        bus=SimpleNamespace(tools=tools, broker=None, url="http://127.0.0.1:8765/"),
        serve_dir=tmp_path,
        session_id=sid,
        resumed=resumed,
        settings={"model": None, "omp_base_url": base_url, "omp_api_key": None},
        mcp_host="127.0.0.1",
        mcp_port=8765,
        agent_token="agent-token",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", [None, "http://127.0.0.1:11434/v1"])
async def test_the_conversation_file_is_recorded_and_a_resume_switches_to_it(
    tmp_path, monkeypatch, base_url
):
    """In both provider modes: omp's own configuration, and a custom endpoint,
    whose throwaway agent directory ``close()`` deletes and so must never hold
    the conversation."""
    clients = []

    def factory(**kwargs):
        fake = FakeRpcClient(**kwargs)
        # What omp does with --session-dir: its conversation file lives there.
        fake.session_file = str(Path(kwargs["session_dir"]) / "conversation.jsonl")
        Path(fake.session_file).write_text("{}\n")
        clients.append(fake)
        return fake

    monkeypatch.setattr(omp_module, "RpcClient", factory)
    first = _omp_launch(tmp_path, EventRecorder(), resumed=False, base_url=base_url)
    await first.start()
    launched = clients[0]
    session_dir = sessions.state_dir(tmp_path) / "omp"
    assert launched.kwargs["session_dir"] == str(session_dir)
    assert "no_session" not in launched.kwargs
    conversation = launched.session_file
    sid = sessions.resolve_continue(tmp_path)
    assert sessions.get_session_info(tmp_path, sid).omp_session_file == conversation
    agent_dir = launched.kwargs["env"].get("PI_CODING_AGENT_DIR")
    assert (agent_dir is not None) == (base_url is not None)
    await first.close()
    if agent_dir is not None:
        assert not Path(agent_dir).exists()
    assert Path(conversation).is_file()

    recorder = EventRecorder()
    resumed = _omp_launch(tmp_path, recorder, resumed=True, base_url=base_url)
    await resumed.start()
    try:
        assert clients[1].switch_calls == [conversation]
        assert resumed.agent_status() == AGENT_READY
        assert not any(isinstance(e, SessionReset) for e in recorder.all)
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_a_conversation_file_omp_never_wrote_is_nothing_to_resume(tmp_path):
    """omp names its file at start and writes it with the first message, so a
    session that ended before one leaves none; that is a fresh start, not a
    failed resume."""
    session, fake, recorder, broker = await _started_session(
        session_dir=tmp_path / "omp", resume=str(tmp_path / "never-written.jsonl")
    )
    try:
        assert fake.switch_calls == []
        assert not any(isinstance(e, SessionReset) for e in recorder.all)
        assert session.agent_status() == AGENT_READY
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_conversation_omp_cannot_open_is_reported_and_the_session_starts_fresh(
    tmp_path, monkeypatch
):
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not a session\n")

    def refuse(self, session_path):
        raise RpcCommandError("Invalid session file")

    monkeypatch.setattr(FakeRpcClient, "switch_session", refuse)
    session, fake, recorder, broker = await _started_session(
        session_dir=tmp_path / "omp", resume=str(broken)
    )
    try:
        reset = next(e for e in recorder.all if isinstance(e, SessionReset))
        assert str(broken) in reset.reason and "Invalid session file" in reset.reason
        assert session.agent_status() == AGENT_READY
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# a profile and a binary of the product's own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_agent_dir_config_dir_and_binary_reach_the_omp_command_line_and_environment(
    tmp_path, monkeypatch
):
    """``PI_CONFIG_DIR`` is joined to ``$HOME`` by omp, so a config directory
    under the home is passed relative to it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    seen = {}

    def factory(**kwargs):
        # The real client builds the argv it would spawn; only running it is
        # left to the fake.
        seen["command"] = RpcClient(**kwargs).command
        seen["env"] = dict(kwargs["env"])
        return FakeRpcClient(**kwargs)

    session = OmpSession(
        EventRecorder(),
        cwd=tmp_path,
        session_id="s-1",
        tool_table=_tool_table(),
        client_factory=factory,
        agent_dir=str(tmp_path / "profile"),
        binary="/opt/omp/bin/omp",
        config_dir=str(home / "loom" / "omp-config"),
        session_dir=tmp_path / "state" / "omp",
    )
    await session.start()
    try:
        command = seen["command"]
        assert command[:3] == ("/opt/omp/bin/omp", "--mode", "rpc")
        assert command[command.index("--session-dir") + 1] == str(tmp_path / "state" / "omp")
        assert "--no-session" not in command
        assert seen["env"]["PI_CODING_AGENT_DIR"] == str(tmp_path / "profile")
        assert seen["env"]["PI_CONFIG_DIR"] == os.path.join("loom", "omp-config")
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"binary": "bin/omp"}, "absolute path"),
        (
            {"agent_dir": "/srv/loom/omp", "base_url": "http://127.0.0.1:11434/v1"},
            "PI_CODING_AGENT_DIR",
        ),
        ({"config_dir": "/etc/omp"}, "must be inside"),
        ({"config_dir": "../elsewhere"}, "must be inside"),
    ],
)
async def test_a_profile_the_omp_process_could_not_use_as_meant_is_refused(kwargs, reason):
    launched = []
    recorder = EventRecorder()
    session = OmpSession(
        recorder,
        cwd="/proj/root",
        session_id="s-1",
        tool_table=_tool_table(),
        client_factory=lambda **kw: launched.append(kw),
        **kwargs,
    )
    await session.start()
    assert launched == []
    assert session.agent_status() == AGENT_UNAVAILABLE
    assert reason in next(e for e in recorder.all if isinstance(e, AgentError)).remediation
    await session.close()


# ---------------------------------------------------------------------------
# a tool result that ends the turn, and a tool table replaced mid-session
# ---------------------------------------------------------------------------


async def _checkpoint_session():
    """A started session whose one tool, ``checkpoint``, ends the turn through
    the real ``_wrap`` and ``ViewerBus`` path."""

    async def checkpoint(args):
        return ok(text="Stop and wait for the human.", end_turn=True)

    bus = ViewerBus(None, url="http://127.0.0.1:8765/")
    wrapped = _wrap(
        SdkMcpTool(name="checkpoint", description="hand back", input_schema={}, handler=checkpoint),
        bus=bus,
        gated=False,
        paused_message="paused",
    )
    table = dict(_tool_table())
    table["checkpoint"] = ToolSpec(
        schema={"type": "object", "properties": {}},
        description="hand back",
        handler=wrapped.handler,
        write=False,
    )
    session, fake, recorder, broker = await _started_session(tool_table=table)
    bus.end_turn_handler = session.end_turn_after_tool
    return session, fake, recorder


async def _run_tool(fake, name, call_id, params=None):
    """Run host tool ``name`` as omp's per-call thread would, for ``call_id``."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, fake.tool(name).execute, params or {}, SimpleNamespace(tool_call_id=call_id)
    )


async def _settle(fake, expected_aborts):
    for _ in range(50):
        if fake.abort_calls >= expected_aborts:
            break
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_tool_result_carrying_end_turn_aborts_the_turn_once_it_is_delivered():
    session, fake, recorder = await _checkpoint_session()
    try:
        await session.submit_turn(_text("draw it, then check in"))
        result = await _run_tool(fake, "checkpoint", "call-cp")
        # What omp receives is an ordinary result; the request travels apart.
        assert result == {
            "content": [{"type": "text", "text": "Stop and wait for the human."}],
            "details": {},
        }
        # A sibling tool running in parallel finishes first: not the result
        # that asked, so nothing stops yet.
        sibling = await _run_tool(fake, "get_view", "call-2", {"q": "x"})
        fake.push_tool_execution_end("call-2", "get_view", sibling)
        await _next_of(recorder, ToolResult)
        await _settle(fake, 1)
        assert fake.abort_calls == 0
        # Once omp has the checkpoint's own result, the turn is stopped.
        fake.push_tool_execution_end("call-cp", "checkpoint", result)
        await _next_of(recorder, ToolResult)
        await _settle(fake, 1)
        assert fake.abort_calls == 1
        fake.push_agent_end()
        ended = await _next_of(recorder, TurnEnd)
        assert (ended.turn, ended.stop_reason) == (1, "ended_by_tool")
        assert session.agent_status() == AGENT_READY
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_an_end_turn_request_whose_result_never_arrived_does_not_end_a_later_turn():
    session, fake, recorder = await _checkpoint_session()
    try:
        await session.submit_turn(_text("check in"))
        await _run_tool(fake, "checkpoint", "call-cp")
        # The human interrupts; omp never reports that call's end, and the
        # run settles.
        fake.push_agent_end()
        await _next_of(recorder, TurnEnd)

        await session.submit_turn(_text("carry on"))
        result = await _run_tool(fake, "get_view", "call-cp", {"q": "x"})
        fake.push_tool_execution_end("call-cp", "get_view", result)
        await _next_of(recorder, ToolResult)
        await _settle(fake, 1)
        assert fake.abort_calls == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_run_s_end_that_the_loop_sees_after_a_new_message_settles_only_its_own_turn():
    """omp ends run 1, and a new message's submit begins before the loop has
    handled that end: turn 1 ends as steered, and the late settle must not
    close turn 2, whose own run settles it."""
    session, fake, recorder, broker = await _started_session()
    try:
        await session.submit_turn(_text("one"))
        fake.push_agent_end()  # queued on the loop, not yet handled
        await session.submit_turn(_text("two"))
        await asyncio.sleep(0.05)
        ends = [(e.turn, e.stop_reason) for e in recorder.all if isinstance(e, TurnEnd)]
        assert ends == [(1, "steered")]
        fake.push_agent_end()
        await asyncio.sleep(0.05)
        ends = [(e.turn, e.stop_reason) for e in recorder.all if isinstance(e, TurnEnd)]
        assert ends == [(1, "steered"), (2, "end")]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_tool_table_registers_the_new_tools_with_the_running_omp():
    session, fake, recorder, broker = await _started_session()
    try:
        table = dict(_tool_table())
        table["ds-wiki__search_parts"] = ToolSpec(
            schema={"type": "object", "properties": {}},
            description="search",
            handler=_read_handler,
            write=False,
        )
        assert await session.set_tool_table(table) is True
        (registered,) = fake.set_custom_tools_calls
        assert {t.name for t in registered} == {"get_view", "add_note", "ds-wiki__search_parts"}
    finally:
        await session.close()
