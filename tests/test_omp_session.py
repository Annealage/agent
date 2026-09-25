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
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from annealage_agent.session.base import (
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentModelChanged,
    AgentStatus,
    PermissionRequest,
    PermissionResolved,
    TextDelta,
    ToolResult,
    ToolUse,
)
from annealage_agent.session.omp import (
    OmpSession,
    _api_key_env_name,
    _build_custom_provider,
    _provider_id,
    _write_agent_dir,
)
from annealage_agent.session.permissions import PermissionBroker
from annealage_agent.tools import ToolSpec


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
        self._listeners = {}
        self.session_id = "omp-sess-1"

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def get_state(self):
        return SimpleNamespace(session_id=self.session_id)

    def prompt(self, message, *, images=None):
        self.prompt_calls.append(SimpleNamespace(message=message, images=images))

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
