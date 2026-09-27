"""Tests for ``session/sdk.py``, driving the real ``ClaudeSDKClient`` through
a fake ``Transport`` (plan section 2, fact 11: a six-method ABC, injectable
as ``ClaudeSDKClient(options=..., transport=...)``).

This is the only file in the suite that imports ``claude_agent_sdk``'s
client directly rather than going through ``session/fake.py``, and it
exists as a canary: if an SDK upgrade renames a message field, changes
what a ``PermissionResult`` must look like, or reshapes the control
protocol's wire dicts, a test here breaks before that reaches a running
agent. What it pins matters more than how much of ``sdk.py`` it exercises.

``FakeTransport`` never spawns a process and never opens a socket; that is
true by construction, not by hoping, because it is a plain ``Transport``
subclass backed by an ``asyncio.Queue`` and its ``connect()`` never calls
anything outside this module. The ``_no_real_transport`` fixture below is a
second, independent guarantee: it makes the SDK's own
``SubprocessCLITransport`` raise if anything ever falls through to it, which
would otherwise mean a real ``claude`` binary got spawned by a test.

``EventRecorder`` gives each test an ``asyncio.Queue`` of the ``AgentEvent``
objects ``SdkSession`` emits, alongside a plain list of everything seen so
far, so a test awaits the next event deterministically instead of guessing
how many ``asyncio.sleep(0)`` hops the fake protocol layers need.
"""

import asyncio
import json

import pytest
from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    Transport,
)
from claude_agent_sdk.types import PermissionRuleValue
from conftest import create_toy_app
from toy_product import build_toy_tools

from annealage_agent.session import sdk as sdk_module
from annealage_agent.session.base import (
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentModelChanged,
    AgentStatus,
    TextDelta,
    ToolResult,
    ToolUse,
    TurnEnd,
    Usage,
)
from annealage_agent.session.permissions import Decision
from annealage_agent.session.sdk import SandboxStatus, SdkSession

# The toy tools that never prompt, in the namespaced form fact 1 requires
# (``mcp__<server>__<tool>``). Hardcoded here rather than read off the tool
# server: this test exists to notice if the list the session is built with
# drifts, so it must not share its source of truth with the code it is
# pinning.
EXPECTED_PRE_ALLOWED_TOOLS = [
    # Read-class: changes nothing.
    "mcp__toy__list_notes",
    "mcp__toy__get_view",
    # View-class: changes only what is on the screen the human is watching, so
    # the pause switch is the control rather than a card per view change.
    "mcp__toy__set_view",
]

# The two that leave something on disk, and therefore must NOT be here. Listed
# so this file states the negative rather than leaving it to be inferred from
# what is missing above: one of these appearing in ``allowed_tools`` would
# silently remove the human's approval card and nothing else would notice.
EXPECTED_NEVER_PRE_ALLOWED = [
    "mcp__toy__add_note",
    "mcp__toy__clear_notes",
]

# The posture from the M5 brief's "decided and not open" section: bash runs
# contained and unprompted, edit/write/network still reach the broker, and
# git is excluded because it needs the real filesystem for the project's own
# repository.
EXPECTED_SANDBOX_SETTINGS = {
    "enabled": True,
    "autoAllowBashIfSandboxed": True,
    "excludedCommands": ["git"],
    # False, so the model cannot drop containment for a command by asking. The
    # option defaults to true, which is why it is set explicitly and asserted
    # here rather than left off.
    "allowUnsandboxedCommands": False,
}


@pytest.fixture(autouse=True)
def _no_real_transport(monkeypatch):
    """Fail loudly, rather than spawning a real ``claude`` subprocess, if
    anything in this module's code path ever falls through to the SDK's own
    ``SubprocessCLITransport`` instead of the fake one a test injected."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "a real SubprocessCLITransport was constructed; the fake "
            "Transport passed to SdkSession was not used"
        )

    monkeypatch.setattr(subprocess_cli, "SubprocessCLITransport", _forbidden)


class FakeTransport(Transport):
    """A ``Transport`` backed by one ``asyncio.Queue``, with no process and
    no socket.

    ``write()`` records every outbound line, parsed from JSON, in
    ``written``, in call order, so a test can assert exactly what the SDK
    sent. It also answers the SDK's own ``initialize`` control request with
    an empty success, because ``ClaudeSDKClient.connect()`` blocks on that
    response before returning and no test here cares what the response
    contains, only that the client is unblocked.

    ``push()`` queues one raw protocol dict for the client to receive next;
    it is how a test plays the part of the CLI child's stdout.
    """

    def __init__(self):
        self.written = []
        self.connected = False
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        self.connected = True

    def is_ready(self) -> bool:
        return self.connected and not self.closed

    async def write(self, data: str) -> None:
        obj = json.loads(data)
        self.written.append(obj)
        if obj.get("type") == "control_request":
            subtype = obj["request"].get("subtype")
            if subtype in ("initialize", "set_model", "interrupt"):
                self._queue.put_nowait(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": obj["request_id"],
                            "response": {},
                        },
                    }
                )

    async def read_messages(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)

    async def end_input(self) -> None:
        pass

    def push(self, message: dict) -> None:
        self._queue.put_nowait(message)


class _FailingConnectTransport(FakeTransport):
    """A transport whose ``connect()`` fails, standing in for a ``claude``
    child that could not be started."""

    async def connect(self) -> None:
        raise RuntimeError("boom: no such host")


class EventRecorder:
    """Collects every ``AgentEvent`` an ``SdkSession`` emits, and lets a
    test await the next one deterministically instead of sleeping and
    hoping the fake protocol layers have caught up."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self.all = []

    def __call__(self, event) -> None:
        self.all.append(event)
        self._queue.put_nowait(event)

    async def next(self, timeout: float = 2.0, include_status: bool = False):
        """The next event, skipping ``AgentStatus`` unless asked for it.

        Status is ambient rather than conversational: it is announced whenever
        the session's own state changes, so it can appear between any two events
        a test cares about. Skipping it here keeps every test that awaits "the
        next event" reading as what it means, and the transitions themselves are
        pinned directly by the tests that assert on ``recorder.all``.
        """
        while True:
            event = await asyncio.wait_for(self._queue.get(), timeout)
            if include_status or not isinstance(event, AgentStatus):
                return event


async def _started_session(**kwargs):
    """A connected ``SdkSession`` over a fresh ``FakeTransport``, with its
    events going to a fresh ``EventRecorder``."""
    transport = kwargs.pop("transport", None) or FakeTransport()
    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd="/proj/root", session_id="toy-sess-1", transport=transport, **kwargs
    )
    await session.start()
    assert session.agent_status() == AGENT_READY
    return session, transport, recorder


def test_fake_transport_is_a_real_transport():
    """``FakeTransport`` satisfies the SDK's own ``Transport`` ABC by
    construction: instantiating a subclass missing one of the six abstract
    methods raises ``TypeError`` at construction time, so this line alone
    proves all six are implemented, before any test ever touches the
    network or a subprocess."""
    transport = FakeTransport()
    assert isinstance(transport, Transport)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_options_wired_into_the_real_client():
    """Every fact the M5 brief pins about the options ``SdkSession`` builds,
    read back off the real ``ClaudeAgentOptions`` the real ``ClaudeSDKClient``
    was constructed with, not off a private method's return value: this is
    the whole point of driving the real client, since an SDK upgrade that
    silently renamed a field would still let a direct call to
    ``_build_options`` "pass" while the client itself was misconfigured.

    The SDK warns that ``can_use_tool`` will never be consulted for the
    tools this test also asserts are pre-allowed; that warning states the
    intended design, not a defect, since pre-allowing them is exactly what
    keeps a view change or a note list off the broker, and is silenced here
    rather than left to clutter the run.
    """

    class _StubBroker:
        async def ask(self, *args, **kwargs):
            raise AssertionError("not exercised by this test")

        def shutdown(self):
            pass

    class _StubBus:
        paused = False

    stub_broker = _StubBroker()
    # Built the way launch.py builds a real session: the allow list is the
    # product tool server's own pre-allowed set, nothing added.
    session, transport, _recorder = await _started_session(
        broker=stub_broker,
        resume="prior-sdk-session-id",
        allowed_tools=build_toy_tools(_StubBus(), "/proj/root").pre_allowed,
    )
    try:
        options = session._client.options

        # Fact 12: without this, only whole assistant messages arrive and
        # the chat pane cannot stream tokens.
        assert options.include_partial_messages is True

        # Fact 1: the product tools that never reach the broker, in their
        # namespaced form. The two that leave something on disk are absent,
        # which is what makes each of them a card the human sees.
        assert options.allowed_tools == EXPECTED_PRE_ALLOWED_TOOLS
        for name in EXPECTED_NEVER_PRE_ALLOWED:
            assert name not in options.allowed_tools

        # Fact 13: the default loads no settings files at all, so a
        # project CLAUDE.md would be silently ignored without this.
        assert options.setting_sources == ["user", "project", "local"]

        # Plan section 3.4: a resolved id, and continue_conversation is
        # never set, because that option is scoped to the CLI's own notion
        # of cwd rather than this project root.
        assert options.resume == "prior-sdk-session-id"
        assert options.continue_conversation is False

        # The posture from the M5 brief's "decided and not open" section.
        assert options.sandbox == EXPECTED_SANDBOX_SETTINGS

        # can_use_tool is session._can_use_tool, sdk.py's own adapter, not
        # broker.ask directly: ask() returns a provider-neutral Decision
        # (session/permissions.py), and this adapter is where that gets
        # converted into what the Claude SDK requires (fact 14).
        assert options.can_use_tool == session._can_use_tool

        # The stderr hook this file's sandbox-status parsing depends on.
        assert options.stderr == session._note_stderr

        # Only the servers this process names: without this the CLI also
        # honours a `.mcp.json` in the served directory, whose entries name
        # commands to spawn.
        assert options.strict_mcp_config is True
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_product_tool_server_is_passed_through_under_its_own_name():
    """The key in ``mcp_servers`` is what the model-visible
    ``mcp__<key>__<tool>`` name is built from (fact 1), so a key that
    disagreed with the server's own name would leave every pre-allowed
    name matching nothing and every one of those tools prompting.
    The product's tool server owns that mapping; this asserts nothing
    rewrites it on the way through, and that every tool the toy classifies
    reaches the session, counted by hand."""

    class _StubBus:
        paused = False

        async def call(self, method, params=None, *, timeout=None):
            raise AssertionError("not exercised by this test")

    toy_tools = build_toy_tools(_StubBus(), "/proj/root")
    session, _transport, _recorder = await _started_session(mcp_servers=toy_tools.mcp_servers)
    try:
        servers = session._client.options.mcp_servers
        assert list(servers) == ["toy"]
        assert servers["toy"]["type"] == "sdk"
        assert servers["toy"]["name"] == "toy"
        assert len(toy_tools.tools) == 5
        # Given a server but no allow list, the session pre-allows nothing:
        # the list comes from the caller, never from a default of its own, so
        # a caller that forgot it gets a card per tool rather than no cards.
        assert session._client.options.allowed_tools == []
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_no_tripwire_hook_without_an_accepted_digest():
    """A session given no accepted digest carries no configuration tripwire, so a
    caller that never ran the gate does not get a control that would deny every
    call for want of a digest to compare against. The credential refusal beside
    it is unconditional and stays."""
    session, _transport, _recorder = await _started_session()
    try:
        # Hooks are installed either way, because the credential refusal is
        # unconditional; what a session with no accepted digest must not carry
        # is the configuration tripwire, which would have nothing to compare a
        # digest against.
        matchers = session._client.options.hooks["PreToolUse"]
        assert matchers[0].hooks == [session._guard_secret_paths]
        assert session._guard_config not in matchers[0].hooks
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_tripwire_is_installed_as_a_pre_tool_use_hook_matching_every_tool():
    """PreToolUse is the only event early enough: a settings allow rule is
    applied before ``can_use_tool``, and the sandbox waves some bash through
    without consulting it at all, so a control bound to either would miss
    exactly the calls that matter."""
    session, _transport, _recorder = await _started_session(trusted_config_digest="accepted-digest")
    try:
        hooks = session._client.options.hooks
        assert list(hooks) == ["PreToolUse"]
        matchers = hooks["PreToolUse"]
        assert len(matchers) == 1
        assert matchers[0].matcher is None
        assert matchers[0].hooks == [session._guard_config, session._guard_secret_paths]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_the_tripwire_says_nothing_while_the_config_is_unchanged(tmp_path):
    """Returning ``{}`` leaves the permission flow exactly as it would be with
    no hook at all, which is what keeps contained bash running unprompted."""
    from annealage_agent.session import workspace_trust as wt

    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd=tmp_path, session_id="s", trusted_config_digest=wt.config_digest(tmp_path)
    )
    result = await session._guard_config({"tool_name": "Bash"}, "tu_1", None)
    assert result == {}
    assert recorder.all == []


@pytest.mark.asyncio
async def test_the_tripwire_denies_every_tool_once_the_config_changes(tmp_path):
    from annealage_agent.session import workspace_trust as wt

    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd=tmp_path, session_id="s", trusted_config_digest=wt.config_digest(tmp_path)
    )
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.local.json").write_text('{"permissions":{"allow":["Bash"]}}')

    result = await session._guard_config({"tool_name": "Bash"}, "tu_1", None)
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert "configuration" in output["permissionDecisionReason"]

    # The human is told too, since a model that reports being blocked is not a
    # channel the human can rely on.
    error = await recorder.next()
    assert isinstance(error, AgentError)
    assert "no longer matches" in error.stderr


@pytest.mark.asyncio
async def test_the_change_is_reported_to_the_human_only_once(tmp_path):
    """The model retries a denied call, so a report per attempt would bury the
    pane in copies of one fact."""
    from annealage_agent.session import workspace_trust as wt

    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd=tmp_path, session_id="s", trusted_config_digest=wt.config_digest(tmp_path)
    )
    (tmp_path / ".mcp.json").write_text('{"mcpServers":{"x":{"command":"sh"}}}')
    for _ in range(3):
        result = await session._guard_config({"tool_name": "Write"}, "tu", None)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert len([e for e in recorder.all if isinstance(e, AgentError)]) == 1


@pytest.mark.asyncio
async def test_a_digest_that_cannot_be_computed_denies(tmp_path, monkeypatch):
    """A control that cannot tell whether the configuration changed has to
    assume it did."""
    from annealage_agent.session import workspace_trust as wt

    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd=tmp_path, session_id="s", trusted_config_digest="accepted-digest"
    )
    monkeypatch.setattr(wt, "config_digest", lambda root: (_ for _ in ()).throw(OSError("boom")))
    result = await session._guard_config({"tool_name": "Read"}, "tu_1", None)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_stream_event_text_delta_becomes_one_text_delta_event():
    """A ``StreamEvent`` carrying a ``content_block_delta``/``text_delta``
    becomes exactly one ``TextDelta`` event with that text."""
    session, transport, recorder = await _started_session()
    try:
        transport.push(
            {
                "type": "stream_event",
                "uuid": "u1",
                "session_id": "sdk-sess-1",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello there"},
                },
            }
        )
        event = await recorder.next()
        assert isinstance(event, TextDelta)
        assert event.text == "Hello there"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_tool_use_and_its_result_become_tool_use_and_tool_result_events():
    """An ``AssistantMessage`` carrying a ``tool_use`` block becomes a
    ``ToolUse`` event, and the ``UserMessage`` echoing its ``tool_result``
    becomes a ``ToolResult`` event correlated by the same
    ``tool_use_id``.

    These arrive on the pump independently of anything this session sent,
    exactly as they would after a turn was already sent by some other
    means, so this test pushes them straight in rather than going through
    ``submit_turn`` (which has its own defect, pinned separately below).
    """
    session, transport, recorder = await _started_session()
    try:
        transport.push(
            {
                "type": "assistant",
                "session_id": "sdk-sess-1",
                "message": {
                    "model": "claude-x",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "mcp__toy__get_view",
                            "input": {"zoom": 2},
                        }
                    ],
                },
            }
        )
        tool_use = await recorder.next()
        assert isinstance(tool_use, ToolUse)
        assert tool_use.tool_use_id == "tu_1"
        assert tool_use.name == "mcp__toy__get_view"
        assert tool_use.input == {"zoom": 2}

        transport.push(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "captured",
                            "is_error": False,
                        }
                    ],
                },
            }
        )
        tool_result = await recorder.next()
        assert isinstance(tool_result, ToolResult)
        assert tool_result.tool_use_id == "tu_1"
        assert tool_result.is_error is False
        assert tool_result.text == "captured"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_submit_turn_sends_the_turn_to_the_real_client():
    """Pinning a defect, left failing on purpose: ``submit_turn`` never
    successfully delivers a turn to a real ``ClaudeSDKClient``.

    ``submit_turn`` calls ``self._client.query({"type": "user", ...})``,
    a plain dict. ``ClaudeSDKClient.query()``'s ``prompt`` parameter is
    typed ``str | AsyncIterable[dict]``: it checks
    ``isinstance(prompt, str)`` and, when that is false, falls through to
    ``async for msg in prompt:`` on the assumption that anything else
    passed in is already an async iterable of messages. A plain dict has
    no ``__aiter__``, so that line raises
    ``TypeError: 'async for' requires an object with __aiter__ method,
    got dict`` before a single byte reaches the transport.

    ``submit_turn``'s own ``except Exception as exc: self._fail(exc, ...)``
    then treats that TypeError as an agent failure: it marks the whole
    session unavailable and emits an ``AgentError``, rather than the turn
    reaching the model. Every call to ``submit_turn`` against the real SDK
    hits this; ``session/fake.py``'s stand-in never calls
    ``ClaudeSDKClient.query()`` at all, so nothing exercises this path
    outside a test built on the real client, which is what this file is
    for.

    Once fixed (wrapping the message in a one-item async generator, or
    passing the string form ``query()`` accepts directly), this assertion
    should hold: the session stays ready and the wire carries the turn.
    """
    session, transport, recorder = await _started_session()
    try:
        await session.submit_turn([{"type": "text", "text": "please look"}])
        assert session.agent_status() == AGENT_READY
        sent = transport.written[-1]
        assert sent["type"] == "user"
        assert sent["message"]["content"] == [{"type": "text", "text": "please look"}]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_model_sends_the_live_control_request_and_emits_the_change():
    """``set_model`` is a genuine control-plane call on the already-connected
    client (``ClaudeSDKClient.set_model``, which sends a
    ``{"subtype": "set_model", "model": ...}`` control request) -- no
    reconnect, no new client, no new transport."""
    session, transport, recorder = await _started_session(model="claude-opus-4")
    try:
        await session.set_model("claude-haiku-5")
        sent = [
            obj["request"]
            for obj in transport.written
            if obj.get("type") == "control_request" and obj["request"].get("subtype") == "set_model"
        ]
        assert len(sent) == 1
        assert sent[0]["model"] == "claude-haiku-5"
        assert session.agent_status() == AGENT_READY
        event = await recorder.next()
        assert isinstance(event, AgentModelChanged)
        assert event.model == "claude-haiku-5"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_model_to_the_current_model_is_a_no_op():
    """No point in a live control-plane round trip, or a redundant event,
    for a switch that changes nothing."""
    session, transport, recorder = await _started_session(model="claude-opus-4")
    try:
        await session.set_model("claude-opus-4")
        assert not any(
            obj.get("type") == "control_request" and obj["request"].get("subtype") == "set_model"
            for obj in transport.written
        )
        assert not any(isinstance(event, AgentModelChanged) for event in recorder.all)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_result_message_becomes_turn_end_with_its_cost():
    """A ``ResultMessage`` becomes a ``TurnEnd`` event carrying the stop
    reason and the per-turn cost the plan says belongs in the UI.

    Pushed straight in rather than following a real ``submit_turn`` call
    (which has its own defect, pinned separately below): this test is
    about what the pump does with a ``result`` message, not about how one
    gets sent.
    """
    session, transport, recorder = await _started_session()
    try:
        session._turn = 1
        transport.push(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 1200,
                "duration_api_ms": 900,
                "is_error": False,
                "num_turns": 1,
                "session_id": "sdk-sess-1",
                "stop_reason": "end_turn",
                "total_cost_usd": 0.031,
            }
        )
        end = await recorder.next()
        assert isinstance(end, TurnEnd)
        assert end.turn == 1
        assert end.stop_reason == "end_turn"
        assert end.cost_usd == 0.031
    finally:
        await session.close()


def _result(total_cost_usd, model_usage=None):
    message = {
        "type": "result",
        "subtype": "success",
        "duration_ms": 1200,
        "duration_api_ms": 900,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sdk-sess-1",
        "stop_reason": "end_turn",
        "total_cost_usd": total_cost_usd,
    }
    if model_usage is not None:
        message["modelUsage"] = model_usage
    return message


def _model_usage(input, output, cache_read, cache_write, window=200000):
    return {
        "inputTokens": input,
        "outputTokens": output,
        "cacheReadInputTokens": cache_read,
        "cacheCreationInputTokens": cache_write,
        "webSearchRequests": 0,
        "costUSD": 0.0,
        "contextWindow": window,
        "maxOutputTokens": 32000,
    }


@pytest.mark.asyncio
async def test_usage_after_a_result_is_the_cli_s_running_total_for_the_conversation():
    """The CLI's result carries running totals for the conversation, which
    it restores on a resume (checked against the bundled CLI: a resumed
    process's first result carried the earlier turns' cost and tokens), so
    ``Usage`` is those totals as they stand, never a sum of results; the
    context's fill is the last main call's prompt against its model's
    window, and a subagent's call is not the main conversation's."""
    session, transport, recorder = await _started_session(resume="prior-sdk-session-id")
    try:
        session._turn = 1
        transport.push(
            {
                "type": "assistant",
                "session_id": "sdk-sess-1",
                "message": {
                    "model": "claude-haiku-4-5-20251001",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 14564,
                        "cache_creation_input_tokens": 90,
                        "output_tokens": 34,
                    },
                },
            }
        )
        transport.push(
            {
                "type": "assistant",
                "session_id": "sdk-sess-1",
                "parent_tool_use_id": "tu_task",
                "message": {
                    "model": "claude-haiku-4-5-20251001",
                    "content": [{"type": "text", "text": "a subagent"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            }
        )
        transport.push(
            _result(
                0.0339,
                {
                    "claude-haiku-4-5": _model_usage(30, 113, 29218, 14733),
                    "claude-haiku-4-5-20251001": _model_usage(898, 13, 0, 0),
                },
            )
        )
        assert isinstance(await recorder.next(), TurnEnd)
        usage = await recorder.next()
        assert isinstance(usage, Usage)
        assert usage.to_wire() == {
            "kind": "usage",
            "cost_usd": 0.0339,
            "tokens": {"input": 928, "output": 126, "cache_read": 29218, "cache_write": 14733},
            "context": {"used_tokens": 14664, "window_tokens": 200000},
        }

        # The next result's totals replace these; they are not added to them.
        session._turn = 2
        transport.push(_result(0.0351, {"claude-haiku-4-5": _model_usage(40, 150, 43900, 14800)}))
        assert isinstance(await recorder.next(), TurnEnd)
        usage = await recorder.next()
        assert usage.cost_usd == 0.0351
        assert usage.tokens["cache_read"] == 43900

        # A CLI that reports neither says nothing it does not know: null, not 0.
        session._turn = 3
        transport.push(_result(None))
        assert isinstance(await recorder.next(), TurnEnd)
        usage = (await recorder.next()).to_wire()
        assert usage["cost_usd"] is None
        assert usage["tokens"] == {
            "input": None,
            "output": None,
            "cache_read": None,
            "cache_write": None,
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_tool_result_carrying_end_turn_interrupts_the_turn_once_the_cli_has_it():
    """``tools.ok(..., end_turn=True)`` reaches the session as
    ``end_turn_after_tool`` while the tool runs; the interrupt waits for the
    tool result to come back through the stream, so the conversation keeps
    it, and the turn then ends as ``ended_by_tool``."""
    session, transport, recorder = await _started_session()

    def interrupts():
        return [
            w
            for w in transport.written
            if w.get("type") == "control_request" and w["request"].get("subtype") == "interrupt"
        ]

    try:
        session._turn = 1
        session.end_turn_after_tool()
        assert interrupts() == []
        transport.push(
            {
                "type": "user",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "wait"}]
                },
            }
        )
        assert isinstance(await recorder.next(), ToolResult)
        for _ in range(50):
            if interrupts():
                break
            await asyncio.sleep(0.01)
        assert len(interrupts()) == 1
        transport.push(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "duration_ms": 1,
                "duration_api_ms": 1,
                "is_error": False,
                "num_turns": 1,
                "session_id": "sdk-sess-1",
                "total_cost_usd": 0.01,
            }
        )
        end = await recorder.next()
        assert isinstance(end, TurnEnd)
        assert (end.turn, end.stop_reason) == (1, "ended_by_tool")
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_each_app_s_write_protected_patterns_reach_its_claude_session(tmp_path):
    """``create_app(write_protected=...)`` is the app's own set, relative to
    the directory it serves, in place of the product's (the toy protects
    nothing): what ``launch.build_session`` gives the session's hook."""
    from annealage_agent import launch, sessions
    from annealage_agent import settings as settings_module

    sid = sessions.create_session(tmp_path)
    built = []

    def build_session(on_event, *, bus):
        built.append(
            launch.build_session(
                "claude",
                on_event,
                bus=bus,
                serve_dir=tmp_path,
                session_id=sid,
                resumed=False,
                settings=settings_module.resolve(tmp_path),
                mcp_host="127.0.0.1",
                mcp_port=8765,
                agent_token="agent",
            )
        )
        return built[0]

    create_toy_app(
        tmp_path,
        token="tok",
        agent_token="agent",
        session_id=sid,
        build_session=build_session,
        write_protected=("designs/*.review.json",),
    )

    async def decision(tool_name, tool_input):
        result = await built[0]._guard_secret_paths(
            {"tool_name": tool_name, "tool_input": tool_input}, "tu_1", None
        )
        return result.get("hookSpecificOutput", {}).get("permissionDecision")

    assert await decision("Write", {"file_path": "designs/amp.review.json"}) == "deny"
    assert await decision("Read", {"file_path": "designs/amp.review.json"}) is None
    assert await decision("Write", {"file_path": "designs/amp.sketch.json"}) is None


def test_a_write_protected_pattern_outside_the_served_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="relative to the served directory"):
        create_toy_app(tmp_path, token="tok", write_protected=("../elsewhere/*.json",))


def test_a_bare_string_for_write_protected_is_refused_not_split_into_characters(tmp_path):
    with pytest.raises(ValueError, match="tuple of glob patterns"):
        create_toy_app(tmp_path, token="tok", write_protected="Makefile")


def test_sandbox_status_missing_list_comes_from_the_childs_own_words(monkeypatch):
    """The startup PATH guess is only a prediction, and the child's report
    supersedes it the moment it arrives (fact 20). The PATH check is
    monkeypatched to say nothing is missing, so this test cannot pass by
    accident on a host that happens to lack ``bwrap``/``socat``: the only
    way ``missing`` can end up naming ``socat`` below is that
    ``_note_stderr`` actually parsed it out of the line the child wrote.
    """
    monkeypatch.setattr(sdk_module, "missing_sandbox_dependencies", lambda: ())
    session = SdkSession(
        lambda event: None, cwd="/proj/root", session_id="toy-sess-1", sandbox=True
    )

    assert session.sandbox_status() == SandboxStatus(requested=True, active=True, missing=())

    session._note_stderr(
        "Sandbox disabled: sandbox is enabled but dependencies are missing: "
        "socat not installed. Commands will run WITHOUT sandboxing. "
        "Network and filesystem restrictions will NOT be enforced."
    )

    status = session.sandbox_status()
    assert status.requested is True
    assert status.active is False
    assert status.missing == ("socat",)


@pytest.mark.asyncio
async def test_connect_failure_becomes_agent_error_not_a_raised_exception():
    """A child that cannot be started (a missing CLI, a bad transport, an
    unauthenticated account) must not raise out of ``start()``: the HTTP
    server has already started independently and must keep serving the
    viewer regardless of what the agent does (plan section 3.4)."""
    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd="/proj/root", session_id="toy-sess-1", transport=_FailingConnectTransport()
    )

    await session.start()  # must not raise

    assert session.agent_status() == AGENT_UNAVAILABLE
    event = await recorder.next()
    assert isinstance(event, AgentError)
    assert "boom: no such host" in event.stderr


@pytest.mark.asyncio
async def test_pump_survives_one_malformed_message():
    """Pinning a defect, left failing on purpose (see the M5 review
    report): a well-formed message sent right after a malformed one is
    never delivered, and the session is marked unavailable instead.

    ``sdk.py``'s own comment on the inner ``try``/``except`` inside
    ``_pump`` says "One malformed message must not end the conversation."
    That guard only wraps ``self._handle(message)``, which runs on an
    already-parsed ``Message`` object. A raw dict whose ``type`` the SDK
    recognises but which is missing a field that type requires (here, an
    ``assistant`` message with no ``message`` key) fails inside
    ``claude_agent_sdk``'s own ``parse_message``, called by
    ``ClaudeSDKClient.receive_messages()`` itself, one level above
    ``_handle`` and outside that inner guard. The exception surfaces at
    ``_pump``'s ``async for message in self._client.receive_messages():``
    line, which only the OUTER ``except Exception as exc: self._fail(exc)``
    catches: that call marks the session unavailable and returns from
    ``_pump`` for good, so nothing sent afterward, however well-formed,
    is ever read again.

    Once fixed, this assertion should hold: the malformed message logs a
    warning and the following ``TextDelta`` still arrives.
    """
    session, transport, recorder = await _started_session()
    try:
        transport.push({"type": "assistant", "session_id": "sdk-sess-1"})
        transport.push(
            {
                "type": "stream_event",
                "uuid": "u2",
                "session_id": "sdk-sess-1",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "still here"},
                },
            }
        )
        event = await recorder.next()
        assert isinstance(event, TextDelta)
        assert event.text == "still here"
        assert session.agent_status() == AGENT_READY
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_becoming_ready_is_announced_so_a_hello_snapshot_is_corrected():
    """A browser opens its socket before the CLI child finishes starting, so the
    hello frame ordinarily reports "connecting". Without this event that first
    answer is also the last one and a working agent reads as stuck starting up.
    """
    recorder = EventRecorder()
    session = SdkSession(recorder, cwd="/proj/root", session_id="s", transport=FakeTransport())
    await session.start()
    try:
        statuses = [e.status for e in recorder.all if isinstance(e, AgentStatus)]
        assert statuses == [AGENT_READY]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_failure_to_start_announces_unavailable():
    recorder = EventRecorder()
    session = SdkSession(
        recorder, cwd="/proj/root", session_id="s", transport=_FailingConnectTransport()
    )
    await session.start()

    statuses = [e.status for e in recorder.all if isinstance(e, AgentStatus)]
    assert statuses == [AGENT_UNAVAILABLE]
    assert session.agent_status() == AGENT_UNAVAILABLE


@pytest.mark.asyncio
async def test_an_unchanged_status_is_not_announced_again():
    """The pane's status line is derived state, so a repeat says nothing."""
    recorder = EventRecorder()
    session = SdkSession(recorder, cwd="/proj/root", session_id="s", transport=FakeTransport())
    await session.start()
    await session.close()
    await session.close()

    statuses = [e.status for e in recorder.all if isinstance(e, AgentStatus)]
    assert statuses == [AGENT_READY, AGENT_UNAVAILABLE]


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_secret_path_hook_denies_a_read_of_a_credential_directory(tmp_path, monkeypatch):
    """The credential refusal reaches the model through the same hook event as
    the configuration tripwire, which is the only point upstream of both a
    settings allow rule and the sandbox's own auto-approval (fact 25)."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    session, _transport, _recorder = await _started_session()
    try:
        result = await session._guard_secret_paths(
            {"tool_name": "Read", "tool_input": {"file_path": str(home / ".ssh" / "id_rsa")}},
            "tu_1",
            None,
        )
        decision = result["hookSpecificOutput"]
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert "credential" in decision["permissionDecisionReason"]
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_secret_path_hook_expresses_no_opinion_on_ordinary_work(tmp_path, monkeypatch):
    """An empty answer leaves the permission flow exactly as it would be without
    the hook, which is what keeps contained bash running unprompted."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    session, _transport, _recorder = await _started_session()
    try:
        assert (
            await session._guard_secret_paths(
                {"tool_name": "Bash", "tool_input": {"command": "python build.py"}}, "tu_1", None
            )
            == {}
        )
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_secret_path_hook_denies_when_it_cannot_decide(monkeypatch):
    """A control that cannot tell whether a call is safe has to assume it is
    not, the same way the digest check does."""
    from annealage_agent.session import secret_paths

    def explode(*args, **kwargs):
        raise RuntimeError("no home directory")

    monkeypatch.setattr(secret_paths, "refusal", explode)
    session, _transport, _recorder = await _started_session()
    try:
        result = await session._guard_secret_paths(
            {"tool_name": "Read", "tool_input": {"file_path": "notes.json"}}, "tu_1", None
        )
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "bug in toy" in result["hookSpecificOutput"]["permissionDecisionReason"]
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_secret_path_hook_is_installed_even_with_no_accepted_digest(
    tmp_path, monkeypatch
):
    """The two hooks answer different questions, so the credential refusal must
    not depend on the served directory having had configuration to accept: a
    folder with no .claude/ at all is the ordinary case."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    session, _transport, _recorder = await _started_session()
    try:
        matchers = session._client.options.hooks["PreToolUse"]
        assert matchers[0].hooks == [session._guard_secret_paths]
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::claude_agent_sdk.CanUseToolShadowedWarning")
async def test_the_hook_refuses_writing_a_file_the_product_protects(
    tmp_path, monkeypatch, swap_product
):
    """``Product.write_protected`` is enforced by the same hook as the
    credential list, so the agent's own Edit and Bash cannot change a file only
    the product's tools may write (a review file), while Read still reads it."""
    import dataclasses

    from toy_product import TOY

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    swap_product(dataclasses.replace(TOY, write_protected=("reviews/*.review.json",)))
    session, _transport, _recorder = await _started_session()

    async def decision(tool_name, tool_input):
        result = await session._guard_secret_paths(
            {"tool_name": tool_name, "tool_input": tool_input}, "tu_1", None
        )
        return result.get("hookSpecificOutput", {}).get("permissionDecision")

    try:
        assert await decision("Edit", {"file_path": "reviews/a.review.json"}) == "deny"
        assert await decision("Bash", {"command": "sed -i s/x/y/ reviews/a.review.json"}) == "deny"
        assert await decision("Read", {"file_path": "reviews/a.review.json"}) is None
        assert await decision("Edit", {"file_path": "reviews/a.sketch.json"}) is None
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# _to_claude_result / _session_rule: the one Claude-specific adapter that
# maps session/permissions.py's provider-neutral ``Decision`` onto the
# ``PermissionResultAllow``/``PermissionResultDeny`` shape
# ``ClaudeAgentOptions.can_use_tool`` requires. ``_StubBroker`` above
# deliberately never calls the real broker, so this is the one place in the
# suite that exercises the adapter functions themselves.
# ---------------------------------------------------------------------------


def test_to_claude_result_allow_with_no_remembered_tool_is_a_bare_allow():
    result = sdk_module._to_claude_result(Decision(allow=True, remember_tool=None))
    assert isinstance(result, PermissionResultAllow)
    assert not result.updated_permissions


def test_to_claude_result_allow_with_a_remembered_tool_carries_a_session_rule():
    """A remembered grant's ``updated_permissions`` is one ``PermissionUpdate``
    that adds a session-scoped allow rule for the whole tool (plan section 5:
    per tool, never per input), which is what lets the SDK's own rule
    matcher, not this module's memory, cover the rest of the running
    session."""
    result = sdk_module._to_claude_result(Decision(allow=True, remember_tool="Write"))
    assert isinstance(result, PermissionResultAllow)
    assert result.updated_permissions is not None
    assert len(result.updated_permissions) == 1
    update = result.updated_permissions[0]
    assert isinstance(update, PermissionUpdate)
    assert update.type == "addRules"
    assert update.behavior == "allow"
    assert update.destination == "session"
    assert len(update.rules) == 1
    rule = update.rules[0]
    assert isinstance(rule, PermissionRuleValue)
    assert rule.tool_name == "Write"
    assert rule.rule_content is None


def test_to_claude_result_deny_carries_the_decisions_message():
    result = sdk_module._to_claude_result(Decision(allow=False, message="nope"))
    assert isinstance(result, PermissionResultDeny)
    assert result.message == "nope"


def test_session_rule_builds_the_same_update_to_claude_result_uses():
    update = sdk_module._session_rule("Write")
    assert isinstance(update, PermissionUpdate)
    assert update.type == "addRules"
    assert update.behavior == "allow"
    assert update.destination == "session"
    assert update.rules == [PermissionRuleValue(tool_name="Write", rule_content=None)]


def test_launch_builds_the_claude_session_with_the_tool_servers_pre_allowed_list(tmp_path):
    """The production path, not a list this test hands the session itself:
    ``launch.py`` building a Claude session from the tool server
    ``create_app`` put on the bus. Its allow list is exactly the read- and
    view-grade tools, namespaced, and no write-grade tool, whose absence is
    what makes each of them reach the human as a card."""
    from types import SimpleNamespace

    from annealage_agent import launch, sessions
    from annealage_agent import settings as settings_module

    class _StubBus:
        paused = False

    session_id = sessions.create_session(tmp_path)
    bus = SimpleNamespace(
        tools=build_toy_tools(_StubBus(), tmp_path, session_id),
        broker=None,
        url="http://127.0.0.1:8765/",
    )
    session = launch.build_session(
        "claude",
        lambda event: None,
        bus=bus,
        serve_dir=tmp_path,
        session_id=session_id,
        resumed=False,
        settings=settings_module.resolve(tmp_path),
        mcp_host="127.0.0.1",
        mcp_port=8765,
        agent_token="agent",
    )
    allowed = session._build_options().allowed_tools
    assert allowed == EXPECTED_PRE_ALLOWED_TOOLS
    for name in EXPECTED_NEVER_PRE_ALLOWED:
        assert name not in allowed


def _session_with_context(backend, tmp_path, session_id):
    """``launch.build_session``'s session for ``backend`` over the toy's tools,
    for the two session-context tests below."""
    from types import SimpleNamespace

    from annealage_agent import launch
    from annealage_agent import settings as settings_module

    bus = SimpleNamespace(
        tools=build_toy_tools(SimpleNamespace(paused=False), tmp_path, session_id),
        broker=None,
        url="http://127.0.0.1:8765/",
    )
    return launch.build_session(
        backend,
        lambda event: None,
        bus=bus,
        serve_dir=tmp_path,
        session_id=session_id,
        resumed=False,
        settings=settings_module.resolve(tmp_path),
        mcp_host="127.0.0.1",
        mcp_port=8765,
        agent_token="agent",
    )


def test_the_product_s_session_context_reaches_each_backend(tmp_path, swap_product):
    """``Product.session_context`` is what the run is about (Loom: the design
    under review). ``launch.py`` calls it once and every backend is started
    with it: Claude's system prompt, Codex's developer instructions (omp's,
    which needs omp-rpc, is the next test). A product with none leaves
    Claude's prompt at the SDK's empty default."""
    import dataclasses

    from toy_product import TOY

    from annealage_agent import sessions

    seen = []

    def context(bus, serve_dir):
        seen.append(serve_dir)
        return "You are reviewing design demo."

    session_id = sessions.create_session(tmp_path)
    claude = _session_with_context("claude", tmp_path, session_id)
    assert claude._build_options().system_prompt is None
    swap_product(dataclasses.replace(TOY, session_context=context))
    claude = _session_with_context("claude", tmp_path, session_id)
    assert claude._build_options().system_prompt == "You are reviewing design demo."
    codex = _session_with_context("codex", tmp_path, session_id)
    assert codex._instructions == "You are reviewing design demo."
    assert seen == [tmp_path] * 2


def test_the_product_s_session_context_reaches_omp(tmp_path, swap_product):
    """The omp leg of the test above: its appended system prompt. omp-rpc
    needs Python 3.11 or later and is installed separately (CONTRIBUTING.md),
    so this skips where it is absent rather than failing the 3.10 run."""
    pytest.importorskip("omp_rpc")
    import dataclasses

    from toy_product import TOY

    from annealage_agent import sessions

    seen = []

    def context(bus, serve_dir):
        seen.append(serve_dir)
        return "You are reviewing design demo."

    swap_product(dataclasses.replace(TOY, session_context=context))
    session_id = sessions.create_session(tmp_path)
    captured = {}

    def omp_client(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("no omp here")

    omp = _session_with_context("omp", tmp_path, session_id)
    omp._client_factory = omp_client
    asyncio.run(omp.start())
    assert captured["append_system_prompt"] == "You are reviewing design demo."
    assert seen == [tmp_path]


# ---------------------------------------------------------------------------
# The backend's own logs (backend_logs)
# ---------------------------------------------------------------------------


def test_backend_logs_are_this_sessions_transcript_and_the_clis_stderr(tmp_path, monkeypatch):
    """The CLI keeps a transcript per session under its config directory, in a
    folder named after the working directory with every character that is not
    an ASCII letter or digit replaced by "-" (the CLI's own rule). Only this
    session's is listed, and the stderr kept in memory is there before any
    transcript exists, which is the case when the CLI never started."""
    import os
    import re

    config = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    project = tmp_path / "my project.v2"
    project.mkdir()
    folder = config / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(project))
    folder.mkdir(parents=True)
    mine = folder / "0f6c1d2e-mine.jsonl"
    mine.write_text("{}\n", encoding="utf-8")
    (folder / "9a8b7c6d-another-session.jsonl").write_text("{}\n", encoding="utf-8")

    session = SdkSession(EventRecorder(), cwd=project, session_id="s")
    session._note_stderr("Invalid API key · Please run /login")
    assert [(e.name, e.kind) for e in session.backend_logs()] == [("Claude stderr", "text")]

    session._remember_sdk_session("0f6c1d2e-mine")
    logs = session.backend_logs()
    assert [(e.name, e.path) for e in logs] == [
        ("Claude transcript", os.path.realpath(mine)),
        ("Claude stderr", None),
    ]
    assert logs[1].text == "Invalid API key · Please run /login"
