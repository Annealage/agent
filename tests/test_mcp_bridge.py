"""Tests for the Codex tool-exposure bridge (Annealage Mesh's
``planning/tickets/phase3_codex-tool-mcp-bridge.md``): ``POST /mcp``
(``http/routes_mcp.py``) and the stdio-to-HTTP proxy
(``session/codex_mcp_stdio_bridge.py``), run against the toy product's tools.

Two layers, tested separately and then together:

- ``/mcp`` in isolation, against a bare ``Microdot`` app (``register_mcp_routes``
  called directly, no ``create_app``): the token/Origin gate, ``tools/list``,
  and ``tools/call`` including the write-class broker gate this bridge is
  responsible for wiring (``routes_mcp.py``'s own docstring explains why that
  gate lives here rather than in ``tools.py``).
- The full chain end to end: a real bound socket serving the toy product's
  app, a real ``httpx`` connection, and a fake app-server (a genuine
  ``mcp.ClientSession`` connected in-memory to the proxy's own ``Server``
  through the SDK's memory transport) - the shape the ticket's own acceptance
  criteria ask for.

The last section proves the proxy subprocess itself exits cleanly when its
stdin closes (what a stdio-launched MCP server sees when its launcher goes
away), with no orphan left behind.
"""

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from conftest import create_toy_app, make_test_client, mcp_client
from microdot import Microdot
from toy_product import NOTES_FILE, build_toy_tools

from annealage_agent import app as agent_app
from annealage_agent.http.routes_mcp import register_mcp_routes
from annealage_agent.session.codex_mcp_stdio_bridge import authority_url, build_server
from annealage_agent.session.fake import FakeSession
from annealage_agent.session.permissions import Decision, PermissionBroker

pytestmark = pytest.mark.asyncio

TOKEN = "mcp-bridge-test-token"
BROWSER_TOKEN = "mcp-bridge-browser-token"


class FakeBus:
    """The two members every toy tool handler depends on
    (``tests/test_tools.py``'s ``FakeBus``, mirrored here rather than
    imported across test files)."""

    def __init__(self):
        self.paused = False

    async def call(self, method, params=None, *, timeout=None):
        return {}


class CountingBroker:
    """Counts calls into ``ask()`` and returns a canned ``Decision`` - what
    the write-class-gate tests below assert against, isolated from a live
    viewer/approval-card round trip (a real ``PermissionBroker`` is exercised
    directly in the end-to-end section further down)."""

    def __init__(self, decision):
        self.calls = []
        self.decision = decision

    async def ask(self, tool_name, input_data, context):
        self.calls.append((tool_name, input_data))
        return self.decision


@pytest.fixture
def project(tmp_path):
    (tmp_path / NOTES_FILE).write_text('["first"]', encoding="utf-8")
    return tmp_path


@pytest.fixture
def toy_tools(project):
    return build_toy_tools(FakeBus(), project, "sess-1")


def _notes(project):
    return json.loads((project / NOTES_FILE).read_text(encoding="utf-8"))


def _mcp_app(
    toy_tools, *, broker, token=TOKEN, allowed_origins=(), hosted_mode=False, hosted_bus=None
):
    async def current_broker():
        return broker

    app = Microdot()
    register_mcp_routes(
        app,
        tools=toy_tools,
        current_broker=current_broker,
        agent_token=token,
        allowed_origins=allowed_origins,
        hosted_mode=hosted_mode,
        hosted_bus=hosted_bus,
        hosted_tool_ops=getattr(hosted_bus, "hosted_tool_ops", {}),
    )
    return app


async def _post(client, body, *, path="/mcp", token=TOKEN):
    query = ("?t=%s" % token) if token is not None else ""
    return await client.post(
        path + query,
        headers={"Content-Type": "application/json"},
        body=json.dumps(body),
    )


# ---------------------------------------------------------------------------
# the token/Origin gate, shared with /ws and /settings
# ---------------------------------------------------------------------------


async def test_no_token_is_refused_like_ws(toy_tools):
    app = _mcp_app(toy_tools, broker=None)
    client = make_test_client(app)
    res = await _post(client, {"method": "tools/list"}, token=None)
    assert res.status_code == 403
    assert res.body == b"forbidden"


async def test_wrong_token_is_refused(toy_tools):
    app = _mcp_app(toy_tools, broker=None, token="the-real-one")
    client = make_test_client(app)
    res = await _post(client, {"method": "tools/list"}, token="not-it")
    assert res.status_code == 403
    assert res.body == b"forbidden"


async def test_disallowed_origin_is_refused(toy_tools):
    app = _mcp_app(toy_tools, broker=None, allowed_origins={"http://127.0.0.1:8765"})
    client = make_test_client(app)
    res = await client.post(
        "/mcp?t=%s" % TOKEN,
        headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
        body=json.dumps({"method": "tools/list"}),
    )
    assert res.status_code == 403


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


async def test_tools_list_returns_every_product_tool_with_schema_and_description(toy_tools):
    app = _mcp_app(toy_tools, broker=None)
    client = make_test_client(app)
    res = await _post(client, {"method": "tools/list"})
    assert res.status_code == 200
    tools = {t["name"]: t for t in res.json["result"]["tools"]}
    assert set(tools) == set(toy_tools.tool_table())
    one = tools["list_notes"]
    assert one["description"]
    assert one["inputSchema"]["type"] == "object"


# ---------------------------------------------------------------------------
# tools/call: read-class bypasses the broker entirely
# ---------------------------------------------------------------------------


async def test_read_class_call_never_touches_the_broker(toy_tools):
    broker = CountingBroker(Decision(allow=True))
    app = _mcp_app(toy_tools, broker=broker)
    client = make_test_client(app)
    res = await _post(
        client, {"method": "tools/call", "params": {"name": "list_notes", "arguments": {}}}
    )
    assert res.status_code == 200
    result = res.json["result"]
    assert not result.get("isError")
    assert result["content"][0]["type"] == "text"
    assert not broker.calls


# ---------------------------------------------------------------------------
# tools/call: write-class reaches PermissionBroker exactly once
# ---------------------------------------------------------------------------


async def test_write_class_call_reaches_broker_exactly_once_when_allowed(toy_tools, project):
    broker = CountingBroker(Decision(allow=True))
    app = _mcp_app(toy_tools, broker=broker)
    client = make_test_client(app)
    res = await _post(
        client,
        {"method": "tools/call", "params": {"name": "add_note", "arguments": {"text": "hi"}}},
    )
    assert res.status_code == 200
    result = res.json["result"]
    assert not result.get("isError")
    assert len(broker.calls) == 1
    tool_name, input_data = broker.calls[0]
    # namespaced, matching Claude's own can_use_tool convention
    # (session/sdk.py) exactly - see routes_mcp.py's docstring for why.
    assert tool_name == "mcp__toy__add_note"
    assert input_data == {"text": "hi"}
    # The handler really ran: the note landed on disk.
    assert _notes(project) == ["first", "hi"]


async def test_write_class_call_is_blocked_and_the_handler_never_runs_when_denied(
    toy_tools, project
):
    broker = CountingBroker(Decision(allow=False, message="not right now"))
    app = _mcp_app(toy_tools, broker=broker)
    client = make_test_client(app)
    res = await _post(
        client,
        {"method": "tools/call", "params": {"name": "add_note", "arguments": {"text": "hi"}}},
    )
    assert res.status_code == 200
    result = res.json["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == "not right now"
    assert len(broker.calls) == 1
    # The handler itself never ran: the notes file is as it was.
    assert _notes(project) == ["first"]


async def test_write_class_call_fails_closed_with_no_broker_configured(toy_tools):
    app = _mcp_app(toy_tools, broker=None)
    client = make_test_client(app)
    res = await _post(
        client,
        {"method": "tools/call", "params": {"name": "add_note", "arguments": {"text": "hi"}}},
    )
    result = res.json["result"]
    assert result["isError"] is True
    assert "no permission broker" in result["content"][0]["text"]


async def test_view_class_call_also_bypasses_the_broker(toy_tools):
    """VIEW_CLASS is pre-allowed for Claude too (session/sdk.py's own
    allowed_tools) - only WRITE_CLASS goes through the broker here, matching
    that convention rather than inventing a stricter one for Codex."""
    broker = CountingBroker(Decision(allow=True))
    app = _mcp_app(toy_tools, broker=broker)
    client = make_test_client(app)
    res = await _post(
        client,
        {"method": "tools/call", "params": {"name": "set_view", "arguments": {"zoom": 2}}},
    )
    assert res.status_code == 200
    assert not broker.calls


# ---------------------------------------------------------------------------
# malformed requests and unknown tools
# ---------------------------------------------------------------------------


async def test_unknown_tool_name_is_an_error_result_not_a_protocol_error(toy_tools):
    app = _mcp_app(toy_tools, broker=None)
    client = make_test_client(app)
    res = await _post(
        client, {"method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}}
    )
    assert res.status_code == 200
    result = res.json["result"]
    assert result["isError"] is True
    assert "no_such_tool" in result["content"][0]["text"]


async def test_unknown_method_is_a_400(toy_tools):
    app = _mcp_app(toy_tools, broker=None)
    client = make_test_client(app)
    res = await _post(client, {"method": "prompts/list"})
    assert res.status_code == 400
    assert res.json["ok"] is False


async def test_missing_method_is_a_400(toy_tools):
    app = _mcp_app(toy_tools, broker=None)
    client = make_test_client(app)
    res = await _post(client, {})
    assert res.status_code == 400


async def test_authority_http_failure_raises_authority_error():
    """The forwarding helper's own failure path: an unreachable/erroring
    authority raises ``AuthorityError`` rather than returning a silent empty
    result. Left to propagate out of the registered ``call_tool`` handler
    uncaught, the ``mcp`` SDK's own ``Server._handle_request`` turns any
    exception into a JSON-RPC error response rather than crashing the stdio
    session (verified by reading that function directly, not merely
    assumed - see ``codex_mcp_stdio_bridge.py``'s own ``AuthorityError``
    docstring), so this only needs to pin that the exception really is
    raised here.
    """
    from annealage_agent.session.codex_mcp_stdio_bridge import AuthorityError, _call_authority

    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        url = authority_url("127.0.0.1", 8765, "/mcp").copy_merge_params({"t": TOKEN})
        with pytest.raises(AuthorityError):
            await _call_authority(client, url, "tools/call", {"name": "x", "arguments": {}})


# ---------------------------------------------------------------------------
# end to end: fake app-server -> proxy Server -> real HTTP -> real /mcp ->
# tool_table()'s handler -> back. The ticket's own acceptance criteria.
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _run_real_server(project, *, agent_token, broker):
    """Serves the toy product's app (``create_toy_app``) through the agent
    layer's ``serve`` as a background task on a real loopback socket, with a
    ``FakeSession`` standing in for the agent (no SDK, no subprocess) and
    ``broker`` wired the same way a product's CLI wires it in its
    ``build_session`` (``bus.broker = broker``, read by ``create_app`` once
    ``build_session`` returns - see ``app.py``'s own comment on that channel).
    Returns ``(port, task)``; the caller cancels ``task`` and awaits it to
    shut down.

    ``agent_token`` is the one ``/mcp`` accepts; the app's browser token is
    ``BROWSER_TOKEN``, which ``/mcp`` must refuse.

    ``sessions.create_session`` scaffolds ``.toy/sessions/<id>/`` first,
    the same way a product's CLI does before ever building an app:
    ``create_app`` in agent mode opens ``events.jsonl`` inside that directory
    unconditionally (``session/events.py``'s ``EventLog``), which does not
    create its own parent directory.
    """
    from annealage_agent import sessions as sessions_module

    session_id = sessions_module.create_session(project)
    port = _free_port()
    ready = asyncio.Event()

    def build_session(on_event, *, bus):
        bus.broker = broker
        return FakeSession(on_event)

    app = create_toy_app(
        project,
        token=BROWSER_TOKEN,
        agent_token=agent_token,
        host="127.0.0.1",
        port=port,
        session_id=session_id,
        build_session=build_session,
    )
    task = asyncio.ensure_future(agent_app.serve(app, "127.0.0.1", port, on_ready=ready.set))
    await asyncio.wait_for(ready.wait(), timeout=5.0)
    return port, task


def _bridge_argv(port):
    """The command line ``CodexSession`` registers the bridge with, minus the
    interpreter's own path: host, port and the MCP server's name and version,
    and no token, which travels in the environment instead."""
    return [
        sys.executable,
        "-m",
        "annealage_agent.session.codex_mcp_stdio_bridge",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--server-name",
        "annealage-toy",
        "--server-version",
        "0",
    ]


async def _stop_real_server(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)


async def test_fake_app_server_lists_and_calls_a_read_tool_through_the_real_http_hop(project):
    token = "e2e-read-token"
    broker = PermissionBroker(lambda e: None, timeout=2.0)
    port, task = await _run_real_server(project, agent_token=token, broker=broker)
    try:
        url = authority_url("127.0.0.1", port, "/mcp").copy_merge_params({"t": token})
        async with httpx.AsyncClient() as client:
            proxy_server = build_server(client, url, name="annealage-toy", version="0")
            async with mcp_client(proxy_server) as session:
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert "list_notes" in names

                result = await session.call_tool("list_notes", {})
                assert not result.is_error
                payload = json.loads(result.content[0].text)
                assert payload["notes"] == ["first"]
    finally:
        await _stop_real_server(task)


async def test_fake_app_server_write_call_reaches_the_broker_exactly_once_end_to_end(project):
    """The full chain's version of the write-class acceptance criterion:
    the fake app-server's call_tool for a write-class tool blocks on a real
    PermissionBroker.ask, which this test answers exactly once
    (broker.decide), and the tool only takes effect after that answer."""
    from annealage_agent.session.base import PermissionRequest

    token = "e2e-write-token"
    events = []
    broker = PermissionBroker(events.append, timeout=5.0, no_viewer_grace=0.05)
    broker.viewer_connected()
    port, task = await _run_real_server(project, agent_token=token, broker=broker)
    try:
        url = authority_url("127.0.0.1", port, "/mcp").copy_merge_params({"t": token})
        async with httpx.AsyncClient(timeout=10.0) as client:
            proxy_server = build_server(client, url, name="annealage-toy", version="0")
            async with mcp_client(proxy_server) as session:

                async def approve():
                    # Poll for the PermissionRequest broker.ask emitted, then
                    # answer it exactly once.
                    for _ in range(200):
                        request = next(
                            (e for e in events if isinstance(e, PermissionRequest)), None
                        )
                        if request is not None:
                            await broker.decide(request.request_id, "allow")
                            return
                        await asyncio.sleep(0.02)
                    raise AssertionError("no PermissionRequest was ever emitted")

                approver = asyncio.ensure_future(approve())
                result = await session.call_tool("add_note", {"text": "from codex"})
                await approver

        assert not result.is_error
        assert _notes(project) == ["first", "from codex"]
        request_count = sum(1 for e in events if isinstance(e, PermissionRequest))
        assert request_count == 1
    finally:
        await _stop_real_server(task)


# ---------------------------------------------------------------------------
# the proxy subprocess itself: exits cleanly with no orphan left behind
# ---------------------------------------------------------------------------


async def test_proxy_subprocess_exits_cleanly_when_its_stdin_closes(project):
    """Simulates 'the parent codex app-server process exits': closing the
    subprocess's stdin is exactly what a stdio-launched MCP server sees when
    its launcher goes away (``mcp.server.stdio``'s own read loop ends on
    EOF, and ``Server.run()`` returns normally, per
    ``session/codex_mcp_stdio_bridge.py``'s own docstring) - proving no
    orphaned proxy process is left running once that happens.
    """
    token = "subproc-exit-token"
    broker = PermissionBroker(lambda e: None, timeout=2.0)
    port, task = await _run_real_server(project, agent_token=token, broker=broker)
    proc = None
    try:
        proc = subprocess.Popen(
            _bridge_argv(port),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, ANNEALAGE_AGENT_TOKEN=token),
        )
        # A short wait so this does not race the subprocess's own startup
        # (stdio_server's async context manager spawning its reader/writer
        # tasks); closing stdin before it is listening would still be a
        # clean EOF either way, this only keeps the "still running" check
        # below meaningful rather than racy.
        await asyncio.sleep(0.3)
        assert proc.poll() is None, "the proxy exited before its stdin was ever closed"

        proc.stdin.close()

        loop = asyncio.get_running_loop()
        returncode = await asyncio.wait_for(loop.run_in_executor(None, proc.wait), timeout=10.0)
        assert returncode == 0
        proc = None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        await _stop_real_server(task)


async def test_proxy_subprocess_authenticates_from_its_environment_not_its_argv(project):
    """The real bridge process, launched the way Codex launches it (the
    command line from ``CodexSession``, the token in the environment Codex
    passes through ``env_vars``), reaches ``/mcp`` and lists the tools. Its
    command line, which every user on the machine can read through ``ps``,
    carries neither the agent token nor the browser token."""
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    token = "subproc-env-token"
    broker = PermissionBroker(lambda e: None, timeout=2.0)
    port, task = await _run_real_server(project, agent_token=token, broker=broker)
    argv = _bridge_argv(port)
    try:
        params = StdioServerParameters(
            command=argv[0],
            args=argv[1:],
            env=dict(os.environ, ANNEALAGE_AGENT_TOKEN=token),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                tools = await asyncio.wait_for(session.list_tools(), timeout=10.0)
        assert "list_notes" in {t.name for t in tools.tools}
        assert not any(token in arg or BROWSER_TOKEN in arg for arg in argv)
    finally:
        await _stop_real_server(task)


async def test_proxy_subprocess_refuses_to_start_without_a_token_in_its_environment():
    """No token, no MCP: the bridge exits before speaking the protocol, so
    Codex reports the server as failed to start rather than as a server with
    no tools. Its argv has nowhere to take a token from any more."""
    env = {key: value for key, value in os.environ.items() if key != "ANNEALAGE_AGENT_TOKEN"}
    proc = await asyncio.to_thread(
        subprocess.run, _bridge_argv(9), input=b"", capture_output=True, env=env, timeout=30
    )
    assert proc.returncode == 2
    assert b"ANNEALAGE_AGENT_TOKEN" in proc.stderr
    assert proc.stdout == b""


# ---------------------------------------------------------------------------
# the two tokens are not interchangeable (D5)
# ---------------------------------------------------------------------------


def _agent_app(project, *, token=BROWSER_TOKEN, agent_token=TOKEN):
    """An agent-mode app with both tokens set, the toy product's real tool
    server and a ``FakeSession``, so ``/mcp`` is mounted exactly as a real run
    mounts it."""
    from annealage_agent import sessions as sessions_module

    return create_toy_app(
        project,
        token=token,
        agent_token=agent_token,
        session_id=sessions_module.create_session(project),
        build_session=lambda on_event, *, bus: FakeSession(on_event),
    )


async def test_mcp_refuses_the_browser_token_and_accepts_the_agent_token(project):
    client = make_test_client(_agent_app(project))
    refused = await _post(client, {"method": "tools/list"}, token=BROWSER_TOKEN)
    assert refused.status_code == 403
    accepted = await _post(client, {"method": "tools/list"}, token=TOKEN)
    assert accepted.status_code == 200
    assert "list_notes" in {t["name"] for t in accepted.json["result"]["tools"]}


async def test_browser_routes_refuse_the_agent_token(project):
    """Every route of the agent layer's that takes the browser token refuses
    the agent token: the socket that carries permission decisions, the
    settings window, uploads and transcript export. Each is also shown to
    accept the browser token, so a refusal here is the token check and not the
    route being unreachable."""
    client = make_test_client(_agent_app(project))
    ws_headers = {
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
    }
    for token, expected in ((TOKEN, 403), (BROWSER_TOKEN, None)):
        res = await client.get("/ws?t=%s" % token, headers=ws_headers)
        assert res.status_code == expected, token
    for token, expected in ((TOKEN, 403), (BROWSER_TOKEN, 200)):
        res = await client.get("/settings?t=%s" % token)
        assert res.status_code == expected, token
    for path in ("/upload", "/session/nope/export"):
        res = await client.post(
            "%s?t=%s" % (path, TOKEN),
            headers={"Content-Type": "application/json"},
            body=b"{}",
        )
        assert res.status_code == 403, path
        res = await client.post(
            "%s?t=%s" % (path, BROWSER_TOKEN),
            headers={"Content-Type": "application/json"},
            body=b"{}",
        )
        assert res.status_code != 403, path


async def test_an_app_whose_two_tokens_are_equal_is_refused(project):
    with pytest.raises(ValueError, match="must differ"):
        _agent_app(project, token="same", agent_token="same")


async def test_what_the_model_reads_with_no_viewer_attached_never_carries_the_browser_token(
    project,
):
    """With no viewer attached, a viewer tool fails and a write-class call is
    refused by the broker, and both tell the model where the viewer is. That
    text reaches the model and the session's event log in the served
    directory, so it names the address alone: with the browser token in it the
    agent could approve its own permission cards. The session is built by
    ``launch.py``, as a real run builds it, so the broker's address is the one
    a real run gives it."""
    from annealage_agent import launch, sessions
    from annealage_agent import settings as settings_module

    session_id = sessions.create_session(project)

    def build_session(on_event, *, bus):
        return launch.build_session(
            "claude",
            on_event,
            bus=bus,
            serve_dir=project,
            session_id=session_id,
            resumed=False,
            settings=settings_module.resolve(project),
            mcp_host="127.0.0.1",
            mcp_port=8765,
            agent_token=TOKEN,
        )

    client = make_test_client(
        create_toy_app(
            project,
            token=BROWSER_TOKEN,
            agent_token=TOKEN,
            session_id=session_id,
            build_session=build_session,
        )
    )
    for name, arguments in (("get_view", {}), ("add_note", {"text": "x"})):
        res = await _post(
            client, {"method": "tools/call", "params": {"name": name, "arguments": arguments}}
        )
        result = res.json["result"]
        assert result["isError"] is True, name
        text = result["content"][0]["text"]
        assert "http://127.0.0.1:8765/" in text, name
        assert BROWSER_TOKEN not in text, name


async def test_hosted_mcp_requires_turn_secret_and_declared_tool_operation(project):
    bus = FakeBus()
    bus.hosted_mode = True
    bus.hosted_tool_ops = {"mcp__toy__list_notes": "project.read"}
    bus.hosted_turn_ops = ("project.read",)
    bus.hosted_turn_exp = time.time() + 30
    bus.hosted_turn_live = True
    bus.hosted_turn_secret = "opaque-turn-secret"
    tools = build_toy_tools(bus, project, "sess-hosted")
    app = _mcp_app(tools, broker=None, hosted_mode=True, hosted_bus=bus)

    denied = await make_test_client(app).post(
        "/mcp",
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
        body=json.dumps({"method": "tools/call", "params": {"name": "list_notes"}}),
    )
    assert denied.status_code == 403

    allowed = await make_test_client(app).post(
        "/mcp",
        headers={
            "Authorization": "Bearer opaque-turn-secret",
            "Content-Type": "application/json",
        },
        body=json.dumps({"method": "tools/call", "params": {"name": "list_notes"}}),
    )
    assert allowed.status_code == 200
    result = json.loads(allowed.body)["result"]
    assert not result["isError"]
