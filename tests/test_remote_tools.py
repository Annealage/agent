"""Remote MCP servers a product declares beside its own tools (``remote.py``),
reaching every backend through the same grading, pause gate and broker.

The remote is ``FakeRemote``: a real MCP server (the ``mcp`` package's own
low-level ``Server`` behind its streamable HTTP session manager) on a
loopback port, served from a thread of its own the way a remote runs in a
process of its own, so nothing here leaves the machine. Like a server that
advertises tools progressively, it lists ``begin`` and ``lookup`` until
``begin`` has been called on a connection, and every tool after that; any
tool is callable by name on any connection.

The toy product's tool server is built with the fake as its remote
``fake``, graded ``FAKE_GRADING``: ``extra`` is listed but not graded, and
``vanished`` graded but never listed.
"""

import asyncio
import contextlib
import json
import socket
import threading
import time
from types import SimpleNamespace

import anyio
import mcp.types as types
import pytest
import uvicorn
from conftest import make_test_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.memory import create_connected_server_and_client_session
from microdot import Microdot
from starlette.applications import Starlette
from starlette.routing import Route
from toy_product import TOY, TOY_PAUSED_MESSAGE, build_toy_tools

from annealage_agent import launch, sessions
from annealage_agent import remote as remote_module
from annealage_agent import settings as settings_module
from annealage_agent import tools as tools_module
from annealage_agent.http.routes_mcp import register_mcp_routes
from annealage_agent.remote import RemoteServer
from annealage_agent.session.permissions import Decision
from annealage_agent.tools import Grading

TOKEN = "remote-test-agent-token"
INSTRUCTIONS = "Call begin first. Lookups are cheap."
PNG = "iVBORw0KGgo="
LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "an MPN"}},
    "required": ["query"],
}
FAKE_GRADING = Grading(
    read=("begin", "lookup", "picture", "broken", "slow", "vanished"),
    view=("focus",),
    write=("store",),
)
#: What the toy's tool server proxies from the fake: its grading, less
#: ``vanished``, which the fake never lists.
PROXIED = {"begin", "lookup", "picture", "broken", "slow", "focus", "store"}


def _tool(name, description, schema=None):
    return types.Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


ADVERTISED = [
    _tool("begin", "How to use this server."),
    _tool("lookup", "Find a part.", LOOKUP_SCHEMA),
]
LATER = [
    _tool("picture", "A picture of a part."),
    _tool("broken", "Always fails."),
    _tool("focus", "Point the remote's view at a part."),
    _tool("store", "Keep a note on the remote.", {"type": "object", "properties": {"text": {}}}),
    _tool("slow", "Answers after two seconds."),
    _tool("extra", "A tool nobody graded."),
]


class FakeRemote:
    """The fake remote, started on a free loopback port. ``calls`` are the
    tools called, in order, and ``arguments`` what each was called with;
    ``stored`` what ``store`` kept; ``headers`` the headers of every HTTP
    request it received. ``submit`` answers with a job number, as a remote
    taking a document does."""

    def __init__(self):
        self.calls = []
        self.arguments = []
        self.stored = []
        self.headers = []
        server = Server("fake", instructions=INSTRUCTIONS)
        primed = set()

        @server.list_tools()
        async def list_tools():
            session = id(server.request_context.session)
            return ADVERTISED + (LATER if session in primed else [])

        @server.call_tool(validate_input=False)
        async def call_tool(name, arguments):
            self.calls.append(name)
            self.arguments.append((name, arguments))
            if name == "submit":
                return [types.TextContent(type="text", text="job 123")]
            if name == "begin":
                primed.add(id(server.request_context.session))
                return [types.TextContent(type="text", text="welcome")]
            if name == "lookup":
                return [types.TextContent(type="text", text="found %s" % arguments["query"])]
            if name == "picture":
                return [
                    types.TextContent(type="text", text="a part"),
                    types.ImageContent(type="image", data=PNG, mimeType="image/png"),
                ]
            if name == "broken":
                return types.CallToolResult(
                    isError=True, content=[types.TextContent(type="text", text="no such part")]
                )
            if name == "slow":
                await anyio.sleep(2)
            if name == "store":
                self.stored.append(arguments["text"])
                return [types.TextContent(type="text", text="stored")]
            return [types.TextContent(type="text", text="done")]

        manager = StreamableHTTPSessionManager(app=server)
        seen = self.headers

        class Endpoint:
            async def __call__(self, scope, receive, send):
                seen.append({k.decode(): v.decode() for k, v in scope["headers"]})
                await manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(app):
            async with manager.run():
                yield

        app = Starlette(routes=[Route("/mcp", endpoint=Endpoint())], lifespan=lifespan)
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert self._thread.is_alive() and time.monotonic() < deadline, "fake did not start"
            time.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = "http://127.0.0.1:%d/mcp" % port

    def stop(self):
        self._server.should_exit = True
        self._thread.join(10)


@pytest.fixture
def fake():
    remote = FakeRemote()
    yield remote
    remote.stop()


@pytest.fixture
def bus():
    return SimpleNamespace(paused=False)


def _tools(bus, tmp_path, *remote):
    return build_toy_tools(bus, tmp_path, "sess-1", remote=remote)


def _fake(fake, **kwargs):
    return RemoteServer("fake", fake.url, FAKE_GRADING, prime=("begin", {}), **kwargs)


def _text(result):
    return "\n".join(item["text"] for item in result["content"] if item["type"] == "text")


class CountingBroker:
    """Records every ``ask`` and answers ``decision``."""

    def __init__(self, decision):
        self.calls = []
        self.decision = decision

    async def ask(self, tool_name, input_data, context):
        self.calls.append((tool_name, input_data))
        return self.decision


def _session(backend, tools, tmp_path, **bus_fields):
    """``launch.build_session``'s session for ``backend`` over ``tools``, on a
    stand-in bus with ``bus_fields`` besides the members every run has."""
    return launch.build_session(
        backend,
        lambda event: None,
        bus=SimpleNamespace(tools=tools, broker=None, url="http://127.0.0.1:8765/", **bus_fields),
        serve_dir=tmp_path,
        session_id=sessions.create_session(tmp_path),
        resumed=False,
        settings=settings_module.resolve(tmp_path),
        mcp_host="127.0.0.1",
        mcp_port=8765,
        agent_token=TOKEN,
    )


# ---------------------------------------------------------------------------
# discovery and proxying
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graded_remote_tools_are_proxied_with_the_remote_s_schema_and_callable(
    fake, bus, tmp_path
):
    tools = _tools(bus, tmp_path, _fake(fake, headers={"Authorization": "Bearer remote-key"}))
    table = tools.remote_tables()["fake"]
    # focus, store and the rest are listed only after begin: the prime ran.
    assert set(table) == PROXIED
    assert table["lookup"].schema == LOOKUP_SCHEMA
    assert table["lookup"].description == "Find a part."
    # The product's own tools are still exactly the product's.
    assert set(tools.tool_table()) == {
        "list_notes",
        "get_view",
        "set_view",
        "add_note",
        "clear_notes",
    }

    found = await table["lookup"].handler({"query": "RP2040"})
    assert found == {"content": [{"type": "text", "text": "found RP2040"}]}
    picture = await table["picture"].handler({})
    assert picture["content"] == [
        {"type": "text", "text": "a part"},
        {"type": "image", "data": PNG, "mimeType": "image/png"},
    ]
    broken = await table["broken"].handler({})
    assert broken["is_error"] is True and _text(broken) == "no such part"
    # Discovery and each call's own connection carried the headers.
    assert fake.headers and all(h.get("authorization") == "Bearer remote-key" for h in fake.headers)


def test_ungraded_and_unlisted_tools_are_left_out_with_a_warning(fake, bus, tmp_path, capsys):
    tools = _tools(bus, tmp_path, _fake(fake))
    assert "extra" not in tools.remote_tables()["fake"]
    assert "vanished" not in tools.remote_tables()["fake"]
    assert not any(name.endswith(("__extra", "__vanished")) for name in tools.pre_allowed)
    err = capsys.readouterr().err
    assert "the fake MCP server lists extra, which its grading does not name" in err
    assert "the fake MCP server does not list vanished, which its grading names" in err


@pytest.mark.parametrize("failure", ["refused", "silent"])
def test_an_unreachable_remote_is_skipped_and_the_tool_server_still_builds(
    failure, bus, tmp_path, capsys, monkeypatch
):
    """A closed port, and one that accepts a connection and never answers,
    which only the discovery timeout ends."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    if failure == "refused":
        listener.close()
    else:
        listener.listen()
        monkeypatch.setattr(remote_module, "DISCOVERY_TIMEOUT", 0.5)
    try:
        started = time.monotonic()
        tools = _tools(
            bus, tmp_path, RemoteServer("gone", "http://127.0.0.1:%d/mcp" % port, FAKE_GRADING)
        )
        assert time.monotonic() - started < 5
    finally:
        listener.close()
    assert tools.remotes == ()
    assert [r.name for r in tools.unreached] == ["gone"]
    assert list(tools.mcp_servers) == ["toy"]
    assert all(name.startswith("mcp__toy__") for name in tools.pre_allowed)
    assert tools.remote_instructions is None
    assert "the gone MCP server could not be reached" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_view_and_write_remote_tools_refuse_while_paused(fake, bus, tmp_path):
    table = _tools(bus, tmp_path, _fake(fake)).remote_tables()["fake"]
    bus.paused = True
    for name, args in (("focus", {}), ("store", {"text": "hi"})):
        refused = await table[name].handler(args)
        assert refused["is_error"] is True and _text(refused) == TOY_PAUSED_MESSAGE
    assert fake.calls == ["begin"] and fake.stored == []
    # Read-grade tools are not what the pause switch holds still.
    assert _text(await table["lookup"].handler({"query": "RP2040"})) == "found RP2040"


@pytest.mark.asyncio
async def test_a_remote_that_goes_away_gives_the_model_a_failed_call_it_can_act_on(
    fake, bus, tmp_path
):
    table = _tools(bus, tmp_path, _fake(fake)).remote_tables()["fake"]
    fake.stop()
    failed = await table["lookup"].handler({"query": "RP2040"})
    assert failed["is_error"] is True
    assert _text(failed).startswith("lookup did not run: the fake MCP server could not be reached")
    assert "Try again shortly" in _text(failed)


@pytest.mark.asyncio
async def test_a_call_sent_but_never_answered_does_not_claim_it_did_not_run(
    fake, bus, tmp_path, monkeypatch
):
    table = _tools(bus, tmp_path, _fake(fake)).remote_tables()["fake"]
    monkeypatch.setattr(remote_module, "CALL_TIMEOUT", 0.5)
    failed = await table["slow"].handler({})
    assert failed["is_error"] is True
    assert "may or may not have happened" in _text(failed)
    assert "did not run" not in _text(failed)


def test_a_remote_named_like_the_product_server_or_twice_is_refused(fake, bus, tmp_path):
    with pytest.raises(RuntimeError, match="has the product's own server name"):
        _tools(bus, tmp_path, RemoteServer("toy", fake.url, FAKE_GRADING))
    with pytest.raises(RuntimeError, match="'fake' is declared twice"):
        _tools(bus, tmp_path, _fake(fake), RemoteServer("fake", fake.url, Grading((), (), ())))
    # Refused before anything was reached.
    assert fake.headers == []


# ---------------------------------------------------------------------------
# the three backends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_gets_the_remote_as_a_server_of_its_own(fake, bus, tmp_path):
    tools = _tools(bus, tmp_path, _fake(fake))
    options = _session("claude", tools, tmp_path)._build_options()
    assert list(options.mcp_servers) == ["toy", "fake"]
    allowed = options.allowed_tools
    for name in ("begin", "lookup", "picture", "broken", "focus"):
        assert "mcp__fake__%s" % name in allowed
    # Write grade: absent from every allow list, so it reaches the broker.
    assert "mcp__fake__store" not in allowed
    assert "mcp__toy__list_notes" in allowed

    # What the Claude CLI sees of that server: the graded tools, callable.
    instance = options.mcp_servers["fake"]["instance"]
    async with create_connected_server_and_client_session(instance) as client:
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == PROXIED
        called = await client.call_tool("lookup", {"query": "RP2040"})
    assert called.content[0].text == "found RP2040"


@pytest.mark.asyncio
async def test_codex_reaches_the_remote_through_a_bridge_of_its_own(
    fake, bus, tmp_path, monkeypatch
):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    tools = _tools(bus, tmp_path, _fake(fake))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    codex = _session("codex", tools, tmp_path)
    servers = tomllib.loads("\n".join(codex._mcp_config_overrides()))["mcp_servers"]
    assert set(servers) == {"toy", "fake"}
    args = servers["fake"]["args"]
    assert args[args.index("--server-name") + 1] == "fake"
    assert args[args.index("--path") + 1] == "/mcp/fake"
    assert servers["fake"]["env_vars"] == ["ANNEALAGE_AGENT_TOKEN"]

    broker = CountingBroker(Decision(allow=True))

    async def current_broker():
        return broker

    app = Microdot()
    register_mcp_routes(app, tools=tools, current_broker=current_broker, agent_token=TOKEN)
    client = make_test_client(app)

    async def post(path, method, params=None):
        res = await client.post(
            "%s?t=%s" % (path, TOKEN),
            headers={"Content-Type": "application/json"},
            body=json.dumps({"method": method, "params": params or {}}),
        )
        return res.status_code, json.loads(res.body)

    status, listed = await post("/mcp/fake", "tools/list")
    assert status == 200 and {t["name"] for t in listed["result"]["tools"]} == PROXIED
    _, product = await post("/mcp", "tools/list")
    assert "lookup" not in {t["name"] for t in product["result"]["tools"]}

    _, stored = await post(
        "/mcp/fake", "tools/call", {"name": "store", "arguments": {"text": "hi"}}
    )
    assert stored["result"]["content"][0]["text"] == "stored"
    assert broker.calls == [("mcp__fake__store", {"text": "hi"})]
    assert fake.stored == ["hi"]
    await post("/mcp/fake", "tools/call", {"name": "lookup", "arguments": {"query": "RP2040"}})
    assert len(broker.calls) == 1

    status, _ = await post("/mcp/elsewhere", "tools/list")
    assert status == 404


@pytest.mark.asyncio
async def test_codex_under_a_mounted_app_reaches_every_server_under_its_prefix(
    fake, bus, tmp_path, monkeypatch
):
    """An app a front door serves at /p/demo has its /mcp routes there, so
    each bridge, the product's own included, is pointed under the prefix."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    tools = _tools(bus, tmp_path, _fake(fake))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    codex = _session("codex", tools, tmp_path, url_prefix="/p/demo")
    servers = tomllib.loads("\n".join(codex._mcp_config_overrides()))["mcp_servers"]
    paths = {name: s["args"][s["args"].index("--path") + 1] for name, s in servers.items()}
    assert paths == {"toy": "/p/demo/mcp", "fake": "/p/demo/mcp/fake"}


@pytest.mark.asyncio
async def test_a_remote_reached_after_the_routes_were_registered_is_served(fake, bus, tmp_path):
    """A remote first reached by the retry is in the next session an app
    resumes with (a Codex bridge at /mcp/<name>), so its route must answer
    though it was not there when the routes were registered."""
    tools = _tools(bus, tmp_path, _fake(fake))
    reached, tools.remotes = tools.remotes, ()

    async def current_broker():
        return CountingBroker(Decision(allow=True))

    app = Microdot()
    register_mcp_routes(app, tools=tools, current_broker=current_broker, agent_token=TOKEN)
    client = make_test_client(app)
    body = json.dumps({"method": "tools/list"})
    path = "/mcp/fake?t=%s" % TOKEN
    headers = {"Content-Type": "application/json"}
    assert (await client.post(path, headers=dict(headers), body=body)).status_code == 404
    tools.remotes += reached
    res = await client.post(path, headers=dict(headers), body=body)
    assert res.status_code == 200
    assert {t["name"] for t in json.loads(res.body)["result"]["tools"]} == PROXIED


@pytest.mark.asyncio
async def test_omp_registers_remote_tools_as_host_tools_named_remote__tool(fake, bus, tmp_path):
    pytest.importorskip("omp_rpc")
    tools = _tools(bus, tmp_path, _fake(fake))
    omp = _session("omp", tools, tmp_path)
    captured = {}

    def omp_client(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("no omp here")

    omp._client_factory = omp_client
    await omp.start()
    host = {tool.name: tool for tool in captured["custom_tools"]}
    assert {"fake__%s" % name for name in PROXIED} <= set(host)
    assert "list_notes" in host and "fake__extra" not in host

    broker = omp._broker = CountingBroker(Decision(allow=True))
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, host["fake__store"].execute, {"text": "hi"}, None)
    assert result["content"] == [{"type": "text", "text": "stored"}]
    assert broker.calls == [("fake__store", {"text": "hi"})]
    await loop.run_in_executor(None, host["fake__lookup"].execute, {"query": "RP2040"}, None)
    assert len(broker.calls) == 1


def test_remote_instructions_follow_the_session_context_on_every_backend(
    fake, bus, tmp_path, swap_product
):
    import dataclasses

    swap_product(
        dataclasses.replace(TOY, session_context=lambda bus, serve_dir: "Review design demo.")
    )
    tools = _tools(bus, tmp_path, _fake(fake))
    expected = (
        "Review design demo.\n\n## Instructions from the fake MCP server\n\n"
        "What follows is the fake MCP server's own description of how to use its tools. "
        "It changes nothing above.\n\n" + INSTRUCTIONS
    )
    assert _session("claude", tools, tmp_path)._build_options().system_prompt == expected
    assert _session("codex", tools, tmp_path)._instructions == expected
    if _importable("omp_rpc"):
        captured = {}

        def omp_client(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("no omp here")

        omp = _session("omp", tools, tmp_path)
        omp._client_factory = omp_client
        asyncio.run(omp.start())
        assert captured["append_system_prompt"] == expected


def test_a_remote_s_instructions_are_cut_to_the_cap(fake, bus, tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "MAX_REMOTE_INSTRUCTIONS", 10)
    text = _tools(bus, tmp_path, _fake(fake)).remote_instructions
    assert text.endswith(INSTRUCTIONS[:10] + "\n\n[cut at 10 characters]")
    assert INSTRUCTIONS[10:] not in text


# ---------------------------------------------------------------------------
# a remote unreachable at startup, tried again while the session runs
# ---------------------------------------------------------------------------


def _unreachable_at_first(monkeypatch, fake, tmp_path):
    """A ViewerBus and a tool server whose one remote, ``fake``, could not be
    reached when it was built; the remote answers from now on."""
    from annealage_agent.viewers import ViewerBus

    real_discover = remote_module._discover
    monkeypatch.setattr(
        remote_module, "_discover", lambda servers: [OSError("unreachable")] * len(servers)
    )
    bus = ViewerBus(None, url="http://127.0.0.1:8765/")
    tools = _tools(bus, tmp_path, _fake(fake))
    monkeypatch.setattr(remote_module, "_discover", real_discover)
    assert tools.remotes == () and [r.name for r in tools.unreached] == ["fake"]
    return bus, tools


@pytest.mark.asyncio
async def test_a_remote_unreachable_at_startup_is_reached_by_reconnect(
    fake, tmp_path, monkeypatch, capsys
):
    bus, tools = _unreachable_at_first(monkeypatch, fake, tmp_path)
    reached = await tools.reconnect()
    assert [r.name for r in reached] == ["fake"]
    assert tools.unreached == ()
    assert {"fake__%s" % name for name in PROXIED} <= set(tools.host_tool_table())
    assert "the fake MCP server, unreachable until now, has been reached" in capsys.readouterr().err
    assert await tools.reconnect() == ()


@pytest.mark.asyncio
async def test_serve_s_retry_gives_an_omp_session_the_remote_on_the_first_turn(
    fake, tmp_path, monkeypatch
):
    from annealage_agent import app as app_module

    class _OmpLike:
        def __init__(self):
            self.tables = []

        async def set_tool_table(self, table):
            self.tables.append(table)
            return True

    bus, tools = _unreachable_at_first(monkeypatch, fake, tmp_path)
    session = _OmpLike()
    # The first turn arrives long before the timer would fire.
    retrying = asyncio.ensure_future(app_module.retry_remotes(tools, bus, session, interval=60))
    blocks = bus.begin_turn([{"type": "text", "text": "find me an LDO"}])
    await asyncio.wait_for(retrying, 15)

    (table,) = session.tables
    assert "fake__lookup" in table and "list_notes" in table
    # The model hears of it on its next turn, with what the remote says of itself.
    (note, _typed) = bus.begin_turn([{"type": "text", "text": "and now?"}])
    assert "The fake MCP server, which could not be reached" in note["text"]
    assert INSTRUCTIONS in note["text"]
    assert len(blocks) == 1


@pytest.mark.asyncio
async def test_a_session_that_cannot_take_new_tools_is_told_nothing_it_cannot_use(
    fake, tmp_path, monkeypatch, capsys
):
    """Claude's SDK servers and Codex's bridges are fixed when the session
    starts: the remote's tools wait for the next start, and the model is not
    told of tools it does not have."""
    from annealage_agent import app as app_module

    bus, tools = _unreachable_at_first(monkeypatch, fake, tmp_path)
    bus.first_turn.set()
    await asyncio.wait_for(app_module.retry_remotes(tools, bus, object(), interval=60), 15)
    assert "cannot take new tools mid-session" in capsys.readouterr().err
    assert bus.begin_turn([{"type": "text", "text": "hi"}]) == [{"type": "text", "text": "hi"}]


@pytest.mark.asyncio
async def test_a_failed_hand_off_to_the_session_is_tried_again_rather_than_lost(
    fake, tmp_path, monkeypatch
):
    from annealage_agent import app as app_module

    class _Flaky:
        def __init__(self):
            self.attempts = 0

        async def set_tool_table(self, table):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("set_host_tools timed out")
            return True

    bus, tools = _unreachable_at_first(monkeypatch, fake, tmp_path)
    bus.first_turn.set()
    session = _Flaky()
    await asyncio.wait_for(app_module.retry_remotes(tools, bus, session, interval=0.05), 15)
    assert session.attempts == 2
    (note, _typed) = bus.begin_turn([{"type": "text", "text": "hi"}])
    assert "The fake MCP server, which could not be reached" in note["text"]


def _importable(module):
    try:
        __import__(module)
    except ImportError:
        return False
    return True
