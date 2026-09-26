"""Tests for ``app.py``: ``serve``, and what ``create_app`` assembles.

Two properties of ``serve`` only a real bound socket can show, rather than
microdot's in-process ``TestClient``: that the listening socket is actually
accepting connections by the time ``on_ready`` fires, and that shutdown on
cancellation returns promptly even while a connection is still open and its
handler has not finished. The rest covers the event log a session publishes
into, the live ``session_info`` a ``hello`` reports, the viewer-only paths,
and the headers every response carries: the Content-Security-Policy computed
from the product page's inline scripts, and the ones beside it.
"""

import asyncio
import contextlib
import socket

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client
from toy_product import TOY_PAGE

from annealage_agent import app as agent_app

pytestmark = pytest.mark.asyncio


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_serve_is_accepting_connections_before_on_ready_fires(tmp_path):
    port = _free_port()
    ready = asyncio.Event()
    probe_done = asyncio.Event()
    probe_result = {}
    # asyncio's event loop holds only a weak reference to a task once it is
    # scheduled; a task with no strong reference anywhere else can be
    # garbage-collected mid-execution (its coroutine closed with
    # GeneratorExit at its current suspension point) at any point the
    # interpreter happens to run a collection. probe_task holds that
    # reference for the test's duration so the probe always runs to
    # completion rather than sometimes being cut short by GC timing.
    probe_task = None

    def on_ready():
        # on_ready runs synchronously inside the server's own coroutine, so
        # a blocking socket.create_connection here would deadlock the loop
        # it is trying to connect to; the connect attempt is scheduled as a
        # separate task and the test awaits its own completion signal
        # (probe_done) rather than guessing how long it needs, since a fixed
        # sleep's margin depends on unrelated load elsewhere in the process
        # (e.g. how much the interpreter has already allocated by the time
        # this runs) and is not a property of this test.
        async def probe():
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                # A streamed file, so this end closes first and the server's
                # side of the port is not left in TIME_WAIT, which would
                # refuse the plain bind below with no listener left at all.
                writer.write(b"GET /agent/static/chat.js HTTP/1.0\r\n\r\n")
                await writer.drain()
                first_line = await reader.readline()
                probe_result["status_line"] = first_line
                writer.close()
                await writer.wait_closed()
            except OSError as exc:
                probe_result["error"] = exc
            finally:
                probe_done.set()

        nonlocal probe_task
        probe_task = asyncio.ensure_future(probe())
        ready.set()

    app = create_toy_app(tmp_path, host=TEST_HOST, port=port)
    task = asyncio.ensure_future(agent_app.serve(app, TEST_HOST, port, on_ready=on_ready))
    try:
        await asyncio.wait_for(ready.wait(), timeout=2.0)
        await asyncio.wait_for(probe_done.wait(), timeout=2.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)
        if probe_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(probe_task, timeout=2.0)

    assert "error" not in probe_result
    assert probe_result.get("status_line", b"").startswith(b"HTTP/1.0 200")

    # Nothing is left listening on the port after shutdown.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


async def test_serve_returns_promptly_when_cancelled_with_a_connection_still_open(
    tmp_path, monkeypatch
):
    # Bounds the shutdown drain wait tightly so the test itself stays fast;
    # the property under test is that the bound is honoured, not its value.
    monkeypatch.setattr(agent_app, "SHUTDOWN_DRAIN_TIMEOUT", 0.1)
    port = _free_port()
    ready = asyncio.Event()

    async def runs_until_cancelled():
        # A product's background coroutine (Mesh's file watchers, say), which
        # never returns on its own either.
        await asyncio.Event().wait()

    app = create_toy_app(tmp_path, host=TEST_HOST, port=port)
    task = asyncio.ensure_future(
        agent_app.serve(
            app, TEST_HOST, port, on_ready=ready.set, background=(runs_until_cancelled,)
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=2.0)

    # A connection that never finishes sending its request headers: the
    # server's handler for it is genuinely in flight and will not complete
    # on its own, which is exactly the situation a slow model transfer or a
    # stalled client leaves behind.
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET / HTTP/1.0\r\n")
    await writer.drain()

    loop = asyncio.get_running_loop()
    start = loop.time()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)
    elapsed = loop.time() - start
    assert elapsed < 1.0, "shutdown did not honour SHUTDOWN_DRAIN_TIMEOUT"

    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()

    # The listening socket itself is gone, even though the dangling
    # connection's handler was still in flight when shutdown returned and
    # its own socket may briefly still hold the port. SO_REUSEADDR is set
    # here for the same reason a product CLI's port check sets it (Annealage
    # Mesh's cli.port_in_use): a new listener on this port must not be
    # refused just because one abandoned, non-listening connection from the
    # previous server has not yet been reaped.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


# ---------------------------------------------------------------------------
# The event log reaches disk, so a session can be resumed and priced
# ---------------------------------------------------------------------------


async def test_agent_mode_writes_the_conversation_to_the_sessions_event_log(tmp_path):
    """The 500-event ring covers a browser reconnecting; only the file covers
    the process exiting. `-c` resumes from it and `-r` prices a session from it,
    so a log with nowhere to go leaves both reading an empty history and
    reporting every session as 0 turns and $0.00.
    """
    from annealage_agent import sessions
    from annealage_agent.session.base import TurnEnd
    from annealage_agent.session.fake import FakeSession

    sid = sessions.create_session(tmp_path)
    built = []

    def build_session(on_event, *, bus):
        session = FakeSession(on_event, session_id=sid)
        built.append(session)
        return session

    create_toy_app(tmp_path, token="tok", session_id=sid, build_session=build_session)
    built[0].emit(TurnEnd(turn=1, stop_reason="end_turn", cost_usd=0.0125))
    built[0].emit(TurnEnd(turn=2, stop_reason="end_turn", cost_usd=0.0075))

    assert sessions.events_path(tmp_path, sid).is_file()

    info = sessions.get_session_info(tmp_path, sid)
    assert info.turn_count == 2
    assert info.cost_usd == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# session_info stays live: a browser tab connecting after a switch sees it
# ---------------------------------------------------------------------------


class _RawSock:
    def __init__(self, initial_bytes):
        self.buffer = initial_bytes
        self.written = []

    async def read(self, n):
        data = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return data

    async def readexactly(self, n):
        return await self.read(n)

    async def readline(self):
        line = b""
        while True:
            byte = await self.read(1)
            if not byte:
                return line
            line += byte
            if line[-1:] == b"\n":
                return line

    async def awrite(self, data):
        self.written.append(bytes(data))


def _decode_text_frame(frame_bytes):
    import struct

    opcode = frame_bytes[0] & 0x0F
    length = frame_bytes[1] & 0x7F
    offset = 2
    if length == 126:
        length = struct.unpack("!H", frame_bytes[2:4])[0]
        offset = 4
    elif length == 127:
        length = struct.unpack("!Q", frame_bytes[2:10])[0]
        offset = 10
    return opcode, frame_bytes[offset : offset + length]


async def _hello_of(app):
    """The ``hello`` frame a tab connecting now would get from ``app``.

    ``TestClient.websocket()``'s fake socket drops anything the server sends
    before its own first ``read()``, which is exactly when ``hello`` is sent
    (test_ws_auth.py's own documented reason for the same workaround), so
    this drives one real ``/ws`` handshake through ``app.dispatch_request``
    with a raw duplex buffer and decodes the literal ``hello`` frame bytes.
    """
    import json

    from microdot import Response
    from microdot.microdot import Request
    from microdot.websocket import WebSocket

    from annealage_agent import protocol

    client = make_test_client(app)
    headers = {
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
        "Origin": "http://127.0.0.1:%d" % DEFAULT_PORT,
    }
    request_bytes = client._render_request("GET", "/ws?t=tok", headers, b"")
    client_hello = json.dumps(
        {"v": protocol.PROTOCOL_VERSION, "type": "hello", "token": "tok", "last_seq": 0}
    )
    hello_frame = WebSocket._encode_websocket_frame(WebSocket.TEXT, client_hello)
    sock = _RawSock(request_bytes + bytes(hello_frame))
    req = await Request.create(client.app, sock, sock, ("127.0.0.1", 1234), scheme=None)
    res = await client.app.dispatch_request(req)
    assert res is Response.already_handled
    # The upgrade handshake's own HTTP response lines go through the same
    # `awrite` call `_RawSock` records, ahead of the first real WebSocket
    # frame; `\x81` (FIN + TEXT opcode) is what distinguishes it from those.
    ws_frames = [f for f in sock.written if f[:1] == b"\x81"]
    assert ws_frames, "the server never sent a hello frame"
    opcode, payload = _decode_text_frame(ws_frames[0])
    assert opcode == WebSocket.TEXT
    return json.loads(payload)


def _app_with_fake_session(served_dir, **kwargs):
    from annealage_agent import sessions
    from annealage_agent.session.fake import FakeSession

    built = []
    sid = sessions.create_session(served_dir)

    def build_session(on_event, *, bus):
        built.append(FakeSession(on_event, session_id=sid))
        return built[-1]

    app = create_toy_app(
        served_dir, token="tok", session_id=sid, build_session=build_session, **kwargs
    )
    return app, built[0]


async def test_a_fresh_connection_after_a_live_model_switch_sees_the_new_model(served_dir):
    """``session_info["model"]`` is built once from ``settings`` at session
    construction; without ``_event_publisher`` writing a live
    ``AgentModelChanged`` back into that same dict, a browser tab connecting
    after the switch only recovers the running model while the event
    announcing it is still inside the replay ring buffer -- once evicted, a
    fresh ``hello`` would permanently show the CLI-configured starting model
    instead of what the session is actually running."""
    from annealage_agent.session.base import AgentModelChanged

    app, session = _app_with_fake_session(served_dir, settings={"model": "claude-opus-4"})
    # The live switch this fix keeps session_info current for -- a real
    # driver emits this once its own control-plane set_model call takes
    # effect (session/omp.py, session/sdk.py, session/codex.py all do).
    session.emit(AgentModelChanged(model="claude-haiku-5"))
    # And the agent's status as it is now, not as it was when the app was
    # built: the page takes it from here, not from replayed agent_status
    # events, which may be an earlier process's.
    session.set_status("unavailable")

    hello = await _hello_of(app)
    # The property under test: a connection opened *after* the switch reads
    # the live model, not the settings-time snapshot ("claude-opus-4") --
    # even though this event has never left the still-full replay ring.
    assert hello["session"]["model"] == "claude-haiku-5"
    assert hello["session"]["agent"] == "unavailable"


async def test_a_page_opened_while_the_agent_is_down_is_told_why(served_dir):
    """The page raises no banner from a replayed ``agent_error``, so a tab
    opened after the agent failed to start (omp not logged in) would show an
    unavailable pane with no reason. The hello carries the latest error while
    the agent is not ready, and nothing once it is."""
    from annealage_agent.session.base import AgentError, AgentStatus

    app, session = _app_with_fake_session(served_dir)
    session.emit(AgentError(stderr="401 no credentials", remediation="log omp in"))
    session.set_status("unavailable")
    session.emit(AgentStatus(status="unavailable"))
    down = await _hello_of(app)
    assert down["session"]["agent_error"] == {
        "remediation": "log omp in",
        "stderr": "401 no credentials",
    }

    session.set_status("ready")
    session.emit(AgentStatus(status="ready"))
    up = await _hello_of(app)
    assert up["session"]["agent_error"] is None


async def test_a_human_turn_the_last_process_never_answered_is_closed_on_resume(served_dir):
    """The process died after logging the human's message and before the
    agent said anything. The resumed app closes that turn, so the page does
    not show it as running, and continues the numbering after it."""
    from annealage_agent import sessions
    from annealage_agent.session.base import TextDelta, TurnEnd, UserTurn
    from annealage_agent.session.events import EventLog, read_records
    from annealage_agent.session.fake import FakeSession

    sid = sessions.create_session(served_dir)
    log = EventLog(str(sessions.events_path(served_dir, sid)))
    log.append(UserTurn(turn=1, blocks=[{"type": "text", "text": "first"}]))
    log.append(TextDelta(turn=1, text="answered"))
    log.append(TurnEnd(turn=1, stop_reason="end", cost_usd=0.0))
    log.append(UserTurn(turn=2, blocks=[{"type": "text", "text": "never answered"}]))
    log.close()

    app = create_toy_app(
        served_dir,
        token="tok",
        session_id=sid,
        build_session=lambda on_event, *, bus: FakeSession(on_event, session_id=sid),
    )
    try:
        ends = [
            (r["event"]["turn"], r["event"]["stop_reason"])
            for r in read_records(sessions.events_path(served_dir, sid))
            if r["event"]["kind"] == "turn_end"
        ]
        assert ends == [(1, "end"), (2, "interrupted")]
        assert app.agent_bus.turn == 2
    finally:
        app.agent_event_log.close()


async def test_a_permission_request_the_last_process_never_resolved_is_closed_on_resume(
    served_dir,
):
    """Killed while a card waited: nobody can answer it now, so the resumed
    app resolves it, and the page replaying the history shows no live card.
    An answered request is left alone."""
    from annealage_agent import sessions
    from annealage_agent.session.base import PermissionRequest, PermissionResolved
    from annealage_agent.session.events import EventLog
    from annealage_agent.session.fake import FakeSession

    sid = sessions.create_session(served_dir)
    log = EventLog(str(sessions.events_path(served_dir, sid)))
    log.append(PermissionRequest(request_id="pr_1", tool="add_note", input={}))
    log.append(PermissionResolved(request_id="pr_1", outcome="allow"))
    log.append(PermissionRequest(request_id="pr_2", tool="add_note", input={}))
    log.close()

    app = create_toy_app(
        served_dir,
        token="tok",
        session_id=sid,
        build_session=lambda on_event, *, bus: FakeSession(on_event, session_id=sid),
    )
    try:
        open_cards = set()
        for _seq, wire in app.agent_event_log.replay(0).events:
            if wire["kind"] == "permission_request":
                open_cards.add(wire["request_id"])
            elif wire["kind"] == "permission_resolved":
                open_cards.discard(wire["request_id"])
        assert open_cards == set()
        resolved = [
            (w["request_id"], w["outcome"])
            for _s, w in app.agent_event_log.replay(0).events
            if w["kind"] == "permission_resolved"
        ]
        assert resolved == [("pr_1", "allow"), ("pr_2", "shutdown")]
    finally:
        app.agent_event_log.close()


async def test_viewer_only_mode_writes_no_event_log(tmp_path):
    """There is no session and no conversation, so there is nothing to persist
    and nothing to create a session directory for."""
    from annealage_agent import sessions

    create_toy_app(tmp_path, token="tok")

    assert not sessions.sessions_dir(tmp_path).exists()


async def test_a_session_factory_that_returns_none_builds_a_working_app(tmp_path):
    """Viewer-only mode passes a factory that answers None, and the app that
    comes back must serve the product's page with no agent attached.

    The distinction this pins is between two things that look alike from the
    outside: ``build_session=None`` (no factory at all) and a factory whose
    answer is None (viewer-only, decided inside the closure a product's CLI
    builds). Only the first was covered while the second was what every
    viewer-only run (Annealage Mesh's ``--no-agent``) actually did, so the
    whole viewer-only path could crash on startup with the suite green.
    """
    calls = []

    def build_session(on_event, *, bus):
        calls.append(bus)
        return

    app = create_toy_app(tmp_path, token="tok", build_session=build_session)

    assert calls, "the factory must still be called; it is what decides"
    assert app.agent_session is None
    res = await make_test_client(app).get("/")
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# What the bus gives a product's tools: attention, and ending a turn
# ---------------------------------------------------------------------------


async def test_attention_is_published_to_the_page_and_recorded_in_the_event_log(tmp_path):
    from annealage_agent import sessions
    from annealage_agent.session.fake import FakeSession

    sid = sessions.create_session(tmp_path)
    app = create_toy_app(
        tmp_path,
        token="tok",
        session_id=sid,
        build_session=lambda on_event, *, bus: FakeSession(on_event, session_id=sid),
    )
    app.agent_bus.attention("Checkpoint: circuit", "Is the power tree right?")
    replay = app.agent_event_log.replay(0)
    assert [event for _seq, event in replay.events] == [
        {"kind": "attention", "title": "Checkpoint: circuit", "body": "Is the power tree right?"}
    ]
    assert '"kind": "attention"' in sessions.events_path(tmp_path, sid).read_text()


async def test_a_tool_s_end_turn_reaches_the_session_that_can_stop_its_turn(tmp_path):
    from annealage_agent import sessions
    from annealage_agent.session.fake import FakeSession

    class _Stoppable(FakeSession):
        ended = 0

        def end_turn_after_tool(self):
            self.ended += 1

    sid = sessions.create_session(tmp_path)
    app = create_toy_app(
        tmp_path,
        token="tok",
        session_id=sid,
        build_session=lambda on_event, *, bus: _Stoppable(on_event, session_id=sid),
    )
    app.agent_bus.request_end_turn()
    assert app.agent_session.ended == 1


# ---------------------------------------------------------------------------
# A fixed token, and the name and origin a proxy fronts the server under
# ---------------------------------------------------------------------------


async def test_an_app_behind_a_proxy_accepts_the_host_and_origin_it_is_given(tmp_path):
    from microdot.test_client import TestClient

    name, origin = "loom.tail1234.ts.net", "https://loom.tail1234.ts.net"
    app = create_toy_app(tmp_path, token="kept-token", extra_origins=(origin,), extra_hosts=(name,))
    proxied = TestClient(app, host=name)
    res = await proxied.get("/settings?t=kept-token", headers={"Origin": origin})
    assert res.status_code == 200
    res = await proxied.get("/settings?t=other-token", headers={"Origin": origin})
    assert res.status_code != 200

    # The same request to an app not told the name is refused before any route.
    plain = create_toy_app(tmp_path, token="kept-token")
    res = await TestClient(plain, host=name).get(
        "/settings?t=kept-token", headers={"Origin": origin}
    )
    assert res.status_code != 200


# ---------------------------------------------------------------------------
# The content security policy
# ---------------------------------------------------------------------------


async def test_every_response_carries_the_policy(tmp_path):
    """Set as an after_request hook rather than on the one HTML route, so a
    response added later cannot arrive without it."""
    client = make_test_client(create_toy_app(tmp_path, token="tok"))

    for path in ("/", "/settings", "/agent/static/chat.js"):
        res = await client.get(path)
        policy = res.headers.get("Content-Security-Policy")
        assert policy, path
        assert "default-src 'none'" in policy
        assert res.headers.get("Referrer-Policy") == "no-referrer"


async def test_the_policy_names_the_import_map_by_hash_not_by_unsafe_inline(tmp_path):
    """The page has exactly one inline script, so it is allowed by content
    rather than by category. 'unsafe-inline' would also allow any script an
    injection managed to place in the markup."""
    policy = agent_app.content_security_policy(TOY_PAGE)
    hashes = agent_app.inline_script_hashes(TOY_PAGE)
    assert len(hashes) == 1
    assert hashes[0] in policy
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy


async def test_the_hash_is_computed_from_the_file_that_is_served(tmp_path):
    """Computed at startup rather than written down, so editing the import map
    cannot leave a policy that blocks the page it is meant to allow."""
    html = tmp_path / "page.html"
    html.write_text(
        '<html><head><script type="importmap">{"imports":{}}</script></head>'
        '<body><script type="module" src="/static/js/main.js"></script></body></html>',
        encoding="utf-8",
    )
    first = agent_app.inline_script_hashes(html)
    assert len(first) == 1, "the sourced script has no body and contributes no hash"

    html.write_text(
        '<html><head><script type="importmap">{"imports":{"three":"/x.js"}}</script>'
        "</head><body></body></html>",
        encoding="utf-8",
    )
    assert agent_app.inline_script_hashes(html) != first


async def test_a_missing_page_yields_a_policy_that_allows_no_inline_script(tmp_path):
    """Failing closed: a policy that could not read the page refuses its inline
    script rather than falling back to allowing every inline script."""
    policy = agent_app.content_security_policy(tmp_path / "absent.html")
    assert "script-src 'self'" in policy
    assert "sha256-" not in policy
    assert "unsafe-inline" not in policy


async def test_the_policy_allows_what_the_page_actually_needs(tmp_path):
    """Each of these is in the policy because something in a product's page
    needs it, and a change that drops one would break the page in a way only a
    browser suite would otherwise catch."""
    policy = agent_app.content_security_policy(TOY_PAGE)
    # A product page may composite over a canvas snapshot through an Image
    # whose src is a data URL (Annealage Mesh's sketch overlay does).
    assert "img-src 'self' data:" in policy
    # /ws is the transport.
    assert "connect-src 'self' ws: wss:" in policy
    # Nothing frames this page, and nothing rewrites its relative URLs.
    assert "frame-ancestors 'none'" in policy
    assert "base-uri 'none'" in policy


async def test_responses_carry_nosniff_header(client):
    res = await client.get("/")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
