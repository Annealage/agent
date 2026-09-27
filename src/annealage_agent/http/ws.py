"""The ``/ws`` route: authentication, the greeting, replay, and frame dispatch.

A bug in this file is not a UI bug. ``/ws`` is the one route a browser tab
turns into a live bidirectional channel, and from the milestone that lands an
agent it carries turns into a Claude Code session with shell access in the
human's project. A check that can be bypassed here is a remote-code-execution
bug, so the checks below are independent, each is commented with what it
defends against and why the others do not cover it, and all of them run
**before** the WebSocket handshake: the Host check (app.py's hook), then the
app's ``BrowserAuth`` (``identity.py``: the Origin, then a tailnet login or
the token).

Refusing before the handshake, rather than upgrading and then closing with a
code, is deliberate and load-bearing in two ways. It means no connection
object, no queue and no writer task is ever created for a request that failed
authentication, so a refused caller cannot consume per-connection resources.
And it is the only form of refusal that microdot's in-process test client can
observe at all: that client's fake socket discards every outbound frame whose
opcode is not TEXT or BINARY, so a test asserting a close code through it
asserts on a frame the harness never delivered and cannot fail. An HTTP status
is checkable; a close code, on that path, is not.

Every refusal returns the same status and the same body. A response that
distinguished "no token" from "wrong token", or "bad Origin" from "bad Host",
would answer questions an unauthenticated caller should not be able to ask.
"""

import asyncio
import json
import sys
import time

from microdot import Response
from microdot.websocket import WebSocket, WebSocketError, websocket_upgrade

from .. import protocol
from ..session.base import AGENT_READY, PauseChanged, TurnEnd, UnknownRequest, UserTurn
from ..viewers import ViewerRegistry

# Inbound frame ceiling, and a line that must stay. Microdot's default,
# ``max_message_length = -1``, means "fall back to Request.max_body_length",
# and app.py sets that to 0 so every request body is streamed rather than
# buffered. An inherited ceiling of zero refuses every frame, so without this
# assignment a viewer's opening ``hello`` is rejected and the socket closes
# with nothing a test client can read off the wire.
#
# 4 MiB rather than something tighter because of one frame kind: the ``result``
# answering a product tool that captures the page's view (Mesh's
# ``capture_view``) carries a screenshot as a base64 data URL, and a
# 1568-pixel-wide render (Mesh's of a shaded part) does not fit
# in the few hundred kilobytes every other frame needs. The consequence of
# getting this wrong is worse than a large buffer: microdot raises
# ``WebSocketError('Message too large')`` on an oversized frame, which this
# route cannot answer, so the connection simply drops and the human's tab
# reconnects with no explanation. Mesh's ``static/js/commands.js`` keeps its own,
# smaller cap and re-encodes a capture until it fits, so a reply that would
# hit this limit is not supposed to be sent at all; this is the backstop.
MAX_WS_MESSAGE = 4 * 1024 * 1024

# How often every connected viewer is pinged.
#
# This is what makes the browser's own liveness watchdog meaningful, and the
# two numbers are a pair: static/ws.js closes a socket that has delivered
# nothing for LIVENESS_TIMEOUT_MS, so without a ping this interval, an idle but
# perfectly healthy connection looks dead and gets closed and reopened on a
# loop. The watchdog there must stay at least twice this value, and says so.
#
# A dead-but-open socket is the case that needs it: a phone that slept or a
# tailnet that rekeyed leaves a connection whose readyState is still OPEN for
# minutes, with no close event, so the only way the page learns is by noticing
# that nothing is arriving any more.
PING_INTERVAL = 5.0
_REFUSED_BODY = "forbidden"
_REFUSED_STATUS = 403


def refusal():
    """The one response every failed authentication check returns."""
    return Response(_REFUSED_BODY, _REFUSED_STATUS)


def host_is_allowed(req, allowed_hosts):
    """True if this request's ``Host`` header may name this server.

    An absent ``Host`` is allowed. HTTP/1.1 requires the header and every
    browser sends it, so absent means a non-browser client (curl, a test
    harness, a script), which is not the caller this check exists for. The
    caller it exists for is a browser that has been pointed at this server by
    a hostname whose DNS answer an attacker controls and has repointed at this
    address after the page loaded. That page's ``Origin`` is the attacker's
    own site, so the Origin check catches the browser case; this check catches
    the request that carries no Origin at all, and it applies to every route
    rather than only ``/ws``, because a rebound name can read ``/manifest``
    and write through ``/submit`` just as readily as it can open a socket.

    Comparison is against the literal set of spellings that can truthfully
    name this bind, with no parsing of the header. A parser would have to
    decide what to do with userinfo, a trailing dot, and unbalanced IPv6
    brackets, and each of those decisions is a chance to extract an address
    from a header that does not actually name it.
    """
    host = req.headers.get("Host")
    if host is None:
        return True
    return host in allowed_hosts


def register_ws(
    app,
    *,
    auth,
    token,
    allowed_hosts,
    registry,
    event_log,
    session_info,
    holder,
):
    """Register ``/ws`` on ``app``.

    ``auth`` is the app's ``identity.BrowserAuth``, the check every browser
    route makes; the ``Human`` it returns for the upgrade is the connection's
    (``conn.human``), and what the human does over it (a turn, a permission
    decision, the pause switch) is recorded with their login. ``token`` is the
    browser token the ``hello`` frame is checked against when the upgrade was
    authenticated by it. With neither a token nor an identity every ``/ws``
    request is refused. That is the safe reading rather than an inconvenient
    one: a socket nobody can authenticate is a socket that should not open.

    ``holder`` is the app's ``AgentHolder`` (``app.py``), read as each
    connection opens and as each frame arrives rather than once here: its
    ``session`` is the live session, ``None`` in viewer-only mode, and a
    connection or an agent frame (``turn``, ``interrupt``, ``permission``,
    ``set_model``) reaching an app its idle timer closed resumes one first
    (``holder.ensure()``). Its ``bus`` is the ``ViewerBus`` holding the human's
    pause switch, handed on only while a session exists, since viewer-only
    mode has no agent tools to pause. The bus is read for the ``hello`` frame
    and written by an inbound ``pause`` frame; this module never calls
    through it, because a ``call`` originates with a tool, never with a
    socket.
    """

    @app.get("/ws")
    async def ws_route(req):
        # Host is enforced for every route by app.py's before_request hook and
        # so has already passed by here. The upgrade is not a plain read, so a
        # tailnet login counts only with an Origin this server serves.
        human = auth.authenticate(req)
        if human is None:
            return refusal()

        ws = await websocket_upgrade(req)
        ws.max_message_length = MAX_WS_MESSAGE
        conn = None
        try:
            # After the upgrade, so the page's plain-HTTP refusal probe
            # (static/ws.js) never starts a session; noted as activity, so
            # the idle sweep does not close the session this page is greeted
            # with before the page counts as connected.
            session = await holder.ensure()
            holder.note_activity()
            bus = holder.bus if session is not None else None
            # Greeting and replay both happen before the connection is
            # registered, and that order is load-bearing. Registering starts a
            # writer task that also sends on this socket, and a replayed event
            # carries an older seq than anything live: if a replay frame were
            # written from here while that writer was draining a live
            # broadcast, the client could see an older seq after a newer one,
            # move its resync position backwards, and replay the same events
            # again on its next reconnect. With nothing registered yet, there
            # is no second writer for those frames to race.
            # The agent's status is read from the session itself, not the
            # snapshot session_info took at startup: the page takes the status,
            # and why the agent is down, from this frame and from live events
            # only (a replayed one may be an earlier process's), so both have
            # to be as they are now.
            agent = session.agent_status() if session is not None else None
            agent = agent or session_info.get("agent", "unavailable")
            await ws.send(
                json.dumps(
                    protocol.build_hello(
                        event_log.current_seq,
                        session_info["id"],
                        session_info.get("sdk_session_id"),
                        session_info["cwd"],
                        agent,
                        paused=bus.paused if bus is not None else False,
                        model=session_info.get("model"),
                        steers=session_info.get("steers", False),
                        agent_error=(
                            session_info.get("agent_error") if agent != AGENT_READY else None
                        ),
                        usage=session_info.get("usage"),
                    )
                )
            )
            # The hello's token is checked only when the token opened the
            # socket: a page signed in by its tailnet login may hold none.
            hello = await _greet(
                ws, event_log, token if human.login is None else None, session_info["id"]
            )
            if hello is None:
                return Response.already_handled
            viewer = hello.get("viewer") or {}
            conn = await registry.add(ws, tab_id=viewer.get("tab_id"), human=human)
            await _serve_connection(ws, conn, registry, event_log, token, holder)
        except WebSocketError:
            # The peer closed, or sent a frame microdot could not read. Not
            # an error worth reporting: a browser tab closing is the ordinary
            # end of every connection.
            pass
        finally:
            if conn is not None:
                await registry.remove(conn)
        return Response.already_handled


def _token_is_allowed(req, token):
    """Constant-time check of the ``t`` query parameter against the run's token.

    The token travels as a query parameter for exactly one reason: a browser
    cannot set headers on a WebSocket handshake, so a header is not available
    on the only route that needs this. It stays out of our own stderr because
    the access log prints ``req.path``, which microdot splits from the query
    string, and never ``req.query_string``.

    A missing token is compared against the configured one anyway rather than
    returning early, so the refusal path costs the same either way and a
    missing token cannot be told apart from a wrong one.

    A request supplying ``t`` more than once is refused outright rather than
    resolved. ``req.args`` is a MultiDict whose lookup returns the first value,
    so ``?t=<real>&t=wrong`` would otherwise authenticate: that is fine on its
    own, since a caller who already knows the token gains nothing, but it means
    the answer to "which value counts" is a property of one parser. Anything in
    front of this server, a reverse proxy or ``tailscale serve``, may pick the
    other one, and a check whose result depends on which component looked is a
    check waiting to disagree with itself. One value, or no.
    """
    if not token:
        return False
    supplied_values = (
        req.args.getlist("t")
        if hasattr(req.args, "getlist")
        else [req.args["t"]]
        if "t" in req.args
        else []
    )
    if len(supplied_values) > 1:
        return False
    supplied = supplied_values[0] if supplied_values else ""
    return _constant_time_equal(supplied, token)


def _constant_time_equal(a, b):
    import hmac

    return hmac.compare_digest(
        a.encode("utf-8", "surrogatepass"), b.encode("utf-8", "surrogatepass")
    )


def _origin_is_allowed(req, allowed_origins):
    """True if this request's ``Origin`` is one this bind serves.

    An absent ``Origin`` is allowed, and the reason is specific: a browser
    always sends one on a WebSocket handshake, so absent means a non-browser
    client, which still has to present the token. (A tailnet login needs an
    ``Origin`` on the handshake; ``identity.BrowserAuth`` says why.) The threat
    this check exists for is a page the human happens to visit opening a socket
    to this server, and that page cannot suppress its own ``Origin``.

    This is not redundant with the token. **A WebSocket handshake is not
    subject to the same-origin policy**, so a malicious page can open this
    socket cross-origin and read every reply; without this check, the only
    thing between such a page and an agent with shell access would be whether
    the token ever appeared somewhere a page could read.

    Membership is exact. Prefix or substring matching would accept
    ``http://127.0.0.1`` (a truncation) and
    ``http://evil.example/http://127.0.0.1:8765`` (a containment), neither of
    which is a value this server ever issued.
    """
    origin = req.headers.get("Origin")
    if origin is None:
        return True
    return origin in allowed_origins


async def _greet(ws, event_log, token, session_id=None):
    """Read the client's ``hello`` and answer it, before any writer exists.

    Returns the hello frame, or ``None`` if the connection was closed here, in
    which case the caller must not register it. Nothing else is read: a client
    that sends some other frame first is answered with a refusal and asked
    again, because ``hello`` is what carries the resync position and there is
    nothing useful to do without it.

    ``token`` is what the hello's ``token`` must equal: the browser token, for
    a socket the token opened. ``None`` checks nothing, for a socket a tailnet
    login opened, whose page may hold no token at all.

    The answer is the history since the client's ``last_seq``: the ring's,
    and before that the event log's file, read off the event loop, which is
    what brings a reloaded page, or any page after a restart, the whole
    conversation (``EventLog.replay_async``). A ``last_seq`` is a position in
    one session's log, so one the client does not say belongs to this
    session (``session_id``, this server's) is no position here, and the
    client is answered from the start. Events appended while the history is
    being read or sent (a turn streaming meanwhile) are caught up with here
    too, round after round, each starting where the last one's snapshot
    ended, until one finds nothing new. A broadcast scheduled before that
    last round may still reach the connection once it is registered, with an
    event this replay already sent; the page drops any event at or below the
    last seq it has (``static/ws.js``). A history that really is unavailable
    (an in-memory log whose ring has moved on) is said so, once.

    A client that connects and then says nothing is never registered, so it
    receives no events and consumes no writer task. That is the right outcome
    rather than an oversight: it is not a viewer until it identifies itself,
    and replay from its ``last_seq`` covers everything it misses in the gap.
    """
    while True:
        raw = await ws.receive()
        frame = await _parse(ws, raw)
        if frame is None:
            continue
        if frame is _CLOSED:
            return None
        if frame["type"] != "hello":
            await ws.send(
                json.dumps(
                    protocol.build_refused("expected a hello frame first, got %s" % frame["type"])
                )
            )
            continue
        # The token is re-checked here even though the query parameter already
        # authenticated the connection. It costs one comparison, and a hello
        # whose token disagrees with the one that opened the socket is a
        # confused or hostile client either way.
        if token is not None and not _constant_time_equal(frame.get("token") or "", token):
            await protocol.close_with_code(
                ws, protocol.CLOSE_VERSION_MISMATCH, "hello token does not match"
            )
            return None
        position = frame.get("last_seq")
        if frame.get("session_id") != session_id:
            position = None
        told = False
        while True:
            replay = await event_log.replay_async(position)
            position = replay.through
            if not replay.events and not replay.truncated:
                return frame
            for seq, event_wire in replay.events:
                await ws.send(json.dumps(protocol.build_event(seq, event_wire)))
            if replay.truncated and not told:
                told = True
                await ws.send(
                    json.dumps(
                        protocol.build_refused(
                            "the history before this point is no longer kept, so it is not shown"
                        )
                    )
                )


#: Returned by _parse when it has already closed the connection.
_CLOSED = object()


async def _parse(ws, raw):
    """Parse and validate one inbound frame.

    Returns the frame, None if it was refused and the connection continues, or
    ``_CLOSED`` if the version mismatch ended it. A protocol-version mismatch
    is the one failure that ends a connection with a coded close rather than an
    HTTP status, because by definition it is discovered after the handshake:
    the version is inside a frame. 4400 is what separates a stale cached page
    from a newer server.
    """
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        await ws.send(json.dumps(protocol.build_refused("frame is not valid JSON")))
        return None
    try:
        ok, result = protocol.validate_inbound(parsed)
    except protocol.ProtocolVersionMismatch as mismatch:
        await protocol.close_with_code(
            ws,
            protocol.CLOSE_VERSION_MISMATCH,
            "protocol version %s, this server speaks %d"
            % (mismatch.args[0] if mismatch.args else "?", protocol.PROTOCOL_VERSION),
        )
        return _CLOSED
    if not ok:
        await ws.send(json.dumps(protocol.build_refused(result)))
        return None
    return result


#: The frames that need a session, which one reaching an idle-closed app
#: resumes first.
_AGENT_FRAMES = frozenset(("turn", "interrupt", "permission", "set_model"))


async def _serve_connection(ws, conn, registry, event_log, token, holder):
    """Read and dispatch frames until the peer goes away.

    Every frame this sends is either a ``refused``, which carries no seq, or a
    terminal coded close. Neither has an ordering relationship with the events
    the connection's writer task is draining onto the same socket, so writing
    them from here cannot reorder anything the client tracks. Frames cannot
    interleave mid-frame either: microdot's ``awrite`` is a single
    ``StreamWriter.write`` of the whole frame followed by a ``drain``, and
    ``write`` buffers synchronously.

    The session each frame goes to is the holder's at that moment, so a
    session resumed while this connection was open is the one it reaches.
    """
    while True:
        raw = await ws.receive()
        frame = await _parse(ws, raw)
        if frame is None:
            continue
        if frame is _CLOSED:
            return
        if frame["type"] in _AGENT_FRAMES:
            await holder.ensure()
        session = holder.session
        bus = holder.bus if session is not None else None
        await _dispatch(ws, conn, registry, event_log, token, frame, session, bus)


async def _dispatch(ws, conn, registry, event_log, token, frame, session=None, bus=None):
    """Route one validated inbound frame."""
    kind = frame["type"]
    if kind == "hello":
        # The opening hello is answered by _greet, before this connection was
        # registered. A second one is not a resync request: replaying from here
        # would write older seqs onto a socket whose writer task is draining
        # newer ones. It is treated as interaction, which re-elects this viewer
        # as the primary, and refused with that said plainly.
        await registry.touch(conn)
        await ws.send(
            json.dumps(
                protocol.build_refused(
                    "hello was already answered; reconnect to resync from a last_seq"
                )
            )
        )
        return
    if kind in ("result", "error"):
        registry.resolve_call(conn, frame)
        return
    if protocol.is_product_frame(kind):
        # The product's page reporting its own state (Mesh's ``state``: camera,
        # visibility, selection, mode). The agent layer does nothing with its
        # content; receiving one is interaction with this tab, which re-elects
        # it as the primary viewer.
        await registry.touch(conn)
        return
    if kind == "pause":
        if bus is None:
            # Viewer-only: there are no agent tools to pause, so a control that
            # appeared to work would be worse than one that says so.
            await ws.send(
                json.dumps(
                    protocol.build_refused(
                        "this server is running viewer-only, so there are no agent tools to pause"
                    )
                )
            )
            return
        await registry.touch(conn)
        if not bus.set_paused(frame["paused"]):
            # Already in that state, which two tabs racing the same control
            # produce routinely. Nothing changed, so nothing is announced: an
            # event per redundant click would make every other tab re-render for
            # no reason.
            return
        # Announced to every viewer, not answered to this one, because the
        # switch is one property of the server and each tab has a control
        # showing it. Through the log, so the seq stays part of the one
        # monotonic stream a reconnecting client replays from, and scheduled
        # behind any broadcast a session has already scheduled, so it reaches
        # each page in seq order (see ViewerRegistry._broadcast_primary).
        event = PauseChanged(paused=bus.paused, by=_login(conn))
        seq = event_log.append(event)
        await asyncio.ensure_future(registry.broadcast(protocol.build_event(seq, event.to_wire())))
        return
    if kind in _AGENT_FRAMES:
        if session is None:
            # Viewer-only: the frames are still defined and validated so one
            # browser build works against both modes, and answering with a
            # reason is what stops a chat pane waiting forever on a turn
            # nothing will ever process.
            await ws.send(
                json.dumps(
                    protocol.build_refused(
                        "this server is running viewer-only, so %s frames are not served" % kind,
                        frame.get("client_id"),
                    )
                )
            )
            return
        await registry.touch(conn)
        # Handed to the session and not awaited for a result: a turn produces
        # events, which arrive over this same socket through the registry, and
        # blocking this connection's read loop on the whole turn would stop it
        # reading the interrupt frame that ends it.
        if kind == "turn":
            await _submit_turn(ws, conn, registry, event_log, frame, session, bus)
            return
        try:
            if kind == "interrupt":
                await session.interrupt()
            elif kind == "set_model":
                await session.set_model(frame["model"])
            else:
                await session.decide_permission(
                    frame["request_id"],
                    frame["decision"],
                    frame.get("message", ""),
                    by=_login(conn),
                )
        except UnknownRequest:
            # Ordinary, not a failure: two tabs held one card and this is the
            # one that lost, or the request expired before the click landed.
            # Answered with what actually happened rather than with the generic
            # message below, because "already decided" tells the human their
            # click changed nothing, which is the whole point of replying.
            await ws.send(
                json.dumps(
                    protocol.build_refused(
                        "that permission request was already decided, by another view or "
                        "by expiring; this decision was not applied"
                    )
                )
            )
        except Exception as exc:
            # A session that fails must not take the socket with it: the viewer
            # half of this page keeps working whatever the agent does.
            sys.stderr.write("warning: session could not handle a %s frame: %r\n" % (kind, exc))
            await ws.send(
                json.dumps(
                    protocol.build_refused(
                        "the agent could not handle that %s frame; see the server output" % kind
                    )
                )
            )
        return
    await ws.send(json.dumps(protocol.build_refused("unhandled frame type: %s" % kind)))


async def _submit_turn(ws, conn, registry, event_log, frame, session, bus):
    """Hand one ``turn`` frame to the session, logging what the human sent.

    Counted, and the product's queued notes put in front of it, here in front
    of every backend (``ViewerBus.begin_turn``), and only for a ready session.
    A turn to a connecting or unavailable session (omp while ``-c`` resumes,
    Codex while its thread starts) is refused here and never reaches the
    session: every backend numbers the turns it is given, so a turn it saw
    that ``bus.turn`` did not count would leave every later ``user_turn`` one
    behind the reply it belongs to, and a message the page was told was not
    sent must not then be answered. A running turn leaves the session ready,
    so a steer counts.

    At that same moment, and before the session sees the turn, the human's
    own blocks go into the event log as a ``user_turn`` numbered ``bus.turn``
    (the number the backend gives the reply), so the history holds them
    ahead of everything the turn produces and every tab, a reload and a
    restart show the human's side of the conversation. A turn that is not
    taken is refused back to this tab with its ``client_id``, so the page
    puts the message back rather than showing it against a later turn. A
    logged turn the session then fails on is ended as ``rejected``, or the
    page would show it running until a restart closed it.

    Published the way a session publishes (``app._event_publisher``): the
    broadcast is scheduled behind any a streaming session has already
    scheduled, so the page receives events in seq order even when a steer
    lands mid-stream.
    """
    blocks = frame["blocks"]
    client_id = frame.get("client_id")
    status = session.agent_status()
    if status != AGENT_READY:
        await ws.send(
            json.dumps(
                protocol.build_refused(
                    "the agent is %s, so this message was not sent" % status, client_id
                )
            )
        )
        return
    turn = None
    if bus is not None:
        blocks_to_send = bus.begin_turn(blocks, by=conn.human)
        turn = bus.turn
        _publish(
            event_log,
            registry,
            UserTurn(
                turn=turn,
                blocks=blocks,
                client_id=client_id,
                viewer=conn.tab_id,
                by=_login(conn),
            ),
        )
    else:
        blocks_to_send = blocks
    try:
        await session.submit_turn(blocks_to_send, viewer=conn.tab_id)
    except Exception as exc:
        # A session that fails must not take the socket with it: the viewer
        # half of this page keeps working whatever the agent does.
        sys.stderr.write("warning: session could not handle a turn frame: %r\n" % (exc,))
        if turn is not None:
            _publish(event_log, registry, TurnEnd(turn=turn, stop_reason="rejected", cost_usd=0.0))
        await ws.send(
            json.dumps(
                protocol.build_refused(
                    "the agent could not handle that turn frame; see the server output", client_id
                )
            )
        )


def _login(conn):
    """The tailnet login of the human on ``conn``, what an event they caused
    records as ``by``; ``None`` for the browser token's holder, whose identity
    is unknown."""
    return conn.human.login if conn.human is not None else None


def _publish(event_log, registry, event):
    """Append ``event`` to the log and schedule its broadcast to every viewer."""
    seq = event_log.append(event)
    asyncio.ensure_future(registry.broadcast(protocol.build_event(seq, event.to_wire())))


def build_registry(**kwargs):
    """A ``ViewerRegistry`` for one app, kept here so app.py does not need to
    know the registry's constructor keywords."""
    return ViewerRegistry(**kwargs)


def ping_frame():
    """A liveness frame, addressed with the wall-clock time the plan's frame
    catalogue shows."""
    return protocol.build_ping(int(time.time()))


async def ping_forever(registry, interval=PING_INTERVAL):
    """Broadcast a ping to every connected viewer, forever.

    One task for the whole process rather than one per connection: a ping
    carries no per-connection state, and the registry's backpressure policy
    already drops queued pings first, so a saturated connection sheds these
    before it sheds anything that matters.

    Without this the browser's liveness watchdog has nothing to distinguish an
    idle connection from a dead one, and closes both.
    """
    while True:
        await asyncio.sleep(interval)
        await registry.broadcast(ping_frame())


__all__ = [
    "MAX_WS_MESSAGE",
    "PING_INTERVAL",
    "WebSocket",
    "build_registry",
    "host_is_allowed",
    "ping_forever",
    "ping_frame",
    "refusal",
    "register_ws",
]
