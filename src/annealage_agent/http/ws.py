"""The ``/ws`` route: authentication, the greeting, replay, and frame dispatch.

A bug in this file is not a UI bug. ``/ws`` is the one route a browser tab
turns into a live bidirectional channel, and from the milestone that lands an
agent it carries turns into a Claude Code session with shell access in the
human's project. A check that can be bypassed here is a remote-code-execution
bug, so the three checks below are independent, each is commented with what it
defends against and why the other two do not cover it, and all three run
**before** the WebSocket handshake.

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
from ..session.base import PauseChanged, UnknownRequest
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
    token,
    allowed_origins,
    allowed_hosts,
    registry,
    event_log,
    session_info,
    session=None,
    bus=None,
):
    """Register ``/ws`` on ``app``.

    ``token`` of ``None`` means no token was configured for this process, and
    every ``/ws`` request is then refused. That is the safe reading rather
    than an inconvenient one: a socket nobody can authenticate is a socket
    that should not open, and the command-line entry point always generates a
    token, so the only way to reach this state is a caller that built an app
    without one.

    ``bus`` is the ``ViewerBus`` holding the human's pause switch, and is
    ``None`` in viewer-only mode, where there are no agent tools to pause. It is
    read for the ``hello`` frame and written by an inbound ``pause`` frame; this
    module never calls through it, because a ``call`` originates with a tool,
    never with a socket.
    """

    @app.get("/ws")
    async def ws_route(req):
        # Order is token, then Origin. Host is enforced for every route by
        # app.py's before_request hook and so has already passed by here.
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()

        ws = await websocket_upgrade(req)
        ws.max_message_length = MAX_WS_MESSAGE
        conn = None
        try:
            # Greeting and replay both happen before the connection is
            # registered, and that order is load-bearing. Registering starts a
            # writer task that also sends on this socket, and a replayed event
            # carries an older seq than anything live: if a replay frame were
            # written from here while that writer was draining a live
            # broadcast, the client could see an older seq after a newer one,
            # move its resync position backwards, and replay the same events
            # again on its next reconnect. With nothing registered yet, there
            # is no second writer for those frames to race.
            await ws.send(
                json.dumps(
                    protocol.build_hello(
                        event_log.current_seq,
                        session_info["id"],
                        session_info.get("sdk_session_id"),
                        session_info["cwd"],
                        session_info.get("agent", "unavailable"),
                        paused=bus.paused if bus is not None else False,
                        model=session_info.get("model"),
                    )
                )
            )
            if not await _greet(ws, event_log, token):
                return Response.already_handled
            conn = await registry.add(ws)
            await _serve_connection(ws, conn, registry, event_log, token, session, bus)
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
    client, which still has to present the token. The threat this check exists
    for is a page the human happens to visit opening a socket to this server,
    and that page cannot suppress its own ``Origin``.

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


async def _greet(ws, event_log, token):
    """Read the client's ``hello`` and answer it, before any writer exists.

    Returns False if the connection was closed here, in which case the caller
    must not register it. Nothing else is read: a client that sends some other
    frame first is answered with a refusal and asked again, because ``hello``
    is what carries the resync position and there is nothing useful to do
    without it.

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
            return False
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
        if not _constant_time_equal(frame.get("token") or "", token or ""):
            await protocol.close_with_code(
                ws, protocol.CLOSE_VERSION_MISMATCH, "hello token does not match"
            )
            return False
        replay = event_log.replay(frame.get("last_seq"))
        for seq, event_wire in replay.events:
            await ws.send(json.dumps(protocol.build_event(seq, event_wire)))
        if replay.truncated:
            await ws.send(
                json.dumps(
                    protocol.build_refused(
                        "history before this point is not in the live ring; page it over HTTP"
                    )
                )
            )
        return True


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


async def _serve_connection(ws, conn, registry, event_log, token, session=None, bus=None):
    """Read and dispatch frames until the peer goes away.

    Every frame this sends is either a ``refused``, which carries no seq, or a
    terminal coded close. Neither has an ordering relationship with the events
    the connection's writer task is draining onto the same socket, so writing
    them from here cannot reorder anything the client tracks. Frames cannot
    interleave mid-frame either: microdot's ``awrite`` is a single
    ``StreamWriter.write`` of the whole frame followed by a ``drain``, and
    ``write`` buffers synchronously.
    """
    while True:
        raw = await ws.receive()
        frame = await _parse(ws, raw)
        if frame is None:
            continue
        if frame is _CLOSED:
            return
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
        # monotonic stream a reconnecting client replays from.
        event = PauseChanged(paused=bus.paused)
        seq = event_log.append(event)
        await registry.broadcast(protocol.build_event(seq, event.to_wire()))
        return
    if kind in ("turn", "interrupt", "permission", "set_model"):
        if session is None:
            # Viewer-only: the frames are still defined and validated so one
            # browser build works against both modes, and answering with a
            # reason is what stops a chat pane waiting forever on a turn
            # nothing will ever process.
            await ws.send(
                json.dumps(
                    protocol.build_refused(
                        "this server is running viewer-only, so %s frames are not served" % kind
                    )
                )
            )
            return
        await registry.touch(conn)
        # Handed to the session and not awaited for a result: a turn produces
        # events, which arrive over this same socket through the registry, and
        # blocking this connection's read loop on the whole turn would stop it
        # reading the interrupt frame that ends it.
        try:
            if kind == "turn":
                await session.submit_turn(frame["blocks"], viewer=conn.tab_id)
            elif kind == "interrupt":
                await session.interrupt()
            elif kind == "set_model":
                await session.set_model(frame["model"])
            else:
                await session.decide_permission(
                    frame["request_id"], frame["decision"], frame.get("message", "")
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
