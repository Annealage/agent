"""Frame construction and validation for the ``/ws`` protocol.

``PROTOCOL_VERSION`` and every frame shape below are the contract in plan
section 3.3, implemented as written. ``http/ws.py`` (not this module)
owns the socket, the auth checks and the dispatch loop; this module only
knows how to build a well-formed frame and how to tell a well-formed
inbound one from a malformed one, with no I/O of its own.

Every frame, in either direction, carries ``{"v": PROTOCOL_VERSION,
"type": ...}``. Validation throughout is whitelist-shaped: a ``type`` this
module does not recognise, or a key beyond the ones a recognised type
allows, is refused rather than passed through or silently dropped,
matching the decision already made for ``PUT /settings``. A refusal
always carries a reason, because a frame refused with no explanation is a
debugging dead end for a phone user with no terminal.

Two shapes here are not in plan section 3.3's catalogue, and both are
flagged rather than buried since they are this module's own additions to
the contract.

``build_refused``: the plan's frame list has no shape for "the server
refused one inbound frame and the connection stays open"; the M4 brief
requires that reason reach the client regardless, and reusing the
existing browser-to-server ``error`` shape (which always carries a call
``id``) would make an uncorrelated protocol refusal look like a failed
reply to some particular ``call``. ``build_refused`` is a new, minimal
type for exactly that one purpose.

The inbound ``pause`` frame: the human's pause switch is enforced in the
server, because the tools it gates run there, so the browser's control has
to be able to say what the human chose. Neither addition changes
``PROTOCOL_VERSION``, and neither needs to: an older page never sends a
``pause`` frame and simply ignores an event kind or frame type it does not
recognise, which is the same tolerance every other addition to this
protocol relies on.
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Any, Callable, Optional, Set, Tuple, Union

PROTOCOL_VERSION = 1

# Close code for a mismatched protocol version (plan section 3.3): how a
# stale cached page is told apart from a newer server. Sent with
# close_with_code, never with plain ws.close().
CLOSE_VERSION_MISMATCH = 4400

# Close code for a connection asked to accept more than its backpressure
# policy (viewers.py) will hold even after every relief it grants runs
# out. 1013 is the standard "try again later" WebSocket close code; the
# client's reconnect-and-replay-from-seq logic (static/ws.js) is what
# "later" means here.
CLOSE_OVERFLOW = 1013

#: Sent to every connected viewer as the process shuts down. Without it, a
#: browser learns the server has gone only by noticing that the pings stopped,
#: which takes the whole liveness timeout: closing the listening socket does
#: not close connections already established on it, and an event loop simply
#: ending leaves those sockets to be collected rather than shut down. A
#: deliberate 1001 turns that wait into an immediate reconnect attempt.
CLOSE_GOING_AWAY = 1001


class ProtocolVersionMismatch(Exception):
    """Raised by ``validate_inbound`` when a frame's ``v`` is not
    ``PROTOCOL_VERSION``.

    Raised rather than returned, unlike an ordinary validation failure,
    because the two are not the same kind of problem for the caller.
    An unknown ``type`` or an unknown key is one bad message from a peer
    both sides still agree how to talk to, and the connection continues
    with a ``refused`` frame telling the client why. A ``v`` mismatch
    means the two sides disagree about the wire format itself, so nothing
    else this frame claims can be trusted either; ``http/ws.py`` catches
    this specifically and closes with ``CLOSE_VERSION_MISMATCH`` instead
    of attempting to respond on a socket, rather than continuing to parse
    frames whose shape it can no longer assume.
    """

    def __init__(self, received: Any):
        self.received = received
        super().__init__(
            "protocol version mismatch: expected %r, got %r" % (PROTOCOL_VERSION, received)
        )


async def close_with_code(ws, code: int, reason: str = "") -> None:
    """Send exactly one coded CLOSE frame on ``ws``, then mark it closed.

    ``WebSocket.close()`` takes no arguments and always sends an empty
    CLOSE payload (verified fact 1); the plan's close codes are not
    reachable through it. A coded close is built by hand instead:
    ``struct.pack("!H", code)`` for the two-byte code RFC 6455 puts first
    in a CLOSE frame's payload, followed by a UTF-8 reason. The reason is
    passed as bytes, not a str, because ``_encode_websocket_frame`` only
    calls ``.encode()`` for the TEXT opcode and passes bytes straight
    through for every other opcode including CLOSE; encoding here, once,
    is what makes that call correct. Truncated to 123 bytes, since a
    control frame's payload is capped at 125 bytes by RFC 6455 and two of
    those are already spent on the code.

    Guarded by ``ws.closed``, the same flag ``WebSocket.close()`` itself
    checks before sending: a second CLOSE frame after the first is a
    protocol violation, and this function and any later ``ws.close()``
    both consult the one flag, so whichever runs first is the only one
    that ever sends. This is the only place in this package that sends a
    coded close; per verified fact 2, ``http/ws.py`` calls
    ``websocket_upgrade`` directly rather than the ``with_websocket``
    decorator specifically so that decorator's own ``finally: await
    ws.close()`` can never append an empty CLOSE after this one.
    """
    if ws.closed:
        return
    ws.closed = True
    payload = struct.pack("!H", code) + reason.encode("utf-8")[:123]
    await ws.send(payload, ws.CLOSE)


# ---------------------------------------------------------------------------
# Outbound (server to browser) frame builders.
# ---------------------------------------------------------------------------


def build_hello(
    seq: int,
    session_id: str,
    sdk_session_id: Optional[str],
    cwd: str,
    agent_status: str,
    paused: bool = False,
    model: Optional[str] = None,
    steers: bool = False,
) -> dict:
    """The greeting sent once, immediately after a successful upgrade.

    ``seq`` is the log's current seq (``EventLog.current_seq``): the point
    in the stream this connection is starting from, with nothing before
    it left to replay on this send (a separate replay of ring-held events
    follows, driven by the client's own ``last_seq``, not by this frame).

    ``paused`` carries the current state of the human's pause switch, so a
    tab opened long after it was set shows it. A ``pause_changed`` event
    announces every later change, but such an event only reaches a client
    that was connected when it happened: replay reaches back 500 events and
    a fresh tab replays nothing at all, so the value belongs in the greeting
    rather than being inferred from the event stream.

    ``model`` is the effective starting model this run's agent session was
    constructed with (``settings.py``'s ``model`` key, Phase 2's per-project
    default), so a fresh tab's picker shows the right value with no flash of
    a wrong default. It is not corrected here on a later live switch; an
    ``AgentModelChanged`` event (``session/base.py``) is what a connected
    client updates its picker from once ``set_model`` actually takes effect.

    ``steers`` says a message sent while a turn is running redirects that
    turn rather than waiting behind it (the omp backend), so the page can
    label its Send button for what it will do.
    """
    return {
        "v": PROTOCOL_VERSION,
        "type": "hello",
        "seq": seq,
        "session": {
            "id": session_id,
            "sdk_session_id": sdk_session_id,
            "cwd": cwd,
            "agent": agent_status,
            "model": model,
            "paused": bool(paused),
            "steers": bool(steers),
        },
        "protocol": PROTOCOL_VERSION,
    }


def build_event(seq: int, event_wire: dict) -> dict:
    """One ``AgentEvent.to_wire()`` result, addressed with its log seq."""
    return {"v": PROTOCOL_VERSION, "type": "event", "seq": seq, "event": event_wire}


def build_call(call_id: str, method: str, params: dict) -> dict:
    """A server-initiated request into the primary viewer's page."""
    return {
        "v": PROTOCOL_VERSION,
        "type": "call",
        "id": call_id,
        "method": method,
        "params": params,
    }


def build_ping(t) -> dict:
    """A liveness frame; ``t`` is whatever timestamp ``http/ws.py`` wants
    the client to see (a Unix time, matching the plan's example)."""
    return {"v": PROTOCOL_VERSION, "type": "ping", "t": t}


def build_refused(reason: str) -> dict:
    """Tells the client one inbound frame was refused, and why.

    Not part of plan section 3.3's catalogue; see this module's docstring
    for why it exists and why it is not the browser-to-server ``error``
    shape reused in the other direction. The connection stays open: this
    is the response to one bad frame, not the version mismatch that ends
    the connection (see ``ProtocolVersionMismatch``).
    """
    return {"v": PROTOCOL_VERSION, "type": "refused", "reason": reason}


# ---------------------------------------------------------------------------
# Inbound (browser to server) frame validation.
# ---------------------------------------------------------------------------


def object_error(obj: Any, allowed: Set[str], required: Set[str], name: str) -> Optional[str]:
    """Whitelist check for one JSON object: not a dict, an unknown key, or
    a missing required key, in that order. Returns a reason, or None."""
    if not isinstance(obj, dict):
        return "%s must be an object" % name
    unknown = set(obj) - allowed
    if unknown:
        return "%s has unknown key(s): %s" % (name, ", ".join(sorted(unknown)))
    missing = required - set(obj)
    if missing:
        return "%s missing required key(s): %s" % (name, ", ".join(sorted(missing)))
    return None


def _check_hello(frame: dict) -> Optional[str]:
    last_seq = frame.get("last_seq")
    if last_seq is not None and not isinstance(last_seq, int):
        return "hello.last_seq must be an integer or absent"
    viewer = frame.get("viewer")
    if viewer is not None:
        return object_error(viewer, {"tab_id", "w", "h"}, {"tab_id"}, "hello.viewer")
    return None


_BLOCK_SPECS = {
    "text": ({"type", "text"}, {"type", "text"}),
    "image_path": ({"type", "path"}, {"type", "path"}),
}


def _check_turn(frame: dict) -> Optional[str]:
    blocks = frame.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return "turn.blocks must be a non-empty array"
    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            return "turn.blocks[%d] must be an object" % i
        block_type = block.get("type")
        spec = _BLOCK_SPECS.get(block_type)
        if spec is None:
            return "turn.blocks[%d] has unknown type: %r" % (i, block_type)
        error = object_error(block, spec[0], spec[1], "turn.blocks[%d]" % i)
        if error:
            return error
    return None


_PERMISSION_DECISIONS = {"allow", "allow_always", "deny"}


def _check_permission(frame: dict) -> Optional[str]:
    decision = frame.get("decision")
    if decision not in _PERMISSION_DECISIONS:
        return "permission.decision must be one of %s" % ", ".join(sorted(_PERMISSION_DECISIONS))
    return None


def _check_error(frame: dict) -> Optional[str]:
    return object_error(frame.get("error"), {"code", "message"}, {"code", "message"}, "error.error")


def _check_pause(frame: dict) -> Optional[str]:
    # ``isinstance(1, bool)`` is False while ``isinstance(True, int)`` is True,
    # so checking for bool specifically refuses a numeric 0 or 1 here. That
    # matches every other check in this module: a client sending something
    # adjacent to the contract is told, rather than having its value coerced
    # into whichever reading this server happens to prefer.
    if not isinstance(frame.get("paused"), bool):
        return "pause.paused must be true or false"
    return None


def _check_set_model(frame: dict) -> Optional[str]:
    # Same non-empty-string convention _check_permission's allowed-decision
    # set and _check_pause's bool check both follow: a client sending
    # something adjacent to the contract (an empty string, a number, null)
    # is told, not coerced. Unlike ``paused``'s enumerable values, a model
    # name is backend/endpoint-dependent and not enumerable here, so this
    # only rules out shapes that could never be a real model id.
    model = frame.get("model")
    if not isinstance(model, str) or not model:
        return "set_model.model must be a non-empty string"
    return None


@dataclasses.dataclass(frozen=True)
class FrameSpec:
    """The shape of one inbound frame type: the keys it may carry and must
    carry beyond ``v``/``type``, and an optional ``check`` that looks inside
    once the flat key shape has passed, returning a refusal reason or None.

    Public because a product describes its own inbound frames with it
    (``Product.inbound_frames``); ``object_error`` above is the whitelist check
    such a ``check`` should use for a nested object, so a product's frame is
    refused in the same words as a generic one.
    """

    allowed: Set[str]
    required: Set[str]
    check: Optional[Callable[[dict], Optional[str]]] = None


# One entry per inbound frame type (plan section 3.3, browser to server,
# plus ``pause``; see this module's docstring).
# ``allowed``/``required`` cover the keys beyond ``v``/``type``, which
# every type carries and neither set repeats. ``check`` runs only once the
# flat key shape already passed, for the shapes that need to look inside a
# nested object or enumerate a value's allowed contents.
_INBOUND_SPECS = {
    "hello": FrameSpec({"token", "last_seq", "viewer"}, {"token"}, _check_hello),
    "turn": FrameSpec({"blocks"}, {"blocks"}, _check_turn),
    "permission": FrameSpec(
        {"request_id", "decision", "message"}, {"request_id", "decision"}, _check_permission
    ),
    "result": FrameSpec({"id", "result"}, {"id", "result"}),
    "error": FrameSpec({"id", "error"}, {"id", "error"}, _check_error),
    "interrupt": FrameSpec(set(), set()),
    "pause": FrameSpec({"paused"}, {"paused"}, _check_pause),
    "set_model": FrameSpec({"model"}, {"model"}, _check_set_model),
}

# The installed product's own inbound frame types (Mesh: ``state``), which
# the page sends to report its own state. Validated exactly as the generic
# ones are; ``http/ws.py`` treats one as interaction with that tab and does
# nothing further with it. Filled by ``register_product_frames``.
_PRODUCT_SPECS = {}


def check_product_frames(specs) -> None:
    """Raise ``ValueError`` if ``specs`` (``{type: FrameSpec}``) cannot be
    registered: a type the generic protocol already defines, or a value that
    is not a ``FrameSpec``. Changes nothing."""
    for frame_type, spec in dict(specs).items():
        if frame_type in _INBOUND_SPECS:
            raise ValueError("product frame type %r is already a protocol frame" % frame_type)
        if not isinstance(spec, FrameSpec):
            raise ValueError("product frame type %r has no FrameSpec" % frame_type)


def register_product_frames(specs) -> None:
    """Make ``specs`` the product's inbound frame types, replacing any
    registered before. Called by ``product.install`` (after
    ``check_product_frames``) and ``product.reset`` only."""
    global _PRODUCT_SPECS
    _PRODUCT_SPECS = dict(specs)


def is_product_frame(frame_type: str) -> bool:
    """Whether ``frame_type`` is one of the installed product's frame types."""
    return frame_type in _PRODUCT_SPECS


def validate_inbound(raw: Any) -> Tuple[bool, Union[dict, str]]:
    """Validate one inbound frame already parsed from JSON.

    Returns ``(True, raw)`` for a well-shaped, recognised frame, or
    ``(False, reason)`` otherwise, so the caller always has a string
    worth sending back via ``build_refused``. Raises
    ``ProtocolVersionMismatch`` instead of returning for a ``v`` mismatch;
    see that exception's docstring for why that one case is not just
    another reason string.

    Validation is whitelist-shaped end to end: an unrecognised ``type`` is
    refused rather than ignored, and a recognised type carrying a key
    beyond the ones listed in ``_INBOUND_SPECS`` is refused rather than
    having the unknown key silently dropped, matching the decision
    already made for ``PUT /settings``.
    """
    if not isinstance(raw, dict):
        return False, "frame is not a JSON object"
    if "v" not in raw:
        return False, "frame missing required key: v"
    if raw["v"] != PROTOCOL_VERSION:
        raise ProtocolVersionMismatch(raw["v"])
    frame_type = raw.get("type")
    spec = _INBOUND_SPECS.get(frame_type) or _PRODUCT_SPECS.get(frame_type)
    if spec is None:
        return False, "unknown frame type: %r" % (frame_type,)
    error = object_error(raw, spec.allowed | {"v", "type"}, spec.required | {"v", "type"}, "frame")
    if error:
        return False, error
    if spec.check is not None:
        error = spec.check(raw)
        if error:
            return False, error
    return True, raw
