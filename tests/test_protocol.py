"""Tests for ``protocol.py``: frame builders, inbound validation, and the
one hand-built wire encoding this package ever sends, a coded CLOSE frame.

Fact 4 of Annealage Mesh's M4 brief makes a coded close unobservable through microdot's
in-process ``TestClient.websocket()``, so ``close_with_code`` is exercised
here directly against a minimal stand-in for microdot's ``WebSocket``, and
cross-checked against microdot's own frame encoder so the bytes this
module hands to ``ws.send`` are proven to decode the way fact 1 says they
do, not just to match a value this test file made up independently.
"""

import json
import struct

import pytest
from microdot.websocket import WebSocket

from annealage_agent.protocol import (
    CLOSE_OVERFLOW,
    CLOSE_VERSION_MISMATCH,
    PROTOCOL_VERSION,
    ProtocolVersionMismatch,
    build_call,
    build_event,
    build_hello,
    build_ping,
    build_refused,
    close_with_code,
    validate_inbound,
)


def _roundtrip(frame):
    """A frame survives being serialised and parsed back unchanged, which
    is the only property that matters for something sent as JSON text."""
    return json.loads(json.dumps(frame))


# ---------------------------------------------------------------------------
# close_with_code
# ---------------------------------------------------------------------------


class _RecordingWebSocket:
    """Enough of microdot's ``WebSocket`` surface for ``close_with_code``:
    the ``CLOSE`` opcode constant it passes to ``send``, the ``closed``
    flag it guards on, and a ``send`` that records its arguments instead
    of writing to a socket."""

    CLOSE = WebSocket.CLOSE

    def __init__(self):
        self.closed = False
        self.sent = []

    async def send(self, data, opcode=None):
        self.sent.append((data, opcode))


@pytest.mark.asyncio
async def test_close_with_code_sends_code_and_reason_as_one_payload():
    ws = _RecordingWebSocket()
    await close_with_code(ws, 4403, "bad token")
    assert ws.closed is True
    assert len(ws.sent) == 1
    payload, opcode = ws.sent[0]
    assert opcode == WebSocket.CLOSE
    assert struct.unpack("!H", payload[:2])[0] == 4403
    assert payload[2:] == b"bad token"


@pytest.mark.asyncio
async def test_close_with_code_payload_decodes_through_microdots_own_encoder():
    """Verified fact 1: this payload, sent as bytes on the CLOSE opcode,
    is what ``_encode_websocket_frame`` (which only calls ``.encode()``
    for TEXT) turns into a real CLOSE frame: byte 0x88, a length byte,
    the two-byte code, then the reason, readable end to end."""
    ws = _RecordingWebSocket()
    await close_with_code(ws, 4403, "bad token")
    payload, _ = ws.sent[0]
    frame = WebSocket._encode_websocket_frame(WebSocket.CLOSE, payload)
    assert frame[0] == 0x88
    assert frame[1] == len(payload)
    assert struct.unpack("!H", frame[2:4])[0] == 4403
    assert frame[4:].decode("utf-8") == "bad token"


@pytest.mark.asyncio
async def test_close_with_code_truncates_reason_to_123_bytes():
    """RFC 6455 caps a control frame payload at 125 bytes; two of those
    are always spent on the code, leaving 123 for the reason."""
    ws = _RecordingWebSocket()
    await close_with_code(ws, CLOSE_OVERFLOW, "x" * 200)
    payload, _ = ws.sent[0]
    assert len(payload) == 125
    assert payload[2:] == b"x" * 123


@pytest.mark.asyncio
async def test_close_with_code_sends_at_most_once():
    """A second CLOSE frame after the first is a protocol violation
    (verified fact 1). ``close_with_code`` guards on ``ws.closed`` exactly
    as ``WebSocket.close()`` does, so calling it twice on one socket must
    not put two CLOSE frames on the wire."""
    ws = _RecordingWebSocket()
    await close_with_code(ws, CLOSE_VERSION_MISMATCH, "stale client")
    await close_with_code(ws, CLOSE_VERSION_MISMATCH, "stale client")
    assert len(ws.sent) == 1


@pytest.mark.asyncio
async def test_close_with_code_is_a_no_op_on_an_already_closed_socket():
    ws = _RecordingWebSocket()
    ws.closed = True
    await close_with_code(ws, CLOSE_OVERFLOW)
    assert ws.sent == []


# ---------------------------------------------------------------------------
# Outbound frame builders
# ---------------------------------------------------------------------------


def test_build_hello_round_trips():
    frame = build_hello(7, "sess-1", "sdk-1", "/tmp/proj", "ready")
    assert frame == {
        "v": PROTOCOL_VERSION,
        "type": "hello",
        "seq": 7,
        "session": {
            "id": "sess-1",
            "sdk_session_id": "sdk-1",
            "cwd": "/tmp/proj",
            "agent": "ready",
            # Present and null unless the agent is down with a reason.
            "agent_error": None,
            # Defaulted, and present rather than omitted: a page that found no
            # `paused` key would have to guess, and the safe guess (not paused)
            # is the one that shows the human a control claiming the agent may
            # drive the viewer when it may not.
            "paused": False,
            # Defaulted to None, same reasoning as `sdk_session_id`: a caller
            # with no agent session yet (or one this project's settings never
            # set a model for) reports that honestly rather than a guess.
            "model": None,
            # Present and false unless the backend steers (omp), so the page
            # labels Send as Send.
            "steers": False,
        },
        "protocol": PROTOCOL_VERSION,
    }
    assert _roundtrip(frame) == frame


def test_build_hello_carries_the_starting_model():
    """A fresh tab's picker reads its initial value from here, not from a
    later `AgentModelChanged` event: this is a snapshot, not the corrected
    value a live `set_model` produces."""
    frame = build_hello(7, "sess-1", "sdk-1", "/tmp/proj", "ready", model="claude-opus-4")
    assert frame["session"]["model"] == "claude-opus-4"
    assert _roundtrip(frame) == frame


def test_build_hello_allows_no_sdk_session_id_yet():
    frame = build_hello(0, "sess-1", None, "/tmp/proj", "connecting")
    assert frame["session"]["sdk_session_id"] is None
    assert _roundtrip(frame) == frame


def test_build_event_round_trips():
    frame = build_event(41, {"kind": "text_delta", "turn": 3, "text": "hi"})
    assert frame == {
        "v": PROTOCOL_VERSION,
        "type": "event",
        "seq": 41,
        "event": {"kind": "text_delta", "turn": 3, "text": "hi"},
    }
    assert _roundtrip(frame) == frame


def test_build_call_round_trips():
    frame = build_call("c_1", "toy.set_view", {"zoom": 2})
    assert frame == {
        "v": PROTOCOL_VERSION,
        "type": "call",
        "id": "c_1",
        "method": "toy.set_view",
        "params": {"zoom": 2},
    }
    assert _roundtrip(frame) == frame


def test_build_ping_round_trips():
    frame = build_ping(1750000000)
    assert frame == {"v": PROTOCOL_VERSION, "type": "ping", "t": 1750000000}
    assert _roundtrip(frame) == frame


def test_build_refused_round_trips():
    frame = build_refused("unknown frame type: 'bogus'")
    assert frame == {
        "v": PROTOCOL_VERSION,
        "type": "refused",
        "reason": "unknown frame type: 'bogus'",
    }
    assert _roundtrip(frame) == frame


def test_a_refused_turn_names_its_client_id():
    frame = build_refused("the agent is connecting", "c-1")
    assert frame["client_id"] == "c-1"
    assert _roundtrip(frame) == frame


# ---------------------------------------------------------------------------
# Inbound validation: well-shaped frames of every recognised type
# ---------------------------------------------------------------------------

VALID_FRAMES = [
    {"v": PROTOCOL_VERSION, "type": "hello", "token": "abc"},
    {
        "v": PROTOCOL_VERSION,
        "type": "hello",
        "token": "abc",
        "last_seq": 12,
        "viewer": {"tab_id": "t1", "w": 800, "h": 600},
    },
    {"v": PROTOCOL_VERSION, "type": "hello", "token": "abc", "viewer": {"tab_id": "t1"}},
    {
        "v": PROTOCOL_VERSION,
        "type": "turn",
        "blocks": [{"type": "text", "text": "why is this wall thin?"}],
    },
    {
        "v": PROTOCOL_VERSION,
        "type": "turn",
        "blocks": [{"type": "image_path", "path": "images/a.png"}],
    },
    {
        "v": PROTOCOL_VERSION,
        "type": "turn",
        "blocks": [
            {"type": "text", "text": "look at"},
            {"type": "image_path", "path": "images/b.png"},
        ],
    },
    {
        "v": PROTOCOL_VERSION,
        "type": "turn",
        "blocks": [{"type": "text", "text": "x"}],
        "client_id": "3f2b8c1e-7a44-4c1d-9b0e-2d6f5a9c8e71",
    },
    {"v": PROTOCOL_VERSION, "type": "permission", "request_id": "pr_1", "decision": "allow"},
    {"v": PROTOCOL_VERSION, "type": "permission", "request_id": "pr_1", "decision": "allow_always"},
    {
        "v": PROTOCOL_VERSION,
        "type": "permission",
        "request_id": "pr_1",
        "decision": "deny",
        "message": "no",
    },
    {"v": PROTOCOL_VERSION, "type": "result", "id": "c_1", "result": {"png": "..."}},
    {
        "v": PROTOCOL_VERSION,
        "type": "error",
        "id": "c_1",
        "error": {"code": "no_canvas", "message": "nope"},
    },
    {"v": PROTOCOL_VERSION, "type": "interrupt"},
    # The installed product's own frame (the toy's ``view``), validated by the
    # FrameSpec the product registered rather than by anything built in.
    {"v": PROTOCOL_VERSION, "type": "view", "view": {}},
    {"v": PROTOCOL_VERSION, "type": "view", "view": {"zoom": 2, "selection": 3}},
    {"v": PROTOCOL_VERSION, "type": "pause", "paused": True},
    {"v": PROTOCOL_VERSION, "type": "pause", "paused": False},
    {"v": PROTOCOL_VERSION, "type": "set_model", "model": "claude-opus-4"},
]


@pytest.mark.parametrize("frame", VALID_FRAMES)
def test_validate_inbound_accepts_every_well_shaped_frame(frame):
    ok, result = validate_inbound(frame)
    assert ok is True
    assert result == frame


# ---------------------------------------------------------------------------
# Inbound validation: malformed frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["not a frame", None, [1, 2, 3], 42, True])
def test_validate_inbound_rejects_non_object_input(raw):
    ok, reason = validate_inbound(raw)
    assert ok is False
    assert isinstance(reason, str) and reason


def test_validate_inbound_rejects_frame_missing_v():
    ok, reason = validate_inbound({"type": "interrupt"})
    assert ok is False
    assert "v" in reason


def test_validate_inbound_raises_on_version_mismatch_instead_of_returning():
    with pytest.raises(ProtocolVersionMismatch) as excinfo:
        # A page from before version 2, left open across an upgrade.
        validate_inbound({"v": 1, "type": "interrupt"})
    assert excinfo.value.received == 1


def test_validate_inbound_rejects_unknown_type_with_reason_naming_it():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "bogus"})
    assert ok is False
    assert "bogus" in reason


def test_validate_inbound_rejects_frame_with_no_type_at_all():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION})
    assert ok is False
    assert isinstance(reason, str) and reason


@pytest.mark.parametrize(
    "frame",
    [
        {"v": PROTOCOL_VERSION, "type": "turn"},
        {"v": PROTOCOL_VERSION, "type": "permission", "request_id": "pr_1"},
        {"v": PROTOCOL_VERSION, "type": "permission", "decision": "allow"},
        {"v": PROTOCOL_VERSION, "type": "result", "id": "c_1"},
        {"v": PROTOCOL_VERSION, "type": "result", "result": {}},
        {"v": PROTOCOL_VERSION, "type": "error", "id": "c_1"},
        {"v": PROTOCOL_VERSION, "type": "view"},
        {"v": PROTOCOL_VERSION, "type": "set_model"},
    ],
)
def test_validate_inbound_rejects_frame_missing_a_required_key(frame):
    ok, reason = validate_inbound(frame)
    assert ok is False
    assert isinstance(reason, str) and reason


@pytest.mark.parametrize(
    "frame",
    [
        {"v": PROTOCOL_VERSION, "type": "interrupt", "extra": 1},
        {"v": PROTOCOL_VERSION, "type": "hello", "token": "a", "extra": 1},
        {
            "v": PROTOCOL_VERSION,
            "type": "turn",
            "blocks": [{"type": "text", "text": "x"}],
            "extra": 1,
        },
        {"v": PROTOCOL_VERSION, "type": "view", "view": {}, "extra": 1},
        {"v": PROTOCOL_VERSION, "type": "set_model", "model": "gpt-5", "extra": 1},
    ],
)
def test_validate_inbound_rejects_unknown_top_level_key(frame):
    ok, reason = validate_inbound(frame)
    assert ok is False
    assert "extra" in reason


def test_validate_inbound_rejects_empty_turn_blocks():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "turn", "blocks": []})
    assert ok is False


def test_validate_inbound_rejects_turn_blocks_not_a_list():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "turn", "blocks": "why is this wall thin?"}
    )
    assert ok is False


def test_validate_inbound_rejects_turn_block_that_is_not_an_object():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "turn", "blocks": ["not an object"]}
    )
    assert ok is False


def test_validate_inbound_rejects_turn_block_of_unknown_type():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "turn", "blocks": [{"type": "video", "path": "x"}]}
    )
    assert ok is False
    assert "video" in reason


def test_validate_inbound_rejects_turn_block_missing_a_required_key():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "turn", "blocks": [{"type": "text"}]}
    )
    assert ok is False


def test_validate_inbound_rejects_turn_block_with_unknown_key():
    ok, reason = validate_inbound(
        {
            "v": PROTOCOL_VERSION,
            "type": "turn",
            "blocks": [{"type": "text", "text": "x", "size": 12}],
        }
    )
    assert ok is False


@pytest.mark.parametrize("client_id", ["", "x" * 65, "has space", "</script>", 7, None])
def test_validate_inbound_rejects_a_turn_client_id_that_is_not_a_short_plain_name(client_id):
    """It is echoed into the event log and every tab's replay."""
    ok, _reason = validate_inbound(
        {
            "v": PROTOCOL_VERSION,
            "type": "turn",
            "blocks": [{"type": "text", "text": "x"}],
            "client_id": client_id,
        }
    )
    assert ok is False


def test_validate_inbound_rejects_permission_decision_outside_the_allowed_set():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "permission", "request_id": "pr_1", "decision": "maybe"}
    )
    assert ok is False
    assert "decision" in reason


def test_validate_inbound_rejects_hello_viewer_that_is_not_an_object():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "hello", "token": "a", "viewer": "tab1"}
    )
    assert ok is False


def test_validate_inbound_rejects_hello_viewer_missing_tab_id():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "hello", "token": "a", "viewer": {"w": 100, "h": 100}}
    )
    assert ok is False


def test_validate_inbound_rejects_hello_last_seq_of_the_wrong_type():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "hello", "token": "a", "last_seq": "12"}
    )
    assert ok is False


def test_validate_inbound_rejects_error_missing_message():
    ok, reason = validate_inbound(
        {"v": PROTOCOL_VERSION, "type": "error", "id": "c_1", "error": {"code": "x"}}
    )
    assert ok is False


def test_validate_inbound_rejects_view_with_an_unknown_key():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "view", "view": {"camera": {}}})
    assert ok is False
    assert "camera" in reason


def test_validate_inbound_rejects_pause_that_is_not_a_boolean():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "pause", "paused": "yes"})
    assert ok is False
    assert "true or false" in reason


def test_validate_inbound_rejects_a_numeric_pause_value():
    """``isinstance(True, int)`` is true, so the bool check has to be explicit
    in this direction. Refused rather than coerced, like every other value
    here: a client sending something adjacent to the contract is told, instead
    of having its value read whichever way this server happens to prefer."""
    ok, _reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "pause", "paused": 1})
    assert ok is False


def test_validate_inbound_rejects_view_value_that_is_not_an_object():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "view", "view": "front"})
    assert ok is False


def test_validate_inbound_rejects_a_non_string_set_model():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "set_model", "model": 5})
    assert ok is False
    assert "non-empty string" in reason


def test_validate_inbound_rejects_an_empty_set_model():
    ok, reason = validate_inbound({"v": PROTOCOL_VERSION, "type": "set_model", "model": ""})
    assert ok is False
    assert "non-empty string" in reason
