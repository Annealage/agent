"""Tests for the pause switch: the frame, the event, and what it actually gates.

The switch is only worth having if the refusal happens in the server, where the
tools run, so these tests follow it end to end within one process: an inbound
``pause`` frame moves the flag on the ``ViewerBus``, the flag is announced to
every viewer, and a product tool that changes something, built on that same bus,
then refuses (the toy product's here). The previous version of this control changed a checkbox and
nothing else, which every test of the browser alone would have passed.

This matters more than it did when the switch was designed. A product's
view-class tools (Annealage Mesh's move the camera and hide parts) are
pre-allowed, so no approval card stands in front of them: this switch is the
human's only control over them, rather than a convenience on top of one.

``_dispatch`` is called directly, as in ``tests/test_permission_frames.py``: the
frame has already been validated by then, so a real socket would add a handshake
without adding anything to the question of what the layer does with the frame.
"""

import json

import pytest
from toy_product import TOY_PAUSED_MESSAGE, build_toy_tools

from annealage_agent import protocol
from annealage_agent.http import ws as ws_module
from annealage_agent.session.base import PauseChanged
from annealage_agent.viewers import ViewerBus, ViewerRegistry

pytestmark = pytest.mark.asyncio


class RecordingSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class StubRegistry:
    """Only what ``_dispatch`` reaches for on this path."""

    def __init__(self):
        self.touched = 0
        self.broadcasts = []

    async def touch(self, conn):
        self.touched += 1

    async def broadcast(self, frame):
        self.broadcasts.append(frame)


class StubEventLog:
    """Assigns seqs the way ``EventLog`` does, so a broadcast frame carries one."""

    def __init__(self):
        self.appended = []

    def append(self, event):
        self.appended.append(event)
        return len(self.appended)


class _Conn:
    tab_id = "tab-1"


def _bus():
    return ViewerBus(ViewerRegistry(), url="http://127.0.0.1:8765/#t=tok")


async def _dispatch(bus, paused, registry=None, event_log=None):
    sock = RecordingSocket()
    registry = registry if registry is not None else StubRegistry()
    event_log = event_log if event_log is not None else StubEventLog()
    await ws_module._dispatch(
        sock,
        _Conn(),
        registry,
        event_log,
        "tok",
        {"v": 1, "type": "pause", "paused": paused},
        None,
        bus,
    )
    return sock.sent, registry, event_log


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


async def test_a_pause_frame_moves_the_flag_and_announces_it():
    bus = _bus()
    sent, registry, event_log = await _dispatch(bus, True)
    assert bus.paused is True
    # Nothing is answered to the tab that clicked: every view has a control
    # showing this, so it is broadcast instead.
    assert sent == []
    assert isinstance(event_log.appended[0], PauseChanged)
    assert registry.broadcasts == [
        {
            "v": protocol.PROTOCOL_VERSION,
            "type": "event",
            "seq": 1,
            "event": {"kind": "pause_changed", "paused": True},
        }
    ]


async def test_unpausing_announces_the_new_value_too():
    bus = _bus()
    await _dispatch(bus, True)
    sent, registry, _ = await _dispatch(bus, False)
    assert bus.paused is False
    assert registry.broadcasts[0]["event"] == {"kind": "pause_changed", "paused": False}


async def test_a_frame_that_changes_nothing_announces_nothing():
    """Two tabs racing one control produce this routinely, and an event per
    redundant click would make every other view re-render for no reason."""
    bus = _bus()
    await _dispatch(bus, True)
    sent, registry, event_log = await _dispatch(bus, True)
    assert bus.paused is True
    assert registry.broadcasts == []
    assert event_log.appended == []
    assert sent == []


async def test_pausing_counts_as_interaction_with_this_view():
    """Same as a turn or a decision: the tab a human just used is the one a
    later ``call`` should go to."""
    bus = _bus()
    _sent, registry, _log = await _dispatch(bus, True)
    assert registry.touched == 1


async def test_viewer_only_mode_says_there_is_nothing_to_pause():
    sock = RecordingSocket()
    await ws_module._dispatch(
        sock,
        _Conn(),
        StubRegistry(),
        StubEventLog(),
        "tok",
        {"v": 1, "type": "pause", "paused": True},
        None,
        None,
    )
    assert len(sock.sent) == 1
    assert sock.sent[0]["type"] == "refused"
    assert "viewer-only" in sock.sent[0]["reason"]


# ---------------------------------------------------------------------------
# The greeting, for a tab that connects after the switch was set
# ---------------------------------------------------------------------------


async def test_the_greeting_carries_the_current_value():
    """A tab that connects after the switch was set may have no event to learn
    it from (a viewer-only run replays only its last 500 events). The
    frame shape itself is covered in ``tests/test_protocol.py``; this is about
    the value coming from the live bus rather than from a default."""
    bus = _bus()
    await _dispatch(bus, True)
    frame = protocol.build_hello(3, "s", None, "/tmp", "ready", paused=bus.paused)
    assert frame["session"]["paused"] is True


# ---------------------------------------------------------------------------
# What it gates
# ---------------------------------------------------------------------------


async def test_the_flag_the_frame_sets_is_the_flag_the_tools_read(tmp_path):
    """The assertion the removed version of this control would have failed: a
    frame arrives, and a tool that changes something, built on that same bus,
    refuses, while one that changes nothing does not.

    ``add_note`` is the one used here because it is both gated and prompting
    (a write-class tool), so it exercises the gate at the point where the two
    policies meet. View-class tools are gated by the same flag; a product's own
    suite covers its tools exhaustively (Annealage Mesh's ``tests/test_tools.py``),
    and those matter more, not less, since the pause switch is the *only*
    control over them."""
    bus = _bus()
    tools = {t.name: t.handler for t in build_toy_tools(bus, tmp_path).tools}

    before = await tools["add_note"]({"text": "hi"})
    assert "is_error" not in before

    await _dispatch(bus, True)
    during = await tools["add_note"]({"text": "hi again"})
    assert during["is_error"] is True
    assert TOY_PAUSED_MESSAGE in during["content"][0]["text"]
    assert "is_error" not in await tools["list_notes"]({})

    await _dispatch(bus, False)
    after = await tools["add_note"]({"text": "and now"})
    assert "is_error" not in after
