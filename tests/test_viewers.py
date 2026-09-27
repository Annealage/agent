"""Tests for ``viewers.py``: pending-call correlation, primary election,
and the backpressure policy in ``ViewerRegistry._enqueue``.

Every test that needs to inspect a connection's outbound queue in a known
state first stalls that connection's writer task with ``_stall_writer``:
cancel it and await the cancellation landing, so nothing ever again calls
``queue.get()``. That is the "fake writer that never drains" Annealage Mesh's
M4 brief asks the backpressure tests to be driven through. It is deliberately
not a substitute writer coroutine, because the real writer task started by
``ViewerRegistry.add`` is what a stalled browser's connection actually has;
stopping it dead is the simplest correct model of a socket nobody is
reading from, and it leaves the queue's raw contents exactly as
``_enqueue`` left them, which every assertion below depends on. Frames
already queued by ``add``/``touch`` themselves (a connection's own
``viewer_primary`` announcement) are cleared out where a test needs to
isolate what one later call to ``broadcast``/``call`` produced.

Every backpressure assertion checks which frames survive and which do not,
never only a queue length: a test that merely proves the queue stayed at
or under its cap says nothing about whether the policy dropped a
``ping`` or, wrongly, a ``turn_end``.

Pending-future immediacy is asserted by checking ``future.done()`` the
statement after ``await registry.remove(...)`` returns, with no
intervening sleep and a deliberately huge injected ``timeout`` (so a test
that regressed to "wait it out" would hang the suite rather than pass by
accident): ``Future.set_exception`` marks a future done synchronously, so
this is a direct check of the immediate-failure path, not a race against
the timeout it is supposed to make irrelevant.
"""

import asyncio
import contextlib
import json
import struct

import pytest
from microdot.websocket import WebSocket

from annealage_agent.identity import Human
from annealage_agent.protocol import CLOSE_OVERFLOW, build_call, build_event, build_ping
from annealage_agent.session.base import PermissionRequest, TextDelta, ToolUse, TurnEnd
from annealage_agent.viewers import (
    NO_VIEWER_MESSAGE,
    CallError,
    NoViewerConnected,
    ViewerBus,
    ViewerGone,
    ViewerRegistry,
)

pytestmark = pytest.mark.asyncio


class _FakeWebSocket:
    """Enough of microdot's ``WebSocket`` surface for ``ViewerRegistry``: the
    ``CLOSE`` opcode constant ``close_with_code`` passes to ``send``, the
    ``closed`` flag it guards on, and a ``send`` that records its
    arguments instead of writing to a real socket. Matches
    ``test_protocol.py``'s ``_RecordingWebSocket``, duplicated here rather
    than imported since each test file owns its own fixtures."""

    CLOSE = WebSocket.CLOSE

    def __init__(self):
        self.closed = False
        self.sent = []

    async def send(self, data, opcode=None):
        self.sent.append((data, opcode))


class _RaisingAfterWebSocket(_FakeWebSocket):
    """A websocket whose ``send`` succeeds normally ``ok_sends`` times, then
    raises on every call after that, modelling a peer whose connection
    dies partway through a session rather than one already dead at
    connect time. ``ok_sends`` lets a test's setup (adding the connection
    enqueues its own ``viewer_primary`` announcement) go through cleanly,
    so the raise lands on a frame the test itself broadcasts."""

    def __init__(self, ok_sends):
        super().__init__()
        self._ok_sends = ok_sends

    async def send(self, data, opcode=None):
        if self._ok_sends <= 0:
            raise ConnectionResetError("peer gone")
        self._ok_sends -= 1
        await super().send(data, opcode)


async def _stall_writer(conn):
    """Cancel ``conn``'s writer task and wait for the cancellation to be
    delivered, so its outbound queue is never drained again. See the
    module docstring for why this, not a substitute writer coroutine, is
    the fake this suite needs."""
    conn.writer_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await conn.writer_task


def _queue_frames(conn):
    """The exact ordered contents of ``conn``'s outbound queue, as a list.
    Only meaningful once ``_stall_writer`` has run, or a concurrently
    scheduled writer task could dequeue an item between this call and the
    assertion reading its result."""
    return list(conn.queue._queue)


async def _wait_for_sent_count(ws, count, timeout=1.0):
    """Poll until ``ws.sent`` holds at least ``count`` entries.

    Used only by the tests exercising the real ``_run_writer`` task
    (below), where a send happens on that task's own schedule rather than
    synchronously within the call under test: a fixed sleep would either
    race a slow scheduler or pad every run with dead time, so this polls
    in small steps up to ``timeout``, which bounds a genuine failure
    (the writer never sends at all) without being the thing under test.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while len(ws.sent) < count:
        if loop.time() >= deadline:
            raise AssertionError(
                "writer sent %d frame(s), expected at least %d" % (len(ws.sent), count)
            )
        await asyncio.sleep(0.005)


def _is_viewer_primary(frame, primary_tab_id):
    return (
        frame.get("type") == "event"
        and frame.get("event", {}).get("kind") == "viewer_primary"
        and frame["event"].get("primary") == primary_tab_id
    )


# Frame fixtures for the backpressure tests. Built once at import time and
# never mutated by anything under test: the non-delta relief path
# (``_evict_ping``) removes queue entries by identity but never edits one,
# and the delta-collapse path always builds a fresh dict for the merged
# entry rather than mutating an existing one, so reusing one object across
# many broadcasts and comparing by ``==`` afterwards is safe.
_PING = build_ping(1)
_FILLER = build_event(
    1, ToolUse(turn=1, tool_use_id="tu_1", name="mcp__toy__x", input={}).to_wire()
)
_PERMISSION = build_event(1, PermissionRequest(request_id="pr_1", tool="Bash", input={}).to_wire())
_TURN_END = build_event(1, TurnEnd(turn=1, stop_reason="end_turn", cost_usd=0.01).to_wire())


def _delta(turn, text):
    return build_event(1, TextDelta(turn=turn, text=text).to_wire())


# ---------------------------------------------------------------------------
# Pending-call correlation
# ---------------------------------------------------------------------------


async def test_call_resolves_with_the_result_from_a_matching_reply():
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())
    call_task = asyncio.ensure_future(registry.call("toy.get_view", {}, timeout=5, url="http://x"))
    await asyncio.sleep(0)  # let call() enqueue and reach its own await
    call_id = next(iter(registry._pending))

    registry.resolve_call(conn, {"type": "result", "id": call_id, "result": {"ok": True}})

    assert await call_task == {"ok": True}


async def test_call_raises_call_error_on_a_matching_error_reply():
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())
    call_task = asyncio.ensure_future(registry.call("toy.get_view", {}, timeout=5, url="x"))
    await asyncio.sleep(0)
    call_id = next(iter(registry._pending))

    registry.resolve_call(
        conn, {"type": "error", "id": call_id, "error": {"code": "no_canvas", "message": "nope"}}
    )

    with pytest.raises(CallError) as excinfo:
        await call_task
    assert excinfo.value.error == {"code": "no_canvas", "message": "nope"}


async def test_late_reply_for_an_unknown_call_id_is_dropped_without_raising(capsys):
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())

    registry.resolve_call(conn, {"type": "result", "id": "c_never_asked", "result": {}})

    # No pending future existed for this id; the connection that sent it
    # did nothing wrong from its own point of view, so this must not raise.
    assert "c_never_asked" in capsys.readouterr().err


async def test_reply_from_a_connection_other_than_the_calls_target_is_dropped():
    """Only the primary is ever asked, so a reply for a pending id that
    arrives from a *different* connection is answering a call that was
    never addressed to it and must not resolve the pending future."""
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    conn_b = await registry.add(_FakeWebSocket(), tab_id="b")
    await registry.touch(conn_a)  # conn_a is primary, so call() targets it
    call_task = asyncio.ensure_future(registry.call("m", {}, timeout=5, url="x"))
    await asyncio.sleep(0)
    call_id = next(iter(registry._pending))

    registry.resolve_call(conn_b, {"type": "result", "id": call_id, "result": {"wrong": True}})
    await asyncio.sleep(0)
    assert not call_task.done()

    registry.resolve_call(conn_a, {"type": "result", "id": call_id, "result": {"right": True}})
    assert await call_task == {"right": True}


async def test_resolve_call_does_not_raise_when_the_future_is_already_done():
    """A future already resolved (its ``call`` having returned, or having
    already been failed by a disconnect) must not be told about a second
    reply; ``asyncio.Future.set_result``/``set_exception`` raise
    ``InvalidStateError`` on a done future, so ``resolve_call`` checking
    ``future.done()`` first is load-bearing, not defensive filler."""
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())
    future = asyncio.get_running_loop().create_future()
    future.set_result("already resolved")
    registry._pending["c_x"] = (future, conn)

    registry.resolve_call(conn, {"type": "result", "id": "c_x", "result": {"new": True}})

    assert future.result() == "already resolved"


# ---------------------------------------------------------------------------
# Closing a connection fails its pending futures immediately
# ---------------------------------------------------------------------------


async def test_removing_a_connection_fails_its_pending_call_future_immediately():
    """The future must be done the instant ``remove`` returns, with no
    intervening await and a ``timeout`` far too large to have expired for
    real: a test that could also pass because the real timeout fired would
    prove nothing about the immediate-failure path this exercises."""
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())
    call_task = asyncio.ensure_future(registry.call("m", {}, timeout=9999, url="x"))
    await asyncio.sleep(0)
    call_id, (future, target) = next(iter(registry._pending.items()))
    assert target is conn
    assert future.done() is False

    await registry.remove(conn)

    assert future.done() is True
    assert isinstance(future.exception(), ViewerGone)
    with pytest.raises(ViewerGone):
        await call_task


async def test_removing_a_connection_does_not_fail_a_call_targeted_at_another_connection():
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    call_a = asyncio.ensure_future(registry.call("m_a", {}, timeout=9999, url="x"))
    await asyncio.sleep(0)
    call_id_a, (future_a, _) = next(iter(registry._pending.items()))

    conn_b = await registry.add(_FakeWebSocket(), tab_id="b")  # conn_b becomes primary
    call_b = asyncio.ensure_future(registry.call("m_b", {}, timeout=9999, url="x"))
    await asyncio.sleep(0)

    await registry.remove(conn_b)

    assert future_a.done() is False
    with pytest.raises(ViewerGone):
        await call_b

    registry.resolve_call(conn_a, {"type": "result", "id": call_id_a, "result": {}})
    await call_a


# ---------------------------------------------------------------------------
# Primary election and re-election
# ---------------------------------------------------------------------------


async def test_add_makes_the_first_connection_primary():
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket(), tab_id="a")
    assert registry._primary is conn


async def test_a_newly_added_connection_becomes_primary_over_an_existing_one():
    registry = ViewerRegistry()
    await registry.add(_FakeWebSocket(), tab_id="a")
    conn_b = await registry.add(_FakeWebSocket(), tab_id="b")
    assert registry._primary is conn_b


async def test_touch_promotes_a_non_primary_connection_to_primary():
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    await registry.add(_FakeWebSocket(), tab_id="b")

    await registry.touch(conn_a)

    assert registry._primary is conn_a


async def test_touch_broadcasts_viewer_primary_to_every_connection_on_a_change():
    """Broadcast-and-take-first-reply is wrong for ``call`` frames because
    two tabs can be showing different views; the whole reason a primary is
    elected at all is so a tab knows whether it is the one being asked,
    which requires every connection, not just the newly primary one, to
    be told."""
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    conn_b = await registry.add(_FakeWebSocket(), tab_id="b")
    await _stall_writer(conn_a)
    await _stall_writer(conn_b)
    conn_a.queue._queue.clear()
    conn_b.queue._queue.clear()

    await registry.touch(conn_a)

    for conn in (conn_a, conn_b):
        frames = _queue_frames(conn)
        assert any(_is_viewer_primary(f, "a") for f in frames), (
            "connection %r was never told conn_a became primary" % conn.tab_id
        )


async def test_touching_the_current_primary_again_does_not_rebroadcast():
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())
    await _stall_writer(conn)
    before = len(_queue_frames(conn))

    await registry.touch(conn)

    assert len(_queue_frames(conn)) == before


async def test_removing_a_non_primary_connection_leaves_the_primary_unchanged():
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    conn_b = await registry.add(_FakeWebSocket(), tab_id="b")  # b is primary

    await registry.remove(conn_a)

    assert registry._primary is conn_b


async def test_removing_the_primary_reelects_the_most_recently_touched_survivor():
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    await registry.add(_FakeWebSocket(), tab_id="b")
    conn_c = await registry.add(_FakeWebSocket(), tab_id="c")
    await registry.touch(conn_a)  # recency: b, c, a; primary=a
    await registry.touch(conn_c)  # recency: b, a, c; primary=c

    await registry.remove(conn_c)  # c was primary; survivors' recency is b, a

    assert registry._primary is conn_a


async def test_removing_the_only_connection_leaves_no_primary():
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())

    await registry.remove(conn)

    assert registry._primary is None


async def test_removing_the_primary_tells_survivors_who_is_primary_now():
    """``http/ws.py`` calls ``remove`` from its own connection loop's
    ``finally`` block for every ordinary disconnect (tab closed, tab
    navigated away, network drop ending the read loop normally): this is
    the common path, not an edge case, so a surviving tab re-elected
    primary by ``remove`` must be told, the same as it would be by
    ``touch`` or by the writer task's own send-failure path.
    """
    registry = ViewerRegistry()
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    conn_b = await registry.add(_FakeWebSocket(), tab_id="b")
    await registry.touch(conn_a)  # conn_a is primary again
    await _stall_writer(conn_a)
    await _stall_writer(conn_b)
    conn_b.queue._queue.clear()  # isolate what remove(conn_a) itself produces

    await registry.remove(conn_a)

    frames = _queue_frames(conn_b)
    assert any(_is_viewer_primary(f, "b") for f in frames), (
        "surviving viewer was never told it became primary after the old primary disconnected"
    )


# ---------------------------------------------------------------------------
# Zero viewers
# ---------------------------------------------------------------------------


async def test_call_with_no_viewers_raises_no_viewer_connected_with_the_actionable_message():
    registry = ViewerRegistry()
    with pytest.raises(NoViewerConnected) as excinfo:
        await registry.call("toy.get_view", {}, timeout=5, url="http://127.0.0.1:8765")
    assert str(excinfo.value) == NO_VIEWER_MESSAGE.format(url="http://127.0.0.1:8765")


async def test_call_after_the_last_viewer_disconnects_raises_no_viewer_connected():
    registry = ViewerRegistry()
    conn = await registry.add(_FakeWebSocket())
    await registry.remove(conn)

    with pytest.raises(NoViewerConnected):
        await registry.call("m", {}, timeout=5, url="http://x")


# ---------------------------------------------------------------------------
# Backpressure: the one policy in _enqueue, driven through a stalled writer
# ---------------------------------------------------------------------------


async def test_overflow_drops_an_incoming_ping_before_trying_any_other_relief():
    registry = ViewerRegistry(queue_maxsize=2)
    conn = await registry.add(_FakeWebSocket())
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_FILLER)
    await registry.broadcast(_PERMISSION)  # full: [filler, permission]
    before = _queue_frames(conn)

    await registry.broadcast(_PING)

    assert _queue_frames(conn) == before, (
        "an incoming ping must be dropped outright, not evicted for"
    )


async def test_overflow_evicts_a_queued_ping_to_make_room_for_a_protected_frame():
    registry = ViewerRegistry(queue_maxsize=2)
    conn = await registry.add(_FakeWebSocket())
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_PING)
    await registry.broadcast(_PERMISSION)  # full: [ping, permission]

    await registry.broadcast(_TURN_END)  # turn_end must never be dropped

    frames = _queue_frames(conn)
    assert _PING not in frames
    assert frames == [_PERMISSION, _TURN_END]


async def test_overflow_prefers_evicting_a_queued_ping_over_collapsing_a_delta():
    """Precedence, not just outcome: with a ping queued, an incoming delta
    is queued alongside the evicted ping's slot rather than being merged
    into whatever delta is already there, because eviction is tried and
    succeeds before collapse is ever attempted."""
    registry = ViewerRegistry(queue_maxsize=2)
    conn = await registry.add(_FakeWebSocket())
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_PING)
    await registry.broadcast(_delta(1, "a"))  # full: [ping, delta("a")]

    await registry.broadcast(_delta(1, "b"))

    frames = _queue_frames(conn)
    assert _PING not in frames
    assert [f["event"]["text"] for f in frames] == ["a", "b"]


async def test_overflow_collapses_a_same_turn_delta_when_no_ping_is_queued():
    registry = ViewerRegistry(queue_maxsize=2)
    conn = await registry.add(_FakeWebSocket())
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_PERMISSION)
    await registry.broadcast(_delta(1, "a"))  # full: [permission, delta("a")]

    await registry.broadcast(_delta(1, "b"))

    frames = _queue_frames(conn)
    assert len(frames) == 2
    assert frames[0] == _PERMISSION
    assert frames[1]["event"]["turn"] == 1
    assert frames[1]["event"]["text"] == "ab"


async def test_overflow_closes_1013_when_no_ping_is_queued_and_the_frame_is_not_a_delta():
    registry = ViewerRegistry(queue_maxsize=2)
    ws = _FakeWebSocket()
    conn = await registry.add(ws)
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_PERMISSION)
    await registry.broadcast(_TURN_END)  # full: [permission, turn_end], no ping to evict

    await registry.broadcast(_FILLER)  # not a ping, not a delta: no relief exists

    assert ws.closed is True
    # The first frame sent may be the viewer_primary add announced.
    payload, opcode = ws.sent[-1]
    assert opcode == ws.CLOSE
    assert struct.unpack("!H", payload[:2])[0] == CLOSE_OVERFLOW
    assert conn.id not in registry._connections


async def test_overflow_closes_1013_when_a_delta_has_no_same_turn_entry_to_collapse_into():
    registry = ViewerRegistry(queue_maxsize=2)
    ws = _FakeWebSocket()
    conn = await registry.add(ws)
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_delta(1, "a"))
    await registry.broadcast(_delta(2, "b"))  # full: [delta(turn1), delta(turn2)], no ping

    await registry.broadcast(
        _delta(3, "c")
    )  # a third, still different, turn: nothing to merge into

    assert ws.closed is True
    payload, opcode = ws.sent[-1]
    assert opcode == ws.CLOSE
    assert struct.unpack("!H", payload[:2])[0] == CLOSE_OVERFLOW


async def test_permission_request_and_turn_end_and_call_are_never_dropped_by_either_relief():
    """The trio Annealage Mesh's M4 brief names by name. Each is pushed onto
    an already saturated queue with nothing evictable and no matching delta
    to collapse into; the policy's only remaining move is to close the
    connection rather than quietly drop any of the three."""
    for protected in (_PERMISSION, _TURN_END, build_call("c_1", "toy.get_view", {})):
        registry = ViewerRegistry(queue_maxsize=1)
        ws = _FakeWebSocket()
        conn = await registry.add(ws)
        await _stall_writer(conn)
        conn.queue._queue.clear()
        await registry.broadcast(_FILLER)  # full: [filler], no ping, not a delta

        await registry.broadcast(protected)

        assert ws.closed is True, (
            "%r was silently dropped instead of closing the connection" % protected
        )


async def test_overflow_evicts_a_queued_ping_to_make_room_for_a_call_frame():
    registry = ViewerRegistry(queue_maxsize=2)
    ws = _FakeWebSocket()
    conn = await registry.add(ws)
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_PING)
    await registry.broadcast(_FILLER)  # full: [ping, filler]

    call_task = asyncio.ensure_future(registry.call("toy.get_view", {}, timeout=5, url="x"))
    await asyncio.sleep(0)

    frames = _queue_frames(conn)
    assert _PING not in frames
    assert any(f.get("type") == "call" for f in frames)

    call_id = next(iter(registry._pending))
    registry.resolve_call(conn, {"type": "result", "id": call_id, "result": {}})
    await call_task


async def test_call_that_cannot_be_enqueued_fails_with_viewer_gone_rather_than_vanishing():
    """A call is never silently dropped: when neither relief mechanism
    frees room for it, the connection is closed and the pending future
    this same ``call()`` registered is failed through the immediate
    disconnect path, exactly as if the browser had genuinely gone away
    mid-call."""
    registry = ViewerRegistry(queue_maxsize=1)
    ws = _FakeWebSocket()
    conn = await registry.add(ws)
    await _stall_writer(conn)
    conn.queue._queue.clear()
    await registry.broadcast(_TURN_END)  # full: [turn_end], nothing evictable

    with pytest.raises(ViewerGone):
        await registry.call("toy.get_view", {}, timeout=5, url="http://x")

    assert ws.closed is True
    assert conn.id not in registry._connections


# ---------------------------------------------------------------------------
# Backpressure: the real writer task, not a stalled stand-in
# ---------------------------------------------------------------------------
#
# The tests above drive _enqueue's overflow policy with the writer stalled,
# to hold a queue in a known raw state; the tests below instead let
# _run_writer actually run, with a small injected flush_interval, to cover
# what only the writer itself does: coalescing same-turn deltas into one
# send, flushing a lone pending delta once the interval passes, and never
# sending a frame queued behind a pending delta ahead of it.


async def test_writer_coalesces_same_turn_deltas_into_one_send_carrying_the_later_seq():
    registry = ViewerRegistry(flush_interval=0.02)
    ws = _FakeWebSocket()
    await registry.add(ws)
    await _wait_for_sent_count(ws, 1)  # let the connection's own viewer_primary go out
    ws.sent.clear()

    await registry.broadcast(build_event(10, TextDelta(turn=1, text="Hel").to_wire()))
    await registry.broadcast(build_event(11, TextDelta(turn=1, text="lo").to_wire()))

    await _wait_for_sent_count(ws, 1)
    assert len(ws.sent) == 1, "two same-turn deltas must merge into a single send"
    frame = json.loads(ws.sent[0][0])
    assert frame["event"]["text"] == "Hello"
    assert frame["seq"] == 11, (
        "the merged frame must carry the later delta's seq, or a reconnecting "
        "client resuming from it would be replayed text this send already delivered"
    )


async def test_writer_flushes_a_lone_pending_delta_within_the_injected_interval():
    registry = ViewerRegistry(flush_interval=0.02)
    ws = _FakeWebSocket()
    await registry.add(ws)
    await _wait_for_sent_count(ws, 1)  # let the connection's own viewer_primary go out
    ws.sent.clear()

    await registry.broadcast(_delta(1, "solo"))

    await _wait_for_sent_count(ws, 1)
    frame = json.loads(ws.sent[0][0])
    assert frame["event"]["text"] == "solo"


async def test_writer_never_sends_a_queued_non_delta_frame_before_a_pending_delta():
    # A long interval, deliberately: this proves turn_end is sent because
    # it is queued behind the pending delta, forcing an immediate flush,
    # not because the flush deadline happened to expire first.
    registry = ViewerRegistry(flush_interval=0.5)
    ws = _FakeWebSocket()
    await registry.add(ws)
    await _wait_for_sent_count(ws, 1)  # let the connection's own viewer_primary go out
    ws.sent.clear()

    await registry.broadcast(_delta(1, "partial"))
    await registry.broadcast(_TURN_END)

    await _wait_for_sent_count(ws, 2)
    kinds = [json.loads(data)["event"]["kind"] for data, _ in ws.sent]
    assert kinds == ["text_delta", "turn_end"], (
        "a pending delta must flush before any frame queued behind it, never after"
    )


async def test_writer_send_failure_removes_the_connection_and_broadcasts_the_new_primary():
    registry = ViewerRegistry(flush_interval=0.02)
    conn_a = await registry.add(_FakeWebSocket(), tab_id="a")
    conn_b = await registry.add(_RaisingAfterWebSocket(ok_sends=1), tab_id="b")  # b is primary
    # Let both connections' own viewer_primary announcements go out (a's own,
    # from its add, and b's, from becoming primary over a) before isolating
    # what conn_b's send failure produces.
    await _wait_for_sent_count(conn_a.ws, 2)
    conn_a.ws.sent.clear()

    await registry.broadcast(_TURN_END)  # gives conn_b's writer something to fail on sending

    await _wait_for_sent_count(conn_a.ws, 2)  # turn_end, then the re-election broadcast
    assert conn_b.id not in registry._connections
    assert registry._primary is conn_a
    frame = json.loads(conn_a.ws.sent[-1][0])
    assert frame["event"]["kind"] == "viewer_primary"
    assert frame["event"]["primary"] == "a"


# --- ViewerBus: what the tool layer sees, and the pause switch it holds ------


async def test_the_bus_binds_the_timeout_and_the_url_a_tool_should_not_have_to_carry():
    registry = ViewerRegistry()
    seen = {}

    async def fake_call(method, params, *, timeout, url):
        seen.update(method=method, params=params, timeout=timeout, url=url)
        return {"ok": True}

    registry.call = fake_call
    bus = ViewerBus(registry, url="http://127.0.0.1:8765/#t=tok", timeout=3.5)

    assert await bus.call("toy.get_view") == {"ok": True}
    assert seen == {
        "method": "toy.get_view",
        "params": {},
        "timeout": 3.5,
        "url": "http://127.0.0.1:8765/#t=tok",
    }


async def test_the_bus_reports_no_viewer_with_the_url_to_open():
    """The wording and the URL are what reach the model as the tool's error, so
    the bus has to be the thing that knows the URL: a tool handler has no idea
    what this process bound to."""
    bus = ViewerBus(ViewerRegistry(), url="http://100.101.1.2:8765/#t=tok")
    with pytest.raises(NoViewerConnected, match="http://100.101.1.2:8765/#t=tok"):
        await bus.call("toy.set_view")


async def test_the_pause_switch_reports_only_a_real_change():
    """The return value is what stops a redundant `pause_changed` event going
    out for a click that set the flag to what it already was, which two tabs
    racing one control produce routinely."""
    bus = ViewerBus(ViewerRegistry(), url="http://127.0.0.1:8765/#t=tok")
    assert bus.paused is False
    assert bus.set_paused(True) is True
    assert bus.paused is True
    assert bus.set_paused(True) is False
    assert bus.set_paused(False) is True
    assert bus.paused is False


async def test_a_primary_change_reaches_every_viewer_behind_events_already_published():
    """A session publishes by appending and scheduling the broadcast; a
    ``viewer_primary`` announced before that broadcast has run must still
    reach each page after it, since the page drops an event at or below the
    last seq it has."""
    from annealage_agent.session.events import EventLog

    log = EventLog()
    registry = ViewerRegistry(event_log=log)
    first = await registry.add(_FakeWebSocket(), tab_id="a")
    second = await registry.add(_FakeWebSocket(), tab_id="b")
    await _stall_writer(first)
    await _stall_writer(second)
    first.queue._queue.clear()
    second.queue._queue.clear()

    event = TextDelta(turn=1, text="streamed")
    seq = log.append(event)
    asyncio.ensure_future(registry.broadcast(build_event(seq, event.to_wire())))
    await registry.touch(first)  # the primary moves: a viewer_primary with a later seq

    for conn in (first, second):
        frames = _queue_frames(conn)
        assert [f["event"]["kind"] for f in frames] == ["text_delta", "viewer_primary"]
        assert [f["seq"] for f in frames] == sorted(f["seq"] for f in frames)


async def test_a_turn_s_sender_given_by_login_is_the_human_with_that_login():
    bus = ViewerBus(ViewerRegistry(), url="http://127.0.0.1:1/")
    assert (bus.turn_human, bus.turn_by) == (None, None)
    bus.begin_turn([], by="sam@example.com")
    assert (bus.turn_human, bus.turn_by) == (Human(login="sam@example.com"), "sam@example.com")
    bus.begin_turn([])
    assert (bus.turn_human, bus.turn_by) == (Human(), None)
