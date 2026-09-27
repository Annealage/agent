"""ViewerRegistry: every browser tab's WebSocket connection, one primary
among them for ``call`` frames, and the backpressure policy that keeps a
slow tab from stalling everyone else's events.

``http/ws.py`` is the only expected caller: it registers a connection with
``add`` on a successful upgrade, calls ``touch`` on every inbound frame so
recency tracks real interaction, feeds inbound ``result``/``error`` frames
to ``resolve_call``, and calls ``remove`` from its own connection loop's
``finally`` block. Nothing in this module reads a socket; it only ever
writes to one, through the one writer task each connection gets.

Multi-viewer is convenience, not collaboration (plan section 5): several
tabs belonging to one human, not several people. That is why exactly one
connection is ever "primary" and why nothing here attaches identity to an
event beyond the tab id ``session.base.AgentEvent.viewer`` already
carries.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any, Dict, Optional

from . import product, protocol
from .protocol import CLOSE_OVERFLOW, build_call, build_event, close_with_code
from .session.base import Attention, ViewerPrimary

# Per-connection outbound queue depth (plan section 3.3). Deep enough to
# absorb a burst of tool-result and text-delta events between two 30 ms
# writer flushes without spilling into the overflow policy on ordinary
# traffic; the overflow policy below exists for the tab that is offline,
# slow, or simply gone, not for this one.
QUEUE_MAXSIZE = 256

# How often the writer task flushes a run of coalesced text_delta events
# for one turn (plan section 3.3). Small enough that streamed text still
# reads as live; large enough that a fast token stream collapses into a
# handful of sends per second instead of one send per token.
FLUSH_INTERVAL = 0.03

# ``url`` is the tokenless address (see ``ViewerBus``), so the message says
# which link carries the token rather than repeating it.
NO_VIEWER_MESSAGE = (
    "no viewer connected; ask the human to open the viewer at {url} "
    "(the link printed when it started, which carries its access token)"
)

# How long one call into the browser is given to come back (plan section 3.3).
# Generous enough for a phone over a tailnet to encode a viewport capture, and
# short enough that a model waiting on a page that will never answer learns so
# while the human is still watching the same turn.
CALL_TIMEOUT = 10.0

# The markers ``ViewerBus.begin_turn`` wraps the product's notes in, and the
# pattern that finds either inside a note's own text (any case, any spacing
# after the bracket), which ``_neutralise_markers`` defuses.
_NOTE_OPEN = "[System note from %s, not written by the human]"
_NOTE_CLOSE = "[End of system note]"
_MARKER_RE = re.compile(r"\[(\s*(?:end\s+of\s+system\s+note|system\s+note\s+from))", re.IGNORECASE)


def _neutralise_markers(text: str) -> str:
    """``text`` with every opening bracket of a note marker turned into a
    quoted form, so text inside a note can neither close it nor open one."""
    return _MARKER_RE.sub(r"(quoted \1", text)


class NoViewerConnected(Exception):
    """Raised by ``ViewerRegistry.call`` when no connection is registered
    to become primary. Carries the exact wording plan section 3.3
    specifies, so a caller (the tool dispatcher, M6) can surface it
    verbatim as a tool error instead of reformatting it and risking a
    slightly different phrase reaching the model each time."""


class ViewerGone(Exception):
    """Raised to fail a pending call whose target connection closed before
    replying. Distinct from ``asyncio.TimeoutError``, which ``call``'s own
    ``asyncio.wait_for`` still raises for the case the reply genuinely
    never arrives: this exception fires the moment the connection is
    known to be gone, rather than making the caller wait out the full
    timeout to learn a fact already known."""


class CallError(Exception):
    """Raised when the primary viewer answers a ``call`` with an inbound
    ``error`` frame instead of a ``result``. Carries the frame's ``error``
    object (``{"code", "message"}``) as ``self.error``, for a caller that
    wants to report the domain-specific code rather than a generic
    failure string."""

    def __init__(self, error: dict):
        self.error = error
        super().__init__(error.get("message", "viewer call failed"))


class _Connection:
    """One browser tab's registration: its socket, its outbound queue, and
    the writer task draining that queue. Not part of this module's public
    API; ``add`` returns instances of this class as opaque handles.

    ``tab_id`` is the id the tab gave in its ``hello``; ``human`` is the
    ``identity.Human`` that opened the socket, whose login is what the events
    they cause over it record."""

    __slots__ = ("id", "ws", "tab_id", "human", "queue", "writer_task", "removed")

    def __init__(self, conn_id: int, ws, tab_id: Optional[str], human, queue_maxsize: int):
        self.id = conn_id
        self.ws = ws
        self.tab_id = tab_id
        self.human = human
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self.writer_task: Optional[asyncio.Task] = None
        self.removed = False


def _is_ping(frame: dict) -> bool:
    return frame.get("type") == "ping"


def _is_text_delta(frame: dict) -> bool:
    return frame.get("type") == "event" and frame.get("event", {}).get("kind") == "text_delta"


class ViewerRegistry:
    """Tracks connections for one session, elects a primary, and enforces
    the backpressure policy on each connection's outbound queue.

    One instance serves every browser tab watching one running process;
    there is no per-session partitioning here because M4 has no
    multi-session concept yet (that arrives with M5's session control).

    Never calls ``task_done()`` or ``join()`` on any connection's
    ``asyncio.Queue``. ``_evict_ping`` and ``_collapse_delta`` remove or
    replace an item that is not at the head by reaching directly into the
    ``collections.deque`` the queue wraps internally (``_queue``), which
    has no way to keep ``Queue._unfinished_tasks`` in step with what it
    does. That is safe only as long as nothing in this class ever drains
    a queue through the ``task_done``/``join`` protocol; introducing that
    protocol here later would need both methods revisited first.
    """

    def __init__(
        self,
        *,
        queue_maxsize: int = QUEUE_MAXSIZE,
        flush_interval: float = FLUSH_INTERVAL,
        event_log=None,
        on_presence=None,
    ):
        self._queue_maxsize = queue_maxsize
        self._flush_interval = flush_interval
        # The shared EventLog, so a self-originated event (currently only
        # viewer_primary) takes a seq from the same monotonic stream a
        # reconnecting client replays from last_seq, rather than one of
        # its own that replay would silently skip. None is accepted, and
        # falls back to a private counter, only so this class stays
        # exercisable on its own in a test that has no EventLog at all.
        self._event_log = event_log
        # Called with the live connection count whenever it changes. The
        # permission broker needs it: a request nobody can answer has to be
        # denied rather than left waiting, and the broker deliberately knows
        # nothing about connections, so presence is pushed to it rather than
        # polled from here.
        self._on_presence = on_presence
        self._fallback_seq = 0
        # Insertion order doubles as recency order: touch() and add() both
        # re-insert a connection at the end, so "most recently interacted
        # with" (the primary-election rule) is always the last value.
        self._connections: Dict[int, _Connection] = {}
        self._next_conn_id = 1
        self._primary: Optional[_Connection] = None
        self._pending: Dict[str, tuple] = {}  # call_id -> (future, _Connection)
        self._next_call_id = 1

    # -- connection lifecycle -------------------------------------------------

    async def add(self, ws, tab_id: Optional[str] = None, human=None) -> _Connection:
        """Register a freshly upgraded connection and start its writer task.

        Becomes primary immediately: connecting is, by definition, the
        most recent interaction there has been with this new connection.
        Returns the handle ``http/ws.py`` must pass to every other method
        for this socket, and must eventually pass to ``remove``.
        """
        conn = _Connection(self._next_conn_id, ws, tab_id, human, self._queue_maxsize)
        self._next_conn_id += 1
        self._connections[conn.id] = conn
        conn.writer_task = asyncio.ensure_future(self._run_writer(conn))
        conn.writer_task.add_done_callback(lambda task, c=conn: self._on_writer_done(c, task))
        await self._set_primary(conn)
        self._notify_presence()
        return conn

    async def remove(self, conn: _Connection) -> None:
        """Unregister ``conn``: fail its pending futures at once, stop its
        writer task, and re-elect and broadcast a new primary if ``conn``
        held that role.

        Idempotent, so calling this from both an explicit close path and a
        connection's own ``finally`` block (both of which the auth and
        overflow paths below trigger, alongside the read loop's normal
        end-of-connection handling) is safe.
        """
        changed = self._discard(conn)
        if changed:
            await self._broadcast_primary()
        if conn.writer_task is not None and not conn.writer_task.done():
            conn.writer_task.cancel()

    async def close_all(
        self, code: int = protocol.CLOSE_GOING_AWAY, reason: str = "server going away"
    ) -> None:
        """Close every connection with a coded close, for shutdown.

        Each connection's writer task is stopped before its close frame is
        sent, so the frame cannot interleave with one the writer was draining,
        and every send is attempted even if an earlier one fails: a peer that
        has already vanished must not stop the others from being told.

        Closing the listening socket is not enough on its own. It stops new
        connections and nothing more, and a WebSocket handler never returns on
        its own, so without this a shutdown leaves every viewer waiting out its
        liveness timeout before it reconnects or falls back to polling.
        """
        for conn in list(self._connections.values()):
            if conn.writer_task is not None and not conn.writer_task.done():
                conn.writer_task.cancel()
            self._discard(conn)
            try:
                await protocol.close_with_code(conn.ws, code, reason)
            except Exception as exc:
                sys.stderr.write(
                    "warning: could not close viewer %s cleanly: %r\n" % (conn.id, exc)
                )

    def _notify_presence(self) -> None:
        """Report the live connection count, swallowing a listener's failure.

        Called from both registration and discard, including the discard that
        runs inside a writer task's own failure path, so it must not raise: a
        broker that throws here would abort a teardown halfway.
        """
        if self._on_presence is None:
            return
        try:
            self._on_presence(len(self._connections))
        except Exception as exc:
            sys.stderr.write("warning: viewer presence listener failed: %r\n" % (exc,))

    def _discard(self, conn: _Connection) -> bool:
        """The synchronous half of removal: drop the connection, fail its
        pending futures, and pick a new primary if needed. Returns True if
        the primary changed, so the caller (always able to await, unlike
        this method) can broadcast ``viewer_primary``.

        Split out from ``remove`` so the writer task's own send-failure
        path can call it directly. That path runs *inside* the very task
        ``remove`` would otherwise cancel; asyncio delivers a task
        cancelling itself as a ``CancelledError`` thrown at that
        coroutine's next ``await``, not at the ``cancel()`` call site, so
        calling the cancelling version of removal from within the task
        being removed risks aborting its own cleanup partway through.
        """
        if conn.removed:
            return False
        conn.removed = True
        self._connections.pop(conn.id, None)
        self._notify_presence()
        # A pending future is failed here, immediately, rather than left
        # to its own asyncio.wait_for timeout (plan section 3.3): a model
        # that waits out a ten-second call timeout to learn the browser
        # is gone has learned nothing a prompt ViewerGone would not have
        # told it in milliseconds.
        for call_id in [cid for cid, (_, target) in self._pending.items() if target is conn]:
            future, _ = self._pending.pop(call_id)
            if not future.done():
                future.set_exception(ViewerGone("viewer connection closed"))
        if self._primary is conn:
            # Most-recently-touched survivor, since _connections is kept
            # in recency order by add()/touch() re-inserting at the end.
            self._primary = next(reversed(self._connections.values()), None)
            return True
        return False

    async def touch(self, conn: _Connection) -> None:
        """Record ``conn`` as the most recent interaction.

        Called by ``http/ws.py`` on every inbound frame. Promotes ``conn``
        to primary if it was not already, broadcasting ``viewer_primary``
        on that change, so a call always goes to whichever tab a human
        most recently drove rather than whichever tab happened to connect
        first (broadcast-and-take-first-reply is wrong here: two tabs
        can show different views, Mesh's different camera poses, so a
        broadcast read of the view would answer with a coin flip).
        """
        if conn.removed:
            return
        await self._set_primary(conn)

    async def _set_primary(self, conn: _Connection) -> None:
        # Re-inserting moves conn to the end of _connections, which is
        # what makes "most recently touched survivor" in _discard work
        # without a separate timestamp per connection.
        self._connections.pop(conn.id, None)
        self._connections[conn.id] = conn
        if self._primary is conn:
            return
        self._primary = conn
        await self._broadcast_primary()

    async def _broadcast_primary(self) -> None:
        primary_tab_id = self._primary.tab_id if self._primary is not None else None
        event = ViewerPrimary(primary=primary_tab_id)
        frame = build_event(self._next_seq(event), event.to_wire())
        # Scheduled, then awaited, rather than awaited directly: a session's
        # events are appended and their broadcasts scheduled (the app's
        # publisher), so one appended before this event may still be waiting
        # to be enqueued. Scheduling behind it keeps every connection's frames
        # in seq order, which the page relies on (static/ws.js drops an event
        # at or below the last seq it has).
        await asyncio.ensure_future(self.broadcast(frame))

    def _next_seq(self, event) -> int:
        """Assign a seq to a self-originated event.

        Delegates to the shared EventLog when one was given at
        construction, appending the event there for real so it takes its
        place in the one monotonic stream a reconnecting client replays
        from ``last_seq``. Falls back to a private counter only when this
        registry was built with no EventLog, which happens only in a test
        exercising ``ViewerRegistry`` in isolation.
        """
        if self._event_log is not None:
            return self._event_log.append(event)
        self._fallback_seq += 1
        return self._fallback_seq

    # -- outbound: broadcast and correlated calls ------------------------------

    async def broadcast(self, frame: dict) -> None:
        """Enqueue ``frame`` for every registered connection.

        Every connected viewer is a broadcast subscriber of the same
        conversation (plan section 3.3): both tabs see the same events,
        only ``call`` frames single one connection out.

        A connection that overflows is closed and removed off this loop's
        own call stack (see ``_enqueue``), so its close never delays
        ``frame`` reaching any connection still to come in this same
        loop. Whatever overflow closes this call triggers are still
        awaited once every connection has had its turn, so this method's
        own caller sees them resolved by the time it returns rather than
        racing a background task it has no handle to.
        """
        overflow_closes = []
        for conn in list(self._connections.values()):
            closer = await self._enqueue(conn, frame)
            if closer is not None:
                overflow_closes.append(closer)
        if overflow_closes:
            await asyncio.gather(*overflow_closes)

    async def call(self, method: str, params: dict, *, timeout: float, url: str) -> Any:
        """Send a ``call`` frame to the primary viewer and await its reply.

        Mirrors the SDK's own pending-future pattern the plan cites
        (``_internal/query.py:131`` and ``:546``): allocate an id,
        register a future keyed by it, send, ``await asyncio.wait_for``,
        and clean up the pending entry in ``finally`` regardless of how
        this call ends, so a timeout or a ``ViewerGone`` never leaves a
        stale entry for some later, unrelated reply to match against.

        Raises ``NoViewerConnected`` if no connection is registered at
        all: with zero viewers there is no state to query, and per plan
        section 3.3 a silent hang here teaches the model nothing.
        """
        if self._primary is None:
            raise NoViewerConnected(NO_VIEWER_MESSAGE.format(url=url))
        conn = self._primary
        call_id = "c_%d" % self._next_call_id
        self._next_call_id += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = (future, conn)
        try:
            await self._enqueue(conn, build_call(call_id, method, params))
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(call_id, None)

    def resolve_call(self, conn: _Connection, frame: dict) -> None:
        """Feed an inbound ``result`` or ``error`` frame to the pending
        future it answers, matched by ``frame["id"]``.

        A reply for an id with no pending future (already timed out,
        already resolved, or simply never asked) is logged once and
        dropped rather than raised: the connection sending it has done
        nothing wrong from its own point of view, and a stray reply is
        not a reason to tear down the socket it arrived on. The same
        applies to a reply whose id is pending but was sent to a
        *different* connection than the one it arrived from, since only
        the primary is ever asked and any other sender is answering a
        call that was never addressed to it.
        """
        call_id = frame.get("id")
        entry = self._pending.get(call_id)
        if entry is None:
            sys.stderr.write(
                "warning: late reply for unknown call id %r from viewer %s; dropping\n"
                % (call_id, conn.id)
            )
            return
        future, target = entry
        if target is not conn:
            sys.stderr.write(
                "warning: reply for call id %r arrived from viewer %s, not the "
                "viewer it was sent to; dropping\n" % (call_id, conn.id)
            )
            return
        if future.done():
            return
        if frame.get("type") == "error":
            future.set_exception(CallError(frame.get("error", {})))
        else:
            future.set_result(frame.get("result"))

    # -- backpressure: the one policy, in one function -------------------------

    async def _enqueue(self, conn: _Connection, frame: dict) -> Optional[asyncio.Task]:
        """Place ``frame`` on ``conn``'s outbound queue, applying the one
        overflow policy this registry has, all of it here:

        On a full queue, a ``ping`` about to be enqueued is simply
        dropped: the next one is seconds away and a client's liveness
        check tolerates a missed beat, so it is never worth evicting
        something else to make room for it. For anything else, a queued
        ``ping`` is evicted first to make room, since it is the one frame
        kind this registry ever discards outright. If that frees no slot
        (nothing queued was a ping) and the incoming frame is a
        ``text_delta`` event, it is folded into a same-turn delta already
        queued instead of taking a new slot; growing one delta's text
        costs nothing a queue slot would. ``call``, ``permission_request``
        and ``turn_end`` frames are never sacrificed to either relief:
        a lost call is a model waiting on a reply that will never come, a
        lost permission request is a decision the model made that the
        human never saw, and a lost turn_end leaves the browser believing
        a turn is still running. When neither relief frees room, the
        connection is asked for more than it can currently drain and is
        closed with ``CLOSE_OVERFLOW``, so the client reconnects and
        replays every missed event from ``seq`` rather than silently
        losing some of them.

        That close is never awaited from here: this method's only callers
        are ``broadcast``, which enqueues to every connection in one loop,
        and ``call``, which enqueues to a single connection and then
        awaits a reply future of its own. Awaiting a coded close inline
        would await a send on the very socket that got a connection here
        in the first place, and a peer whose TCP window is full may not
        error out on that send for as long as the platform's retransmit
        timers run; ``broadcast`` would then stall on one connection's
        close before it ever reached the next one in its loop. The close
        and the removal are handed to a detached task instead, returned
        here so ``broadcast`` can still wait for it once every connection
        in its own loop has had a turn (see ``broadcast``'s docstring);
        ``call`` has no use for the return value, since its own pending
        future already resolves once ``_close_overflowed`` reaches
        ``remove``.
        """
        try:
            conn.queue.put_nowait(frame)
            return None
        except asyncio.QueueFull:
            pass

        if _is_ping(frame):
            return None

        if self._evict_ping(conn.queue):
            try:
                conn.queue.put_nowait(frame)
                return None
            except asyncio.QueueFull:
                pass

        if _is_text_delta(frame) and self._collapse_delta(conn.queue, frame):
            return None

        return asyncio.ensure_future(self._close_overflowed(conn))

    async def _close_overflowed(self, conn: _Connection) -> None:
        """Send the ``CLOSE_OVERFLOW`` close and unregister ``conn``, off
        the call stack that discovered the overflow (see ``_enqueue``).
        ``remove`` cancels ``conn``'s writer task and, if ``conn`` held
        that role, re-elects and broadcasts a new primary.
        """
        await close_with_code(conn.ws, CLOSE_OVERFLOW)
        await self.remove(conn)

    @staticmethod
    def _evict_ping(queue: asyncio.Queue) -> bool:
        """Remove one queued ``ping`` frame from ``queue``, if any.

        ``asyncio.Queue`` has no public way to remove an item that is not
        at the head, so this reaches into the ``collections.deque`` it
        wraps internally (``_queue``) rather than draining the whole
        queue into a list and rebuilding it, which would briefly make an
        already-full queue look empty to a concurrent ``get()`` and
        reorder every frame behind the one removed. Safe here because
        this registry only ever uses ``put_nowait``/``get``, never the
        blocking ``put()``, whose own internal bookkeeping (waking
        blocked putters) this bypass does not touch.
        """
        for item in queue._queue:
            if _is_ping(item):
                queue._queue.remove(item)
                return True
        return False

    @staticmethod
    def _collapse_delta(queue: asyncio.Queue, frame: dict) -> bool:
        """Fold ``frame``, a ``text_delta`` event, into a same-turn
        ``text_delta`` already queued, in place of adding a second item.

        Searched from the tail end, not the head: the queue is drained
        head-first, so the *last* same-turn delta already queued is the
        one chronologically nearest to ``frame``. Appending to an earlier
        one instead would splice ``frame``'s text into the middle of the
        turn's stream, ahead of whatever a later, still-queued delta for
        the same turn already holds, reordering the text a reader sees.

        The scan stops the moment it reaches a non-delta frame, matching
        the writer's own coalescer (``_run_writer``), which flushes
        ``pending`` before sending any non-delta frame: a non-delta frame
        already queued behind some earlier same-turn delta means that
        delta will reach the browser before whatever the non-delta frame
        represents (a tool call, a permission request, a turn ending), so
        collapsing ``frame`` into it would deliver ``frame``'s text
        alongside content the browser must see first. Falling through to
        the overflow close in that case is correct: nothing here can
        collapse ``frame`` in without reordering something.

        Replaces the queued item outright (by index, into the deque)
        rather than mutating the dict found there: every frame handed to
        ``_enqueue`` is this connection's own copy (``broadcast`` and
        ``call`` never share one dict object across two connections), so
        mutating in place here would be safe on that account too, but
        rebuilding the dict keeps this method correct even if a future
        caller of ``_enqueue`` ever changes that.

        The merged frame carries ``frame``'s own seq, for the same reason
        the writer's coalescer does: the merged text already includes
        everything up to that point, so a client resuming from this seq
        must not be replayed text this frame already delivered.
        """
        target = frame["event"].get("turn")
        for i in range(len(queue._queue) - 1, -1, -1):
            item = queue._queue[i]
            if not _is_text_delta(item):
                return False
            if item["event"].get("turn") == target:
                merged_event = dict(item["event"])
                merged_event["text"] = merged_event.get("text", "") + frame["event"].get("text", "")
                merged_frame = dict(item)
                merged_frame["event"] = merged_event
                merged_frame["seq"] = frame["seq"]
                queue._queue[i] = merged_frame
                return True
        return False

    def _on_writer_done(self, conn: _Connection, task: asyncio.Task) -> None:
        """Report and clean up after a writer task that ended on its own,
        rather than through ``remove``'s cancellation.

        ``_run_writer`` already catches its own ``CancelledError`` and
        returns normally, and a send failure inside ``_send`` is already
        handled by discarding the connection there; both of those end the
        task with no exception, so this callback sees ``task.exception()
        is None`` for both and does nothing further. Anything else here is
        a bug the writer loop did not anticipate (a malformed frame
        reaching ``_collapse_delta``, say): without this callback that
        failure would surface only as "Task exception was never
        retrieved" at garbage collection, while ``conn`` stayed registered
        and possibly primary with nothing left draining its queue or ever
        going to answer a ``call`` sent to it.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        sys.stderr.write("error: viewer %s writer task failed: %r; removing\n" % (conn.id, exc))
        asyncio.ensure_future(self.remove(conn))

    async def _run_writer(self, conn: _Connection) -> None:
        """Drain ``conn``'s outbound queue and send frames to its socket.

        This is the one place ``text_delta`` coalescing and the 30 ms
        flush interval live. A dequeued ``text_delta`` is held as
        ``pending`` rather than sent immediately; a further delta for the
        *same* turn is merged into it, and anything else dequeued (a
        different turn's delta, or any non-delta frame) flushes whatever
        is pending first, so frames are never sent out of the order they
        were produced in. A pending delta with nothing new to merge is
        flushed on its own once ``self._flush_interval`` passes, via
        ``asyncio.wait_for`` racing the next dequeue against that
        deadline; the deadline is never a bare sleep; it is only ever the
        timeout of a wait that is also trying to make progress.

        Ends when this task is cancelled (``remove`` does this once the
        connection is gone) or a send fails (the socket itself is gone,
        which ``_discard`` treats identically to an explicit removal).
        """
        pending: Optional[dict] = None
        deadline: Optional[float] = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                if deadline is not None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        frame, timed_out = None, True
                    else:
                        try:
                            frame = await asyncio.wait_for(conn.queue.get(), timeout=remaining)
                            timed_out = False
                        except asyncio.TimeoutError:
                            frame, timed_out = None, True
                else:
                    frame = await conn.queue.get()
                    timed_out = False

                if timed_out:
                    if pending is not None:
                        if not await self._send(conn, pending):
                            return
                        pending, deadline = None, None
                    continue

                if _is_text_delta(frame):
                    if pending is not None and pending["event"].get("turn") == frame["event"].get(
                        "turn"
                    ):
                        pending = dict(pending)
                        pending["event"] = dict(pending["event"])
                        pending["event"]["text"] = pending["event"].get("text", "") + frame[
                            "event"
                        ].get("text", "")
                        # The merged text already includes everything up to
                        # frame's own contribution, so the merged frame must
                        # carry frame's seq, not the seq the run started
                        # with: a client resuming from a coalesced send's
                        # seq must not be replayed text this send already
                        # delivered.
                        pending["seq"] = frame["seq"]
                    else:
                        if pending is not None and not await self._send(conn, pending):
                            return
                        pending = frame
                        deadline = loop.time() + self._flush_interval
                    continue

                if pending is not None:
                    if not await self._send(conn, pending):
                        return
                    pending, deadline = None, None
                if not await self._send(conn, frame):
                    return
        except asyncio.CancelledError:
            return

    async def _send(self, conn: _Connection, frame: dict) -> bool:
        """Send one frame; return False (and tear the connection down) on
        failure. Runs inside ``conn``'s own writer task, so on failure it
        calls the synchronous ``_discard`` directly rather than ``remove``,
        which would try to cancel this very task from inside itself; see
        ``_discard``'s docstring for why that is unsafe here.
        """
        try:
            await conn.ws.send(json.dumps(frame))
            return True
        except Exception:
            changed = self._discard(conn)
            if changed:
                await self._broadcast_primary()
            return False


class ViewerBus:
    """What the tool layer sees of the browser: one correlated call into the
    primary viewer, and the human's pause switch.

    Every product tool handler depends on this rather than on ``ViewerRegistry``,
    which is what lets ``tests/test_tools.py`` drive all of them against a
    recorder with the same two members and no socket at all. It also means no
    handler has to carry the viewer URL that ``NoViewerConnected`` names, or
    decide a timeout: both are properties of the run, fixed here once.

    ``url`` is where the page is served, **without** the browser token: it is
    written into text the model reads (``NoViewerConnected``, a permission
    request refused for want of a viewer) and from there into the session's
    event log inside the served directory, and the browser token is what
    approves a permission request over ``/ws``. The human opens the link the
    startup banner printed, which carries it.

    **The pause switch lives here, in the server, and that is the whole
    point of it.** A flag the browser kept would not be enforcement: the tools
    run in this process, and a page is free to be old, cached, or simply a
    second tab that never saw the click. The browser's control sends an
    inbound ``pause`` frame, ``http/ws.py`` sets the flag through this object,
    and the tool registry reads it before running any tool that changes
    anything. What the human sees in the topbar is then a display of a decision
    recorded here, not the decision itself.

    This switch, rather than an approval card, is the control over the
    view-grade tools (Mesh's move the camera and hide parts): those are
    pre-allowed, because the
    human is watching the screen they change and a card per camera move would
    be clicked unread. So the switch is not a convenience. It is the only thing
    standing between the model and the view while the human is working in it,
    and the tools it gates are a superset of the ones that prompt.

    Read-class tools are deliberately not gated. Pausing exists so the human
    can line up a view or edit a pin comment without the agent moving things
    out from under them; a model that goes on reading while paused is doing no
    harm and is better informed when the pause lifts.

    **The turn counter and the notes.** ``turn`` is the conversation's
    current turn number: 0 before the first human turn of a new session, and
    on a resumed one it continues from the last turn in the session's history
    (``create_app`` seeds it from the event log), so the next turn never
    reuses a number the page already shows. ``turn_at_open`` is that seed (0
    for a new session), so ``turn - turn_at_open`` is how many turns this
    process has accepted. A tool can stamp the number on what it writes
    (Annealage Loom's ``Loom-Turn:`` commit trailer, which offsets it by
    ``turn_at_open``). ``http/ws.py`` counts a turn through ``begin_turn``,
    which is also where the notes ``queue_note`` collected are prefixed to
    it: one place in front of every backend, so a note reaches the model the
    same way whichever one runs. A viewer-only or external-agent run never
    accepts a turn, so its ``turn`` stays at the seed and its notes are never
    sent.

    ``attention`` publishes an ``Attention`` event through ``publish``, the
    app's event publisher (``None`` for a bus built outside ``create_app``,
    whose ``attention`` then does nothing). ``request_end_turn`` is how a tool
    result's ``end_turn`` (``tools.ok``) reaches the session, through
    ``end_turn_handler``, which ``create_app`` sets to the session's
    ``end_turn_after_tool`` when it has one.

    ``write_protected`` is the app's write-protected patterns
    (``create_app``), which ``launch.build_session`` hands to the session.

    ``url_prefix`` is the path the app is mounted under in a front door
    (``""`` at the root; ``url`` already ends in it), which
    ``launch.build_session`` hands to a Codex session for its bridge's
    ``--path``.
    """

    def __init__(
        self,
        registry: ViewerRegistry,
        *,
        url: str,
        url_prefix: str = "",
        timeout: float = CALL_TIMEOUT,
        publish: Optional[Any] = None,
        turn: int = 0,
    ):
        self._registry = registry
        self._url = url
        self.url_prefix = url_prefix
        self._timeout = timeout
        self._paused = False
        self._publish = publish
        # A resumed session's last turn (create_app reads it off the event
        # log), so the next one continues the numbering rather than reusing 1.
        self._turn = int(turn)
        self._turn_at_open = self._turn
        self._notes: list = []
        # The login of the human who sent the turn now starting (begin_turn).
        self.turn_by: Optional[str] = None
        self._turn_start_callbacks: list = []
        self.end_turn_handler: Optional[Any] = None
        # None: no app set it, so the session takes the product's patterns.
        self.write_protected: Optional[tuple] = None
        # Set by the first ``begin_turn``: the app's remote-server retry waits
        # on it, so a remote unreachable at startup is tried again the moment
        # the session is first used rather than only on its timer.
        self.first_turn = asyncio.Event()

    @property
    def url(self) -> str:
        """Where the page is served, with no token; see the class docstring."""
        return self._url

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def turn(self) -> int:
        """The conversation's current turn number (class docstring)."""
        return self._turn

    @property
    def turn_at_open(self) -> int:
        """``turn`` when this process opened the conversation: 0 for a new
        session, the history's last turn for a resumed one."""
        return self._turn_at_open

    def queue_note(self, text: str) -> None:
        """Prefix ``text`` to the next human turn sent to the model, as a
        marked system note (see ``begin_turn``). Notes queued before one turn
        all go with it, in order, and none goes twice."""
        text = str(text).strip()
        if text:
            self._notes.append(text)

    def on_turn_start(self, callback) -> None:
        """Run ``callback(turn)`` (synchronous) each time a human turn is
        accepted, with the new turn number, before that turn's notes are
        taken: a note the callback queues goes with this very turn. A
        callback that raises is logged and never stops the turn. ``turn_by``
        already names who sent it when the callback runs."""
        self._turn_start_callbacks.append(callback)

    def begin_turn(self, blocks: list, by: Optional[str] = None) -> list:
        """Count one human turn, run the ``on_turn_start`` callbacks, and
        return ``blocks`` with every queued note in front of them, as one text
        block of its own marked as coming from the product rather than the
        human. Clears the queue. ``http/ws.py`` calls it only once the turn is
        going to a ready session, with ``by``, the sender's tailnet login
        (``None`` for the browser token's holder), which is ``turn_by`` from
        then until the next turn: what a product attributes the turn's
        consequences to (``UserTurn.by`` is the same login, logged just after).

        A note can carry text the product did not write (a remote MCP
        server's own instructions, ``app.retry_remotes``), and a backend may
        join the blocks into one message (omp's prompt text), so both marker
        strings are neutralised inside the notes: nothing in a note can end
        the note early and have what follows read as the human's."""
        self._turn += 1
        self.turn_by = by
        self.first_turn.set()
        for callback in list(self._turn_start_callbacks):
            try:
                callback(self._turn)
            except Exception as exc:
                sys.stderr.write("warning: a turn-start callback failed: %r\n" % (exc,))
        notes, self._notes = self._notes, []
        if not notes:
            return blocks
        opening = _NOTE_OPEN % product.current().title
        body = "\n\n".join(_neutralise_markers(note) for note in notes)
        text = "%s\n\n%s\n\n%s" % (opening, body, _NOTE_CLOSE)
        return [{"type": "text", "text": text}] + list(blocks)

    def attention(self, title: str, body: str) -> None:
        """Tell the human the agent needs them: a browser notification and a
        flashing title on every open page, and a record in the event log."""
        if self._publish is not None:
            self._publish(Attention(title=str(title), body=str(body)))

    def request_end_turn(self) -> None:
        """A tool result asked to end the agent's turn (``tools.ok``'s
        ``end_turn``); called from inside that tool's handler."""
        if self.end_turn_handler is not None:
            self.end_turn_handler()

    def set_paused(self, paused: bool) -> bool:
        """Record the human's choice; return True if it changed anything.

        The return value is what stops a redundant ``pause_changed`` event
        going out for a click that set the flag to what it already was, which
        two tabs racing the same control produce routinely.
        """
        paused = bool(paused)
        if paused == self._paused:
            return False
        self._paused = paused
        return True

    async def call(
        self, method: str, params: Optional[dict] = None, *, timeout: Optional[float] = None
    ) -> Any:
        """Ask the primary viewer to run ``method`` and return its reply.

        Raises what ``ViewerRegistry.call`` raises, untranslated:
        ``NoViewerConnected``, ``ViewerGone``, ``CallError`` and
        ``asyncio.TimeoutError``. Turning those into tool errors is the
        registry's job (``tools.py``'s ``_wrap``), in one place, because the four
        of them mean four different things to a model and a wrapper here would
        have to flatten them to say anything at all.
        """
        return await self._registry.call(
            method,
            params or {},
            timeout=self._timeout if timeout is None else timeout,
            url=self._url,
        )
