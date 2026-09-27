"""EventLog: monotonic seq, a bounded in-memory ring, and an optional
append-only ``events.jsonl``.

Every server-to-browser event carries a seq number that never repeats and
never goes backwards for a given log (plan section 3.4). A reconnecting
browser sends the last seq it saw; this module replays what happened since
then from the in-memory ring, or from the file where the ring no longer
reaches (a long gap, or a restart), and never silently drops a difference
it cannot fill.

``EventLog`` takes an optional ``path``: with one it persists to
``events.jsonl`` beneath a session's own ``<state dir>/sessions/<sid>/``
directory, and without one it is a pure in-memory ring, which is what lets
it be exercised directly against a ``tmp_path`` with no session, served
directory, or agent involved at all.

This is not the exchange-file threat model ``files.py``'s
``safe_fixed_file`` and ``read_fixed_file`` defend: those
files sit in a served project directory an outside party can also write
into, so every open there re-validates identity against a race. An
``events.jsonl`` lives inside the product's own ``<state dir>/`` control
directory, which nothing but this process ever writes to, so there is no
name to race and no symlink substitution to guard against; the file is
opened once, by name, and held for the log's lifetime.

``read_records``, ``render_transcript`` and ``export_transcript``, below,
turn one session's ``events.jsonl`` back into a document a person can read
or archive. They live here rather than in a separate module because a
transcript is a projection of exactly the file ``EventLog`` writes, and
reading it back is the direct counterpart to appending to it.
``export_transcript`` writes through ``files.create_review_file``, which
gives a generated destination in a served project directory the same
containment ``files.create_image_file`` gives a model-supplied one: the
directory is created by this process on first use, but the served project
directory around it is not otherwise trusted, so a ``review/`` replaced by
a symlink is refused rather than followed.

A rendered transcript reads as the conversation, the human's side included.
``http/ws.py`` logs each turn the human sends as a ``user_turn`` event
(``session/base.py``'s ``UserTurn``) before the session sees it, carrying
the blocks exactly as the page sent them: what the human typed, and an
attached image as its path. The product's own notes, which
``ViewerBus.begin_turn`` puts in front of a turn on its way to the model, are
not part of it, so a transcript shows what the human said rather than
everything the model was sent.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .. import files, product, sessions

# Bounded history kept in memory for a reconnect to replay without a disk
# read. 500 events comfortably outlasts a normal reconnect gap (dropped
# WiFi, laptop lid, a phone locking) without growing without bound across
# a long session; a gap wider than that is read back from the file, or,
# for a log with no file, reported rather than guessed at, via
# Replay.truncated below.
RING_SIZE = 500


@dataclasses.dataclass(frozen=True)
class Replay:
    """The result of asking an ``EventLog`` to replay from a client's ``last_seq``.

    ``events`` is every ``(seq, wire_dict)`` pair newer than ``last_seq`` that
    this log can still produce, in seq order, ready to send straight back
    over the socket: the ring's own, preceded, when the client is further
    back than the ring reaches, by the file's records for the gap (see
    ``EventLog.replay``). ``truncated`` is True only when some of that
    history is unavailable: the log has no file (viewer-only mode's, or a
    test's) and the ring has already dropped events the client is missing,
    or the file could not be read back as far as the ring reaches. The
    caller must then say so rather than replay a partial history that looks
    complete. ``events`` is still populated in that case with whatever is
    available, since a gap in the oldest history is no reason to also
    withhold the newer events.

    ``through`` is the log's ``current_seq`` when the replay was taken:
    everything up to it is in ``events`` or reported missing, so it is where
    a following replay (``http/ws.py``'s catch-up) starts, whatever was
    appended while this one was being read or sent.
    """

    events: List[Tuple[int, dict]]
    truncated: bool
    through: int = 0


#: The event kinds whose ``turn`` numbers a conversation turn (``EventLog``
#: reads a resumed session's turns from these alone). ``user_turn`` is one,
#: so a turn the human sent that got no reply at all (the process was
#: killed first) still counts and is closed as interrupted on the next start.
_TURN_KINDS = frozenset(("user_turn", "text_delta", "tool_use", "turn_end"))


class EventLog:
    """Append-only event history for one session's lifetime.

    ``path``, when given, is opened once for appending (``O_APPEND`` makes
    each write atomic with respect to any other append to the same file
    descriptor, which matters only for the "one process, one writer"
    invariant this class itself maintains: nothing here defends against a
    second process writing the same path, that is ``<state dir>/lock``'s job,
    owned elsewhere). Re-opening the same path in a fresh ``EventLog``
    picks up numbering where the file left off, so a restarted process
    (or, in tests, a second ``EventLog`` standing in for one) never
    reissues a seq a client may already have.
    """

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path) if path is not None else None
        self._ring: collections.deque = collections.deque(maxlen=RING_SIZE)
        self._seq = 0
        self._fd: Optional[int] = None
        #: The highest turn number in the file, and every turn that has no
        #: ``turn_end`` there (a process killed mid-turn): where a resumed
        #: session's numbering continues and which turns ``app.create_app``
        #: closes, so a new turn never reuses the number of one the page
        #: already shows from history, nor does history look like it runs.
        self.last_turn = 0
        self.unfinished_turns: tuple = ()
        #: Every ``permission_request`` in the file with no
        #: ``permission_resolved`` after it: asked by a process that died
        #: before anyone answered, which ``app.create_app`` closes, so a page
        #: replaying the history is not shown a card nobody can answer.
        self.unresolved_requests: tuple = ()
        #: Called with each appended event's wire dict, after it is recorded:
        #: every event a page sees passes through here, whichever of the
        #: app's publishers sent it, so this is where the app's status
        #: summary (``AgentHolder``) watches them. One that raises is
        #: reported and never stops the append.
        self.observers: list = []
        if self._path is not None:
            self._seq = self._recover_seq()
            self._fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def _recover_seq(self) -> int:
        """The highest seq already written to ``self._path``, or 0, noting
        the turns on the way (``last_turn``, ``unfinished_turns``).

        A new ``EventLog`` for a path that already has content must not
        restart numbering at 0: a client that saw seq 50 before a restart
        and reconnects with ``last_seq=50`` would otherwise be replayed
        seq 1 through 50 again, duplicating events it has already
        rendered. A malformed trailing line, from a process killed mid
        write, is skipped rather than raising: the recovered seq only
        needs to be at least as high as anything a client could have
        already been sent, and a torn last line was never sent to one
        either, so skipping it costs nothing a client could have seen.
        """
        if not self._path.exists():
            return 0
        highest = 0
        started, ended = set(), set()
        requests: dict = {}  # request_id -> None, in the order they were asked
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                seq = record.get("seq")
                if isinstance(seq, int) and seq > highest:
                    highest = seq
                event = record.get("event")
                if not isinstance(event, dict):
                    continue
                kind = event.get("kind")
                request_id = event.get("request_id")
                if kind == "permission_request" and isinstance(request_id, str):
                    requests[request_id] = None
                elif kind == "permission_resolved":
                    requests.pop(request_id, None)
                if kind not in _TURN_KINDS:
                    continue
                turn = event.get("turn")
                if isinstance(turn, int) and not isinstance(turn, bool):
                    started.add(turn)
                    if kind == "turn_end":
                        ended.add(turn)
        self.last_turn = max(started, default=0)
        self.unfinished_turns = tuple(sorted(started - ended))
        self.unresolved_requests = tuple(requests)
        return highest

    @property
    def current_seq(self) -> int:
        """The seq of the most recently appended event, or 0 if none yet.

        This is what a ``hello`` frame reports as its own ``seq``: the
        point in the stream a freshly connected client is starting from,
        with nothing before it to replay.
        """
        return self._seq

    def append(self, event) -> int:
        """Assign the next seq to ``event`` and record it. Returns that seq.

        ``event`` is anything with a ``to_wire()`` method returning a
        JSON-able dict (``session.base.AgentEvent`` and its subclasses);
        this module has no other dependency on that class, so a caller
        with some other object shaped the same way works identically.
        """
        self._seq += 1
        seq = self._seq
        wire = event.to_wire()
        self._ring.append((seq, wire))
        if self._fd is not None:
            line = json.dumps({"seq": seq, "event": wire}) + "\n"
            os.write(self._fd, line.encode("utf-8"))
        for observe in self.observers:
            try:
                observe(wire)
            except Exception as exc:
                sys.stderr.write("warning: an event log observer failed: %r\n" % (exc,))
        return seq

    def replay(self, last_seq: Optional[int]) -> Replay:
        """Everything newer than ``last_seq``, in seq order, plus whether
        any of it is unavailable. ``last_seq=None`` means the client has no
        prior history at all (a first-ever connection, a reloaded page),
        treated the same as 0.

        The ring answers for what it holds. What it no longer holds, or
        never held (a restarted log starts with an empty ring: see
        ``_recover_seq``), comes from the file: the records after
        ``last_seq`` and older than the ring's oldest, with each run of one
        turn's ``text_delta`` events joined into one (``_coalesce``), so a
        long session's history is a few frames per turn rather than one per
        streamed chunk. The ring is copied and the file's bound fixed in one
        step (``_plan``), so the file's part ends exactly where the ring's
        begins whenever the file is read: an event appended meanwhile is in
        neither, and ``Replay.through`` says where the next replay starts.
        """
        plan = self._plan(last_seq)
        gap = self._read_between(plan.after, plan.before) if plan.before is not None else []
        return plan.replay(gap)

    async def replay_async(self, last_seq: Optional[int]) -> Replay:
        """``replay``, with the file read in a worker thread, so a long
        history does not stall every stream on the event loop while one tab
        connects. The file's part is bounded by the ring's oldest seq at the
        moment of the call, and every record below that is already whole in
        the file (``append`` writes the line before it returns), so appends
        while the thread reads change nothing it returns."""
        plan = self._plan(last_seq)
        gap = []
        if plan.before is not None:
            gap = await asyncio.to_thread(self._read_between, plan.after, plan.before)
        return plan.replay(gap)

    def _plan(self, last_seq: Optional[int]) -> "_ReplayPlan":
        """What a replay from ``last_seq`` is made of, fixed at one moment:
        nothing here waits, so the ring copy, the file's bound and
        ``through`` all describe the same point in the stream."""
        if last_seq is None or last_seq < 0 or last_seq > self._seq:
            # A last_seq higher than any seq this log has issued came from a
            # different run of the log (most likely a new session after a
            # restart, counting up from a lower seq): the client has none of
            # this log's events, so it is answered as a fresh one, rather
            # than with an empty reply that would read as "already caught
            # up". A negative one is no position at all.
            last_seq = 0
        ring = [(seq, wire) for seq, wire in self._ring if seq > last_seq]
        # The first seq the ring can answer for; with an empty ring, the next
        # one to be issued.
        oldest = self._ring[0][0] if self._ring else self._seq + 1
        # Past the ring's reach, everything between last_seq and oldest fell
        # off it (or was written by an earlier process). It is not silently
        # skippable: it comes from the file, and whatever the file cannot
        # supply is reported rather than handed over as a history with a
        # hole in it.
        before = oldest if last_seq < oldest - 1 else None
        return _ReplayPlan(after=last_seq, before=before, ring=ring, through=self._seq)

    def _read_between(self, after: int, before: int) -> List[Tuple[int, dict]]:
        """The file's records with ``after < seq < before``, coalesced; none
        for a log with no file.

        The file is in seq order (this class is its one writer), so reading
        stops at the first record the ring already holds. Every record below
        the ring's oldest is already in the file: ``append`` writes the line
        in the same call that puts the event in the ring.
        """
        if self._path is None:
            return []
        records = []
        for record in read_records(self._path):
            seq = record["seq"]
            if seq >= before:
                break
            if seq > after:
                records.append((seq, record["event"]))
        return _coalesce(records)

    def close(self) -> None:
        """Close the append-only file descriptor, if one is open.

        Idempotent, so a caller that closes explicitly and a shutdown
        path that closes again defensively do not double-close a
        descriptor number the OS may since have reissued to something
        else entirely.
        """
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@dataclasses.dataclass(frozen=True)
class _ReplayPlan:
    """One replay's parts, fixed by ``EventLog._plan``: the ring's events
    after ``after``, and, when ``before`` is set, the file's records with
    ``after < seq < before`` still to read."""

    after: int
    before: Optional[int]
    ring: List[Tuple[int, dict]]
    through: int

    def replay(self, gap: List[Tuple[int, dict]]) -> Replay:
        if self.before is None:
            return Replay(events=self.ring, truncated=False, through=self.through)
        reached = gap[-1][0] if gap else self.after
        return Replay(
            events=gap + self.ring, truncated=reached < self.before - 1, through=self.through
        )


def _coalesce(records: List[Tuple[int, dict]]) -> List[Tuple[int, dict]]:
    """``records`` with each run of consecutive same-turn ``text_delta``
    events joined into one, which carries the run's joined text and its last
    seq; everything else passes through untouched.

    The same merge ``viewers.ViewerRegistry``'s writer makes on a live
    connection, for the same reasons: the page appends a delta's text to its
    turn whatever size it is, so one joined delta renders exactly as the run
    did, and the last seq is the one a client may resume from, since the
    joined text already includes everything up to it.
    """
    out: List[Tuple[int, dict]] = []
    run: List[str] = []  # the texts of the delta run out[-1] stands for

    def close_run():
        if len(run) > 1:
            seq, event = out[-1]
            out[-1] = (seq, dict(event, text="".join(run)))
        run.clear()

    for seq, event in records:
        is_delta = event.get("kind") == "text_delta"
        if run and is_delta and event.get("turn") == out[-1][1].get("turn"):
            run.append(event.get("text", ""))
            out[-1] = (seq, out[-1][1])
            continue
        close_run()
        out.append((seq, event))
        if is_delta:
            run.append(event.get("text", ""))
    close_run()
    return out


# ---------------------------------------------------------------------------
# Transcript rendering and export: turning an events.jsonl back into a
# document, and writing that document under review/.
# ---------------------------------------------------------------------------


# The two shapes a transcript can be written in: prose meant to be read, or
# the underlying event records themselves, one JSON object per line, for a
# caller that wants the wire shapes rather than a rendering of them.
TRANSCRIPT_FORMATS = ("markdown", "jsonl")

# How much of a session's history a transcript shows. "text" is the
# conversation as a person would read it: what the human sent, what the
# agent said, and which tools it named. "full" adds everything else
# events.jsonl carries: each tool's input and result, each permission
# request's outcome, and the cost of each turn.
TRANSCRIPT_INCLUDE = ("text", "full")

# The least inclusive TRANSCRIPT_INCLUDE level at which each AgentEvent kind
# appears in a transcript. A kind absent from this table never appears in a
# transcript at any level: review_changed, a product's own events (Mesh's
# models_changed), pause_changed and viewer_primary describe the browser's
# view of a running server, not the conversation, and agent_status,
# session_reset and agent_error describe the session's own lifecycle rather
# than anything said or done within it.
_KIND_MIN_INCLUDE = {
    "user_turn": "text",
    "text_delta": "text",
    "tool_use": "text",
    "tool_result": "full",
    "permission_request": "full",
    "permission_resolved": "full",
    "turn_end": "full",
}


def _kind_kept(kind, include: str) -> bool:
    minimum = _KIND_MIN_INCLUDE.get(kind)
    if minimum is None:
        return False
    return TRANSCRIPT_INCLUDE.index(include) >= TRANSCRIPT_INCLUDE.index(minimum)


def read_records(path) -> Iterator[dict]:
    """Yield ``{"seq": int, "event": dict}`` from an ``events.jsonl``-shaped
    file, in file order.

    A missing file yields nothing rather than raising: a session with no
    events yet is an empty transcript, not an error, the same reading
    ``sessions._turn_stats`` gives an absent ``events.jsonl``. A line that
    fails to parse, or parses to something other than a seq/event pair, is
    skipped: it is the torn trailing line a process killed mid-write can
    leave (see ``EventLog._recover_seq``), and it was never delivered to a
    connected client either, so a transcript built from what a client could
    actually have seen loses nothing by skipping it too.
    """
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return
    with f:
        for line in f:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            seq = record.get("seq")
            event = record.get("event")
            if isinstance(seq, int) and isinstance(event, dict):
                yield {"seq": seq, "event": event}


def _render_jsonl(kept: list) -> str:
    return "".join(json.dumps(record) + "\n" for record in kept)


def _human_lines(blocks) -> List[str]:
    """A ``user_turn``'s blocks as a blockquote headed with who said it: the
    text as typed, line for line, and an attached image as its path."""
    body: List[str] = []
    for block in blocks if isinstance(blocks, list) else ():
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            body.extend(str(block.get("text", "")).splitlines() or [""])
        elif block.get("type") == "image_path":
            body.append("[image: %s]" % block.get("path", ""))
    return ["> **Human:**"] + [("> " + line) if line else ">" for line in body]


def _render_markdown(kept: list, include: str, session_id, project_dir, exported_at) -> str:
    lines = ["# %s transcript" % product.current().title]
    meta = []
    if session_id is not None:
        meta.append("- session: %s" % session_id)
    if project_dir is not None:
        meta.append("- project: %s" % project_dir)
    if exported_at is not None:
        meta.append("- exported: %s" % exported_at)
    if meta:
        lines.append("")
        lines.extend(meta)

    full = include == "full"
    open_turn = None
    text_buffer: List[str] = []

    def flush_text():
        if text_buffer:
            lines.append("")
            lines.append("".join(text_buffer))
            text_buffer.clear()

    for record in kept:
        event = record["event"]
        kind = event.get("kind")
        turn = event.get("turn")
        if turn is not None and turn != open_turn:
            flush_text()
            lines.append("")
            lines.append("## Turn %s" % turn)
            open_turn = turn

        if kind == "text_delta":
            text_buffer.append(event.get("text", ""))
            continue
        flush_text()

        if kind == "user_turn":
            lines.append("")
            lines.extend(_human_lines(event.get("blocks")))
        elif kind == "tool_use":
            lines.append("")
            lines.append("Tool call: %s" % event.get("name", ""))
            if full:
                lines.append("```json")
                lines.append(json.dumps(event.get("input", {}), indent=2, sort_keys=True))
                lines.append("```")
        elif kind == "tool_result":
            status = "error" if event.get("is_error") else "ok"
            lines.append("")
            lines.append("Tool result (%s):" % status)
            lines.append("```")
            lines.append(event.get("text", ""))
            lines.append("```")
        elif kind == "permission_request":
            lines.append("")
            lines.append("Permission requested: %s" % event.get("tool", ""))
        elif kind == "permission_resolved":
            lines.append("Permission resolved: %s" % event.get("outcome", ""))
        elif kind == "turn_end":
            cost = event.get("cost_usd")
            cost = cost if isinstance(cost, (int, float)) else 0.0
            lines.append("")
            lines.append(
                "Turn %s ended: %s, cost $%.4f" % (turn, event.get("stop_reason", ""), cost)
            )
            open_turn = None

    flush_text()
    return "\n".join(lines) + "\n"


def render_transcript(
    records,
    *,
    fmt: str = "markdown",
    include: str = "text",
    session_id: Optional[str] = None,
    project_dir: Optional[str] = None,
    exported_at: Optional[str] = None,
) -> str:
    """Render ``records`` (as ``read_records`` yields them) as one document.

    ``fmt="markdown"`` produces prose meant to be read: what the human sent
    (``user_turn``, quoted), the model's text, joined across the
    ``text_delta`` chunks that streamed it, and one line naming each tool
    call. At ``include="full"`` this adds each tool's input and result, each
    permission request's outcome, and each turn's cost; ``include="text"``
    leaves all of that out, showing only the conversation's words and which
    tools the model reached for.

    ``fmt="jsonl"`` instead emits the surviving records themselves, one JSON
    object per line, unmodified: the same ``TRANSCRIPT_INCLUDE`` rule decides
    which records survive, but nothing about a kept record's own fields is
    stripped, so a caller wanting the raw wire shapes gets them exactly as
    ``events.jsonl`` holds them.

    ``session_id``, ``project_dir`` and ``exported_at`` are folded into the
    markdown document's header when given, and have no effect on ``jsonl``
    output, which carries no header of its own.

    Raises ``ValueError`` if ``fmt`` is not in ``TRANSCRIPT_FORMATS`` or
    ``include`` is not in ``TRANSCRIPT_INCLUDE``.
    """
    if fmt not in TRANSCRIPT_FORMATS:
        raise ValueError("fmt must be one of %s, not %r" % (TRANSCRIPT_FORMATS, fmt))
    if include not in TRANSCRIPT_INCLUDE:
        raise ValueError("include must be one of %s, not %r" % (TRANSCRIPT_INCLUDE, include))

    kept = [r for r in records if _kind_kept(r.get("event", {}).get("kind"), include)]

    if fmt == "jsonl":
        return _render_jsonl(kept)
    return _render_markdown(kept, include, session_id, project_dir, exported_at)


# Extension a transcript is written with, keyed by TRANSCRIPT_FORMATS.
_TRANSCRIPT_EXTENSION = {"markdown": "md", "jsonl": "jsonl"}

# How many disambiguating suffixes export_transcript tries before giving up.
# Two exports of the same project in the same second, at the same fmt, is the
# only way to reach a second attempt at all, so this bounds a loop that in
# practice runs once.
EXPORT_NAME_ATTEMPTS = 20


def export_transcript(
    project_dir,
    session_id: str,
    *,
    fmt: str = "markdown",
    include: str = "text",
    now: Optional[float] = None,
) -> Path:
    """Render one session's ``events.jsonl`` and write it under ``review/``.

    Reads ``sessions.events_path(project_dir, session_id)`` through
    ``read_records``; a session with no events yet, or no directory at all,
    renders as an otherwise-empty transcript rather than failing.

    The file is named ``transcript-<stamp>.<md|jsonl>``, where ``<stamp>`` is
    ``now`` (epoch seconds, ``time.time()`` if not given) formatted as UTC
    ``YYYYMMDDTHHMMSSZ``: no colons, since a colon in a filename is hostile
    on Windows and inside an archive, and the compact form is still ISO 8601.
    A second export landing on the same stamp and the same ``fmt`` gets a
    ``-2``, ``-3``, ... suffix rather than overwriting the first.

    Writes through ``files.create_review_file``, so ``review/`` is created on
    first use and a symlinked or non-directory ``review/`` is refused. That
    refusal reaches the caller as ``OSError`` rather than a return value:
    this function has nothing sensible to do but write the file it was
    asked for, so there is no partial result to hand back instead.
    """
    if fmt not in _TRANSCRIPT_EXTENSION:
        raise ValueError("fmt must be one of %s, not %r" % (TRANSCRIPT_FORMATS, fmt))
    serve_dir = files.resolve_serve_dir(project_dir)
    records = list(read_records(sessions.events_path(serve_dir, session_id)))
    when = time.time() if now is None else now
    text = render_transcript(
        records,
        fmt=fmt,
        include=include,
        session_id=session_id,
        project_dir=str(serve_dir),
        exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when)),
    )
    data = text.encode("utf-8")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(when))
    extension = _TRANSCRIPT_EXTENSION[fmt]
    base = "transcript-%s" % stamp

    name = None
    for attempt in range(1, EXPORT_NAME_ATTEMPTS + 1):
        name = (
            "%s.%s" % (base, extension) if attempt == 1 else "%s-%d.%s" % (base, attempt, extension)
        )
        try:
            created = files.create_review_file(serve_dir, name)
        except FileExistsError:
            continue
        if created is None:
            raise OSError(
                "review/ under %s is not a directory this process can write into" % serve_dir
            )
        fd, target = created
        try:
            written = 0
            while written < len(data):
                written += os.write(fd, data[written:])
        finally:
            os.close(fd)
        return target
    raise FileExistsError(name)
