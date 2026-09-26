"""EventLog: monotonic seq, a bounded in-memory ring, and an optional
append-only ``events.jsonl``.

Every server-to-browser event carries a seq number that never repeats and
never goes backwards for a given log (plan section 3.4). A reconnecting
browser sends the last seq it saw; this module decides how much of what
happened since then can be replayed straight back over the socket versus
having to be paged from disk, and never silently drops the difference.

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

A rendered transcript never shows what a human typed. Nothing in
``session/base.py``'s ``AgentEvent`` set records the content of an inbound
turn: ``http/ws.py`` hands a turn frame's ``blocks`` straight to
``AgentSession.submit_turn`` without logging them anywhere first, so the
events this module reads back are exactly the ones a session produced on
its own. A transcript exported from here is a record of what the agent
said and did, not a transcript of the conversation the way a human would
read one back.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .. import files, product, sessions

# Bounded history kept in memory for a reconnect to replay without a disk
# read. 500 events comfortably outlasts a normal reconnect gap (dropped
# WiFi, laptop lid, a phone locking) without growing without bound across
# a long session; a gap wider than that is reported rather than guessed
# at, via Replay.truncated below.
RING_SIZE = 500


@dataclasses.dataclass(frozen=True)
class Replay:
    """The result of asking an ``EventLog`` to replay from a client's ``last_seq``.

    ``events`` is every ring-held ``(seq, wire_dict)`` pair newer than
    ``last_seq``, in seq order, ready to send straight back over the
    socket. ``truncated`` is True when ``last_seq`` predates what the ring
    still holds: some events between ``last_seq`` and the oldest one
    returned exist only in the append-only file (or, if no path was
    given, nowhere at all), and the caller must say so rather than
    replaying a partial history that looks complete. ``events`` is still
    populated in that case with whatever the ring does have, since
    telling the browser about a gap in its oldest history is no reason to
    also withhold the newer events that are available. ``truncated`` is
    also True when ``last_seq`` is below the log's own current position
    but the ring is empty (a restart recovers the seq counter but not the
    ring's contents), and when ``last_seq`` is higher than any seq this
    log has ever issued, which can only mean it belongs to a different
    run of the log than the one now being asked. In both cases real
    events exist between ``last_seq`` and the log's current position that
    this reply cannot hand back, which is a gap regardless of the ring's
    own state.
    """

    events: List[Tuple[int, dict]]
    truncated: bool


#: The event kinds whose ``turn`` numbers a conversation turn (``EventLog``
#: reads a resumed session's turns from these alone).
_TURN_KINDS = frozenset(("text_delta", "tool_use", "turn_end"))


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
                if not isinstance(event, dict) or event.get("kind") not in _TURN_KINDS:
                    continue
                turn = event.get("turn")
                if isinstance(turn, int) and not isinstance(turn, bool):
                    started.add(turn)
                    if event["kind"] == "turn_end":
                        ended.add(turn)
        self.last_turn = max(started, default=0)
        self.unfinished_turns = tuple(sorted(started - ended))
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
        return seq

    def replay(self, last_seq: Optional[int]) -> Replay:
        """Everything the ring holds newer than ``last_seq``, plus whether
        that leaves a gap. ``last_seq=None`` means the client has no prior
        history at all (a first-ever connection), treated the same as 0.
        """
        if last_seq is None:
            last_seq = 0
        if last_seq > self._seq:
            # last_seq is higher than any seq this log has ever issued.
            # This log cannot be the one that produced it, most likely
            # because the process restarted and this is a fresh log
            # counting up from a lower recovered seq: nothing here can
            # answer for what came after a point it never reached, so the
            # whole ring is handed back rather than an empty reply that
            # would read as "already caught up".
            return Replay(events=list(self._ring), truncated=True)
        if not self._ring:
            # An empty ring says nothing about whether last_seq is caught
            # up: a restarted, path-backed log recovers self._seq from
            # the file but never repopulates the ring (see
            # _recover_seq), so last_seq below self._seq here means real
            # events exist only in the file, not that nothing has
            # happened since last_seq.
            return Replay(events=[], truncated=last_seq < self._seq)
        oldest = self._ring[0][0]
        # A gap exists whenever the ring's oldest kept event is not the
        # very next one the client is expecting: everything between
        # last_seq and oldest fell off the ring before this replay and is
        # not silently skippable, so the caller is told rather than
        # handed a history with a hole already cut out of it.
        truncated = last_seq < oldest - 1
        events = [(seq, wire) for seq, wire in self._ring if seq > last_seq]
        return Replay(events=events, truncated=truncated)

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


# ---------------------------------------------------------------------------
# Transcript rendering and export: turning an events.jsonl back into a
# document, and writing that document under review/.
# ---------------------------------------------------------------------------


# The two shapes a transcript can be written in: prose meant to be read, or
# the underlying event records themselves, one JSON object per line, for a
# caller that wants the wire shapes rather than a rendering of them.
TRANSCRIPT_FORMATS = ("markdown", "jsonl")

# How much of a session's history a transcript shows. "text" is the
# conversation as a person would read it: what the agent said, and which
# tools it named. "full" adds everything else events.jsonl carries: each
# tool's input and result, each permission request's outcome, and the cost
# of each turn.
TRANSCRIPT_INCLUDE = ("text", "full")

# The least inclusive TRANSCRIPT_INCLUDE level at which each AgentEvent kind
# appears in a transcript. A kind absent from this table never appears in a
# transcript at any level: review_changed, a product's own events (Mesh's
# models_changed), pause_changed and viewer_primary describe the browser's
# view of a running server, not the conversation, and agent_status,
# session_reset and agent_error describe the session's own lifecycle rather
# than anything said or done within it.
_KIND_MIN_INCLUDE = {
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

        if kind == "tool_use":
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

    ``fmt="markdown"`` produces prose meant to be read: the model's text,
    joined across the ``text_delta`` chunks that streamed it, and one line
    naming each tool call. At ``include="full"`` this adds each tool's input
    and result, each permission request's outcome, and each turn's cost;
    ``include="text"`` leaves all of that out, showing only the model's own
    words and which tools it reached for.

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
