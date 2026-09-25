"""The on-disk shape of ``<state dir>/sessions/`` and ``<state dir>/state.json``,
and the lookups ``-c``/``-r`` (plan section 3.4) resolve a session id through.

``<state dir>`` is the installed product's ``state_dirname`` under the
project directory (Mesh: ``.mesh/``). One session is one directory,
``<state dir>/sessions/<sid>/``, holding
``events.jsonl`` (opened and written by ``session/events.py``'s
``EventLog``; this module only reads it back) and ``meta.json`` (owned by
this module: the handful of facts ``events.jsonl`` cannot answer on its
own). ``<sid>`` is the product's own identifier, generated once by
``new_session_id`` and used as the directory name, the id ``-r`` lists and
accepts, and the key ``<state dir>/state.json`` remembers for ``-c``. It is
never the same value as ``sdk_session_id``, which is the underlying
``claude`` CLI's own conversation id: the two are recorded side by side in
``meta.json`` because a session exists (and has a directory, an event
log, a place in ``-r``'s listing) before the SDK client has connected and
reported an id of its own, and because ``continue_conversation=True`` is
scoped to the CLI's notion of cwd (plan section 3.4) while a resume needs
an id that stays meaningful regardless of where this process happens to
be run from.

``meta.json``'s ``project_key`` is a defence against a state directory
copied or moved out of the project tree it was created for: it is computed
once, from the resolved project directory, at session creation, and every
lookup in this module recomputes the caller's own current project key and
refuses to resume or list a session recorded against a different one.
Resuming the wrong conversation under an id that looks right is worse than
reporting it as unknown.

Turn count and cost are derived from ``events.jsonl`` rather than kept a
second time in ``meta.json``: ``EventLog`` is already the one writer of
that file, and a count kept separately in ``meta.json`` would need someone
to update it in step with every turn, which is exactly the two-writer
hazard the front-end store contract elsewhere in this project exists to
avoid. ``first_user_text`` has no home in ``events.jsonl`` (nothing in
``session/base.py``'s ``AgentEvent`` set records what the human typed) and
is the one field this module keeps in ``meta.json``, written at most once
per session by ``record_first_user_text``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import List, Optional

from . import product

SESSIONS_DIRNAME = "sessions"
STATE_FILENAME = "state.json"
META_FILENAME = "meta.json"
EVENTS_FILENAME = "events.jsonl"

# A snippet this long is enough to recognise a conversation in an ``-r``
# listing without meta.json carrying an unbounded amount of what may be
# sensitive project detail.
FIRST_USER_TEXT_LIMIT = 200


def state_dir(project_dir) -> Path:
    """The product's per-project state directory: sessions, ``state.json``,
    the lock, ``permissions.toml`` and the project settings file."""
    return Path(project_dir) / product.current().state_dirname


def sessions_dir(project_dir) -> Path:
    return state_dir(project_dir) / SESSIONS_DIRNAME


def session_dir(project_dir, sid: str) -> Path:
    return sessions_dir(project_dir) / sid


def events_path(project_dir, sid: str) -> Path:
    return session_dir(project_dir, sid) / EVENTS_FILENAME


def meta_path(project_dir, sid: str) -> Path:
    return session_dir(project_dir, sid) / META_FILENAME


def state_path(project_dir) -> Path:
    return state_dir(project_dir) / STATE_FILENAME


def project_key_for_directory(directory) -> str:
    """A stable identifier for a resolved project directory.

    Two different relative arguments naming the same directory resolve to
    the same key, which is what lets a session's recorded key be compared
    against a fresh ``Path.resolve()`` of whatever directory this process
    is serving right now, regardless of how that directory was named on
    the command line.
    """
    resolved = str(Path(directory).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def new_session_id() -> str:
    """A fresh session id: sortable by start time, and unique even against
    another session started in the same second.

    The timestamp prefix is what makes the mtime-fallback scan in
    ``resolve_continue`` and the newest-first ordering in ``list_sessions``
    agree with the id itself rather than only with the filesystem's own
    mtime, which a copy or a backup restore can disturb.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return "%s-%s" % (stamp, secrets.token_hex(4))


@dataclasses.dataclass(frozen=True)
class SessionInfo:
    """Everything ``-r``'s listing, and a resolved ``-r SID``, need about
    one session. ``sdk_session_id`` is ``None`` for a session whose SDK
    client never connected (a crash before the first turn, or a viewer-only
    run that never had one); such a session is still resumable as a product
    session id, just not as a conversation the SDK can pick back up, and
    the caller resolving it is the one that decides what to do about that.
    """

    session_id: str
    sdk_session_id: Optional[str]
    started_at: Optional[str]
    turn_count: int
    cost_usd: float
    first_user_text: Optional[str]


def _read_json(path: Path) -> Optional[dict]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _write_json_atomic(path: Path, data: dict) -> None:
    """Replace ``path`` with ``data``, never leaving a partially written file
    for a concurrent reader to observe. ``os.replace`` is atomic on both
    POSIX and Windows when source and destination share a filesystem, which
    a sibling temp file in the same directory guarantees."""
    tmp = path.with_suffix(path.suffix + ".tmp-%d" % os.getpid())
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _turn_stats(path: Path) -> tuple:
    """``(turn_count, cost_usd)`` derived from an ``events.jsonl``.

    Counts and sums ``turn_end`` events only; every other kind is
    irrelevant to either figure. A line that fails to parse is skipped
    rather than raising, matching ``EventLog._recover_seq``'s tolerance of
    a torn trailing line from a process killed mid write: such a line was
    never delivered to a client either, so skipping it costs nothing a
    reader of this history could have already seen. Missing file reads as
    no turns yet, not an error.
    """
    turn_count = 0
    cost_usd = 0.0
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        return turn_count, cost_usd
    with f:
        for line in f:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            event = record.get("event")
            if isinstance(event, dict) and event.get("kind") == "turn_end":
                turn_count += 1
                cost = event.get("cost_usd")
                if isinstance(cost, (int, float)):
                    cost_usd += cost
    return turn_count, cost_usd


def create_session(project_dir, sid: Optional[str] = None) -> str:
    """Create ``<state dir>/sessions/<sid>/`` and its initial ``meta.json``.

    ``sid`` is generated if not given. ``events.jsonl`` is not created
    here: ``session/events.py``'s ``EventLog`` opens it with ``O_CREAT``
    on first use, and creating an empty one in advance would only be a
    second place that has to agree with the first about the file's name.
    """
    sid = sid or new_session_id()
    directory = session_dir(project_dir, sid)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": sid,
        "sdk_session_id": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_key": project_key_for_directory(project_dir),
        "first_user_text": None,
    }
    _write_json_atomic(meta_path(project_dir, sid), meta)
    return sid


def set_sdk_session_id(project_dir, sid: str, sdk_session_id: str) -> None:
    """Record the underlying SDK conversation id once the client reports
    one, so a later ``-c``/``-r`` in this project can pass it as
    ``resume=``. Called by whatever constructs the real session
    (``session/sdk.py``, via the wiring in ``app.py``); this module never
    fabricates the value itself (plan section 3.4: "a resolved id stays
    inspectable on disk").
    """
    path = meta_path(project_dir, sid)
    meta = _read_json(path) or {"session_id": sid}
    meta["sdk_session_id"] = sdk_session_id
    _write_json_atomic(path, meta)


def record_first_user_text(project_dir, sid: str, text: str) -> None:
    """Record a snippet of the first inbound turn, once, for ``-r``'s
    listing. A call after the first is a no-op: the field exists to answer
    "which conversation was this", which the first turn already answers as
    well as any later one, and overwriting it would make a long-running
    session's listing entry describe whatever was typed most recently
    instead of what it actually opened with.
    """
    path = meta_path(project_dir, sid)
    meta = _read_json(path) or {"session_id": sid}
    if meta.get("first_user_text"):
        return
    meta["first_user_text"] = text[:FIRST_USER_TEXT_LIMIT]
    _write_json_atomic(path, meta)


def get_session_info(project_dir, sid: str) -> Optional[SessionInfo]:
    """Look up one session by id, scoped to ``project_dir``.

    Returns ``None`` when ``sid`` names no session directory at all, when
    its ``meta.json`` is missing or will not parse, or when its recorded
    ``project_key`` does not match ``project_dir`` resolved right now: all
    three read as "unknown to this project" to a caller resolving ``-r
    SID``, which is deliberately the same outcome as a plainly nonexistent
    id rather than a more specific error that would confirm to a caller
    which of the three is true.
    """
    directory = session_dir(project_dir, sid)
    if not directory.is_dir():
        return None
    meta = _read_json(meta_path(project_dir, sid))
    if meta is None:
        return None
    if meta.get("project_key") != project_key_for_directory(project_dir):
        return None
    turn_count, cost_usd = _turn_stats(events_path(project_dir, sid))
    return SessionInfo(
        session_id=sid,
        sdk_session_id=meta.get("sdk_session_id"),
        started_at=meta.get("started_at"),
        turn_count=turn_count,
        cost_usd=cost_usd,
        first_user_text=meta.get("first_user_text"),
    )


def list_sessions(project_dir) -> List[SessionInfo]:
    """Every session belonging to ``project_dir``, newest first.

    "Belonging to" is ``get_session_info``'s project-key check: a foreign
    session directory (copied in from elsewhere, or left behind by a
    project that used to live at this path before being replaced) is
    silently excluded rather than listed as if it were reachable, which it
    is not, per that function's contract.
    """
    root = sessions_dir(project_dir)
    if not root.is_dir():
        return []
    infos = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        info = get_session_info(project_dir, entry.name)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda info: info.session_id, reverse=True)
    return infos


def record_last_session(project_dir, sid: str) -> None:
    """Remember ``sid`` as the one ``-c`` should resolve to next.

    Merges into ``state.json`` rather than replacing it, since that file
    also holds viewer preferences (plan section 3.8) this module has no
    reason to know the shape of and every reason not to clobber.
    """
    path = state_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _read_json(path) or {}
    state["last_session"] = sid
    _write_json_atomic(path, state)


def resolve_continue(project_dir) -> Optional[str]:
    """The session id ``-c`` should resume, or ``None`` if this project has
    none.

    ``state.json``'s ``last_session`` is trusted first, but only if
    ``get_session_info`` still recognises it (it may have been for a
    session directory since removed, or recorded under a stale
    ``project_key`` from before a project was moved). Failing that, every
    session directory belonging to this project is scanned and the one
    with the latest ``started_at`` wins; ``started_at`` rather than mtime,
    because a copy, a backup restore or a slow filesystem can all disturb
    mtime ordering without disturbing when a session actually began.
    """
    state = _read_json(state_path(project_dir)) or {}
    last = state.get("last_session")
    if isinstance(last, str) and get_session_info(project_dir, last) is not None:
        return last

    candidates = list_sessions(project_dir)
    if not candidates:
        return None
    candidates.sort(key=lambda info: info.started_at or "", reverse=True)
    return candidates[0].session_id
