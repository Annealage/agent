"""Tests for ``session/events.py``: seq numbering, the in-memory ring, replay,
and the append-only file.

Nothing here needs asyncio; ``EventLog`` is plain, synchronous bookkeeping.
``TextDelta`` (from ``session/base.py``) stands in for "any event with a
``to_wire()``", since ``EventLog.append`` has no dependency of its own on
which concrete event type is used.
"""

import json

from annealage_agent.session.base import TextDelta
from annealage_agent.session.events import RING_SIZE, EventLog


def _evt(text="x", turn=1):
    return TextDelta(turn=turn, text=text)


# ---------------------------------------------------------------------------
# seq numbering
# ---------------------------------------------------------------------------


def test_current_seq_starts_at_zero_with_nothing_appended():
    log = EventLog()
    assert log.current_seq == 0


def test_append_assigns_increasing_seq_starting_at_one():
    log = EventLog()
    assert log.append(_evt("a")) == 1
    assert log.append(_evt("b")) == 2
    assert log.append(_evt("c")) == 3
    assert log.current_seq == 3


def test_seq_is_monotonic_across_a_restart_of_the_log_object(tmp_path):
    """A fresh ``EventLog`` opened against a path that already has content
    must continue numbering from where the file left off: a client that
    saw seq 3 before a restart and reconnects with ``last_seq=3`` must
    never be replayed seq 1 through 3 again."""
    path = tmp_path / "events.jsonl"
    first = EventLog(str(path))
    for i in range(3):
        first.append(_evt(str(i)))
    first.close()

    second = EventLog(str(path))
    assert second.current_seq == 3
    assert second.append(_evt("d")) == 4
    assert second.current_seq == 4
    second.close()


def test_recover_seq_skips_a_malformed_trailing_line(tmp_path):
    """A process killed mid-write can leave a torn final line. That line
    was never delivered to any client either, so recovery only needs to
    reach the highest seq a complete line actually recorded."""
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append(_evt("a"))
    log.append(_evt("b"))
    log.close()
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"seq": 3, "event": {"kind": "text_delta"')  # torn, no closing brace

    restarted = EventLog(str(path))
    assert restarted.current_seq == 2
    assert restarted.append(_evt("c")) == 3


# ---------------------------------------------------------------------------
# ring bound
# ---------------------------------------------------------------------------


def test_ring_is_bounded_at_ring_size():
    log = EventLog()
    for i in range(RING_SIZE + 100):
        log.append(_evt(str(i)))
    assert log.current_seq == RING_SIZE + 100

    replay = log.replay(0)
    assert len(replay.events) == RING_SIZE
    assert replay.truncated is True
    first_seq, _ = replay.events[0]
    last_seq, _ = replay.events[-1]
    assert first_seq == 101
    assert last_seq == RING_SIZE + 100


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_from_a_last_seq_inside_the_ring_returns_only_the_newer_events():
    log = EventLog()
    for i in range(10):
        log.append(_evt(str(i)))

    replay = log.replay(5)
    assert replay.truncated is False
    assert [seq for seq, _ in replay.events] == [6, 7, 8, 9, 10]


def test_replay_with_no_last_seq_returns_everything_the_ring_holds():
    log = EventLog()
    for i in range(3):
        log.append(_evt(str(i)))
    replay = log.replay(None)
    assert replay.truncated is False
    assert [seq for seq, _ in replay.events] == [1, 2, 3]


def test_replay_at_the_ring_boundary_is_not_reported_truncated():
    """``last_seq`` naming exactly the event before the oldest one the ring
    still holds is a clean handoff, not a gap: everything the client is
    missing is right there in the ring."""
    log = EventLog()
    for i in range(RING_SIZE + 1):
        log.append(_evt(str(i)))
    assert log.replay(1).truncated is False
    assert [seq for seq, _ in log.replay(1).events] == list(range(2, RING_SIZE + 2))


def test_replay_from_a_last_seq_older_than_the_ring_reports_truncation():
    """Events between ``last_seq`` and what the ring still holds fell off
    the ring before this replay; the caller (``http/ws.py``) must be told
    the browser needs to page the rest over HTTP rather than being handed
    a history with a silent hole cut out of the front of it. Whatever the
    ring does still hold is returned alongside the truncation flag, since
    a gap in the oldest history is no reason to also withhold newer
    events that are available."""
    log = EventLog()
    for i in range(RING_SIZE + 50):
        log.append(_evt(str(i)))

    replay = log.replay(0)
    assert replay.truncated is True
    assert len(replay.events) == RING_SIZE
    assert replay.events != []


def test_replay_immediately_after_a_restart_must_report_truncation_not_a_false_all_clear(
    tmp_path,
):
    """A restarted ``EventLog`` recovers ``current_seq`` from the file but
    starts its in-memory ring empty (``__init__`` never repopulates the
    ring from disk). A client reconnecting right after that restart, with
    a ``last_seq`` below the recovered seq, must be told ``truncated``:
    every event between its ``last_seq`` and the recovered seq exists
    only in the append-only file, not in the empty ring, so an empty ring
    on its own must never be read as "nothing to report"."""
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    for i in range(3):
        log.append(_evt(str(i)))
    log.close()

    restarted = EventLog(str(path))
    assert restarted.current_seq == 3

    replay = restarted.replay(1)
    assert replay.truncated is True


# ---------------------------------------------------------------------------
# append-only file
# ---------------------------------------------------------------------------


def test_append_writes_one_json_line_per_event_when_a_path_is_given(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append(_evt("hello", turn=7))
    log.append(_evt("world", turn=7))
    log.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first == {"seq": 1, "event": {"kind": "text_delta", "turn": 7, "text": "hello"}}
    assert second == {"seq": 2, "event": {"kind": "text_delta", "turn": 7, "text": "world"}}


def test_no_path_means_no_file_is_created(tmp_path):
    log = EventLog()
    log.append(_evt("a"))
    assert list(tmp_path.iterdir()) == []
    log.close()


def test_close_is_idempotent(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append(_evt("a"))
    log.close()
    log.close()
