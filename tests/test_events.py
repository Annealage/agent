"""Tests for ``session/events.py``: seq numbering, the in-memory ring, replay,
and the append-only file.

Nothing here needs asyncio; ``EventLog`` is plain, synchronous bookkeeping.
``TextDelta`` (from ``session/base.py``) stands in for "any event with a
``to_wire()``", since ``EventLog.append`` has no dependency of its own on
which concrete event type is used.
"""

import json

import pytest

from annealage_agent.session import events as events_module
from annealage_agent.session.base import TextDelta, ToolUse, TurnEnd, UserTurn
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
    """With no file behind it, events between ``last_seq`` and what the ring
    still holds are gone; the caller (``http/ws.py``) must be told rather
    than handed a history with a silent hole cut out of the front of it.
    Whatever the ring does still hold is returned alongside the truncation
    flag, since a gap in the oldest history is no reason to also withhold
    newer events that are available."""
    log = EventLog()
    for i in range(RING_SIZE + 50):
        log.append(_evt(str(i)))

    replay = log.replay(0)
    assert replay.truncated is True
    assert len(replay.events) == RING_SIZE
    assert replay.events != []


def test_a_restarted_log_replays_its_history_from_the_file(tmp_path):
    """A restarted ``EventLog`` recovers ``current_seq`` from the file but
    starts its in-memory ring empty. A client reconnecting right after that
    restart, with a ``last_seq`` below the recovered seq, is sent what it is
    missing out of the file, not an empty reply that reads as "caught up"."""
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    for i in range(3):
        log.append(_evt(str(i)))
    log.close()

    restarted = EventLog(str(path))
    assert restarted.current_seq == 3

    replay = restarted.replay(1)
    assert replay.truncated is False
    assert replay.events == [(3, {"kind": "text_delta", "turn": 1, "text": "12"})]


def test_a_gap_the_file_cannot_fill_is_reported(tmp_path):
    """The history behind the ring is gone (the file was removed under a
    running log): the gap is reported, never handed over as a complete
    history with a hole in it."""
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    for i in range(3):
        log.append(_evt(str(i)))
    log.close()
    restarted = EventLog(str(path))
    path.unlink()

    assert restarted.replay(0).truncated is True


def test_the_file_and_the_ring_join_with_no_seq_skipped_or_repeated(tmp_path, monkeypatch):
    """Past the ring, the file supplies exactly the seqs the ring no longer
    holds; each run of one turn's deltas from the file comes back as one
    event with the run's last seq, and together they render the same text."""
    monkeypatch.setattr(events_module, "RING_SIZE", 3)
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    streamed = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    for text in streamed:
        log.append(_evt(text))

    replay = log.replay(0)
    assert replay.truncated is False
    assert [seq for seq, _ in replay.events] == [7, 8, 9, 10]
    assert "".join(wire["text"] for _, wire in replay.events) == "".join(streamed)
    # A client that saw the joined event resumes after its seq, not before.
    assert [seq for seq, _ in log.replay(7).events] == [8, 9, 10]
    log.close()


@pytest.mark.asyncio
async def test_appends_while_the_file_is_read_are_neither_skipped_nor_repeated(
    tmp_path, monkeypatch
):
    """The file's part of a replay is read in a worker thread while the
    stream goes on: events appended meanwhile (enough to push more off the
    ring) come in the next replay, from ``through``, each exactly once."""
    monkeypatch.setattr(events_module, "RING_SIZE", 3)
    log = EventLog(str(tmp_path / "events.jsonl"))
    for turn in range(1, 11):
        log.append(_evt(str(turn), turn=turn))
    read_between = log._read_between

    def appending_meanwhile(after, before):
        for turn in range(11, 16):
            log.append(_evt(str(turn), turn=turn))
        return read_between(after, before)

    monkeypatch.setattr(log, "_read_between", appending_meanwhile)
    first = await log.replay_async(0)
    monkeypatch.setattr(log, "_read_between", read_between)
    second = await log.replay_async(first.through)

    assert first.through == 10
    seqs = [seq for seq, _ in first.events + second.events]
    assert seqs == list(range(1, 16))
    assert not first.truncated and not second.truncated
    log.close()


def test_coalescing_keeps_every_other_event_and_every_turn_s_own_text(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    for event in (
        UserTurn(turn=1, blocks=[{"type": "text", "text": "hi"}], client_id="c1"),
        _evt("Let me ", turn=1),
        _evt("look.", turn=1),
        ToolUse(turn=1, tool_use_id="t1", name="read_file", input={}),
        _evt("Done.", turn=1),
        TurnEnd(turn=1, stop_reason="end", cost_usd=0.0),
        _evt("Next ", turn=2),
        _evt("one.", turn=2),
    ):
        log.append(event)
    log.close()

    replayed = EventLog(str(path)).replay(0).events
    assert [(seq, wire["kind"]) for seq, wire in replayed] == [
        (1, "user_turn"),
        (3, "text_delta"),
        (4, "tool_use"),
        (5, "text_delta"),
        (6, "turn_end"),
        (8, "text_delta"),
    ]
    assert [wire["text"] for _, wire in replayed if wire["kind"] == "text_delta"] == [
        "Let me look.",
        "Done.",
        "Next one.",
    ]
    assert replayed[0][1]["blocks"] == [{"type": "text", "text": "hi"}]


def test_a_human_turn_with_no_reply_counts_and_is_unfinished(tmp_path):
    """The process died before the agent said anything: the turn still
    counts, so the next one is numbered after it, and it is unfinished, so
    the next start closes it (``app.create_app``)."""
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append(UserTurn(turn=1, blocks=[{"type": "text", "text": "a"}]))
    log.append(TurnEnd(turn=1, stop_reason="end", cost_usd=0.0))
    log.append(UserTurn(turn=2, blocks=[{"type": "text", "text": "b"}]))
    log.close()

    restarted = EventLog(str(path))
    assert restarted.last_turn == 2
    assert restarted.unfinished_turns == (2,)
    restarted.close()


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
