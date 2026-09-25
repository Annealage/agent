"""Tests for transcript export: ``session/events.py``'s ``read_records``,
``render_transcript`` and ``export_transcript``, and ``files.create_review_file``.

``render_transcript`` is exercised directly against hand-built
``{"seq": ..., "event": ...}`` records, the same shape ``read_records``
yields, so its rendering rules are pinned independently of any file on disk.
``export_transcript`` and ``read_records`` are exercised against a real
``events.jsonl`` under a session directory built with ``sessions``, and
``create_review_file`` is exercised directly, the way ``create_image_file``
is.
"""

import json
import os
import time

import pytest

from annealage_agent import files, sessions
from annealage_agent.session import events
from annealage_agent.session.base import (
    AgentStatus,
    PermissionRequest,
    PermissionResolved,
    TextDelta,
    ToolResult,
    ToolUse,
    TurnEnd,
)


def _rec(seq, event):
    return {"seq": seq, "event": event.to_wire()}


def _session_with_events(project_dir, *event_objs):
    """A fresh session under ``project_dir`` with ``event_objs`` appended to
    its ``events.jsonl``, in order."""
    sid = sessions.create_session(project_dir)
    log = events.EventLog(str(sessions.events_path(project_dir, sid)))
    for event in event_objs:
        log.append(event)
    log.close()
    return sid


# --- read_records -----------------------------------------------------------


def test_read_records_yields_seq_and_event_in_file_order(tmp_path):
    path = tmp_path / "events.jsonl"
    log = events.EventLog(str(path))
    log.append(TextDelta(turn=1, text="hello"))
    log.append(ToolUse(turn=1, tool_use_id="t1", name="read_file", input={"path": "a"}))
    log.close()

    records = list(events.read_records(path))
    assert [r["seq"] for r in records] == [1, 2]
    assert records[0]["event"]["kind"] == "text_delta"
    assert records[0]["event"]["text"] == "hello"
    assert records[1]["event"]["name"] == "read_file"


def test_read_records_on_a_missing_file_yields_nothing(tmp_path):
    assert list(events.read_records(tmp_path / "absent.jsonl")) == []


def test_read_records_skips_a_torn_trailing_line(tmp_path):
    """A process killed mid-write can leave a truncated final line. That
    line was never delivered to any client either, so a transcript loses
    nothing a client could have seen by skipping it too."""
    path = tmp_path / "events.jsonl"
    good = json.dumps({"seq": 1, "event": TextDelta(turn=1, text="a").to_wire()})
    torn = '{"seq": 2, "event": {"kind": "text_delta", "tex'
    path.write_text(good + "\n" + torn, encoding="utf-8")

    records = list(events.read_records(path))
    assert [r["seq"] for r in records] == [1]


def test_read_records_skips_lines_missing_seq_or_event(tmp_path):
    path = tmp_path / "events.jsonl"
    lines = [
        json.dumps({"event": {"kind": "text_delta"}}),  # no seq
        json.dumps({"seq": 1}),  # no event
        json.dumps({"seq": "x", "event": {}}),  # seq not an int
        json.dumps([1, 2]),  # not an object at all
        json.dumps({"seq": 2, "event": {"kind": "text_delta", "text": "ok"}}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = list(events.read_records(path))
    assert [r["seq"] for r in records] == [2]


# --- render_transcript -------------------------------------------------------


def test_render_transcript_rejects_an_unknown_format():
    with pytest.raises(ValueError):
        events.render_transcript([], fmt="pdf")


def test_render_transcript_rejects_an_unknown_include_level():
    with pytest.raises(ValueError):
        events.render_transcript([], include="verbose")


def test_render_transcript_markdown_of_an_empty_session_still_has_a_header():
    text = events.render_transcript([], fmt="markdown", session_id="abc")
    assert text.startswith("# Toy transcript")
    assert "abc" in text


def test_render_transcript_jsonl_of_an_empty_session_is_empty():
    assert events.render_transcript([], fmt="jsonl") == ""


def test_render_transcript_markdown_joins_text_deltas_and_names_a_tool_call():
    records = [
        _rec(1, TextDelta(turn=1, text="Let me check ")),
        _rec(2, TextDelta(turn=1, text="the file.")),
        _rec(3, ToolUse(turn=1, tool_use_id="t1", name="read_file", input={"path": "a.json"})),
        _rec(4, ToolResult(tool_use_id="t1", is_error=False, text="contents")),
        _rec(5, TurnEnd(turn=1, stop_reason="end_turn", cost_usd=0.01)),
    ]
    text = events.render_transcript(records, fmt="markdown", include="text")
    assert "Let me check the file." in text
    assert "read_file" in text
    # include="text" is the model's words and which tools it named, nothing more.
    assert "a.json" not in text
    assert "contents" not in text
    assert "cost" not in text


def test_render_transcript_markdown_full_includes_tool_input_and_result_and_cost():
    records = [
        _rec(1, ToolUse(turn=1, tool_use_id="t1", name="read_file", input={"path": "a.json"})),
        _rec(2, ToolResult(tool_use_id="t1", is_error=False, text="contents")),
        _rec(3, PermissionRequest(request_id="p1", tool="read_file", input={})),
        _rec(4, PermissionResolved(request_id="p1", outcome="allow")),
        _rec(5, TurnEnd(turn=1, stop_reason="end_turn", cost_usd=0.0123)),
    ]
    text = events.render_transcript(records, fmt="markdown", include="full")
    assert "a.json" in text
    assert "contents" in text
    assert "allow" in text
    assert "0.0123" in text


def test_render_transcript_jsonl_keeps_only_records_the_include_level_allows():
    records = [
        _rec(1, TextDelta(turn=1, text="hi")),
        _rec(2, ToolResult(tool_use_id="t1", is_error=False, text="x")),
    ]
    at_text = events.render_transcript(records, fmt="jsonl", include="text")
    assert [json.loads(line)["seq"] for line in at_text.splitlines()] == [1]

    at_full = events.render_transcript(records, fmt="jsonl", include="full")
    assert [json.loads(line)["seq"] for line in at_full.splitlines()] == [1, 2]


def test_render_transcript_jsonl_line_matches_the_record_unmodified():
    record = _rec(7, ToolUse(turn=2, tool_use_id="t9", name="write_file", input={"a": 1}))
    text = events.render_transcript([record], fmt="jsonl", include="full")
    assert json.loads(text.strip()) == record


def test_render_transcript_excludes_lifecycle_events_at_any_include_level():
    """agent_status describes the session's own lifecycle, not anything said
    or done within it, so it never appears in a transcript."""
    records = [_rec(1, AgentStatus(status="ready"))]
    assert events.render_transcript(records, fmt="jsonl", include="full") == ""


# --- export_transcript -------------------------------------------------------


def test_export_transcript_writes_markdown_by_default(tmp_path):
    sid = _session_with_events(tmp_path, TextDelta(turn=1, text="hello"))
    target = events.export_transcript(tmp_path, sid, now=1_700_000_000.0)
    assert target.suffix == ".md"
    assert target.parent == tmp_path / files.REVIEW_DIRNAME
    assert "hello" in target.read_text(encoding="utf-8")


def test_export_transcript_writes_jsonl(tmp_path):
    sid = _session_with_events(tmp_path, TextDelta(turn=1, text="hello"))
    target = events.export_transcript(tmp_path, sid, fmt="jsonl", now=1_700_000_000.0)
    assert target.suffix == ".jsonl"
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"]["text"] == "hello"


def test_export_transcript_rejects_an_unknown_format(tmp_path):
    sid = sessions.create_session(tmp_path)
    with pytest.raises(ValueError):
        events.export_transcript(tmp_path, sid, fmt="pdf")


def test_export_transcript_include_full_threads_through_to_the_render(tmp_path):
    sid = _session_with_events(
        tmp_path,
        ToolUse(turn=1, tool_use_id="t1", name="read_file", input={"path": "secret.json"}),
    )
    at_text = events.export_transcript(tmp_path, sid, now=1_700_000_000.0)
    assert "secret.json" not in at_text.read_text(encoding="utf-8")

    at_full = events.export_transcript(tmp_path, sid, include="full", now=1_700_000_100.0)
    assert "secret.json" in at_full.read_text(encoding="utf-8")


def test_export_transcript_of_a_session_with_no_events_file_yet(tmp_path):
    """``create_session`` never creates ``events.jsonl`` (``EventLog`` opens it
    with ``O_CREAT`` on first use), so a session that never had a turn is a
    real case, not a synthetic one, and must still export cleanly."""
    sid = sessions.create_session(tmp_path)
    target = events.export_transcript(tmp_path, sid, now=1_700_000_000.0)
    assert target.read_text(encoding="utf-8").startswith("# Toy transcript")


def test_export_transcript_filename_stamp_is_deterministic_under_now(tmp_path):
    sid = _session_with_events(tmp_path, TextDelta(turn=1, text="hi"))
    now = 1_700_000_000.0
    target = events.export_transcript(tmp_path, sid, now=now)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    assert target.name == "transcript-%s.md" % stamp


def test_export_transcript_creates_review_once_and_reuses_it(tmp_path):
    sid = _session_with_events(tmp_path, TextDelta(turn=1, text="one"))
    first = events.export_transcript(tmp_path, sid, now=1_700_000_000.0)
    review_dir = tmp_path / files.REVIEW_DIRNAME
    before = os.stat(review_dir, follow_symlinks=False)

    second = events.export_transcript(tmp_path, sid, now=1_700_000_100.0)
    after = os.stat(review_dir, follow_symlinks=False)

    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert first != second
    assert first.exists() and second.exists()


def test_export_transcript_two_calls_at_the_same_stamp_do_not_collide(tmp_path):
    sid1 = _session_with_events(tmp_path, TextDelta(turn=1, text="first"))
    sid2 = _session_with_events(tmp_path, TextDelta(turn=1, text="second"))
    now = 1_700_000_000.0

    first = events.export_transcript(tmp_path, sid1, now=now)
    second = events.export_transcript(tmp_path, sid2, now=now)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    assert first.name == "transcript-%s.md" % stamp
    assert second.name == "transcript-%s-2.md" % stamp
    assert "first" in first.read_text(encoding="utf-8")
    assert "second" in second.read_text(encoding="utf-8")


def test_export_transcript_refuses_a_symlinked_review_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / files.REVIEW_DIRNAME).symlink_to(outside, target_is_directory=True)
    sid = sessions.create_session(tmp_path)

    with pytest.raises(OSError):
        events.export_transcript(tmp_path, sid, now=1_700_000_000.0)
    assert list(outside.iterdir()) == []


def test_export_transcript_refuses_a_review_path_that_is_a_regular_file(tmp_path):
    (tmp_path / files.REVIEW_DIRNAME).write_text("not a directory")
    sid = sessions.create_session(tmp_path)

    with pytest.raises(OSError):
        events.export_transcript(tmp_path, sid, now=1_700_000_000.0)


# --- files.create_review_file ------------------------------------------------


def test_create_review_file_writes_into_review_and_makes_the_directory(tmp_path):
    fd, target = files.create_review_file(tmp_path, "transcript-x.md")
    try:
        os.write(fd, b"# hi")
    finally:
        os.close(fd)
    assert target == tmp_path / files.REVIEW_DIRNAME / "transcript-x.md"
    assert target.read_bytes() == b"# hi"
    assert oct(target.stat().st_mode)[-3:] == "644"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.md",  # traversal
        "sub/transcript.md",  # a directory component
        ".hidden.md",  # a dotfile
        "transcript.txt",  # an extension this package does not write
        "transcript",  # no extension at all
        "",  # nothing
        "transcript.md\x00.txt",  # a NUL, in case a lower layer truncates at it
    ],
)
def test_create_review_file_refuses_a_name_it_would_not_write(tmp_path, name):
    assert files.create_review_file(tmp_path, name) is None


def test_create_review_file_refuses_a_symlinked_review_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / files.REVIEW_DIRNAME).symlink_to(outside, target_is_directory=True)

    assert files.create_review_file(project, "transcript-x.md") is None
    assert list(outside.iterdir()) == []


def test_create_review_file_refuses_a_review_path_that_is_a_regular_file(tmp_path):
    (tmp_path / files.REVIEW_DIRNAME).write_text("not a directory")
    assert files.create_review_file(tmp_path, "transcript-x.md") is None


def test_create_review_file_raises_rather_than_overwriting(tmp_path):
    fd, _target = files.create_review_file(tmp_path, "transcript-x.md")
    os.close(fd)
    with pytest.raises(FileExistsError):
        files.create_review_file(tmp_path, "transcript-x.md")


def test_create_review_file_refuses_a_review_directory_swapped_during_the_open(
    tmp_path, monkeypatch
):
    """O_NOFOLLOW guards the file being created, not the ``review/``
    directory it is created in: an intermediate path component is followed
    regardless of that flag. If ``review/`` is replaced by a symlink after
    the pre-open checks but before the open itself lands, the post-open
    identity re-check must still catch it, rather than trusting a directory
    it validated a moment earlier."""
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / files.REVIEW_DIRNAME).mkdir()

    real_open = os.open

    def swap_then_open(path, flags, mode=0o777):
        if str(path).endswith("swap.md"):
            review_dir = project / files.REVIEW_DIRNAME
            review_dir.rmdir()
            review_dir.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", swap_then_open)

    assert files.create_review_file(project, "swap.md") is None
    assert list(outside.iterdir()) == []
