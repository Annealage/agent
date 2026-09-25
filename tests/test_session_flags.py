"""Tests for the lookups a product's ``-c``/``--continue`` and ``-r``/``--resume``
flags resolve a session id through: ``sessions.py``'s ``resolve_continue``,
``get_session_info`` and ``list_sessions`` (Annealage Mesh's plan section 3.4).

What a person running the product's command sees (exit codes, messages, the
listing's layout) is the product CLI's to report and is tested in that
product's suite; these pin what the CLI is handed to report. Sessions are
built on disk directly with caller-chosen ``started_at`` values, so ordering
between several sessions is pinned by an explicit value rather than by how
fast real wall-clock timestamps happen to separate back-to-back calls.
"""

import json

from annealage_agent import sessions


def _write_session(serve_dir, sid, started_at, first_user_text=None, turn_events=()):
    """Build one ``.toy/sessions/<sid>/`` directly, with a caller-chosen
    ``started_at``."""
    directory = sessions.session_dir(serve_dir, sid)
    directory.mkdir(parents=True)
    meta = {
        "session_id": sid,
        "sdk_session_id": None,
        "started_at": started_at,
        "project_key": sessions.project_key_for_directory(serve_dir),
        "first_user_text": first_user_text,
    }
    sessions.meta_path(serve_dir, sid).write_text(json.dumps(meta), encoding="utf-8")
    if turn_events:
        with open(sessions.events_path(serve_dir, sid), "w", encoding="utf-8") as f:
            for cost in turn_events:
                f.write(json.dumps({"event": {"kind": "turn_end", "cost_usd": cost}}) + "\n")


def test_sessions_live_under_the_installed_products_state_directory(tmp_path):
    sid = sessions.create_session(tmp_path)
    assert sessions.session_dir(tmp_path, sid) == tmp_path / ".toy" / "sessions" / sid
    assert sessions.session_dir(tmp_path, sid).is_dir()


# --------------------------------------------------------------------------
# -c / --continue
# --------------------------------------------------------------------------


def test_continue_with_no_prior_session_resolves_to_nothing(tmp_path):
    assert sessions.resolve_continue(tmp_path) is None
    # Resolving creates nothing for a run that never got further.
    assert not (tmp_path / ".toy").exists()


def test_continue_picks_the_most_recently_started_of_several(tmp_path):
    """Named so that id order disagrees with start order ("sid-old" sorts
    last), which a resolver ordering by id rather than ``started_at`` fails."""
    _write_session(tmp_path, "sid-old", "2026-08-01T00:00:00Z")
    _write_session(tmp_path, "sid-mid", "2026-08-05T00:00:00Z")
    _write_session(tmp_path, "sid-new", "2026-08-10T00:00:00Z")

    assert sessions.resolve_continue(tmp_path) == "sid-new"


# --------------------------------------------------------------------------
# -r / --resume SID
# --------------------------------------------------------------------------


def test_resume_unknown_id_is_not_found(tmp_path):
    assert sessions.get_session_info(tmp_path, "no-such-session") is None


def test_resume_known_id_is_found(tmp_path):
    _write_session(tmp_path, "sid-target", "2026-08-01T00:00:00Z")
    info = sessions.get_session_info(tmp_path, "sid-target")
    assert info is not None
    assert info.session_id == "sid-target"
    assert info.started_at == "2026-08-01T00:00:00Z"


# --------------------------------------------------------------------------
# bare -r: the listing
# --------------------------------------------------------------------------


def test_the_listing_carries_every_session_with_its_turns_cost_and_opening_text(tmp_path):
    _write_session(
        tmp_path,
        "sid-a",
        "2026-08-01T00:00:00Z",
        first_user_text="please check this note",
        turn_events=(0.5, 1.25),
    )
    _write_session(tmp_path, "sid-b", "2026-08-02T00:00:00Z")

    infos = {info.session_id: info for info in sessions.list_sessions(tmp_path)}

    assert sorted(infos) == ["sid-a", "sid-b"]
    assert infos["sid-a"].turn_count == 2
    assert infos["sid-a"].cost_usd == 1.75
    assert infos["sid-a"].first_user_text == "please check this note"
    assert infos["sid-b"].turn_count == 0
    assert infos["sid-b"].cost_usd == 0.0


def test_the_listing_of_a_project_with_no_sessions_is_empty(tmp_path):
    assert sessions.list_sessions(tmp_path) == []
