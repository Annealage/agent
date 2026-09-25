"""``POST /session/<sid>/export``: the human's Export button
(``http/routes_chat.py``).

Separate from ``tests/test_export.py`` by subject: that file pins the rendering
and the containment of the write itself, this one pins the route over it. The
route is reachable by anyone holding the browser token and asks nobody's
permission, because a human pressing their own Export button has already
decided; what it must still do is refuse a request without the token or from
a foreign Origin, refuse a session it does not know and options it does not
accept, naming the allowed values, and write nothing in any of those cases.
A product's own tool over the same export (Mesh's ``export_transcript``) is
tested in that product's suite.
"""

import json

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client

from annealage_agent import files, sessions
from annealage_agent.session import events
from annealage_agent.session.base import TextDelta, ToolResult, ToolUse, TurnEnd

pytestmark = pytest.mark.asyncio

TOKEN = "the-real-export-token-Value_123"


def json_headers():
    """A fresh header dict per request; see ``tests/test_settings_routes.py``
    for why sharing one silently corrupts later request lengths."""
    return {"Content-Type": "application/json"}


@pytest.fixture
def project(served_dir):
    """A served directory with one session holding a short conversation."""
    sid = sessions.create_session(served_dir)
    log = events.EventLog(str(sessions.events_path(served_dir, sid)))
    log.append(TextDelta(turn=1, text="the second note needs a date"))
    log.append(ToolUse(turn=1, tool_use_id="t1", name="mcp__toy__set_view", input={"zoom": 2}))
    log.close()
    return served_dir, sid


@pytest.fixture
def client(project):
    served_dir, _sid = project
    return make_test_client(
        create_toy_app(served_dir, token=TOKEN, host=TEST_HOST, port=DEFAULT_PORT)
    )


def body_of(res):
    return json.loads(res.body.decode("utf-8"))


def _export(client, sid, *, token=TOKEN, body=None):
    if body is None:
        return client.post("/session/%s/export?t=%s" % (sid, token))
    return client.post(
        "/session/%s/export?t=%s" % (sid, token), headers=json_headers(), body=json.dumps(body)
    )


async def test_export_refused_without_the_token(project):
    served_dir, sid = project
    client = make_test_client(
        create_toy_app(served_dir, token=TOKEN, host=TEST_HOST, port=DEFAULT_PORT)
    )
    res = await _export(client, sid, token="not-the-token")
    assert res.status_code == 403
    assert not (served_dir / files.REVIEW_DIRNAME).exists()


async def test_export_refused_from_a_disallowed_origin(client, project):
    served_dir, sid = project
    res = await client.post(
        "/session/%s/export?t=%s" % (sid, TOKEN), headers={"Origin": "http://evil.example"}
    )
    assert res.status_code == 403
    assert not (served_dir / files.REVIEW_DIRNAME).exists()


async def test_export_with_no_body_writes_a_markdown_transcript(client, project):
    """The body is optional: every field has a default, so the button can post
    nothing at all."""
    served_dir, sid = project
    res = await _export(client, sid)
    assert res.status_code == 200
    payload = body_of(res)
    assert payload["ok"] is True
    assert payload["format"] == "markdown"
    assert payload["include"] == "text"
    assert payload["path"].startswith("%s/transcript-" % files.REVIEW_DIRNAME)
    assert payload["path"].endswith(".md")

    written = served_dir / payload["path"]
    assert written.is_file()
    assert written.stat().st_size == payload["bytes"]
    text = written.read_text(encoding="utf-8")
    assert "the second note needs a date" in text
    assert sid in text


async def test_export_honours_format_and_include(client, project):
    served_dir, sid = project
    payload = body_of(await _export(client, sid, body={"format": "jsonl", "include": "full"}))
    assert payload["path"].endswith(".jsonl")
    lines = [
        json.loads(line)
        for line in (served_dir / payload["path"]).read_text(encoding="utf-8").splitlines()
    ]
    kinds = [record["event"]["kind"] for record in lines]
    assert "text_delta" in kinds
    assert "tool_use" in kinds


async def test_export_include_text_keeps_the_call_and_drops_its_result(client, project):
    """The two levels split at the tool's *result*, not at the tool call: a
    reader of a text transcript sees what was said and which tools were
    reached for, and only a full one carries what those tools returned, the
    permission outcomes and the cost."""
    served_dir, sid = project
    log = events.EventLog(str(sessions.events_path(served_dir, sid)))
    log.append(ToolResult(tool_use_id="t1", is_error=False, text="zoomed the view"))
    log.append(TurnEnd(turn=1, stop_reason="end_turn", cost_usd=0.0123))
    log.close()

    def kinds(payload):
        return [
            json.loads(line)["event"]["kind"]
            for line in (served_dir / payload["path"]).read_text(encoding="utf-8").splitlines()
        ]

    text_only = kinds(body_of(await _export(client, sid, body={"format": "jsonl"})))
    full = kinds(body_of(await _export(client, sid, body={"format": "jsonl", "include": "full"})))
    assert "tool_use" in text_only
    assert "tool_result" not in text_only
    assert "turn_end" not in text_only
    assert "tool_result" in full
    assert "turn_end" in full


async def test_export_of_an_unknown_session_is_a_404(client, project):
    served_dir, _sid = project
    res = await _export(client, "20260101-000000-abcdef")
    assert res.status_code == 404
    assert "no session" in body_of(res)["error"]
    assert not (served_dir / files.REVIEW_DIRNAME).exists()


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"format": "md"}, "format must be one of"),
        ({"include": "everything"}, "include must be one of"),
        ({"formats": "markdown"}, "unknown body field"),
    ],
)
async def test_export_refuses_bad_options_and_names_the_allowed_values(
    client, project, body, fragment
):
    served_dir, sid = project
    res = await _export(client, sid, body=body)
    assert res.status_code == 400
    assert fragment in body_of(res)["error"]
    assert not (served_dir / files.REVIEW_DIRNAME).exists()


async def test_two_exports_in_the_same_second_do_not_collide(client, project):
    """Every transcript is a record of a moment, so a second export must not
    overwrite the first even when the timestamp in its name is identical."""
    served_dir, sid = project
    first = body_of(await _export(client, sid))
    second = body_of(await _export(client, sid))
    assert first["path"] != second["path"]
    assert (served_dir / first["path"]).is_file()
    assert (served_dir / second["path"]).is_file()


async def test_export_refuses_a_symlinked_review_directory(client, project, tmp_path):
    """review/ is created by this route, so it is also a name an attacker with
    write access to the project can get there first with."""
    served_dir, sid = project
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (served_dir / files.REVIEW_DIRNAME).symlink_to(outside, target_is_directory=True)
    res = await _export(client, sid)
    assert res.status_code == 500
    assert "could not write the transcript" in body_of(res)["error"]
    assert list(outside.iterdir()) == []
