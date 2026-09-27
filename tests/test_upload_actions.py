"""A PDF the human uploads and an action they start on it (``uploads.py``):
the call goes in front of them as a permission card, as the agent's own
write-grade calls do, the bytes are put in the call on the server only once
they approve, and the agent hears what happened on its next turn.

The remote is ``test_remote_tools.FakeRemote``, declared by the toy's tool
server as ``datum``; its ``submit`` answers ``job 123``.
"""

import asyncio
import base64
import json
import os
import stat
import time

import pytest
import toy_product
from conftest import create_toy_app, make_test_client
from test_remote_tools import FAKE_GRADING, FakeRemote

from annealage_agent import sessions, uploads
from annealage_agent.remote import RemoteServer
from annealage_agent.session.fake import FakeSession
from annealage_agent.session.permissions import PermissionBroker
from annealage_agent.uploads import UploadAction

pytestmark = pytest.mark.asyncio

TOKEN = "upload-actions-browser-token"
AGENT_TOKEN = "upload-actions-agent-token"
PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
TOOL = "mcp__datum__submit"
DATUM = UploadAction(
    name="datum",
    label="Submit to Datum",
    tool=("datum", "submit"),
    build_args=lambda upload, human: {"project_tag": "psu", "doc_type_hint": "datasheet"},
)


@pytest.fixture
def remote(monkeypatch):
    fake = FakeRemote()
    monkeypatch.setattr(
        toy_product, "REMOTE", (RemoteServer("datum", fake.url, FAKE_GRADING, prime=("begin", {})),)
    )
    yield fake
    fake.stop()


@pytest.fixture
def app(served_dir, remote):
    """An app whose session's broker has a page to ask, and whose events are
    kept in ``app.events`` as they are published."""
    sid = sessions.create_session(served_dir)

    def build_session(on_event, *, bus):
        bus.broker = PermissionBroker(on_event, timeout=5.0, no_viewer_grace=0.05)
        bus.broker.viewer_connected()
        return FakeSession(on_event, session_id=sid)

    built = create_toy_app(
        served_dir,
        token=TOKEN,
        agent_token=AGENT_TOKEN,
        session_id=sid,
        build_session=build_session,
        upload_actions=(DATUM,),
    )
    built.events = []
    built.agent_event_log.observers.append(built.events.append)
    yield built
    built.agent_event_log.close()


async def _upload(client, body=PDF, token=TOKEN, name="LM2596 datasheet.pdf"):
    return await client.post(
        "/upload?t=%s&kind=document&name=%s" % (token, name.replace(" ", "%20")), body=body
    )


async def _start(client, upload, token=TOKEN, headers=None):
    return await client.post(
        "/upload/action?t=%s" % token,
        headers={"Content-Type": "application/json", **(headers or {})},
        body=json.dumps({"upload": upload, "action": "datum"}).encode(),
    )


async def _next(app, kind):
    for _ in range(500):
        found = [e for e in app.events if e["kind"] == kind]
        if found:
            return found[-1]
        await asyncio.sleep(0.01)
    raise AssertionError("no %s event" % kind)


async def test_an_approved_action_sends_the_bytes_from_the_server_and_tells_the_agent(
    app, remote, served_dir
):
    client = make_test_client(app)
    res = await _upload(client)
    assert res.status_code == 200, res.json
    upload = res.json
    assert (upload["name"], upload["bytes"], upload["media_type"]) == (
        "LM2596-datasheet.pdf",
        len(PDF),
        "application/pdf",
    )
    assert upload["actions"] == [
        {"name": "datum", "label": "Submit to Datum", "accepts": "application/pdf"}
    ]
    # Kept outside the served tree, and never served.
    kept = app.agent_documents.directory / upload["upload"]
    assert kept.read_bytes() == PDF
    assert (await client.get("/asset/%s" % upload["upload"])).status_code == 404

    started = await _start(client, upload["upload"])
    assert started.status_code == 202, started.json
    card = await _next(app, "permission_request")
    # The tool, the file's name and size, and every other argument; not the bytes.
    assert card["tool"] == TOOL
    assert card["action"] == "Submit to Datum"
    assert card["rememberable"] is False
    assert card["input"] == {
        "project_tag": "psu",
        "doc_type_hint": "datasheet",
        "filename": "LM2596-datasheet.pdf",
        "content_base64": "(the file's %d bytes, read when you allow this)" % len(PDF),
    }
    assert "submit" not in remote.calls, "nothing leaves before the human approves"

    await app.agent_bus.broker.decide(card["request_id"], "allow")
    ended = await _next(app, "upload_action")
    assert (ended["outcome"], ended["text"], ended["file"], ended["tool"]) == (
        "done",
        "job 123",
        "LM2596-datasheet.pdf",
        TOOL,
    )
    (called,) = [args for name, args in remote.arguments if name == "submit"]
    assert called == {
        "project_tag": "psu",
        "doc_type_hint": "datasheet",
        "filename": "LM2596-datasheet.pdf",
        "content_base64": base64.b64encode(PDF).decode(),
    }
    assert ended["upload"] == upload["upload"]
    assert not kept.exists(), "the document is removed once its action ends"
    (note,) = app.agent_bus._notes
    assert note.startswith('The human used "Submit to Datum" on LM2596-datasheet.pdf')
    assert note.endswith("It answered:\njob 123")


async def test_a_declined_action_sends_nothing_and_tells_the_agent_nothing(app, remote):
    client = make_test_client(app)
    upload = (await _upload(client)).json["upload"]
    await _start(client, upload)
    card = await _next(app, "permission_request")
    await app.agent_bus.broker.decide(card["request_id"], "deny", "not that one")
    ended = await _next(app, "upload_action")
    assert (ended["outcome"], ended["text"]) == ("denied", "not that one")
    assert "submit" not in remote.calls
    assert app.agent_bus._notes == []
    assert not (app.agent_documents.directory / upload).exists()


async def test_a_standing_grant_for_the_agent_does_not_answer_the_human_s_action(app, remote):
    """The agent's "always allow" for the tool covers the agent's calls; the
    human's own file leaving still gets its card, and an "always" sent for
    it is kept as a one-time allow."""
    broker = app.agent_bus.broker
    agent_call = asyncio.ensure_future(broker.ask(TOOL, {"query": "x"}, None))
    agents = await _next(app, "permission_request")
    await broker.decide(agents["request_id"], "allow_always")
    assert (await agent_call).allow
    assert (await broker.ask(TOOL, {}, None)).allow, "the grant answers the agent"

    client = make_test_client(app)
    upload = (await _upload(client)).json["upload"]
    await _start(client, upload)
    for _ in range(500):
        cards = [e for e in app.events if e["kind"] == "permission_request" and e.get("action")]
        if cards:
            break
        await asyncio.sleep(0.01)
    assert cards, "the human's action was answered without a card"
    await broker.decide(cards[0]["request_id"], "allow_always")
    await _next(app, "upload_action")
    assert TOOL in broker._granted_tools
    await _start(client, (await _upload(client)).json["upload"])
    for _ in range(500):
        if (
            len([e for e in app.events if e["kind"] == "permission_request" and e.get("action")])
            > 1
        ):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the second action was answered without a card")


async def test_a_file_that_is_not_a_pdf_is_refused_and_nothing_is_kept(app, served_dir):
    client = make_test_client(app)
    res = await _upload(client, body=b"<html><script>alert(1)</script></html>")
    assert res.status_code == 415
    assert "not a PDF" in res.json["error"]
    documents = uploads.documents_dir(served_dir)
    assert not documents.exists() or list(documents.iterdir()) == []


async def test_a_document_changed_on_disk_before_approval_is_not_sent(app, remote):
    """The file is kept outside the served tree, but whatever reaches it
    between the upload and the human's Allow, the bytes sent must be the ones
    the card was raised for: a changed file is refused, not sent."""
    client = make_test_client(app)
    upload = (await _upload(client)).json["upload"]
    await _start(client, upload)
    card = await _next(app, "permission_request")
    kept = app.agent_documents.directory / upload
    kept.write_bytes(PDF.replace(b"obj", b"sec"))
    await app.agent_bus.broker.decide(card["request_id"], "allow")
    ended = await _next(app, "upload_action")
    assert ended["outcome"] == "failed"
    assert "changed on disk after it was uploaded, so nothing was sent" in ended["text"]
    assert "submit" not in remote.calls
    assert not kept.exists()


async def test_documents_are_kept_privately_outside_the_workspace_and_removed_when_done(
    app, remote, served_dir
):
    client = make_test_client(app)
    first = (await _upload(client)).json["upload"]
    directory = app.agent_documents.directory
    assert directory == uploads.documents_dir(served_dir)
    assert not directory.is_relative_to(served_dir)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert (directory / first).read_bytes() == PDF

    # The chip's remove: only the browser, and the file goes.
    delete = "/upload/%s?t=%s" % (first, TOKEN)
    assert (await client.delete("/upload/%s?t=%s" % (first, AGENT_TOKEN))).status_code == 403
    assert (await client.delete(delete)).status_code == 200
    assert not (directory / first).exists()
    assert (await client.delete(delete)).status_code == 404
    assert (await _start(client, first)).status_code == 404

    # A declined action removes it too.
    second = (await _upload(client)).json["upload"]
    await _start(client, second)
    card = await _next(app, "permission_request")
    await app.agent_bus.broker.decide(card["request_id"], "deny")
    await _next(app, "upload_action")
    assert list(directory.iterdir()) == []


async def test_a_new_app_removes_documents_left_more_than_a_day_ago(served_dir, remote):
    directory = uploads.documents_dir(served_dir)
    directory.mkdir(parents=True)
    old = directory / "20260101-000000-0123abcd-old.pdf"
    recent = directory / "20260927-000000-4567cdef-recent.pdf"
    for path in (old, recent):
        path.write_bytes(PDF)
    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(old, (two_days_ago, two_days_ago))
    built = create_toy_app(
        served_dir,
        token=TOKEN,
        session_id=sessions.create_session(served_dir),
        build_session=lambda on_event, *, bus: None,
        upload_actions=(DATUM,),
    )
    built.agent_event_log.close()
    assert sorted(p.name for p in directory.iterdir()) == [recent.name]


async def test_only_the_browser_can_upload_a_document_or_start_an_action(app, remote):
    client = make_test_client(app)
    assert (await _upload(client, token=AGENT_TOKEN)).status_code == 403
    upload = (await _upload(client)).json["upload"]
    assert (await _start(client, upload, token=AGENT_TOKEN)).status_code == 403
    foreign = await _start(client, upload, headers={"Origin": "https://elsewhere.example"})
    assert foreign.status_code == 403
    assert not [e for e in app.events if e["kind"] == "permission_request"]


async def test_an_app_offering_no_action_takes_no_document(served_dir):
    client = make_test_client(create_toy_app(served_dir, token=TOKEN))
    res = await _upload(client)
    assert res.status_code == 400
    assert "kind must be one of" in res.json["error"]


async def test_an_action_naming_a_remote_the_tool_server_does_not_declare_is_refused(served_dir):
    with pytest.raises(ValueError, match="does not declare"):
        create_toy_app(
            served_dir,
            token=TOKEN,
            session_id=sessions.create_session(served_dir),
            build_session=lambda on_event, *, bus: None,
            upload_actions=(DATUM,),
        )
