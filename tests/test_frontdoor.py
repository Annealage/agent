"""``frontdoor.FrontDoor``: several apps in one process, each under its own
``/p/<id>/``.

What a human relies on is that the apps are whole and separate: each has its
own served directory, conversation, review and tools, and nothing sent under
one prefix lands in another's; that one link (the browser token, or one
login nonce) opens every app; that a page reached without its trailing slash
is sent to it; and that the front door starts and stops every app it holds,
one mounted while serving included, built off the event loop as a product
builds one.
"""

import asyncio
import contextlib
import json
import socket

import pytest
from conftest import DEFAULT_PORT, TEST_AUTHORITY, TEST_HOST, create_toy_app, make_test_client
from microdot import Request
from microdot.test_client import TestClient
from toy_product import NOTES_FILE, TOY_PAGE, TOY_REVIEW_FILE, toy_review_store

from annealage_agent import app as agent_app
from annealage_agent import sessions
from annealage_agent.frontdoor import FrontDoor, mount_prefix
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio

BROWSER_TOKEN = "front-door-browser-token"
AGENT_TOKEN = "front-door-agent-token"
JSON = {"Content-Type": "application/json"}


def _front(port=DEFAULT_PORT):
    return FrontDoor(
        TOY_PAGE, token=BROWSER_TOKEN, agent_token=AGENT_TOKEN, host=TEST_HOST, port=port
    )


def _workspace(tmp_path, name, notes):
    served = tmp_path / name
    served.mkdir()
    (served / NOTES_FILE).write_text(json.dumps(notes), encoding="utf-8")
    return served


def _app(front, app_id, served, port=DEFAULT_PORT, built=None):
    def build(on_event, *, bus):
        session = FakeSession(on_event)
        if built is not None:
            built.append(session)
        return session

    return create_toy_app(
        served,
        token=BROWSER_TOKEN,
        agent_token=AGENT_TOKEN,
        host=TEST_HOST,
        port=port,
        session_id=sessions.create_session(served),
        build_session=build,
        review_store=toy_review_store(served),
        login=front.login,
        url_prefix=mount_prefix(app_id),
    )


@pytest.fixture
def two(tmp_path):
    """A front door holding apps ``a`` and ``b`` over separate directories."""
    front = _front()
    dirs = {
        "a": _workspace(tmp_path, "a", ["alpha"]),
        "b": _workspace(tmp_path, "b", ["beta"]),
    }
    apps = {app_id: _app(front, app_id, served) for app_id, served in dirs.items()}
    for app_id, app in apps.items():
        front.mount(app_id, app)
    return front, apps, dirs, make_test_client(front.app)


def _body(res):
    return json.loads(res.body.decode("utf-8"))


async def _call_tool(client, prefix, name, arguments=None):
    res = await client.post(
        "%s/mcp?t=%s" % (prefix, AGENT_TOKEN),
        headers=dict(JSON),
        body=json.dumps(
            {"method": "tools/call", "params": {"name": name, "arguments": arguments or {}}}
        ),
    )
    assert res.status_code == 200
    return json.loads(_body(res)["result"]["content"][0]["text"])


def _logged_kinds(app):
    return [wire["kind"] for _seq, wire in app.agent_event_log.replay(0).events]


# ---------------------------------------------------------------------------
# two apps, one server, nothing shared that belongs to one of them
# ---------------------------------------------------------------------------


async def test_each_app_keeps_its_own_tools_review_uploads_and_conversation(two):
    front, apps, dirs, client = two

    # The tool table under each prefix is that app's, over its own directory.
    assert (await _call_tool(client, "/p/a", "list_notes"))["notes"] == ["alpha"]
    assert (await _call_tool(client, "/p/b", "list_notes"))["notes"] == ["beta"]

    # A comment made on one app's page is in its review only.
    res = await client.post(
        "/p/a/review?t=%s" % BROWSER_TOKEN,
        headers=dict(JSON),
        body=json.dumps({"anchor": {"card": "front", "x": 60, "y": 10}, "text": "thin"}),
    )
    assert res.status_code == 200
    assert len(_body(await client.get("/p/a/review?t=%s" % BROWSER_TOKEN))["comments"]) == 1
    assert _body(await client.get("/p/b/review?t=%s" % BROWSER_TOKEN))["comments"] == []
    assert (dirs["a"] / TOY_REVIEW_FILE).exists() and not (dirs["b"] / TOY_REVIEW_FILE).exists()

    # An upload lands in its own app's directory, at a URL under its prefix
    # that the other app does not serve.
    res = await client.post(
        "/p/a/upload?t=%s" % BROWSER_TOKEN, body=b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00pixels"
    )
    url = _body(res)["url"]
    name = url.rpartition("/")[2]
    assert url == "/p/a/asset/" + name
    assert (dirs["a"] / "images" / name).is_file()
    assert not (dirs["b"] / "images").exists()
    assert (await client.get(url)).status_code == 200
    assert (await client.get("/p/b/asset/" + name)).status_code == 404

    # Each conversation is its own session writing its own event log.
    apps["a"].agent_bus.attention("Checkpoint", "OK?")
    assert "attention" in _logged_kinds(apps["a"])
    assert "attention" not in _logged_kinds(apps["b"])
    # And each names its own address to the model and the broker.
    assert apps["a"].agent_bus.url == "http://%s/p/a/" % TEST_AUTHORITY


async def test_a_page_without_its_trailing_slash_is_sent_to_it(two):
    _front, _apps, _dirs, client = two
    res = await client.get("/p/a")
    assert res.status_code == 302
    assert res.headers["Location"] == "/p/a/"
    res = await client.get("/p/b?from=list")
    assert res.headers["Location"] == "/p/b/?from=list"
    res = await client.get("/p/a/")
    assert res.status_code == 200
    assert res.body == TOY_PAGE.read_bytes()


async def test_one_link_opens_every_app(two):
    front, _apps, _dirs, client = two
    for prefix in ("/p/a", "/p/b"):
        res = await client.get("%s/settings?t=%s" % (prefix, BROWSER_TOKEN))
        assert res.status_code == 200, prefix

    # A nonce the front door issues is traded under any prefix, once.
    nonce = front.login.issue()
    res = await client.post("/p/b/login", headers=dict(JSON), body=json.dumps({"nonce": nonce}))
    assert _body(res) == {"ok": True, "token": BROWSER_TOKEN}
    res = await client.post("/p/a/login", headers=dict(JSON), body=json.dumps({"nonce": nonce}))
    assert res.status_code == 403
    nonce = front.login.issue()
    res = await client.post("/login", headers=dict(JSON), body=json.dumps({"nonce": nonce}))
    assert _body(res)["token"] == BROWSER_TOKEN


async def test_the_front_page_lists_every_app_s_status(two):
    _front, apps, _dirs, client = two
    assert (await client.get("/apps")).status_code == 403
    listed = _body(await client.get("/apps?t=%s" % BROWSER_TOKEN))
    assert set(listed) == {"a", "b"}
    assert listed["a"]["agent"] == "ready" and listed["a"]["viewers"] == 0
    apps["b"].agent_bus.attention("Checkpoint", "OK?")
    listed = _body(await client.get("/apps?t=%s" % BROWSER_TOKEN))
    assert listed["b"]["waiting"] is True and listed["b"]["attention"] == "Checkpoint: OK?"
    assert listed["a"]["waiting"] is False


async def test_the_front_door_guards_what_no_app_answers_for(two, capsys, monkeypatch):
    _front, _apps, _dirs, client = two
    # A path under a prefix that names no route never reaches the app's own
    # handlers, so the front door's headers are what it gets.
    res = await client.get("/p/a/nowhere")
    assert res.status_code == 404
    assert "default-src 'none'" in res.headers["Content-Security-Policy"]
    assert res.headers["Cache-Control"] == "no-store"
    # A rebound name is refused by the front door's own routes as by an app's.
    rebound = TestClient(two[0].app, host="rebound.example:%d" % DEFAULT_PORT)
    for path in ("/", "/apps?t=%s" % BROWSER_TOKEN, "/p/a", "/p/a/settings?t=%s" % BROWSER_TOKEN):
        assert (await rebound.get(path)).status_code == 403, path

    # One access log line per request, mounted or not.
    capsys.readouterr()
    await client.get("/p/a/settings?t=%s" % BROWSER_TOKEN)
    await client.get("/apps?t=%s" % BROWSER_TOKEN)
    err = capsys.readouterr().err
    assert err.count("/p/a/settings") == 1 and err.count("/apps") == 1

    # microdot refuses an oversized body before it finds the route, so the
    # front door's handler answers for every app, in the apps' JSON shape.
    monkeypatch.setattr(Request, "max_content_length", 8)
    res = await client.post("/p/a/review?t=%s" % BROWSER_TOKEN, headers=dict(JSON), body="x" * 64)
    assert res.status_code == 413
    assert _body(res)["ok"] is False


async def test_an_app_is_mounted_once_under_the_prefix_it_was_built_for(tmp_path):
    front = _front()
    served = _workspace(tmp_path, "a", [])
    app = _app(front, "a", served)
    with pytest.raises(ValueError, match="url_prefix='/p/b'"):
        front.mount("b", app)
    with pytest.raises(ValueError, match="one path segment"):
        front.mount("a/b", app)
    # A '.' is a regex wildcard in a microdot route: /p/a.b would also answer
    # for /p/axb, so neither the id nor the prefix may carry one.
    with pytest.raises(ValueError, match="one path segment"):
        front.mount("a.b", app)
    with pytest.raises(ValueError, match="url_prefix"):
        _app(front, "a.b", _workspace(tmp_path, "dotted", []))
    other = FrontDoor(
        TOY_PAGE, token=BROWSER_TOKEN, agent_token=AGENT_TOKEN, host=TEST_HOST, port=DEFAULT_PORT
    )
    with pytest.raises(ValueError, match="login=front.login"):
        other.mount("a", app)
    front.mount("a", app)
    with pytest.raises(ValueError, match="already mounted"):
        front.mount("a", _app(front, "a", _workspace(tmp_path, "a2", [])))


# ---------------------------------------------------------------------------
# serving: every app started once listening, stopped on the way out
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _get(port, path):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(("GET %s HTTP/1.0\r\nHost: 127.0.0.1:%d\r\n\r\n" % (path, port)).encode())
    await writer.drain()
    response = await reader.read()
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return response


async def test_serve_starts_every_app_and_stops_them_all(tmp_path, capsys):
    port = _free_port()
    front = _front(port)
    built, ran, stopped = [], [], []

    async def background():
        ran.append("a")
        await asyncio.Event().wait()

    # Built off the loop, as a product builds one while others are served.
    app_a = await asyncio.to_thread(_app, front, "a", _workspace(tmp_path, "a", []), port, built)
    app_a.agent_on_stop.append(lambda: stopped.append("a"))
    front.mount("a", app_a, background=(background,))
    ready = asyncio.Event()
    task = asyncio.ensure_future(front.serve(on_ready=ready.set))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        assert built[0].started == 1 and ran == ["a"]

        # Mounted while serving: started then, and served at once.
        app_b = await asyncio.to_thread(
            _app, front, "b", _workspace(tmp_path, "b", []), port, built
        )

        async def settle_b():
            stopped.append("b")

        app_b.agent_on_stop.append(settle_b)
        front.mount("b", app_b)
        await front.start_app("b")
        assert built[1].started == 1
        assert (await _get(port, "/p/b/")).startswith(b"HTTP/1.0 200")
        assert b"Location: /p/b/" in await _get(port, "/p/b")

        # One that fails to start while serving is reported, not lost.
        def broken_watcher():
            raise RuntimeError("no watcher")

        app_c = await asyncio.to_thread(
            _app, front, "c", _workspace(tmp_path, "c", []), port, built
        )
        front.mount("c", app_c, background=(broken_watcher,))
        await _until_logged(capsys, "the app c did not start")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)

    assert [session.closed for session in built] == [1, 1, 1]
    assert sorted(stopped) == ["a", "b"]
    assert {status["agent"] for status in front.apps().values()} == {agent_app.AGENT_CLOSED}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


async def _until_logged(capsys, text):
    seen = ""
    for _ in range(200):
        seen += capsys.readouterr().err
        if text in seen:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("%r was never written to stderr" % text)
