"""``GET`` and ``POST /review`` (``http/routes_review.py``), and the review as
the app wires it: one store for the routes, the tools and the watcher, and
``review_changed`` published through the app's own event log.

The routes carry the human's words about their project, so they sit behind
the browser token like every other browser route, never the agent token, and
refuse with the same opaque answer ``/ws`` gives.
"""

import asyncio
import json
import socket

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client
from toy_product import TOY_REVIEW_FILE, toy_review_store

from annealage_agent import app as agent_app
from annealage_agent import sessions
from annealage_agent.review import Capabilities
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio

TOKEN = "review-browser-token"
AGENT_TOKEN = "review-agent-token"
FRONT = {"card": "front", "x": 60, "y": 10}


def _app(served_dir, *, store=True, token=TOKEN, port=DEFAULT_PORT, **kwargs):
    review = toy_review_store(served_dir) if store is True else store
    return create_toy_app(
        served_dir,
        token=token,
        agent_token=AGENT_TOKEN,
        host=TEST_HOST,
        port=port,
        review_store=review,
        **kwargs,
    )


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((TEST_HOST, 0))
        return s.getsockname()[1]


def _body(res):
    return json.loads(res.body.decode("utf-8"))


def _post(client, body, *, token=TOKEN, headers=None):
    return client.post(
        "/review?t=%s" % token,
        headers=dict({"Content-Type": "application/json"}, **(headers or {})),
        body=json.dumps(body),
    )


# --- the browser token, and nothing else ----------------------------------------


@pytest.mark.parametrize(
    "path, token_configured",
    [
        ("/review", True),
        ("/review?t=not-the-token", True),
        ("/review?t=%s" % AGENT_TOKEN, True),
        ("/review?t=", False),
    ],
    ids=["no token", "wrong token", "the agent token", "no token configured"],
)
async def test_get_is_refused_without_the_browser_token(served_dir, path, token_configured):
    client = make_test_client(_app(served_dir, token=TOKEN if token_configured else None))
    res = await client.get(path)
    assert res.status_code == 403
    assert res.body == b"forbidden"


async def test_post_is_refused_with_the_agent_token_and_writes_nothing(served_dir):
    """The agent holds the agent token; a comment written with it would be the
    model's words filed as the human's."""
    client = make_test_client(_app(served_dir))
    res = await _post(client, {"anchor": FRONT, "text": "mine"}, token=AGENT_TOKEN)
    assert res.status_code == 403
    assert not (served_dir / TOY_REVIEW_FILE).exists()


async def test_a_refused_origin_looks_like_a_refused_token(served_dir):
    client = make_test_client(_app(served_dir))
    no_token = await client.get("/review")
    bad_origin = await client.get("/review?t=%s" % TOKEN, headers={"Origin": "http://evil.example"})
    assert no_token.status_code == bad_origin.status_code == 403
    assert no_token.body == bad_origin.body


# --- what the page reads and writes ------------------------------------------------


async def test_get_serves_every_comment_in_the_product_neutral_shape(served_dir):
    store = toy_review_store(served_dir)
    store.add_comment(anchor=FRONT, text="thin", author="human")
    store.add_comment(anchor={"card": "back", "x": 1, "y": 2}, text="here", author="model")
    store.resolve_comment(1, "thicker")
    res = await make_test_client(_app(served_dir, store=store)).get("/review?t=%s" % TOKEN)
    assert res.status_code == 200
    assert _body(res) == {
        "ok": True,
        "anchor_space": "toy-card",
        "capabilities": {
            "can_resolve": True,
            "can_delete_own": True,
            "human_adds_via_api": True,
            "max_open_model_callouts": 50,
        },
        "comments": [
            {
                "id": 1,
                "anchor": {"card": "front", "x": 60.0, "y": 10.0},
                "ref": "box-b",
                "text": "thin",
                "author": "human",
                "status": "resolved",
                "resolution": "thicker",
            },
            {
                "id": 2,
                "anchor": {"card": "back", "x": 1.0, "y": 2.0},
                "text": "here",
                "author": "model",
                "status": "open",
            },
        ],
    }


async def test_get_reports_an_unreadable_review_rather_than_an_empty_one(served_dir):
    (served_dir / TOY_REVIEW_FILE).write_text("{ broken", encoding="utf-8")
    res = await make_test_client(_app(served_dir)).get("/review?t=%s" % TOKEN)
    assert res.status_code == 409
    assert "fix it by hand" in _body(res)["error"]
    assert (served_dir / TOY_REVIEW_FILE).read_text(encoding="utf-8") == "{ broken"


async def test_a_product_without_a_review_answers_404(served_dir):
    client = make_test_client(_app(served_dir, store=None))
    assert (await client.get("/review?t=%s" % TOKEN)).status_code == 404
    assert (await _post(client, {"anchor": FRONT, "text": "t"})).status_code == 404


async def test_post_adds_the_human_s_comment_with_what_is_under_it(served_dir):
    res = await _post(make_test_client(_app(served_dir)), {"anchor": FRONT, "text": " thin "})
    assert res.status_code == 200
    assert _body(res)["comment"] == {
        "id": 1,
        "anchor": {"card": "front", "x": 60.0, "y": 10.0},
        "ref": "box-b",
        "text": "thin",
        "author": "human",
        "status": "open",
    }


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"anchor": {"card": "side", "x": 1, "y": 1}, "text": "t"}, "no card 'side'"),
        ({"anchor": FRONT, "text": "   "}, "needs text"),
        ({"anchor": FRONT, "text": "t", "author": "model"}, "unknown body field: author"),
        ({"anchor": "front", "text": "t"}, '"anchor"'),
        (["not", "an", "object"], "JSON object"),
    ],
)
async def test_post_refuses_a_bad_comment_and_writes_nothing(served_dir, body, fragment):
    res = await _post(make_test_client(_app(served_dir)), body)
    assert res.status_code == 400
    assert fragment in _body(res)["error"]
    assert not (served_dir / TOY_REVIEW_FILE).exists()


async def test_post_is_refused_where_the_page_adds_comments_its_own_way(served_dir):
    store = toy_review_store(served_dir)
    store.capabilities = Capabilities(can_delete_own=True)
    res = await _post(
        make_test_client(_app(served_dir, store=store)), {"anchor": FRONT, "text": "t"}
    )
    assert res.status_code == 405
    assert not (served_dir / TOY_REVIEW_FILE).exists()


# --- review_changed, as the app publishes it ---------------------------------------


async def _run_watcher(app):
    task = asyncio.ensure_future(app.agent_review_watcher.run())
    await asyncio.sleep(0.05)  # past the priming sample
    return task


async def _stop(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _kinds(app):
    return [wire["kind"] for _seq, wire in app.agent_event_log.replay(0).events]


async def _logged_kinds(app):
    for _ in range(200):
        if "review_changed" in _kinds(app):
            break
        await asyncio.sleep(0.01)
    return _kinds(app)


async def test_a_comment_posted_by_the_page_is_announced_through_the_app_s_event_log(served_dir):
    """Through the one event log a reconnecting page replays from, so the seq
    it resyncs by covers this event too."""
    app = _app(served_dir)
    task = await _run_watcher(app)
    try:
        res = await _post(make_test_client(app), {"anchor": FRONT, "text": "t"})
        assert res.status_code == 200
        assert await _logged_kinds(app) == ["review_changed"]
    finally:
        await _stop(task)


async def test_a_callout_from_the_model_s_tool_is_announced(served_dir):
    """The tool server is built over the app's own store (``bus.review_store``),
    so the tool's write reaches the watcher directly."""

    def build_session(on_event, *, bus):
        return FakeSession(on_event)

    app = _app(
        served_dir,
        session_id=sessions.create_session(served_dir),
        build_session=build_session,
    )
    assert app.agent_bus.review_store is app.agent_review_store
    task = await _run_watcher(app)
    try:
        handler = {t.name: t.handler for t in app.agent_tools.tools}["add_callout"]
        result = await handler({"card": "back", "x": 5, "y": 5, "text": "look"})
        assert "is_error" not in result
        assert await _logged_kinds(app) == ["review_changed"]
    finally:
        await _stop(task)


async def test_an_edit_around_the_store_is_announced_too(served_dir):
    """A hand edit, a git checkout, an external agent: nothing the store
    hears about, which is why the watcher samples as well."""
    app = _app(served_dir)
    app.agent_review_watcher._interval = 0.02
    task = await _run_watcher(app)
    try:
        other = toy_review_store(served_dir)  # not the app's: no listener
        other.add_comment(anchor=FRONT, text="from elsewhere", author="human")
        assert await _logged_kinds(app) == ["review_changed"]
    finally:
        await _stop(task)


async def test_serve_runs_the_review_watcher(served_dir):
    port = _free_port()
    app = _app(served_dir, port=port)
    app.agent_review_watcher._interval = 0.02
    ready = asyncio.Event()
    task = asyncio.ensure_future(agent_app.serve(app, TEST_HOST, port, on_ready=ready.set))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        await asyncio.sleep(0.05)
        toy_review_store(served_dir).add_comment(anchor=FRONT, text="t", author="human")
        assert await _logged_kinds(app) == ["review_changed"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
