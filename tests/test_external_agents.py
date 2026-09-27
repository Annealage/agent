"""``create_app(external_agents=True)``: a run with no embedded agent whose
tools an agent in another process calls through ``/mcp``
(``session/external.py``).

The human-facing property is that such an agent is gated exactly like an
embedded one: its write-grade calls become permission cards the page answers,
a call made with no page open is refused at once, and the page's pause and
permission frames reach the broker. Without the flag, a viewer-only run still
builds no tools and serves no ``/mcp``.
"""

import asyncio
import json

import pytest
from conftest import create_toy_app, make_test_client
from toy_product import NOTES_FILE, toy_review_store

from annealage_agent import settings as settings_module
from annealage_agent.http import ws as ws_module
from annealage_agent.session.base import AGENT_UNAVAILABLE
from annealage_agent.session.external import ExternalAgentSession, NoEmbeddedAgent

pytestmark = pytest.mark.asyncio

BROWSER_TOKEN = "external-browser-token"
AGENT_TOKEN = "external-agent-token"


def _app(served_dir, **kwargs):
    return create_toy_app(
        served_dir,
        token=BROWSER_TOKEN,
        agent_token=AGENT_TOKEN,
        review_store=toy_review_store(served_dir),
        **kwargs,
    )


def _mcp(client, body, token=AGENT_TOKEN):
    return client.post(
        "/mcp?t=%s" % token,
        headers={"Content-Type": "application/json"},
        body=json.dumps(body),
    )


def _call(name, arguments):
    return {"method": "tools/call", "params": {"name": name, "arguments": arguments}}


def _logged(app, kind):
    return [wire for _seq, wire in app.agent_event_log.replay(0).events if wire["kind"] == kind]


def _notes(served_dir):
    return json.loads((served_dir / NOTES_FILE).read_text(encoding="utf-8"))


async def test_viewer_only_without_the_flag_serves_no_tools(served_dir):
    app = _app(served_dir)
    assert app.agent_session is None and app.agent_tools is None
    res = await _mcp(make_test_client(app), {"method": "tools/list"})
    assert res.status_code == 404


async def test_the_flag_serves_the_product_s_tools_with_no_agent_of_its_own(served_dir):
    app = _app(served_dir, external_agents=True)
    assert isinstance(app.agent_session, ExternalAgentSession)
    assert app.agent_session.agent_status() == AGENT_UNAVAILABLE
    res = await _mcp(make_test_client(app), {"method": "tools/list"})
    assert res.status_code == 200
    names = {tool["name"] for tool in json.loads(res.body)["result"]["tools"]}
    assert {"list_notes", "add_note", "list_comments", "resolve_comment"} <= names
    # The agent token and nothing else: the browser token approves cards.
    assert (
        await _mcp(make_test_client(app), {"method": "tools/list"}, BROWSER_TOKEN)
    ).status_code == 403


async def test_a_write_grade_call_waits_for_the_page_s_answer(served_dir):
    app = _app(served_dir, external_agents=True)
    session = app.agent_session
    session.on_viewer_presence(1)
    client = make_test_client(app)

    call = asyncio.ensure_future(_mcp(client, _call("add_note", {"text": "from outside"})))
    for _ in range(200):
        requests = _logged(app, "permission_request")
        if requests:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("no permission_request reached the page")
    assert _notes(served_dir) == ["first"], "nothing happens before the human answers"
    (request,) = requests
    assert request["tool"] == "mcp__toy__add_note"

    await session.decide_permission(request["request_id"], "allow")
    res = await asyncio.wait_for(call, timeout=5.0)
    assert not json.loads(res.body)["result"].get("isError")
    assert _notes(served_dir) == ["first", "from outside"]


async def test_a_card_the_human_leaves_expires_after_the_approval_timeout(served_dir):
    app = _app(
        served_dir,
        external_agents=True,
        settings=settings_module.resolve(served_dir, flags={"approval_timeout": 1}),
    )
    app.agent_session.on_viewer_presence(1)
    res = await asyncio.wait_for(
        _mcp(make_test_client(app), _call("add_note", {"text": "unanswered"})), timeout=5.0
    )
    result = json.loads(res.body)["result"]
    assert result["isError"] and "within 1 seconds" in json.dumps(result)
    assert _notes(served_dir) == ["first"]


async def test_a_write_grade_call_with_no_page_open_is_refused_at_once(served_dir):
    app = _app(served_dir, external_agents=True)
    res = await asyncio.wait_for(
        _mcp(make_test_client(app), _call("add_note", {"text": "unseen"})), timeout=5.0
    )
    result = json.loads(res.body)["result"]
    assert result["isError"]
    assert BROWSER_TOKEN not in json.dumps(result)
    assert _notes(served_dir) == ["first"]
    assert _logged(app, "permission_request") == []


async def test_the_page_s_frames_reach_the_external_session(served_dir):
    """Pause is served (the external agent's view tools are gated by it), and
    a turn is refused rather than dropped."""
    app = _app(served_dir, external_agents=True)
    assert app.agent_bus is not None

    class Sock:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    class Registry:
        async def touch(self, conn):
            pass

        async def broadcast(self, frame):
            pass

    class Conn:
        tab_id = "tab-1"

    sock = Sock()
    await ws_module._dispatch(
        sock,
        Conn(),
        Registry(),
        app.agent_event_log,
        BROWSER_TOKEN,
        {"v": 1, "type": "pause", "paused": True},
        app.agent_session,
        app.agent_bus,
    )
    assert app.agent_bus.paused is True and sock.sent == []

    with pytest.raises(NoEmbeddedAgent):
        await app.agent_session.submit_turn([{"type": "text", "text": "hi"}])


async def test_shutting_down_denies_what_the_external_agent_still_waits_on(served_dir):
    app = _app(served_dir, external_agents=True)
    app.agent_session.on_viewer_presence(1)
    call = asyncio.ensure_future(_mcp(make_test_client(app), _call("add_note", {"text": "late"})))
    for _ in range(200):
        if _logged(app, "permission_request"):
            break
        await asyncio.sleep(0.01)
    await app.agent_session.close()
    res = await asyncio.wait_for(call, timeout=5.0)
    assert json.loads(res.body)["result"]["isError"]
    assert _notes(served_dir) == ["first"]
