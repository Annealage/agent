"""Tests for ``POST /login`` (``http/routes_login.py``): the single-use
nonce the auto-opened browser URL carries instead of the browser token.

The CLI side, that ``webbrowser.open`` is given a nonce and never the token,
and the page side, that ``#n=`` logs a tab in once and scrubs the address
bar, belong to the product that owns the CLI and the page (Annealage Mesh:
its ``tests/test_cli.py`` and ``tests/test_viewer_e2e.py``).
"""

import json

import pytest
from conftest import create_toy_app, make_test_client

from annealage_agent import sessions
from annealage_agent.http.routes_login import LoginNonces
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio

BROWSER_TOKEN = "login-browser-token"
AGENT_TOKEN = "login-agent-token"


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _client(served_dir, login):
    """An agent-mode app with both tokens, so the agent token exists to be
    refused."""
    app = create_toy_app(
        served_dir,
        token=BROWSER_TOKEN,
        agent_token=AGENT_TOKEN,
        session_id=sessions.create_session(served_dir),
        build_session=lambda on_event, *, bus: FakeSession(on_event),
        login=login,
    )
    return make_test_client(app)


async def _login(client, nonce):
    return await client.post(
        "/login",
        headers={"Content-Type": "application/json"},
        body=json.dumps({"nonce": nonce}),
    )


async def test_a_nonce_logs_in_once_and_is_then_refused(served_dir):
    login = LoginNonces()
    client = _client(served_dir, login)
    nonce = login.issue()

    first = await _login(client, nonce)
    assert first.status_code == 200
    assert first.json == {"ok": True, "token": BROWSER_TOKEN}

    again = await _login(client, nonce)
    assert again.status_code == 403


async def test_an_expired_nonce_is_refused(served_dir):
    clock = _Clock()
    login = LoginNonces(ttl=60.0, clock=clock)
    client = _client(served_dir, login)
    nonce = login.issue()

    clock.now += 60.0
    res = await _login(client, nonce)
    assert res.status_code == 403


async def test_a_nonce_is_redeemable_until_just_before_it_expires(served_dir):
    clock = _Clock()
    login = LoginNonces(ttl=60.0, clock=clock)
    client = _client(served_dir, login)
    nonce = login.issue()

    clock.now += 59.0
    res = await _login(client, nonce)
    assert res.status_code == 200


@pytest.mark.parametrize("credential", [AGENT_TOKEN, BROWSER_TOKEN, "", None, 7])
async def test_the_route_accepts_nothing_but_an_outstanding_nonce(served_dir, credential):
    """Neither token is a nonce: the agent token must never be exchangeable
    for the browser token, and anything else is refused the same way."""
    login = LoginNonces()
    login.issue()
    client = _client(served_dir, login)
    res = await _login(client, credential)
    assert res.status_code == 403
    assert res.body == b"forbidden"


async def test_the_agent_token_in_the_query_opens_nothing_either(served_dir):
    login = LoginNonces()
    client = _client(served_dir, login)
    res = await client.post(
        "/login?t=%s" % AGENT_TOKEN,
        headers={"Content-Type": "application/json"},
        body=json.dumps({}),
    )
    assert res.status_code == 403
