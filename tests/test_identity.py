"""Signing in by tailnet login (``identity.py``): ``tailscale serve`` adds
``Tailscale-User-Login`` to every request it proxies, WebSocket upgrades
included, and a login on the users file is the human, by name, beside the
browser token.

What these defend is the trust rule, since the headers are ambient authority
(serve adds them to whatever the browser sends, a request another site makes
it send included): a login counts only with an ``Origin`` this server serves,
and must come with one on anything that is not a plain read; the token keeps
working as it did; neither opens the other's routes; and what a signed-in
human does is recorded as theirs.
"""

import asyncio
import json
import os
import struct

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client
from microdot import Response
from microdot.microdot import Request
from microdot.websocket import WebSocket
from toy_product import NOTES_FILE, TOY_PAGE, toy_review_store

from annealage_agent import app as agent_app
from annealage_agent import protocol, sessions
from annealage_agent.frontdoor import FrontDoor, mount_prefix
from annealage_agent.identity import Human, TailscaleIdentity, load_users
from annealage_agent.session.base import UserTurn
from annealage_agent.session.events import EventLog
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio

TOKEN = "identity-browser-token"
AGENT_TOKEN = "identity-agent-token"
ORIGIN = "http://127.0.0.1:%d" % DEFAULT_PORT
LOGIN = "andrew@example.com"
SIGNED_IN = {"Tailscale-User-Login": LOGIN, "Tailscale-User-Name": "Andrew Leech"}
JSON = {"Content-Type": "application/json"}
UPGRADE = {
    "Upgrade": "websocket",
    "Connection": "Upgrade",
    "Sec-WebSocket-Version": "13",
    "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
}
FRONT = {"card": "front", "x": 60, "y": 10}


def _app(served_dir, identity=None, **kwargs):
    kwargs.setdefault("review_store", toy_review_store(served_dir))
    return create_toy_app(
        served_dir,
        token=TOKEN,
        agent_token=AGENT_TOKEN,
        identity=identity if identity is not None else TailscaleIdentity([LOGIN]),
        **kwargs,
    )


def _agent_app(served_dir):
    """An app with a scripted session, so a turn is taken and logged."""
    return _app(
        served_dir,
        session_id=sessions.create_session(served_dir),
        build_session=lambda on_event, *, bus: FakeSession(on_event),
    )


def _body(res):
    return json.loads(res.body.decode("utf-8"))


def _logged(app, kind):
    return [wire for _seq, wire in app.agent_event_log.replay(0).events if wire["kind"] == kind]


class _RawSock:
    """One connection's bytes: the upgrade request and the client's frames in,
    everything the server wrote out (``TestClient.websocket`` drops what the
    server sends before its first read, the hello included)."""

    def __init__(self, initial_bytes):
        self.buffer = initial_bytes
        self.written = []

    async def read(self, n):
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    async def readexactly(self, n):
        return await self.read(n)

    async def readline(self):
        line = b""
        while not line.endswith(b"\n"):
            byte = await self.read(1)
            if not byte:
                break
            line += byte
        return line

    async def awrite(self, data):
        self.written.append(bytes(data))


async def _converse(app, frames, *, headers, query=""):
    """Open ``/ws`` on ``app`` with ``headers``, send ``frames`` and hang up.
    Returns the route's response and the close code the server sent, if any."""
    client = make_test_client(app)
    request = client._render_request("GET", "/ws" + query, {**UPGRADE, **headers}, b"")
    inbound = b"".join(
        bytes(
            WebSocket._encode_websocket_frame(
                WebSocket.TEXT, json.dumps({"v": protocol.PROTOCOL_VERSION, **frame})
            )
        )
        for frame in frames
    )
    sock = _RawSock(request + inbound)
    req = await Request.create(app, sock, sock, ("127.0.0.1", 1234), scheme=None)
    res = await app.dispatch_request(req)
    closes = [f for f in sock.written if f[:1] == b"\x88"]
    close_code = struct.unpack("!H", closes[0][2:4])[0] if closes else None
    return res, close_code


HELLO = {"type": "hello", "last_seq": 0, "viewer": {"tab_id": "tab-1"}}


def _turn(text):
    return {"type": "turn", "blocks": [{"type": "text", "text": text}], "client_id": "c-1"}


# ---------------------------------------------------------------------------
# who the page is signed in as
# ---------------------------------------------------------------------------


async def test_an_allowed_login_signs_the_page_in_by_name(served_dir):
    client = make_test_client(_app(served_dir))
    res = await client.get("/whoami", headers=dict(SIGNED_IN))
    assert res.status_code == 200
    assert _body(res) == {"login": LOGIN, "name": "Andrew Leech", "via": "tailscale"}
    # With the token as well the login still names the human.
    res = await client.get("/whoami?t=%s" % TOKEN, headers=dict(SIGNED_IN))
    assert _body(res)["via"] == "tailscale"
    # Without a login, the token's holder is nobody in particular.
    res = await client.get("/whoami?t=%s" % TOKEN)
    assert _body(res) == {"login": None, "name": None, "via": "token"}


async def test_logins_compare_case_insensitively_and_names_are_decoded(served_dir):
    client = make_test_client(_app(served_dir))
    res = await client.get(
        "/whoami",
        headers={
            "Tailscale-User-Login": "Andrew@Example.COM",
            "Tailscale-User-Name": "=?utf-8?q?Zo=C3=AB_Leech?=",
        },
    )
    assert _body(res) == {"login": "Andrew@Example.COM", "name": "Zoë Leech", "via": "tailscale"}
    # No name: the login stands in for it.
    res = await client.get("/whoami", headers={"Tailscale-User-Login": LOGIN})
    assert _body(res)["name"] == LOGIN


@pytest.mark.parametrize(
    "login",
    ["someone@example.com", "", "%s, someone@example.com" % LOGIN, "%s %s" % (LOGIN, LOGIN)],
    ids=["unlisted", "empty", "folded-pair", "two-words"],
)
async def test_a_login_not_on_the_list_opens_nothing(served_dir, login):
    client = make_test_client(_app(served_dir))
    assert (await client.get("/whoami", headers={"Tailscale-User-Login": login})).status_code == 403
    res = await client.post(
        "/review",
        headers={"Tailscale-User-Login": login, "Origin": ORIGIN, **JSON},
        body=json.dumps({"anchor": FRONT, "text": "thin"}),
    )
    assert res.status_code == 403


async def test_a_star_allows_any_login_serve_vouches_for(served_dir):
    client = make_test_client(_app(served_dir, identity=TailscaleIdentity(["*"])))
    res = await client.get("/whoami", headers={"Tailscale-User-Login": "visitor@github"})
    assert _body(res)["login"] == "visitor@github"
    # A request serve added no login to (a tagged device) is still nobody.
    assert (await client.get("/whoami")).status_code == 403
    with pytest.raises(ValueError, match="only entry"):
        TailscaleIdentity(["*", LOGIN])


async def test_a_login_that_only_unicode_case_folds_onto_an_allowed_one_is_refused(served_dir):
    """U+212A KELVIN SIGN lower-cases to ``k``: another identity provider's
    login spelt with it must not be taken for the allowed ASCII one."""
    kelvin = "\u212aate@corp.example"
    for identity in (TailscaleIdentity(["kate@corp.example"]), TailscaleIdentity(["*"])):
        assert not identity.allows(kelvin)
        client = make_test_client(_app(served_dir, identity=identity))
        res = await client.get("/whoami", headers={"Tailscale-User-Login": kelvin})
        assert res.status_code == 403
    assert TailscaleIdentity(["kate@corp.example"]).allows("Kate@Corp.Example")
    with pytest.raises(ValueError, match="not a tailnet login"):
        TailscaleIdentity([kelvin])


# ---------------------------------------------------------------------------
# the Origin rule: ambient authority needs the page's own Origin
# ---------------------------------------------------------------------------


async def test_a_login_without_an_origin_does_not_write(served_dir):
    """A write the browser was made to send by another site can come without
    an Origin (a form, a no-cors fetch); a login on it proves nothing."""
    app = _app(served_dir)
    client = make_test_client(app)
    comment = json.dumps({"anchor": FRONT, "text": "thin"})
    res = await client.post("/review", headers={**SIGNED_IN, **JSON}, body=comment)
    assert res.status_code == 403
    res = await client.put(
        "/settings", headers={**SIGNED_IN, **JSON}, body=json.dumps({"changes": {}})
    )
    assert res.status_code == 403
    assert app.agent_review_store.list_comments().comments == ()
    # The page's own write carries its Origin, and goes through.
    res = await client.post(
        "/review", headers={**SIGNED_IN, "Origin": ORIGIN, **JSON}, body=comment
    )
    assert res.status_code == 200


async def test_a_login_without_an_origin_does_not_open_the_socket(served_dir):
    """A browser always sends an Origin on a WebSocket handshake."""
    app = _agent_app(served_dir)
    res, _ = await _converse(app, [HELLO, _turn("hi")], headers=dict(SIGNED_IN))
    assert res.status_code == 403
    assert _logged(app, "user_turn") == []


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
async def test_another_site_s_image_or_link_does_not_read_as_the_human(served_dir, site):
    """An ``<img>`` or a link on another site sends a plain GET with no
    Origin; the page cannot read the answer, but can see whether an image
    loaded. The browser says where it came from."""
    client = make_test_client(_app(served_dir))
    res = await client.get("/review", headers={**SIGNED_IN, "Sec-Fetch-Site": site})
    assert res.status_code == 403
    # The page's own fetch, and the human typing the address, still read.
    for own in ("same-origin", "none"):
        res = await client.get("/review", headers={**SIGNED_IN, "Sec-Fetch-Site": own})
        assert res.status_code == 200, own


async def test_a_login_with_a_foreign_origin_is_refused_everywhere(served_dir):
    app = _agent_app(served_dir)
    evil = {**SIGNED_IN, "Origin": "https://evil.example"}
    client = make_test_client(app)
    assert (await client.get("/whoami", headers=evil)).status_code == 403
    assert (await client.get("/review", headers=evil)).status_code == 403
    res, _ = await _converse(app, [HELLO, _turn("hi")], headers=evil)
    assert res.status_code == 403
    assert _logged(app, "user_turn") == []


async def test_the_token_without_an_origin_works_as_it_always_has(served_dir):
    app = _app(served_dir)
    client = make_test_client(app)
    res = await client.post(
        "/review?t=%s" % TOKEN, headers=dict(JSON), body=json.dumps({"anchor": FRONT, "text": "t"})
    )
    assert res.status_code == 200
    assert "by" not in _body(res)["comment"]


# ---------------------------------------------------------------------------
# neither credential opens the other's routes
# ---------------------------------------------------------------------------


async def test_a_login_never_opens_mcp(served_dir):
    client = make_test_client(_app(served_dir, external_agents=True))
    res = await client.post(
        "/mcp",
        headers={**SIGNED_IN, "Origin": ORIGIN, **JSON},
        body=json.dumps({"method": "tools/list"}),
    )
    assert res.status_code == 403


async def test_the_agent_token_never_opens_a_browser_route(served_dir):
    client = make_test_client(_app(served_dir, external_agents=True))
    for path in ("/whoami", "/review", "/settings", "/agent/logs"):
        res = await client.get("%s?t=%s" % (path, AGENT_TOKEN))
        assert res.status_code == 403, path


# ---------------------------------------------------------------------------
# the socket: the hello's token, and what the human does over it
# ---------------------------------------------------------------------------


async def test_a_page_signed_in_by_login_needs_no_token_and_its_turn_is_theirs(served_dir):
    app = _agent_app(served_dir)
    # A product's turn-start work (Loom's sweep of outside changes) is the
    # sender's too: it runs before the turn is logged, so it reads the bus,
    # which has the whole human, their name as serve gave it included.
    senders = []
    bus = app.agent_bus
    bus.on_turn_start(lambda turn: senders.append((bus.turn_by, bus.turn_human)))
    res, close_code = await _converse(
        app, [HELLO, _turn("add a cap")], headers={**SIGNED_IN, "Origin": ORIGIN}
    )
    assert res is Response.already_handled and close_code is None
    (turn,) = _logged(app, "user_turn")
    assert (turn["by"], turn["viewer"]) == (LOGIN, "tab-1")
    assert senders == [(LOGIN, Human(login=LOGIN, name="Andrew Leech"))]


async def test_a_socket_the_token_opened_still_needs_the_token_in_its_hello(served_dir):
    app = _agent_app(served_dir)
    senders = []
    app.agent_bus.on_turn_start(lambda turn: senders.append(app.agent_bus.turn_human))
    res, close_code = await _converse(
        app, [HELLO, _turn("hi")], headers={"Origin": ORIGIN}, query="?t=%s" % TOKEN
    )
    assert res is Response.already_handled
    assert close_code == protocol.CLOSE_VERSION_MISMATCH
    assert _logged(app, "user_turn") == []
    # With it, the turn is taken, and nobody in particular sent it.
    res, close_code = await _converse(
        app,
        [{**HELLO, "token": TOKEN}, _turn("hi")],
        headers={"Origin": ORIGIN},
        query="?t=%s" % TOKEN,
    )
    assert close_code is None
    (turn,) = _logged(app, "user_turn")
    assert "by" not in turn
    assert senders == [Human()] and app.agent_bus.turn_by is None


async def test_a_permission_decision_records_who_made_it(served_dir):
    app = _app(served_dir, external_agents=True)
    app.agent_session.on_viewer_presence(1)
    client = make_test_client(app)
    call = asyncio.ensure_future(
        client.post(
            "/mcp?t=%s" % AGENT_TOKEN,
            headers=dict(JSON),
            body=json.dumps(
                {
                    "method": "tools/call",
                    "params": {"name": "add_note", "arguments": {"text": "from outside"}},
                }
            ),
        )
    )
    for _ in range(200):
        if _logged(app, "permission_request"):
            break
        await asyncio.sleep(0.01)
    (request,) = _logged(app, "permission_request")

    permission = {"type": "permission", "request_id": request["request_id"], "decision": "allow"}
    await _converse(app, [HELLO, permission], headers={**SIGNED_IN, "Origin": ORIGIN})
    await asyncio.wait_for(call, timeout=5.0)

    (resolved,) = _logged(app, "permission_resolved")
    assert (resolved["outcome"], resolved["by"]) == ("allow", LOGIN)
    notes = json.loads((served_dir / NOTES_FILE).read_text(encoding="utf-8"))
    assert notes[-1] == "from outside"


# ---------------------------------------------------------------------------
# the review records who wrote a comment and who set its status
# ---------------------------------------------------------------------------


async def test_a_signed_in_human_s_comment_and_status_carry_their_login(served_dir):
    app = _app(served_dir)
    client = make_test_client(app)

    def signed_in():
        # Fresh each time: the test client writes Content-Length into the dict.
        return {**SIGNED_IN, "Origin": ORIGIN, **JSON}

    res = await client.post(
        "/review", headers=signed_in(), body=json.dumps({"anchor": FRONT, "text": "too thin"})
    )
    comment = _body(res)["comment"]
    assert comment["by"] == LOGIN
    status_url = "/review/%d" % comment["id"]
    res = await client.post(
        status_url, headers=signed_in(), body=json.dumps({"status": "resolved"})
    )
    assert _body(res)["comment"]["status_by"] == LOGIN
    stored = json.loads(app.agent_review_store.path.read_text(encoding="utf-8"))["comments"][0]
    assert (stored["by"], stored["status"], stored["status_by"]) == (LOGIN, "resolved", LOGIN)

    # Reopened by the token's holder, whose name nobody knows: the login that
    # resolved it is not left standing as the one who set the status.
    await client.post(
        status_url + "?t=%s" % TOKEN,
        headers={"Origin": ORIGIN, **JSON},
        body=json.dumps({"status": "open"}),
    )
    stored = json.loads(app.agent_review_store.path.read_text(encoding="utf-8"))["comments"][0]
    assert (stored["status"], stored["by"]) == ("open", LOGIN)
    assert "status_by" not in stored


# ---------------------------------------------------------------------------
# where identity may be enabled at all
# ---------------------------------------------------------------------------


async def test_an_identity_is_refused_on_a_bind_that_is_not_loopback(served_dir):
    identity = TailscaleIdentity([LOGIN])
    with pytest.raises(ValueError, match="loopback"):
        create_toy_app(served_dir, token=TOKEN, host="192.0.2.10", identity=identity)
    with pytest.raises(ValueError, match="loopback"):
        FrontDoor(
            TOY_PAGE,
            token=TOKEN,
            agent_token=AGENT_TOKEN,
            host="0.0.0.0",
            port=DEFAULT_PORT,
            identity=identity,
        )


async def test_serve_refuses_an_identity_on_the_address_it_actually_binds(served_dir):
    """``create_app``'s host only decides the allowlists; the socket is
    ``serve``'s, so the rule is checked there too, before anything listens."""
    app = _app(served_dir)
    with pytest.raises(ValueError, match="loopback"):
        await asyncio.wait_for(agent_app.serve(app, "0.0.0.0", DEFAULT_PORT), timeout=5.0)


async def test_a_front_door_signs_in_by_login_and_takes_only_apps_with_its_identity(tmp_path):
    identity = TailscaleIdentity([LOGIN])
    front = FrontDoor(
        TOY_PAGE,
        token=TOKEN,
        agent_token=AGENT_TOKEN,
        host=TEST_HOST,
        port=DEFAULT_PORT,
        identity=identity,
    )
    client = make_test_client(front.app)
    assert _body(await client.get("/whoami", headers=dict(SIGNED_IN)))["login"] == LOGIN
    assert (await client.get("/apps", headers=dict(SIGNED_IN))).status_code == 200

    served = tmp_path / "a"
    served.mkdir()
    other = create_toy_app(
        served,
        token=TOKEN,
        agent_token=AGENT_TOKEN,
        login=front.login,
        url_prefix=mount_prefix("a"),
        identity=TailscaleIdentity([LOGIN]),
    )
    with pytest.raises(ValueError, match="identity"):
        front.mount("a", other)


# ---------------------------------------------------------------------------
# the users file
# ---------------------------------------------------------------------------


def _users(tmp_path, text, mode=0o644):
    path = tmp_path / "users"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


async def test_the_users_file_is_one_login_per_line_with_comments(tmp_path):
    identity = load_users(
        _users(tmp_path, "# who may sign in\n\nAndrew@Example.com  # me\nsomeone@github\n")
    )
    assert identity.logins == frozenset({"andrew@example.com", "someone@github"})
    assert identity.allows(LOGIN) and not identity.allows("stranger@github")
    with pytest.raises(ValueError, match="line 1"):
        load_users(_users(tmp_path, "two logins@here\n"))
    with pytest.raises(ValueError, match="line 2"):
        load_users(_users(tmp_path, "kate@corp.example\n\u212aate@corp.example\n"))


async def test_an_empty_users_file_signs_nobody_in_and_the_token_still_works(tmp_path, served_dir):
    client = make_test_client(_app(served_dir, identity=load_users(_users(tmp_path, ""))))
    assert (await client.get("/whoami", headers=dict(SIGNED_IN))).status_code == 403
    assert (await client.get("/whoami?t=%s" % TOKEN)).status_code == 200


@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666])
async def test_a_users_file_others_can_write_is_refused(tmp_path, mode):
    with pytest.raises(ValueError, match="writable by other users"):
        load_users(_users(tmp_path, LOGIN + "\n", mode))


async def test_a_users_file_readable_by_others_is_fine(tmp_path):
    assert load_users(_users(tmp_path, LOGIN + "\n", 0o644)).allows(LOGIN)


async def test_a_symlinked_or_missing_users_file_is_refused(tmp_path):
    target = _users(tmp_path, LOGIN + "\n")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_users(link)
    with pytest.raises(ValueError, match="cannot be opened"):
        load_users(tmp_path / "absent")


async def test_a_users_file_owned_by_someone_else_is_refused(tmp_path, monkeypatch):
    path = _users(tmp_path, LOGIN + "\n")
    monkeypatch.setattr(os, "getuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(ValueError, match="another user"):
        load_users(path)


# ---------------------------------------------------------------------------
# an event log written before anyone was named still replays
# ---------------------------------------------------------------------------


async def test_an_event_log_from_before_attribution_replays_beside_a_new_one(tmp_path):
    path = tmp_path / "events.jsonl"
    old = [
        {"kind": "user_turn", "turn": 1, "blocks": [{"type": "text", "text": "hi"}]},
        {"kind": "permission_resolved", "request_id": "pr_1", "outcome": "allow"},
        {"kind": "pause_changed", "paused": True},
    ]
    path.write_text(
        "".join(json.dumps({"seq": i + 1, "event": e}) + "\n" for i, e in enumerate(old)),
        encoding="utf-8",
    )
    log = EventLog(str(path))
    log.append(UserTurn(turn=2, blocks=[{"type": "text", "text": "again"}], by=LOGIN))
    log.close()
    replayed = [wire for _seq, wire in EventLog(str(path)).replay(0).events]
    assert replayed[:3] == old
    assert replayed[3]["by"] == LOGIN
