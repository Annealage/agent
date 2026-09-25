"""Tests for the ``/ws`` auth trio (token, Origin, Host) and the close-frame
encoder those checks are expected to use once refused.

A bug in this path is not a UI bug: ``/ws`` is the one route a browser tab
turns into a live channel that drives an agent (Annealage Mesh's plan section 3.3), so a
check that can be bypassed here is a remote-code-execution bug once M5
lands. Every refusal below is asserted by driving a real HTTP exchange
through microdot and reading the status code back, not by calling an
internal predicate and checking its return value; a predicate can be right
while the route that is supposed to call it is wired wrong, or not wired at
all.

Ground truth this file is built on, established by source-reading microdot
2.6.2 and by encoding real frames rather than trusted from a docstring:

  1. ``WebSocket.close()`` takes no arguments and always sends an empty
     CLOSE frame; a coded close (4400, 1013, ...) can only be sent by
     building the frame by hand, which is what ``protocol.close_with_code``
     does.
  2. A pre-handshake auth failure (missing/wrong token, disallowed Origin,
     mismatched Host) refuses before ``websocket_upgrade`` runs, as an
     ordinary HTTP error response, never as a close code: there is no
     WebSocket connection yet at the point any of these three checks runs,
     so there is nothing to send a close frame on.
  3. ``websocket_upgrade(request)`` can be called from partway through a
     route, after auth checks already ran, rather than only via the
     ``with_websocket`` decorator wrapping the whole route.
  4. ``TestClient.websocket()``'s fake socket discards every outbound frame
     whose opcode is not TEXT or BINARY (``FakeWebSocket.awrite``). A test
     that opens a connection through ``TestClient.websocket()`` and then
     asserts on a close code is asserting on a frame the harness itself
     never delivered; such a test cannot fail even if the server sends the
     wrong code, or none at all, which makes it worse than no test. Every
     assertion here about a close code instead either (a) reads the status
     code of the pre-handshake HTTP response for the refusal cases, where
     no WebSocket frame is involved at all, or (b) drives a raw byte
     buffer through ``app.dispatch_request`` directly (``_RawSock`` below)
     and reads the literal bytes an ``awrite`` call was given, for the one
     case (protocol version mismatch) where a close code is sent after a
     successful upgrade.
  5. ``max_message_length`` must be set explicitly; not exercised by this
     file since it is a ``viewers.py``/``ws.py`` backpressure concern, not
     an auth one.

One further quirk found empirically while building this file, not among
the five above: a server ``send()`` issued before the server's own first
``receive()`` is also silently dropped by ``TestClient.websocket()``,
because its fake socket only starts recording sent frames once ``read()``
has been called at least once. This is a second, independent reason
``TestClient.websocket()`` is unsuitable for asserting anything about what
a freshly-upgraded connection sends before it has read anything, on top of
the opcode filtering in fact 4.

The M4 brief's own bind-mode section (line 94, in Annealage Mesh's copy) says an
unlisted Origin is refused "with close code 4403"; verified fact 2 above
says all three auth checks refuse before the handshake, never as a close
code. This file treats fact 2 as authoritative, since the brief states
outright that the five verified facts were established by source-reading
and are not to be second-guessed against other prose in the same document.
Every Origin-refusal test below therefore asserts an HTTP status, not a
close code 4403, and this contradiction should be reconciled in the brief
by whoever owns it.

The app under test is the toy product's (``conftest.create_toy_app``),
whose ``/ws`` reads ``token`` from the ``t`` query parameter, computes the
allowed Origin set from ``host`` and ``port`` (plus ``extra_origins``) the
way ``net.allowed_origins`` does, and validates the inbound ``Host`` header
against ``host``:``port``. Each test builds its app through the
``make_client`` fixture below from inside its own body, so a signature
mismatch fails each test individually, with the ``TypeError`` naming the
actual mismatch, rather than one collection error hiding every property
this file pins.

The "brief", "M4" and "M5" named above are Annealage Mesh's planning
history, from before this layer was extracted from it.
"""

import json
import struct

import pytest
from conftest import create_toy_app
from microdot import Response
from microdot.microdot import Request
from microdot.test_client import TestClient
from microdot.websocket import WebSocket

from annealage_agent import protocol

pytestmark = pytest.mark.asyncio

TOKEN = "the-real-token-Value_123"
PORT = 8765
LOOPBACK_HOST = "127.0.0.1"
LAN_HOST = "192.0.2.10"
EXTRA_ORIGIN = "https://tail-abc.ts.net"

LOOPBACK_ORIGIN = "http://127.0.0.1:%d" % PORT
LAN_ORIGIN = "http://%s:%d" % (LAN_HOST, PORT)


@pytest.fixture
def make_client(served_dir):
    """Build a ``TestClient`` around the toy app's ``create_toy_app``
    documented in this module's docstring. Called from inside each test
    body, not from fixture setup, so a signature mismatch is reported as
    that test failing, not as an error attributed to fixture teardown."""

    def _make(*, token=TOKEN, host=LOOPBACK_HOST, port=PORT, extra_origins=(EXTRA_ORIGIN,)):
        app = create_toy_app(
            served_dir, token=token, host=host, port=port, extra_origins=extra_origins
        )
        # The client's own Host header names the bind this app was built for,
        # the way a browser talking to it would. TestClient otherwise sends
        # "example.com:1234", which the Host check refuses on every route, so
        # without this every acceptance test below would be asserting on a
        # rebinding refusal rather than on the check it means to exercise.
        return TestClient(app, host="%s:%d" % (host, port))

    return _make


def _ws_query(token=TOKEN):
    return "/ws?t=%s" % token


def _ws_headers(origin=LOOPBACK_ORIGIN, host=None):
    headers = {
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
        "Origin": origin,
    }
    if host is not None:
        headers["Host"] = host
    return headers


def _assert_refused(res):
    # A pre-handshake refusal is an ordinary HTTP error response; 403 is
    # the status the brief specifies for all three auth checks. Asserting
    # the exact code, not just "not accepted", catches a route that
    # refuses by raising an uncaught exception (a 500, since microdot's
    # sentinel-return path is bypassed entirely on an exception) as
    # sharply as one that refuses with the wrong 4xx.
    assert res.status_code == 403


def _assert_accepted(res):
    # A route that reaches websocket_upgrade and then returns
    # Response.already_handled (the microdot convention every WebSocket
    # route follows) comes back through TestClient with status_code left
    # as None, since TestResponse never fabricates one for the sentinel.
    # Anything else, including a genuine 200, means the route did not
    # follow that convention or did not reach the upgrade at all.
    assert res.status_code is None


# ---------------------------------------------------------------------------
# protocol.close_with_code: real code, tested directly, no server required.
# ---------------------------------------------------------------------------


class _FakeWs:
    """The two attributes and one method close_with_code touches: `.closed`,
    `.CLOSE` and `.send`. Recording sent frames here, rather than routing
    through microdot at all, is what makes these three tests pass or fail
    on close_with_code's own logic alone, independent of whether ws.py
    exists."""

    CLOSE = WebSocket.CLOSE

    def __init__(self):
        self.closed = False
        self.sent = []

    async def send(self, data, opcode=None):
        self.sent.append((opcode, data))


async def test_close_with_code_encodes_rfc6455_close_payload():
    ws = _FakeWs()
    await protocol.close_with_code(ws, 4400, "version mismatch")
    assert ws.closed is True
    assert len(ws.sent) == 1
    opcode, payload = ws.sent[0]
    assert opcode == WebSocket.CLOSE
    assert payload[:2] == struct.pack("!H", 4400)
    assert payload[2:] == b"version mismatch"


async def test_close_with_code_truncates_reason_to_123_bytes():
    # RFC 6455 caps a control frame's payload at 125 bytes; 2 of those are
    # spent on the code, leaving 123 for the reason. A reason that runs
    # past that is silently truncated rather than raising or overflowing
    # the frame, since the encoder that would be asked to send it has no
    # way to report failure back to whatever built the reason string.
    ws = _FakeWs()
    await protocol.close_with_code(ws, 1013, "x" * 200)
    _, payload = ws.sent[0]
    assert len(payload) == 2 + 123
    assert payload[2:] == b"x" * 123


async def test_close_with_code_is_a_noop_once_already_closed():
    # A second CLOSE frame after the first is an RFC 6455 protocol
    # violation; the guard exists so a caller that reaches close_with_code
    # twice on one socket (e.g. a version-mismatch close racing an
    # unrelated disconnect close) sends exactly one CLOSE, not two.
    ws = _FakeWs()
    await protocol.close_with_code(ws, 4400, "first")
    await protocol.close_with_code(ws, 1013, "second")
    assert len(ws.sent) == 1


# ---------------------------------------------------------------------------
# Token checks.
# ---------------------------------------------------------------------------


async def test_correct_token_is_accepted(make_client):
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers())
    _assert_accepted(res)


async def test_missing_token_refused(make_client):
    client = make_client()
    res = await client.get("/ws", headers=_ws_headers())
    _assert_refused(res)


async def test_wrong_token_same_length_refused(make_client):
    client = make_client()
    wrong = "x" * len(TOKEN)
    res = await client.get(_ws_query(wrong), headers=_ws_headers())
    _assert_refused(res)


async def test_token_one_character_different_refused(make_client):
    client = make_client()
    wrong = TOKEN[:-1] + ("a" if TOKEN[-1] != "a" else "b")
    res = await client.get(_ws_query(wrong), headers=_ws_headers())
    _assert_refused(res)


async def test_token_that_is_a_prefix_of_the_real_one_refused(make_client):
    client = make_client()
    res = await client.get(_ws_query(TOKEN[:-4]), headers=_ws_headers())
    _assert_refused(res)


async def test_real_token_with_extra_trailing_bytes_refused(make_client):
    client = make_client()
    res = await client.get(_ws_query(TOKEN + "x"), headers=_ws_headers())
    _assert_refused(res)


async def test_token_with_trailing_whitespace_refused(make_client):
    client = make_client()
    # '%20' rather than a literal space: the raw request line this becomes
    # must stay well-formed regardless of what the token check itself does
    # with the decoded value.
    res = await client.get(_ws_query(TOKEN + "%20"), headers=_ws_headers())
    _assert_refused(res)


async def test_token_supplied_only_as_a_header_is_not_accepted(make_client):
    # The contract is a query parameter (a page's WebSocket constructor has
    # no way to set a custom header on the handshake at all); a token that
    # arrives some other way must not be treated as equivalent to arriving
    # correctly, which would suggest a second, undocumented code path reads
    # credentials from wherever this header maps to.
    client = make_client()
    headers = _ws_headers()
    headers["Authorization"] = "Bearer " + TOKEN
    res = await client.get("/ws", headers=headers)
    _assert_refused(res)


async def test_missing_and_wrong_token_produce_identical_responses(make_client):
    # A response that differs between "no token given" and "wrong token
    # given" (a different status code, or a body that echoes back which
    # check failed) hands an attacker a free oracle for guessing the real
    # token one property at a time; both must look the same from outside.
    client = make_client()
    missing = await client.get("/ws", headers=_ws_headers())
    wrong = await client.get(_ws_query("wrong-token-value"), headers=_ws_headers())
    assert missing.status_code == wrong.status_code == 403
    assert missing.body == wrong.body


# ---------------------------------------------------------------------------
# Origin checks.
# ---------------------------------------------------------------------------


async def test_missing_origin_header_is_accepted(make_client):
    # A non-browser client (a CLI, a test harness, curl) never sends an
    # Origin header at all; the check exists to stop a browser page on
    # another origin from opening this socket under the victim's cookies
    # and network position, not to require every client to send one.
    client = make_client()
    headers = _ws_headers()
    del headers["Origin"]
    res = await client.get(_ws_query(), headers=headers)
    _assert_accepted(res)


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:%d" % PORT,
        "http://localhost:%d" % PORT,
        "http://[::1]:%d" % PORT,
    ],
)
async def test_loopback_origin_forms_accepted_for_loopback_bind(make_client, origin):
    client = make_client(host=LOOPBACK_HOST)
    res = await client.get(_ws_query(), headers=_ws_headers(origin=origin))
    _assert_accepted(res)


async def test_extra_origin_from_cli_flag_accepted(make_client):
    client = make_client(extra_origins=(EXTRA_ORIGIN,))
    res = await client.get(_ws_query(), headers=_ws_headers(origin=EXTRA_ORIGIN))
    _assert_accepted(res)


async def test_unlisted_origin_refused(make_client):
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(origin="https://evil.example"))
    _assert_refused(res)


async def test_unlisted_origin_refusal_is_http_403_not_a_close_code(make_client):
    # Pins the resolution of this file's documented brief contradiction:
    # verified fact 2 says every auth refusal happens before the
    # handshake, so there is no WebSocket connection yet to send a close
    # code 4403 on. If this ever starts failing because a real
    # implementation upgrades first and then closes with 4403, that is the
    # brief's line 94 having been implemented instead of its own verified
    # facts, and this test is the one that should catch it.
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(origin="https://evil.example"))
    assert res.status_code == 403


async def test_origin_that_is_a_prefix_of_an_allowed_one_refused(make_client):
    client = make_client()
    # A truncated allowed origin is not itself a value the allowlist
    # contains; string prefix-matching, rather than exact set membership,
    # would be the bug this test exists to catch.
    res = await client.get(_ws_query(), headers=_ws_headers(origin="http://127.0.0.1"))
    _assert_refused(res)


async def test_origin_with_allowed_one_as_a_substring_refused(make_client):
    client = make_client()
    res = await client.get(
        _ws_query(), headers=_ws_headers(origin="http://evil.example/http://127.0.0.1:%d" % PORT)
    )
    _assert_refused(res)


async def test_origin_scheme_mismatch_refused(make_client):
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(origin="https://127.0.0.1:%d" % PORT))
    _assert_refused(res)


async def test_origin_port_mismatch_refused(make_client):
    client = make_client()
    res = await client.get(
        _ws_query(), headers=_ws_headers(origin="http://127.0.0.1:%d" % (PORT + 1))
    )
    _assert_refused(res)


async def test_origin_case_difference_refused(make_client):
    # RFC 6454 origins are compared with a case-sensitive scheme and host
    # per the spec's ASCII-serialization rules for the common case here
    # (no IDN host involved); refusing on any case difference is the safe
    # default, since accepting one requires deciding which components may
    # vary in case and why, which nothing in the brief asks for.
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(origin="HTTP://127.0.0.1:%d" % PORT))
    _assert_refused(res)


async def test_origin_ipv6_wrong_port_refused(make_client):
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(origin="http://[::1]:%d" % (PORT + 1)))
    _assert_refused(res)


async def test_lan_bind_does_not_accept_loopback_origin(make_client):
    # Companion to test_bind_modes.py's origin-set test for the same
    # property, exercised here through the actual route rather than
    # through net.allowed_origins directly: a page that only ever needed
    # loopback access to itself must not be able to drive a server bound
    # to a LAN address just because it happens to claim Origin
    # http://127.0.0.1:<port>.
    client = make_client(host=LAN_HOST)
    res = await client.get(
        _ws_query(), headers=_ws_headers(origin=LOOPBACK_ORIGIN, host="%s:%d" % (LAN_HOST, PORT))
    )
    _assert_refused(res)


# ---------------------------------------------------------------------------
# Host checks. DNS rebinding is the threat: a page an attacker controls,
# loaded from a hostname whose DNS record they also control, can be
# repointed after the browser's same-origin check has already passed so
# that a later request actually reaches this server; validating Host
# against the bind's own address is what a same-origin check on the Origin
# header alone cannot catch, since Origin describes where the page came
# from and says nothing about which server it is now actually talking to.
# ---------------------------------------------------------------------------


async def test_host_mismatch_refused_on_ws(make_client):
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(host="evil.example:%d" % PORT))
    _assert_refused(res)


async def test_host_mismatch_refused_on_an_ordinary_route(make_client):
    # The check applies to every route, not only /ws: a rebound DNS name
    # is just as capable of driving a product's own routes as it is of
    # opening /ws, and each of those is itself a real capability (in
    # Annealage Mesh, /submit writes a pin comment and /manifest enumerates
    # every model in the served directory). The toy's page stands in for
    # them here.
    client = make_client()
    res = await client.get("/", headers={"Host": "evil.example:%d" % PORT})
    assert res.status_code == 403


async def test_correct_host_accepted_on_ws(make_client):
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(host="%s:%d" % (LOOPBACK_HOST, PORT)))
    _assert_accepted(res)


async def test_correct_host_accepted_on_an_ordinary_route(make_client):
    client = make_client()
    res = await client.get("/", headers={"Host": "%s:%d" % (LOOPBACK_HOST, PORT)})
    assert res.status_code == 200


async def test_host_localhost_accepted_for_loopback_bind(make_client):
    client = make_client(host=LOOPBACK_HOST)
    res = await client.get(_ws_query(), headers=_ws_headers(host="localhost:%d" % PORT))
    _assert_accepted(res)


async def test_host_localhost_refused_for_nonloopback_bind(make_client):
    # "localhost" resolves on the attacking browser's own machine, which is
    # never the machine this server is bound to once the bind is a LAN or
    # tailscale address; accepting it there would accept a Host header that
    # can never legitimately describe this server.
    client = make_client(host=LAN_HOST)
    res = await client.get(
        _ws_query(), headers=_ws_headers(origin=LAN_ORIGIN, host="localhost:%d" % PORT)
    )
    _assert_refused(res)


async def test_host_right_address_wrong_port_refused(make_client):
    client = make_client()
    res = await client.get(
        _ws_query(), headers=_ws_headers(host="%s:%d" % (LOOPBACK_HOST, PORT + 1))
    )
    _assert_refused(res)


async def test_host_with_embedded_userinfo_refused(make_client):
    # "user@host" is not valid in a Host header at all (RFC 7230 section
    # 5.4 defines Host as uri-host [ ":" port ], with no userinfo
    # component); a check that parses this permissively enough to still
    # extract "127.0.0.1" and accept it would be reusing a URL parser built
    # for a context where userinfo is meaningful, on a header where it
    # is not, and getting a same-looking-but-wrong answer as a result.
    client = make_client()
    res = await client.get(
        _ws_query(), headers=_ws_headers(host="attacker@%s:%d" % (LOOPBACK_HOST, PORT))
    )
    _assert_refused(res)


async def test_host_with_trailing_dot_refused(make_client):
    # "127.0.0.1." and "127.0.0.1" are the same resolved address under DNS's
    # root-dot convention for a hostname, but Host-header comparison here is
    # against the literal bind address, not through a resolver; accepting a
    # trailing dot means accepting the first of a family of syntactic
    # variants of the same string with no defined stopping point.
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers(host="%s.:%d" % (LOOPBACK_HOST, PORT)))
    _assert_refused(res)


async def test_host_ipv6_loopback_form_accepted_for_loopback_bind(make_client):
    client = make_client(host=LOOPBACK_HOST)
    res = await client.get(_ws_query(), headers=_ws_headers(host="[::1]:%d" % PORT))
    _assert_accepted(res)


async def test_host_ipv6_bracket_confusion_refused(make_client):
    # A close-but-wrong bracketing is not the same string as "[::1]:8765"
    # even though every character is drawn from a plausible IPv6 Host
    # header; a parser that strips brackets before comparing without
    # requiring them to be balanced and positioned correctly could be
    # tricked into extracting "::1" from a header that does not actually
    # name that address.
    client = make_client(host=LOOPBACK_HOST)
    res = await client.get(_ws_query(), headers=_ws_headers(host="[::1:%d]" % PORT))
    _assert_refused(res)


# ---------------------------------------------------------------------------
# Protocol version mismatch: the one close-code case reachable here, since
# it happens after a successful upgrade rather than being a pre-handshake
# auth refusal. Verified fact 4 rules out TestClient.websocket() for this;
# _RawSock drives app.dispatch_request directly and records every awrite
# call verbatim so the CLOSE frame's actual bytes can be decoded.
# ---------------------------------------------------------------------------


class _RawSock:
    """A duplex byte buffer standing in for a real connection's reader and
    writer. ``buffer`` is pre-loaded with one HTTP request's bytes followed
    by one raw WebSocket frame, so a single ``dispatch_request`` call can
    complete the upgrade and then read that frame as the connection's
    first inbound message. ``written`` accumulates every ``awrite`` call's
    bytes with no opcode filtering, unlike ``TestClient``'s own fake
    socket (verified fact 4)."""

    def __init__(self, initial_bytes):
        self.buffer = initial_bytes
        self.written = []

    async def read(self, n):
        data = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return data

    async def readexactly(self, n):
        return await self.read(n)

    async def readline(self):
        line = b""
        while True:
            byte = await self.read(1)
            if not byte:
                return line
            line += byte
            if line[-1:] == b"\n":
                return line

    async def awrite(self, data):
        self.written.append(bytes(data))


def _decode_close_frame(frame_bytes):
    """Decode one server-to-client CLOSE frame's raw bytes (as produced by
    WebSocket._encode_websocket_frame, unmasked since a server frame never
    is) into (opcode, close_code, reason)."""
    opcode = frame_bytes[0] & 0x0F
    length = frame_bytes[1] & 0x7F
    body = frame_bytes[2 : 2 + length]
    close_code = struct.unpack("!H", body[:2])[0]
    reason = body[2:]
    return opcode, close_code, reason


async def test_protocol_version_mismatch_closes_with_4400(make_client):
    client = make_client()
    request_bytes = client._render_request("GET", _ws_query(), dict(_ws_headers()), b"")
    bad_frame = WebSocket._encode_websocket_frame(
        WebSocket.TEXT, json.dumps({"v": protocol.PROTOCOL_VERSION + 1, "type": "hello"})
    )
    sock = _RawSock(request_bytes + bytes(bad_frame))
    req = await Request.create(client.app, sock, sock, ("127.0.0.1", 1234), scheme=None)
    res = await client.app.dispatch_request(req)
    assert res is Response.already_handled, (
        "route did not follow the websocket_upgrade + already_handled "
        "convention; no upgrade completed for this request"
    )
    assert sock.written, (
        "server sent nothing after an inbound frame with the wrong "
        "protocol version; expected exactly one CLOSE frame"
    )
    opcode, close_code, reason = _decode_close_frame(sock.written[-1])
    assert opcode == WebSocket.CLOSE
    assert close_code == protocol.CLOSE_VERSION_MISMATCH
    assert reason != b""


# ---------------------------------------------------------------------------
# The access log must never contain the token, on any route including
# /ws: a query string carries it, and app.py's _access_log logs only
# req.path, never req.query_string.
# ---------------------------------------------------------------------------


async def _run_and_capture_stderr(client, capsys, method, path, **kwargs):
    getattr_method = getattr(client, method)
    await getattr_method(path, **kwargs)
    return capsys.readouterr().err


async def test_access_log_never_contains_token_on_root(make_client, capsys):
    client = make_client()
    err = await _run_and_capture_stderr(client, capsys, "get", "/?t=" + TOKEN)
    assert TOKEN not in err


async def test_access_log_never_contains_token_on_settings(make_client, capsys):
    client = make_client()
    err = await _run_and_capture_stderr(client, capsys, "get", "/settings?t=" + TOKEN)
    assert TOKEN not in err


async def test_access_log_never_contains_token_on_agent_static(make_client, capsys):
    client = make_client()
    err = await _run_and_capture_stderr(client, capsys, "get", "/agent/static/chat.js?t=" + TOKEN)
    assert TOKEN not in err


async def test_access_log_never_contains_token_on_login(make_client, capsys):
    client = make_client()
    err = await _run_and_capture_stderr(
        client,
        capsys,
        "post",
        "/login?t=" + TOKEN,
        body={"nonce": "not-a-nonce"},
        headers={"Content-Type": "application/json"},
    )
    assert TOKEN not in err


async def test_access_log_never_contains_token_on_ws_when_refused(make_client, capsys):
    client = make_client()
    err = await _run_and_capture_stderr(
        client, capsys, "get", "/ws?t=" + TOKEN + "x", headers=_ws_headers()
    )
    assert TOKEN not in err


async def test_access_log_never_contains_token_on_ws_when_accepted(make_client, capsys):
    client = make_client()
    err = await _run_and_capture_stderr(client, capsys, "get", _ws_query(), headers=_ws_headers())
    assert TOKEN not in err


# ---------------------------------------------------------------------------
# Repeated query parameters. Added after a raw-socket probe against a running
# server showed "?t=<real>&t=wrong" completing the handshake while
# "?t=wrong&t=<real>" was refused: microdot's MultiDict lookup returns the
# first value, so the answer to "which one counts" was a property of one
# parser rather than of this check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "/ws?t=%s&t=wrong" % TOKEN,
        "/ws?t=wrong&t=%s" % TOKEN,
        "/ws?t=%s&t=%s" % (TOKEN, TOKEN),
    ],
)
async def test_a_repeated_token_parameter_is_refused_whichever_order(make_client, query):
    # Refused even when both values are the real token, and even when the real
    # one comes first: the point is not which value wins but that nothing in
    # front of this server can disagree with it about that.
    client = make_client()
    res = await client.get(query, headers=_ws_headers())
    _assert_refused(res)


async def test_a_single_token_parameter_still_works(make_client):
    # The companion assertion, so the test above cannot pass merely because
    # every /ws request is being refused for some unrelated reason.
    client = make_client()
    res = await client.get(_ws_query(), headers=_ws_headers())
    _assert_accepted(res)
