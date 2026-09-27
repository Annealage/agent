"""In-process HTTP contract tests for ``GET`` and ``PUT /settings``.

Separate from ``tests/test_settings.py`` by subject: that file pins how
four layers resolve into one value, this one pins the surface a browser reaches that resolution
through, including the token gate it shares with ``/ws`` and the refusals that
must survive being reachable over a network rather than only from a function
call.

Two properties here are not about resolution at all and are the reason this
route exists in the shape it does. A ``GET`` reports what **this run** is
using, never a value freshly read off disk, because reporting the file would
claim a port the server is not listening on; and anything saved that differs
appears under ``pending``, because a settings window that could not confirm
its own write would be useless. Both are asserted below against a real write.
"""

import json

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client
from toy_product import TOY

from annealage_agent import settings

pytestmark = pytest.mark.asyncio

TOKEN = "the-real-settings-token-Value_123"


def json_headers():
    """A fresh header dict per request.

    ``TestClient._process_body`` writes ``Content-Length`` into the dict it is
    handed and only when the key is absent, so a shared dict would carry the
    first request's length into every later one and the route would read a body
    of the wrong size.
    """
    return {"Content-Type": "application/json"}


@pytest.fixture
def make_client(served_dir):
    """Build a ``TestClient`` for ``served_dir`` with a token of the caller's
    choosing, and optionally the resolved settings a run would have started
    with, so a test can pin what a flag does to provenance without a CLI."""

    def _make(*, token=TOKEN, resolved=None):
        return make_test_client(
            create_toy_app(
                served_dir, token=token, host=TEST_HOST, port=DEFAULT_PORT, settings=resolved
            )
        )

    return _make


def body_of(res):
    """The parsed JSON body of a response.

    ``Response`` carries no ``.json`` (only microdot's ``TestResponse`` does,
    and only when the route returned one), so the bytes are parsed here.
    """
    return json.loads(res.body.decode("utf-8"))


def _put(client, changes, *, token=TOKEN, body=None):
    payload = json.dumps({"changes": changes} if body is None else body)
    return client.put("/settings?t=%s" % token, headers=json_headers(), body=payload)


# --- the token gate, shared with /ws -----------------------------------------


async def test_get_refused_when_no_token_configured(make_client):
    """A server with no token refuses /settings rather than treating an absent
    check as an open door. Diagnostics is behind this gate for a reason: it
    names the project root and every reachable address of the machine."""
    res = await make_client(token=None).get("/settings")
    assert res.status_code == 403
    assert res.body == b"forbidden"


async def test_get_refused_with_no_token_supplied(make_client):
    res = await make_client().get("/settings")
    assert res.status_code == 403


async def test_get_refused_with_wrong_token(make_client):
    res = await make_client().get("/settings?t=not-the-token")
    assert res.status_code == 403


async def test_put_refused_with_wrong_token_before_the_body_is_read(make_client, served_dir):
    """A wrong token refuses without writing, so a refusal cannot be a way to
    edit config: the file must still not exist afterwards."""
    res = await _put(make_client(), {"port": 9999}, token="not-the-token")
    assert res.status_code == 403
    assert not settings.project_config_path(served_dir).exists()
    assert not settings.user_settings_path().exists()


async def test_refusals_are_indistinguishable_from_each_other(make_client):
    """The token failure and the Origin failure return the same opaque
    response, so a caller cannot learn which check it failed."""
    no_token = await make_client().get("/settings")
    bad_origin = await make_client().get(
        "/settings?t=%s" % TOKEN, headers={"Origin": "http://evil.example"}
    )
    assert no_token.status_code == bad_origin.status_code == 403
    assert no_token.body == bad_origin.body


# --- GET: what is in effect, and where it came from --------------------------


async def test_get_reports_every_key_with_provenance_and_diagnostics(make_client):
    res = await make_client().get("/settings?t=%s" % TOKEN)
    assert res.status_code == 200
    payload = body_of(res)
    assert payload["ok"] is True
    assert set(payload["settings"]) == set(settings.KEYS_BY_NAME)
    for entry in payload["settings"].values():
        assert entry["from"] in (settings.FLAG, settings.PROJECT, settings.USER, settings.DEFAULT)
        assert entry["effect"] in ("restart", "load")
        assert set(entry) == {
            "value",
            "from",
            "effect",
            "editable",
            "type",
            "description",
            "choices",
            "nullable",
        }
    assert payload["settings"]["port"]["value"] == DEFAULT_PORT
    assert payload["settings"]["port"]["from"] == settings.DEFAULT
    assert payload["pending"] == {}


async def test_get_lays_out_the_window_from_each_keys_own_section(make_client):
    """The window names no key itself: which keys it shows, under which
    heading and as which choices all come from here. The toy's ``units`` joins
    the generic Viewer section ahead of the chat pane's preference, a key with
    no section (the omp endpoint and its API key) is not laid out at all, and
    a key with a closed set of values carries it."""
    payload = body_of(await make_client().get("/settings?t=%s" % TOKEN))
    assert payload["sections"] == [
        {"title": "Server", "keys": ["host", "port", "open_browser"]},
        {
            "title": "Agent",
            "keys": ["model", "effort", "permission_mode", "approval_timeout", "backend"],
        },
        {"title": "Viewer", "keys": ["units", "tool_cards_collapsed"]},
    ]
    assert payload["settings"]["units"]["choices"] == ["mm", "in"]
    assert payload["settings"]["units"]["nullable"] is False
    assert payload["settings"]["backend"]["choices"] == ["claude", "codex", "omp"]
    assert payload["settings"]["backend"]["nullable"] is True
    assert payload["product"]["title"] == "Toy"
    assert payload["product"]["state_dirname"] == ".toy"


async def test_get_diagnostics_carries_the_project_and_is_json_able(make_client, served_dir):
    payload = body_of(await make_client().get("/settings?t=%s" % TOKEN))
    facts = payload["diagnostics"]
    assert facts["project_root"] == str(served_dir)
    assert facts["bind"] == TEST_HOST
    assert facts["port"] == DEFAULT_PORT
    assert facts["toy_version"] == TOY.version
    # Round-trips, because this whole payload has already crossed a socket by
    # the time a test sees it and a field that could not be serialised would
    # have failed the request rather than this assertion.
    assert json.loads(json.dumps(facts)) == facts


async def test_get_reports_a_flag_as_the_source_and_says_a_restart_is_needed(make_client):
    """A value given as a flag this run must be reported as coming from the
    flag, because the settings window has to be able to say that editing the
    field cannot take effect until a restart."""
    resolved = settings.resolve(None, flags={"port": 9101})
    payload = body_of(await make_client(resolved=resolved).get("/settings?t=%s" % TOKEN))
    assert payload["settings"]["port"]["value"] == 9101
    assert payload["settings"]["port"]["from"] == settings.FLAG
    assert payload["settings"]["port"]["effect"] == "restart"


async def test_get_survives_a_config_file_that_stops_parsing_mid_run(make_client, served_dir):
    """The window is where someone goes to find out what is wrong, so a
    hand-edited file that no longer parses is reported as a field rather than
    taking the whole response down.

    The client is built before the file is broken, which is the order the real
    flow has: ``cli.py`` resolves settings at startup and reports a malformed
    file as a startup error, so the case this route has to survive is a file
    that stops parsing while the server is already running."""
    client = make_client()
    path = settings.project_config_path(served_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("model = \n", encoding="utf-8")
    res = await client.get("/settings?t=%s" % TOKEN)
    assert res.status_code == 200
    payload = body_of(res)
    assert payload["ok"] is True
    assert "saved_error" in payload
    assert str(path) in payload["saved_error"]
    assert payload["settings"]["port"]["value"] == DEFAULT_PORT


# --- PUT: what it refuses ----------------------------------------------------


@pytest.mark.parametrize(
    "changes,fragment",
    [
        ({"nonsense": 1}, "not a toy setting"),
        ({"port": "eight thousand"}, "not an integer"),
        ({"port": True}, "not an integer"),
        ({"open_browser": "yes"}, "not a boolean"),
        ({"units": "q"}, "must be one of"),
        ({"permission_mode": "bypassPermissions"}, "never be written"),
        ({"token": "anything"}, "never persisted"),
    ],
)
async def test_put_refuses_and_writes_nothing(make_client, served_dir, changes, fragment):
    res = await _put(make_client(), changes)
    assert res.status_code == 400
    payload = body_of(res)
    assert payload["ok"] is False
    assert fragment in payload["error"]
    assert not settings.project_config_path(served_dir).exists()
    assert not settings.user_settings_path().exists()


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"changes": []}, "changes"),
        ({"changes": {}}, "nothing to write"),
        ({"settings": {"port": 9000}}, "unknown body field"),
        ([1, 2, 3], "must be a JSON object"),
    ],
)
async def test_put_refuses_a_malformed_body(make_client, body, fragment):
    res = await _put(make_client(), None, body=body)
    assert res.status_code == 400
    assert fragment in body_of(res)["error"]


async def test_put_refuses_a_body_that_is_not_json(make_client):
    res = await make_client().put(
        "/settings?t=%s" % TOKEN, headers=json_headers(), body="{not json"
    )
    assert res.status_code == 400
    assert "invalid JSON" in body_of(res)["error"]


async def test_put_refuses_a_body_with_no_declared_length(make_client):
    """The body is read off the raw stream, so a request that declares no
    length is refused rather than read; on a real connection such a read never
    returns."""
    res = await make_client().put("/settings?t=%s" % TOKEN)
    assert res.status_code == 411


async def test_put_refuses_one_bad_key_in_an_otherwise_good_batch(make_client, served_dir):
    """Validation covers the whole batch before anything is written, so the
    good change in a mixed batch does not land either."""
    res = await _put(make_client(), {"port": 9000, "units": "sideways"})
    assert res.status_code == 400
    assert not settings.user_settings_path().exists()


# --- PUT: what it writes -----------------------------------------------------


async def test_put_writes_each_key_to_its_own_layer_and_says_which(make_client, served_dir):
    res = await _put(make_client(), {"port": 9000, "permission_mode": "plan"})
    assert res.status_code == 200
    payload = body_of(res)
    assert payload["written"] == {"user": ["port"], "project": ["permission_mode"]}
    assert "port = 9000" in settings.user_settings_path().read_text(encoding="utf-8")
    project_text = settings.project_config_path(served_dir).read_text(encoding="utf-8")
    assert 'permission_mode = "plan"' in project_text


async def test_put_reports_the_saved_value_as_pending_not_as_in_effect(make_client):
    """The write lands, and the response still reports the port this run is
    actually listening on. Claiming the new value as effective would be a lie
    about the running server, which is the whole reason pending exists."""
    payload = body_of(await _put(make_client(), {"port": 9000}))
    assert payload["settings"]["port"]["value"] == DEFAULT_PORT
    assert payload["settings"]["port"]["from"] == settings.DEFAULT
    assert payload["pending"]["port"] == {"value": 9000, "from": settings.USER}


async def test_a_later_get_still_reports_the_write_as_pending(make_client):
    """Pending is computed from disk on every read, not remembered from the
    write, so a second client sees it too."""
    client = make_client()
    await _put(client, {"port": 9000})
    payload = body_of(await client.get("/settings?t=%s" % TOKEN))
    assert payload["pending"]["port"]["value"] == 9000


async def test_a_load_effect_key_is_in_effect_at_once_and_never_pending(make_client):
    """units is applied by each page as it loads, so for a browser asking now
    the saved value *is* the one in effect. Reporting the startup value and
    calling the new one pending would leave every later page applying a
    preference the human had already changed, which is what the two effect
    classes exist to tell apart."""
    payload = body_of(await _put(make_client(), {"units": "in"}))
    assert payload["settings"]["units"]["effect"] == "load"
    assert payload["settings"]["units"]["value"] == "in"
    assert payload["settings"]["units"]["from"] == settings.USER
    assert "units" not in payload["pending"]


async def test_put_does_not_promote_a_file_over_a_flag_that_still_outranks_it(make_client):
    """A flag given this run outranks the file the window just wrote, so the
    re-resolution behind pending must supply the same flags; without that, a
    write would appear to have taken effect when the flag still wins."""
    resolved = settings.resolve(None, flags={"port": 9101})
    payload = body_of(await _put(make_client(resolved=resolved), {"port": 9000}))
    assert payload["settings"]["port"]["value"] == 9101
    assert payload["settings"]["port"]["from"] == settings.FLAG
    # 9000 is on disk but the flag outranks it, so nothing is pending: this run
    # and the next both use 9101 until the flag is dropped.
    assert "port" not in payload["pending"]
