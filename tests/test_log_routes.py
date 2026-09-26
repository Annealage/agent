"""What the agent backend itself said, reaching the human: every agent error on
the server's stderr (a service's journal), and ``GET /agent/logs`` serving the
backend's own logs to the page.

The remediation a session attaches to an ``AgentError`` is its guess at what
went wrong; the backend's own words are the evidence. These pin that the
evidence gets out, once per distinct error on stderr, and that the route which
serves the logs to the browser serves nothing else: only the browser token
opens it, only the session's own entries are reachable, and only their end.
"""

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client

from annealage_agent import sessions
from annealage_agent.session import logfiles
from annealage_agent.session.base import AgentError, BackendLog
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio

TOKEN = "browser-token-for-the-logs_123"
AGENT_TOKEN = "agent-token-for-the-logs_456"

NO_MODELS = (
    "RpcProcessExitError: RPC process exited with code 1. Stderr: No models available. "
    "Use /login to configure a provider."
)


def _app(served_dir, logs=()):
    """An agent-mode app over ``served_dir`` whose session lists ``logs``."""
    sid = sessions.create_session(served_dir)
    built = []

    def build_session(on_event, *, bus):
        session = FakeSession(on_event, session_id=sid)
        session.logs = list(logs)
        built.append(session)
        return session

    app = create_toy_app(
        served_dir,
        token=TOKEN,
        agent_token=AGENT_TOKEN,
        host=TEST_HOST,
        port=DEFAULT_PORT,
        session_id=sid,
        build_session=build_session,
    )
    return app, built[0]


def _journal(err):
    return [line for line in err.splitlines() if line.startswith("agent error:")]


# --- the journal ---------------------------------------------------------------


async def test_a_repeated_startup_error_reaches_stderr_once_with_the_backends_own_text(
    served_dir, capsys
):
    """The omp service that could not start said why only in the event log;
    its journal had nothing. Each turn sent to the dead agent, or a backend
    reporting the same failure again, must not bury that one line either."""
    _app_, session = _app(served_dir)
    failed = AgentError(
        stderr=NO_MODELS,
        remediation="the omp process exited before it was ready; its stderr says why",
    )
    for _ in range(3):
        session.emit(failed)

    assert _journal(capsys.readouterr().err) == [
        "agent error: the omp process exited before it was ready; its stderr says why: " + NO_MODELS
    ]

    # A different error in between makes the first one news again.
    session.emit(AgentError(stderr="", remediation="the agent is not running"))
    session.emit(failed)
    assert _journal(capsys.readouterr().err) == [
        "agent error: the agent is not running",
        "agent error: the omp process exited before it was ready; its stderr says why: "
        + NO_MODELS,
    ]


# --- the log routes: who may read them -----------------------------------------


async def test_the_log_routes_refuse_the_agent_token_a_missing_one_and_a_foreign_page(
    served_dir,
):
    """The logs hold the conversation and whatever the provider said. The
    agent token is handed to processes beside the agent's shell, so it must
    not open them any more than no token does."""
    log = served_dir / "omp.log"
    log.write_text("the conversation\n", encoding="utf-8")
    app, _session = _app(served_dir, [BackendLog("omp log", "file", path=str(log))])
    client = make_test_client(app)

    for path in ("/agent/logs", "/agent/logs/omp%20log"):
        for query in ("", "?t=" + AGENT_TOKEN, "?t=wrong"):
            res = await client.get(path + query)
            assert res.status_code == 403, path + query
            assert res.body == b"forbidden"
        res = await client.get(path + "?t=" + TOKEN, headers={"Origin": "http://evil.example"})
        assert res.status_code == 403
        assert res.body == b"forbidden"


# --- the log routes: what they serve -------------------------------------------


async def test_the_listing_is_the_sessions_own_entries(served_dir):
    log = served_dir / "omp.log"
    log.write_text("{}\n", encoding="utf-8")
    app, _session = _app(
        served_dir,
        [
            BackendLog("omp log", "file", path=str(log), format="jsonl"),
            BackendLog("omp stderr", "text", text="No models available."),
        ],
    )
    res = await make_test_client(app).get("/agent/logs?t=" + TOKEN)

    assert res.status_code == 200
    assert res.json["logs"] == [
        {"name": "omp log", "kind": "file", "path": str(log), "format": "jsonl"},
        {"name": "omp stderr", "kind": "text", "path": None, "format": "text"},
    ]


async def test_a_log_is_served_as_its_capped_end_starting_at_a_whole_line(served_dir):
    """A 10 MB omp log must not become a 10 MB response, and what is cut is the
    beginning: the failure is at the end."""
    log = served_dir / "big.log"
    lines = ["line %06d %s" % (i, "x" * 60) for i in range(8000)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.stat().st_size > logfiles.TAIL_BYTES
    app, _session = _app(served_dir, [BackendLog("omp log", "file", path=str(log))])

    res = await make_test_client(app).get("/agent/logs/omp%20log?t=" + TOKEN)

    body = res.json
    assert res.status_code == 200
    assert body["truncated"] is True
    assert body["size"] == log.stat().st_size
    assert len(body["text"].encode("utf-8")) <= logfiles.TAIL_BYTES
    assert body["text"].endswith(lines[-1] + "\n")
    assert body["text"].split("\n", 1)[0] in lines


async def test_stderr_kept_in_memory_is_served_whole_when_it_fits(served_dir):
    app, _session = _app(served_dir, [BackendLog("omp stderr", "text", text=NO_MODELS)])
    res = await make_test_client(app).get("/agent/logs/omp%20stderr?t=" + TOKEN)
    assert res.status_code == 200
    assert res.json["text"] == NO_MODELS
    assert res.json["truncated"] is False


async def test_a_log_is_found_by_its_name_after_the_list_has_grown(served_dir):
    """The page learns the list when the window opens, and the list grows while
    it is open (omp's conversation file appears with the first message). A log
    asked for then must still be the one the page showed under that name."""
    app, session = _app(served_dir, [BackendLog("omp stderr", "text", text=NO_MODELS)])
    client = make_test_client(app)
    listed = (await client.get("/agent/logs?t=" + TOKEN)).json["logs"]
    assert [entry["name"] for entry in listed] == ["omp stderr"]

    conversation = served_dir / "conversation.jsonl"
    conversation.write_text('{"role":"user"}\n', encoding="utf-8")
    session.logs.insert(
        0, BackendLog("omp conversation", "file", path=str(conversation), format="jsonl")
    )

    res = await client.get("/agent/logs/omp%20stderr?t=" + TOKEN)
    assert res.status_code == 200
    assert res.json["name"] == "omp stderr"
    assert res.json["text"] == NO_MODELS


async def test_nothing_but_the_sessions_entries_can_be_read(served_dir):
    """No request names a file: a name the session does not list (an index, a
    path, a traversal), or an entry that has become a symlink to something
    else, all get nothing."""
    secret = served_dir / "secret.txt"
    secret.write_text("SECRET", encoding="utf-8")
    planted = served_dir / "planted.log"
    planted.symlink_to(secret)
    app, _session = _app(served_dir, [BackendLog("omp log", "file", path=str(planted))])
    client = make_test_client(app)

    for path in (
        "/agent/logs/omp%20log",
        "/agent/logs/0",
        "/agent/logs/omp",
        "/agent/logs/omp%20log/../../secret.txt",
        "/agent/logs/..%2Fsecret.txt",
        "/agent/logs/%2F" + str(secret).lstrip("/").replace("/", "%2F"),
        "/agent/logs/" + str(secret).lstrip("/"),
    ):
        res = await client.get(path + "?t=" + TOKEN)
        assert res.status_code == 404, path
        assert b"SECRET" not in res.body


async def test_a_run_with_no_session_lists_no_logs(served_dir):
    app = create_toy_app(served_dir, token=TOKEN, host=TEST_HOST, port=DEFAULT_PORT)
    res = await make_test_client(app).get("/agent/logs?t=" + TOKEN)
    assert res.status_code == 200
    assert res.json["logs"] == []


async def test_the_settings_diagnostics_name_the_running_sessions_logs(served_dir):
    """The diagnostics block, which a product's doctor shares, carries the
    names and paths, never the text."""
    log = served_dir / "omp.log"
    log.write_text("{}\n", encoding="utf-8")
    app, _session = _app(
        served_dir,
        [
            BackendLog("omp log", "file", path=str(log), format="jsonl"),
            BackendLog("omp stderr", "text", text=NO_MODELS),
        ],
    )
    res = await make_test_client(app).get("/settings?t=" + TOKEN)

    assert res.status_code == 200
    assert res.json["diagnostics"]["backend_logs"] == [
        {"name": "omp log", "kind": "file", "path": str(log), "format": "jsonl"},
        {"name": "omp stderr", "kind": "text", "path": None, "format": "text"},
    ]
    assert "No models available" not in res.body.decode("utf-8")
