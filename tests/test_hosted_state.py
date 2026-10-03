"""Hosted durable records (``hosted_state.py``): what a human writes, handed to
the injected ``state_sink`` with their principal id and display snapshot.

The shape tests drive the routes and ``/ws`` dispatch as a hosted worker does,
with a fake verifier standing in for the hosted library. Standalone behaviour
(no sink, a tailnet login, the token's holder) must stay exactly as it was.
"""

import asyncio
import json
import re
import time
from types import SimpleNamespace

import pytest
from conftest import create_toy_app, make_test_client
from toy_product import toy_review_store

from annealage_agent import hosted_state, settings
from annealage_agent.http import ws as ws_module
from annealage_agent.identity import Human
from annealage_agent.session.events import EventLog
from annealage_agent.session.fake import FakeSession
from annealage_agent.viewers import ViewerBus

pytestmark = pytest.mark.asyncio

PRINCIPAL = "usr_0b0c2f6e-6c1c-4f6a-9d2a-5d6f0e7a4a11"
REV = "a" * 40
FRONT = {"card": "front", "x": 60, "y": 10}
RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class _Verifier:
    def verify(self, token, operation):
        if token != "delegation":
            raise ValueError("invalid delegation")
        return SimpleNamespace(
            sub=PRINCIPAL,
            wsp="wsp_test",
            prj="prj_test",
            ops=("project.read", "project.comment"),
            job="job_test",
            rev=REV,
            exp=int(time.time()) + 60,
        )


class _Sink:
    def __init__(self):
        self.records = []
        self.fail = 0

    def __call__(self, record):
        if self.fail:
            self.fail -= 1
            raise RuntimeError("store down")
        self.records.append(record)


def _headers(display="Ada L."):
    headers = {"Authorization": "Bearer delegation", "Content-Type": "application/json"}
    if display is not None:
        headers["X-Annealage-Principal-Display"] = display
    return headers


@pytest.fixture
def hosted(served_dir, tmp_path_factory):
    sink = _Sink()
    store = toy_review_store(served_dir)
    app = create_toy_app(
        served_dir,
        hosted_mode=True,
        hosted_verifier=_Verifier(),
        hosted_state_dir=tmp_path_factory.mktemp("worker-state"),
        settings=settings.resolve(tmp_path_factory.mktemp("service")),
        review_store=store,
        state_sink=sink,
    )
    return SimpleNamespace(client=make_test_client(app), sink=sink, store=store)


def _body(res):
    return json.loads(res.body.decode("utf-8"))


# --- comments and their status ---------------------------------------------------


async def test_a_hosted_comment_and_its_status_change_reach_the_sink_with_the_principal(hosted):
    res = await hosted.client.post(
        "/review", headers=_headers(), body=json.dumps({"anchor": FRONT, "text": "too thin"})
    )
    assert res.status_code == 200
    comment_id = _body(res)["comment"]["id"]
    res = await hosted.client.post(
        "/review/%d" % comment_id, headers=_headers(), body=json.dumps({"status": "resolved"})
    )
    assert res.status_code == 200

    comment, status = hosted.sink.records
    assert comment["record_id"] == "comment:%d" % comment_id
    assert comment == {
        "record_id": "comment:1",
        "kind": "comment",
        "principal_id": PRINCIPAL,
        "display": "Ada L.",
        "payload": {
            "id": 1,
            "anchor": {"card": "front", "x": 60.0, "y": 10.0},
            "text": "too thin",
            "author": "human",
            "ref": "box-b",
        },
        "rev": REV,
        "ts": comment["ts"],
    }
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", comment["ts"])
    assert status["kind"] == "comment_status"
    assert status["payload"] == {"comment_id": comment["record_id"], "status": "resolved"}
    assert (status["principal_id"], status["display"]) == (PRINCIPAL, "Ada L.")
    assert "rev" not in status
    assert RECORD_ID.fullmatch(status["record_id"])
    assert status["record_id"].startswith("comment_status:1:")


async def test_two_status_changes_of_one_comment_are_two_records(hosted):
    await hosted.client.post(
        "/review", headers=_headers(), body=json.dumps({"anchor": FRONT, "text": "x"})
    )
    for status in ("resolved", "open"):
        await hosted.client.post(
            "/review/1", headers=_headers(), body=json.dumps({"status": status})
        )
    ids = [r["record_id"] for r in hosted.sink.records if r["kind"] == "comment_status"]
    assert len(set(ids)) == 2
    assert [r["payload"]["status"] for r in hosted.sink.records[1:]] == ["resolved", "open"]


async def test_the_local_comment_survives_a_sink_that_fails_and_is_offered_again(hosted, capsys):
    hosted.sink.fail = 1
    res = await hosted.client.post(
        "/review", headers=_headers(), body=json.dumps({"anchor": FRONT, "text": "first"})
    )
    assert res.status_code == 200
    assert hosted.sink.records == []
    assert "first" in (hosted.store.get_comment(1)).text
    assert "store down" in capsys.readouterr().err

    await hosted.client.post(
        "/review", headers=_headers(), body=json.dumps({"anchor": FRONT, "text": "second"})
    )
    # The undelivered one goes first, so a status never precedes its comment.
    assert [r["record_id"] for r in hosted.sink.records] == ["comment:1", "comment:2"]


async def test_a_hosted_human_without_a_valid_principal_is_not_recorded(
    served_dir, tmp_path_factory, capsys
):
    class Verifier(_Verifier):
        def verify(self, token, operation):
            claims = super().verify(token, operation)
            claims.sub = "usr_test"
            return claims

    sink = _Sink()
    store = toy_review_store(served_dir)
    client = make_test_client(
        create_toy_app(
            served_dir,
            hosted_mode=True,
            hosted_verifier=Verifier(),
            hosted_state_dir=tmp_path_factory.mktemp("worker-state"),
            settings=settings.resolve(tmp_path_factory.mktemp("service")),
            review_store=store,
            state_sink=sink,
        )
    )
    res = await client.post(
        "/review", headers=_headers(), body=json.dumps({"anchor": FRONT, "text": "kept"})
    )
    assert res.status_code == 200
    assert sink.records == []
    assert store.get_comment(1).text == "kept"
    assert "no valid principal" in capsys.readouterr().err


async def test_a_standalone_login_is_never_a_principal(served_dir):
    """A tailnet login is legacy attribution: nothing is emitted for it, and a
    sink is refused outright outside hosted mode."""
    with pytest.raises(ValueError, match="requires hosted mode"):
        create_toy_app(served_dir, token="t", state_sink=_Sink())
    assert hosted_state.attribution(Human(login="ada@example.com", name="Ada")) is None
    assert hosted_state.attribution(Human()) is None


# --- chat turns and permission decisions -----------------------------------------


class _Registry:
    async def touch(self, conn):
        pass

    async def broadcast(self, frame):
        pass


class _Socket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


def _conn(human):
    return SimpleNamespace(tab_id="tab-1", human=human)


def _hosted_human(name="Ada L."):
    return Human(name=name, principal_id=PRINCIPAL, workspace_id="wsp_test", project_id="prj_test")


async def _send(frame, human, sink, session, bus=None, log=None):
    log = log if log is not None else EventLog()
    recorder = hosted_state.StateRecorder(sink)
    await ws_module._dispatch(
        _Socket(),
        _conn(human),
        _Registry(),
        log,
        "tok",
        dict({"v": 1}, **frame),
        session,
        bus if bus is not None else ViewerBus(None, url="http://127.0.0.1:8765/"),
        recorder,
    )
    return log


def _logged(log, kind):
    return [wire for _seq, wire in log.replay(0).events if wire["kind"] == kind]


async def test_a_hosted_turn_is_recorded_by_session_and_turn_before_the_agent_sees_it():
    sink, seen = _Sink(), []

    class Session(FakeSession):
        async def submit_turn(self, blocks, viewer=None):
            seen.append(list(sink.records))
            await super().submit_turn(blocks, viewer)

    session = Session(lambda event: None, session_id="sess-1")
    blocks = [{"type": "text", "text": "carry on"}]
    log = await _send(
        {"type": "turn", "blocks": blocks, "client_id": "c-1"}, _hosted_human(), sink, session
    )
    await asyncio.sleep(0)

    (record,) = sink.records
    assert seen == [[record]]
    assert record == {
        "record_id": "user_turn:sess-1:1",
        "kind": "user_turn",
        "principal_id": PRINCIPAL,
        "display": "Ada L.",
        "payload": {"session_id": "sess-1", "turn": 1, "blocks": blocks, "client_id": "c-1"},
        "ts": record["ts"],
    }
    (event,) = _logged(log, "user_turn")
    assert (event["principal_id"], event["display"]) == (PRINCIPAL, "Ada L.")


async def test_a_turn_with_an_unsafe_session_id_still_gets_a_valid_stable_id():
    first = hosted_state.record_id("user_turn", "sess/with spaces and ü", 3)
    assert first == hosted_state.record_id("user_turn", "sess/with spaces and ü", 3)
    assert RECORD_ID.fullmatch(first) and first.startswith("user_turn:")
    assert hosted_state.record_id("user_turn", "sess/other", 3) != first
    long = hosted_state.record_id("user_turn", "s" * 300, 1)
    assert RECORD_ID.fullmatch(long)


async def test_a_standalone_turn_logs_the_login_only_and_records_nothing():
    sink = _Sink()
    session = FakeSession(lambda event: None)
    blocks = [{"type": "text", "text": "hi"}]
    log = await _send(
        {"type": "turn", "blocks": blocks}, Human(login="ada@example.com"), sink, session
    )
    assert sink.records == []
    (event,) = _logged(log, "user_turn")
    assert event["by"] == "ada@example.com"
    assert "principal_id" not in event and "display" not in event


async def test_a_permission_decision_is_recorded_once_even_if_a_second_view_clicks_too():
    sink = _Sink()
    session = FakeSession(lambda event: None)
    frame = {"type": "permission", "request_id": "pr_ab12_1", "decision": "allow"}

    class Broker:
        def pending_requests(self):
            return [SimpleNamespace(request_id="pr_ab12_1", tool="mcp__toy__write_note")]

    bus = ViewerBus(None, url="http://127.0.0.1:8765/")
    bus.broker = Broker()
    await _send(frame, _hosted_human(), sink, session, bus)
    await _send(frame, _hosted_human(), sink, session, bus)

    (record,) = sink.records
    assert record == {
        "record_id": "permission_decision:pr_ab12_1",
        "kind": "permission_decision",
        "principal_id": PRINCIPAL,
        "display": "Ada L.",
        "payload": {
            "request_id": "pr_ab12_1",
            "decision": "allow",
            "tool": "mcp__toy__write_note",
        },
        "ts": record["ts"],
    }


# --- the sink wrapper and the label rules -----------------------------------------


async def test_display_is_attribution_text_that_is_cleaned_or_replaced():
    assert hosted_state.attribution(Human(principal_id=PRINCIPAL, name="  Ada  ")) == (
        PRINCIPAL,
        "Ada",
    )
    for bad in (None, "", "a\u2028b", "a\x00b", "\u202eevil", " " * 5):
        assert hosted_state.attribution(Human(principal_id=PRINCIPAL, name=bad))[1] == PRINCIPAL
    long = hosted_state.attribution(Human(principal_id=PRINCIPAL, name="é" * 300))[1]
    assert len(long) == hosted_state.DISPLAY_LIMIT
    assert hosted_state.attribution(Human(principal_id="usr_public")) == (
        "usr_public",
        "usr_public",
    )
    for bad_principal in ("ada", "usr_ABC", PRINCIPAL.upper(), "svc_" + PRINCIPAL[4:]):
        assert hosted_state.attribution(Human(principal_id=bad_principal)) is None


async def test_a_record_the_sink_keeps_rejecting_is_dropped_after_its_attempts(capsys):
    def sink(record):
        raise ValueError("RecordConflict")

    wrapped = hosted_state.Sink(sink)
    for _ in range(hosted_state.MAX_ATTEMPTS):
        wrapped.emit({"record_id": "x"}, "x")
    err = capsys.readouterr().err
    assert err.count("sink failed for x") == hosted_state.MAX_ATTEMPTS
    assert "gave up on x" in err
    wrapped.emit({"record_id": "y"}, "y")
    assert "sink failed for x" not in capsys.readouterr().err


async def test_no_sink_means_no_records_and_no_noise(capsys):
    recorder = hosted_state.StateRecorder(None)
    assert not recorder.enabled
    recorder.comment(
        _hosted_human(),
        SimpleNamespace(id=1, anchor={}, text="t", author="human", ref=None, extra={}),
    )
    assert capsys.readouterr().err == ""
