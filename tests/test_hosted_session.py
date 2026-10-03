"""Hosted session control (``http/routes_session.py``, ``AgentHolder.launch`` /
``cancel``) and the lifecycle ``session_event`` records it produces.

The app is a hosted agent app over a fake session, with a verifier standing in
for the hosted library. What matters here: who may launch or cancel, that a
cancel is not undone by a page or a tool call, that a launch reopens it, that
nothing is recorded without a sink, and that local (standalone) apps have
none of these routes.
"""

import asyncio
import json
import re
import time
from types import SimpleNamespace

import pytest
from conftest import create_toy_app, make_test_client

from annealage_agent import sessions, settings
from annealage_agent.session.base import AgentError, UserTurn
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio

PRINCIPAL = "usr_0b0c2f6e-6c1c-4f6a-9d2a-5d6f0e7a4a11"
OTHER = "usr_5a1d0d52-3a64-4c53-b1d2-2f7d5a0c9e77"
RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class _Verifier:
    def __init__(self, ops=("project.read", "project.exec"), sub=PRINCIPAL):
        self.ops = ops
        self.sub = sub

    def verify(self, token, operation):
        if token != "delegation":
            raise ValueError("invalid delegation")
        return SimpleNamespace(
            sub=self.sub,
            wsp="wsp_test",
            prj="prj_test",
            ops=self.ops,
            job="job_test",
            rev="a" * 40,
            exp=int(time.time()) + 60,
        )


class _Sink:
    def __init__(self):
        self.records = []

    def __call__(self, record):
        self.records.append(record)

    def events(self):
        return [r for r in self.records if r["kind"] == "session_event"]


def _headers(display="Ada L."):
    return {
        "Authorization": "Bearer delegation",
        "Content-Type": "application/json",
        "X-Annealage-Principal-Display": display,
    }


def _body(res):
    return json.loads(res.body.decode("utf-8"))


class _App:
    def __init__(self, app, sink, sessions):
        self.app = app
        self.client = make_test_client(app)
        self.sink = sink
        self.sessions = sessions

    async def call(self, method, path, headers=None):
        return await getattr(self.client, method)(path, headers=headers or _headers())


def _build(served_dir, tmp_path_factory, *, verifier=None, sink=True, **kwargs):
    built = []
    sink = _Sink() if sink else None

    def build_session(on_event, *, bus):
        session = FakeSession(on_event, session_id="hosted-session")
        built.append(session)
        return session

    state_dir = tmp_path_factory.mktemp("worker-state")
    sessions.create_session(state_dir, "hosted-session")
    app = create_toy_app(
        served_dir,
        session_id="hosted-session",
        build_session=build_session,
        hosted_mode=True,
        hosted_verifier=verifier or _Verifier(),
        hosted_state_dir=state_dir,
        hosted_tool_ops={"mcp__toy__list_notes": "project.read"},
        settings=settings.resolve(tmp_path_factory.mktemp("service"), flags={"backend": "omp"}),
        state_sink=sink,
        **kwargs,
    )
    return _App(app, sink, built)


@pytest.fixture
def hosted(served_dir, tmp_path_factory):
    return _build(served_dir, tmp_path_factory)


async def test_the_report_names_the_session_without_backend_or_model_detail(hosted):
    res = await hosted.call("get", "/agent/session")
    assert res.status_code == 200
    session = _body(res)["session"]
    assert session["session_id"] == "hosted-session"
    assert session["agent"] == "ready"
    assert session["cancelled"] is False
    assert set(session) == {
        "session_id",
        "agent",
        "cancelled",
        "turn",
        "turn_running",
        "waiting",
        "last_activity",
    }


async def test_each_route_needs_a_valid_delegation_naming_its_operation(
    served_dir, tmp_path_factory
):
    read_only = _build(served_dir, tmp_path_factory, verifier=_Verifier(ops=("project.read",)))
    assert (await read_only.call("get", "/agent/session")).status_code == 200
    assert (await read_only.call("post", "/agent/session")).status_code == 403
    assert (await read_only.call("post", "/agent/session/cancel")).status_code == 403
    assert read_only.sessions[0].closed == 0
    assert read_only.sink.records == []

    anonymous = {"Content-Type": "application/json"}
    for method, path in (("get", "/agent/session"), ("post", "/agent/session")):
        assert (await read_only.call(method, path, headers=anonymous)).status_code == 403


async def test_launching_a_live_session_opens_nothing_and_records_nothing(hosted):
    res = await hosted.call("post", "/agent/session")
    assert res.status_code == 200
    assert _body(res)["launched"] is False
    assert len(hosted.sessions) == 1
    assert hosted.sink.events() == []


async def test_a_cancelled_session_stays_closed_until_launched_again(hosted):
    holder = hosted.app.agent_holder
    res = await hosted.call("post", "/agent/session/cancel")
    body = _body(res)
    assert (res.status_code, body["cancelled"]) == (200, True)
    assert body["session"]["agent"] == "closed" and body["session"]["cancelled"] is True
    assert hosted.sessions[0].closed == 1

    # A page connecting (ensure with retry) does not resurrect it, and a second
    # cancel has nothing to end, so it records nothing.
    assert await holder.ensure(retry=True) is None
    assert len(hosted.sessions) == 1
    assert _body(await hosted.call("post", "/agent/session/cancel"))["cancelled"] is False

    launched = _body(await hosted.call("post", "/agent/session"))
    assert launched["launched"] is True
    assert launched["session"]["agent"] in ("ready", "connecting")
    assert launched["session"]["cancelled"] is False
    assert len(hosted.sessions) == 2 and holder.session is hosted.sessions[1]

    cancel, launch = hosted.sink.events()[0], hosted.sink.events()[1]
    assert (cancel["payload"]["event"], launch["payload"]["event"]) == ("cancel", "launch")


async def test_cancel_ends_a_running_turn_as_interrupted_and_ends_the_tool_grant(hosted):
    app = hosted.app
    app.agent_event_log.append(UserTurn(turn=1, blocks=[{"type": "text", "text": "go"}]))
    app.agent_bus.hosted_turn_live = True
    app.agent_bus.hosted_turn_ops = ("project.read",)
    app.agent_bus.hosted_turn_secret = "turn-secret"

    body = _body(await hosted.call("post", "/agent/session/cancel"))

    assert body["interrupted_turns"] == [1]
    assert body["session"]["turn_running"] is False
    ended = [
        wire for _s, wire in app.agent_event_log.replay(0).events if wire["kind"] == "turn_end"
    ]
    assert [(w["turn"], w["stop_reason"]) for w in ended] == [(1, "interrupted")]
    assert app.agent_bus.hosted_turn_live is False
    assert app.agent_bus.hosted_turn_secret is None
    assert hosted.sink.events()[0]["payload"]["interrupted_turns"] == [1]


async def test_launch_and_cancel_are_recorded_for_the_requesting_principal(hosted):
    await hosted.call("post", "/agent/session/cancel")
    await hosted.call("post", "/agent/session")
    cancel, launch = hosted.sink.events()
    assert cancel["kind"] == "session_event"
    assert cancel["principal_id"] == PRINCIPAL and cancel["display"] == "Ada L."
    assert cancel["payload"] == {
        "session_id": "hosted-session",
        "event": "cancel",
        "interrupted_turns": [],
    }
    assert launch["payload"] == {"session_id": "hosted-session", "event": "launch"}
    assert RECORD_ID.fullmatch(cancel["record_id"])
    assert cancel["record_id"].startswith("session_event:hosted-session:cancel:")
    assert cancel["record_id"] != launch["record_id"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z", cancel["ts"])


async def test_a_backend_failure_and_a_shutdown_are_attributed_to_the_principal_driving_it(hosted):
    await hosted.call("post", "/agent/session")  # names the principal; opens nothing
    hosted.sessions[0].emit(AgentError(stderr="boom secret", remediation="omp stopped; use Retry"))
    await asyncio.sleep(0.05)
    (error,) = hosted.sink.events()
    assert error["payload"] == {
        "session_id": "hosted-session",
        "event": "error",
        "message": "omp stopped; use Retry",
    }
    assert "secret" not in json.dumps(error)
    assert error["principal_id"] == PRINCIPAL

    await hosted.app.agent_stop()
    end = hosted.sink.events()[-1]
    assert end["payload"] == {
        "session_id": "hosted-session",
        "event": "end",
        "reason": "shutdown",
    }


async def test_a_session_event_nobody_caused_without_a_known_principal_is_not_invented(
    hosted, capsys
):
    hosted.sessions[0].emit(AgentError(stderr="", remediation="omp stopped"))
    await asyncio.sleep(0.05)
    assert hosted.sink.events() == []
    assert "no valid principal" in capsys.readouterr().err


async def test_the_control_routes_work_and_record_nothing_without_a_sink(
    served_dir, tmp_path_factory
):
    app = _build(served_dir, tmp_path_factory, sink=False)
    assert (await app.call("post", "/agent/session/cancel")).status_code == 200
    assert (await app.call("post", "/agent/session")).status_code == 200
    assert len(app.sessions) == 2


async def test_a_standalone_app_has_no_session_control_routes(served_dir):
    sessions.create_session(served_dir, "s")
    client = make_test_client(
        create_toy_app(
            served_dir,
            token="tok",
            session_id="s",
            build_session=lambda on_event, *, bus: FakeSession(on_event),
        )
    )
    for method, path in (
        ("get", "/agent/session?t=tok"),
        ("post", "/agent/session?t=tok"),
        ("post", "/agent/session/cancel?t=tok"),
    ):
        assert (await getattr(client, method)(path)).status_code == 404


async def test_a_usage_sink_outside_hosted_mode_is_refused(served_dir):
    with pytest.raises(ValueError, match="usage_sink.*requires hosted mode"):
        create_toy_app(served_dir, token="tok", usage_sink=lambda event: None)
