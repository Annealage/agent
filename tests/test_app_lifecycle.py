"""One app's own lifecycle (``app.AgentHolder``), which a front door serving
many apps relies on each app to run by itself.

An app idle past its timeout, with no page open, no turn running and nothing
waiting on the human, closes its session and that session's broker; a page
connecting starts a new session from ``resume_session`` and everything that
needs one (``/mcp`` among them) follows it, since a closed broker denies for
good. ``app.agent_status()`` is what a front door's list shows, and moves as
the conversation does. ``serve`` drives an app through ``agent_start`` and
``agent_stop``, whose ``agent_on_stop`` is the product's own teardown.
"""

import asyncio
import contextlib
import json
import socket

import pytest
from conftest import TEST_HOST, create_toy_app, make_test_client
from toy_product import NOTES_FILE

from annealage_agent import app as agent_app
from annealage_agent import sessions
from annealage_agent.session.base import AGENT_READY, AgentModelChanged, TurnEnd, UserTurn
from annealage_agent.session.external import ExternalAgentSession
from annealage_agent.session.permissions import PermissionBroker

pytestmark = pytest.mark.asyncio

BROWSER_TOKEN = "lifecycle-browser-token"
AGENT_TOKEN = "lifecycle-agent-token"
IDLE = 0.2


class _Session(ExternalAgentSession):
    """A ready session whose approvals go through a real broker, which its
    ``close`` shuts down for good as every backend's does."""

    def __init__(self, on_event, broker, resumed):
        super().__init__(on_event, broker)
        self.broker = broker
        self.resumed = resumed
        self.emit = on_event
        self.started = 0
        self.closed = 0
        self.models = []

    def agent_status(self):
        return AGENT_READY

    async def set_model(self, model):
        self.models.append(model)
        self.emit(AgentModelChanged(model=model))

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1
        await super().close()


class _SlowClose(_Session):
    """Its close waits for ``gate``, as a backend's does for its child."""

    gate = None

    async def close(self):
        await self.gate.wait()
        await super().close()


def _factory(built, *, resumed, cls=_Session):
    def build(on_event, *, bus):
        broker = PermissionBroker(on_event, viewer_url=bus.url, timeout=30)
        bus.broker = broker
        built.append(cls(on_event, broker, resumed))
        return built[-1]

    return build


def _app(served_dir, built, cls=_Session, **kwargs):
    kwargs.setdefault("resume_session", _factory(built, resumed=True, cls=cls))
    kwargs.setdefault("idle_timeout", IDLE)
    return create_toy_app(
        served_dir,
        token=BROWSER_TOKEN,
        agent_token=AGENT_TOKEN,
        session_id=sessions.create_session(served_dir),
        build_session=_factory(built, resumed=False, cls=cls),
        **kwargs,
    )


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


def _logged(app, kind):
    return [wire for _seq, wire in app.agent_event_log.replay(0).events if wire["kind"] == kind]


class _Socket:
    """What the registry writes a connected page's frames to."""

    closed = False

    async def send(self, data, opcode=None):
        pass


def _ws_headers():
    return {
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
    }


async def _mcp_call(client, name, arguments):
    res = await client.post(
        "/mcp?t=%s" % AGENT_TOKEN,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"method": "tools/call", "params": {"name": name, "arguments": arguments}}),
    )
    return json.loads(res.body.decode("utf-8"))["result"]


async def test_an_idle_app_closes_and_a_page_connecting_resumes_a_new_session(served_dir):
    built = []
    app = _app(served_dir, built)
    await app.agent_start()
    try:
        # A connected page keeps the app open however long it sits there.
        conn = await app.agent_registry.add(_Socket())
        await asyncio.sleep(IDLE * 3)
        assert app.agent_status()["agent"] == AGENT_READY
        await app.agent_registry.remove(conn)

        await _until(lambda: app.agent_status()["agent"] == agent_app.AGENT_CLOSED)
        first = built[0]
        assert first.closed == 1 and app.agent_session is None
        denied = await first.broker.ask("mcp__toy__add_note", {"text": "late"}, None)
        assert not denied.allow and "shutting down" in denied.message

        # A page connecting starts the next session, resumed; the page stays,
        # so nothing below races the idle timer.
        client = make_test_client(app)
        await client.get("/ws?t=%s" % BROWSER_TOKEN, headers=_ws_headers())
        await app.agent_registry.add(_Socket())
        second = app.agent_session
        assert [session.resumed for session in built] == [False, True]
        assert second is built[1] and app.agent_bus.broker is second.broker
        await _until(lambda: second.started == 1)

        # A write through /mcp now asks through the new session's broker, and
        # the new session's answer decides it.
        call = asyncio.ensure_future(_mcp_call(client, "add_note", {"text": "after"}))
        await _until(lambda: _logged(app, "permission_request"))
        request_id = _logged(app, "permission_request")[-1]["request_id"]
        assert app.agent_status()["waiting"] is True
        await second.decide_permission(request_id, "allow")
        assert not (await asyncio.wait_for(call, 5)).get("isError")
        notes = json.loads((served_dir / NOTES_FILE).read_text(encoding="utf-8"))
        assert notes == ["first", "after"]
    finally:
        await app.agent_stop()


async def test_an_app_with_nothing_to_resume_from_never_closes_for_idle(served_dir):
    built = []
    app = _app(served_dir, built, resume_session=None, idle_timeout=0.05)
    await app.agent_start()
    try:
        await asyncio.sleep(0.3)
        assert app.agent_status()["agent"] == AGENT_READY and built[0].closed == 0
    finally:
        await app.agent_stop()


async def test_a_resumed_session_comes_back_on_the_model_the_human_switched_to(served_dir):
    """Every factory builds from the run's settings, so the switch the human
    made before the app closed is made again on the resumed session, and the
    page is told so by the same event a switch from the page publishes."""
    built = []
    app = _app(served_dir, built)
    await app.agent_start()
    try:
        await built[0].set_model("claude-haiku-5")
        await _until(lambda: app.agent_status()["agent"] == agent_app.AGENT_CLOSED)
        before_resume = len(_logged(app, "agent_model_changed"))
        client = make_test_client(app)
        await client.get("/ws?t=%s" % BROWSER_TOKEN, headers=_ws_headers())
        await app.agent_registry.add(_Socket())
        await _until(lambda: len(built) == 2 and built[1].models == ["claude-haiku-5"])
        published = _logged(app, "agent_model_changed")[before_resume:]
        assert [event["model"] for event in published] == ["claude-haiku-5"]
    finally:
        await app.agent_stop()


@pytest.mark.parametrize("idle_timeout", [0, -1.0])
async def test_an_idle_timeout_that_is_not_positive_is_refused(served_dir, idle_timeout):
    with pytest.raises(ValueError, match="idle_timeout"):
        _app(served_dir, [], idle_timeout=idle_timeout)


async def test_a_stop_during_an_idle_close_finishes_that_close(served_dir):
    """The sweep closing the session is cancelled by the stop; the close it
    started (a backend's child going down) still completes, once."""
    built = []
    _SlowClose.gate = asyncio.Event()
    app = _app(served_dir, built, cls=_SlowClose)
    await app.agent_start()
    await _until(lambda: app.agent_status()["agent"] == agent_app.AGENT_CLOSED)
    stopping = asyncio.ensure_future(app.agent_stop())
    await asyncio.sleep(0.05)
    assert not stopping.done() and built[0].closed == 0
    _SlowClose.gate.set()
    await asyncio.wait_for(stopping, 5)
    assert built[0].closed == 1


async def test_a_cancelled_stop_still_closes_and_runs_the_teardown_hooks(served_dir):
    """A caller cancelled while the app stops (a front door interrupted
    twice) does not cut the teardown short: the session still closes, the
    product's hooks (its lock release) still run, and a second stop waits for
    that same teardown."""
    built, released = [], []
    _SlowClose.gate = asyncio.Event()
    app = _app(served_dir, built, cls=_SlowClose, idle_timeout=None)
    app.agent_on_stop.append(lambda: released.append(built[0].closed))
    await app.agent_start()
    stopping = asyncio.ensure_future(app.agent_stop())
    await asyncio.sleep(0.05)
    stopping.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await stopping
    assert released == []
    _SlowClose.gate.set()
    await asyncio.wait_for(app.agent_stop(), 5)
    assert built[0].closed == 1 and released == [1]


async def test_the_status_follows_a_turn_a_permission_and_an_idle_close(served_dir):
    built = []
    app = _app(served_dir, built)
    seen = []
    app.agent_status_listeners.append(
        lambda: seen.append(
            tuple(app.agent_status()[key] for key in ("agent", "turn_running", "waiting"))
        )
    )
    assert app.agent_status() == {
        "agent": AGENT_READY,
        "turn_running": False,
        "waiting": False,
        "attention": None,
        "last_activity": app.agent_status()["last_activity"],
        "viewers": 0,
    }
    await app.agent_start()
    try:
        session = built[0]
        session.on_viewer_presence(1)
        session.emit(UserTurn(turn=1, blocks=[{"type": "text", "text": "add a note"}]))
        asking = asyncio.ensure_future(session.broker.ask("mcp__toy__add_note", {}, None))
        await _until(lambda: _logged(app, "permission_request"))
        request_id = _logged(app, "permission_request")[0]["request_id"]
        await session.decide_permission(request_id, "allow")
        assert (await asking).allow
        session.emit(TurnEnd(turn=1, stop_reason="end_turn", cost_usd=0.0))
        # The agent stops to ask the human; that is waiting too, and stays so
        # after the idle timer closes the session, until the human answers.
        app.agent_bus.attention("Checkpoint: circuit", "Is the power tree right?")
        assert app.agent_status()["attention"] == "Checkpoint: circuit: Is the power tree right?"
        await _until(lambda: app.agent_status()["agent"] == agent_app.AGENT_CLOSED)
    finally:
        await app.agent_stop()

    changes = [state for i, state in enumerate(seen) if i == 0 or seen[i - 1] != state]
    assert changes == [
        ("ready", True, False),
        ("ready", True, True),
        ("ready", True, False),
        ("ready", False, False),
        ("ready", False, True),
        ("closed", False, True),
    ]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_serve_stops_the_app_and_runs_its_teardown_hooks(served_dir, capsys):
    port = _free_port()
    built, order = [], []
    app = _app(served_dir, built, host=TEST_HOST, port=port)

    def failing():
        order.append("failing")
        raise RuntimeError("settle failed")

    async def release():
        order.append("release (session closed %d)" % built[0].closed)

    app.agent_on_stop.extend([failing, release])
    ready = asyncio.Event()
    task = asyncio.ensure_future(agent_app.serve(app, TEST_HOST, port, on_ready=ready.set))
    await asyncio.wait_for(ready.wait(), 5)
    assert built[0].started == 1
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)

    # One hook failing does not keep the next from running, and both run
    # after the session is closed.
    assert order == ["failing", "release (session closed 1)"]
    assert "settle failed" in capsys.readouterr().err
    assert app.agent_status()["agent"] == agent_app.AGENT_CLOSED
    await app.agent_stop()
    assert built[0].closed == 1 and order == ["failing", "release (session closed 1)"]
