"""Tests for ``session/permissions.py``: the ``can_use_tool`` broker.

Every test asserts on the ``Decision`` object ``ask``/``decide`` produce,
never on an internal predicate such as ``_granted_tools`` or
``_viewer_count`` in isolation: a predicate can be correct while the code
path that is supposed to consult it is wired wrong, and ``Decision`` is the
one provider-neutral contract every backend's own session adapter converts
into whatever its SDK requires (``session/sdk.py``'s ``_to_claude_result``
and ``_session_rule``, for the Claude SDK, have their own dedicated
coverage in ``tests/test_sdk_session.py``), so it is the only contract
worth pinning here.

Nothing in this file imports ``claude_agent_sdk``: that is itself part of
what this module's decoupling from the Claude SDK buys, and a stray import
creeping back in here would be exactly the regression to catch.

Each test builds its own broker via ``_broker()``, which records every
emitted event in a plain list rather than wiring a real ``EventLog`` or
``ViewerRegistry``, matching ``session/base.py``'s own description of
``on_event`` as a call-and-return callback with no reference back into the
broker. A request's id is read off the recorded ``PermissionRequest``
event, never assumed from the allocator's own counting scheme, so a test
does not silently start pinning ``"pr_1"`` as part of the contract.

Nothing here sleeps out a real timeout: the timeout path uses a broker
built with a timeout of a few hundredths of a second, so ``asyncio.wait_for``
itself is what elapses, not a test-authored delay racing against it.
"""

import asyncio

import pytest

from annealage_agent.session.base import PermissionRequest, PermissionResolved
from annealage_agent.session.permissions import (
    DEFAULT_DENY_MESSAGE,
    NEVER_REMEMBERED,
    Decision,
    PermissionBroker,
    UnknownRequest,
)

pytestmark = pytest.mark.asyncio


def _broker(**kwargs):
    """A broker recording every emitted event in ``events``, returned
    alongside it so a test can read a request's id off the event the
    broker itself produced rather than assuming an id format."""
    events = []
    broker = PermissionBroker(events.append, **kwargs)
    return broker, events


async def _pending_request(broker, events, tool_name="Edit", input_data=None):
    """Start ``ask`` as a background task, yield once so it reaches its
    own ``await`` inside ``asyncio.wait_for``, and return the task plus the
    ``PermissionRequest`` it emitted. Requires a viewer already connected
    and the tool not already granted, or ``ask`` would resolve immediately
    and there would be nothing pending to return."""
    before = len(events)
    task = asyncio.ensure_future(broker.ask(tool_name, input_data or {}, None))
    await asyncio.sleep(0)
    assert not task.done()
    assert len(events) == before + 1
    request = events[-1]
    assert isinstance(request, PermissionRequest)
    assert request.tool == tool_name
    return task, request


def _assert_allow(result):
    assert isinstance(result, Decision)
    assert result.allow is True


def _assert_deny(result):
    assert isinstance(result, Decision)
    assert result.allow is False


def _assert_remembers(result, tool_name):
    """A remembered grant's ``Decision`` names the tool a backend's own
    remembered-grant mechanism (the Claude SDK's session-scoped
    ``PermissionUpdate``, mapped by ``session/sdk.py``'s
    ``_to_claude_result``; other backends' own, different mechanisms) should
    also cover for the rest of the running session."""
    _assert_allow(result)
    assert result.remember_tool == tool_name


# ---------------------------------------------------------------------------
# decide(): allow, deny
# ---------------------------------------------------------------------------


async def test_decide_allow_resolves_ask_to_an_allow_result():
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    await broker.decide(request.request_id, "allow")
    result = await task

    _assert_allow(result)
    assert result.remember_tool is None


async def test_decide_deny_carries_the_given_message_to_the_model():
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    await broker.decide(request.request_id, "deny", "not touching that file")
    result = await task

    _assert_deny(result)
    assert result.message == "not touching that file"


async def test_decide_deny_with_no_message_falls_back_to_a_default():
    """The pane's deny button may be clicked with the reason field left
    blank; the model must still receive an explanation, per fact 2 of Annealage
    Mesh's agent-chat plan, rather
    than an empty string reading as a fresh kind of infrastructure error."""
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    await broker.decide(request.request_id, "deny")
    result = await task

    _assert_deny(result)
    assert result.message == DEFAULT_DENY_MESSAGE
    assert result.message != ""


# ---------------------------------------------------------------------------
# allow_always: persistence and the Bash carve-out
# ---------------------------------------------------------------------------


async def test_allow_always_persists_and_grants_a_later_request_without_asking(tmp_path):
    permissions_path = tmp_path / ".toy" / "permissions.toml"
    broker, events = _broker(permissions_path=permissions_path)
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    await broker.decide(request.request_id, "allow_always")
    result = await task

    _assert_remembers(result, "Write")
    assert permissions_path.exists()
    assert "Write" in permissions_path.read_text(encoding="utf-8")
    # The header names the installed product, not a hard-coded one.
    assert "annealage-toy" in permissions_path.read_text(encoding="utf-8")

    # Same broker, same process: a later request for the same tool is
    # granted from the in-memory set with no viewer needed at all, since
    # the granted-tool check in ``ask`` runs before the no-viewer check.
    broker.viewer_disconnected()
    second = await broker.ask("Write", {}, None)
    _assert_remembers(second, "Write")
    # A grant answered from memory opens no request, so it announces neither a
    # request nor a resolution: the only pair on the wire is the first one.
    assert [type(e).__name__ for e in events] == ["PermissionRequest", "PermissionResolved"]

    # A fresh broker over the same file, simulating a process restart,
    # loads the grant from disk and grants it without ever creating a
    # request either.
    reopened, reopened_events = _broker(permissions_path=permissions_path)
    third = await reopened.ask("Write", {}, None)
    _assert_remembers(third, "Write")
    assert reopened_events == []


async def test_allow_always_for_bash_is_downgraded_and_never_persisted(tmp_path, capsys):
    assert "Bash" in NEVER_REMEMBERED
    permissions_path = tmp_path / ".toy" / "permissions.toml"
    broker, events = _broker(permissions_path=permissions_path)
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Bash")

    await broker.decide(request.request_id, "allow_always")
    result = await task

    _assert_allow(result)
    assert result.remember_tool is None
    assert not permissions_path.exists()
    assert "downgraded" in capsys.readouterr().err

    # Not remembered even in this process: a second Bash call on the same
    # broker still has to go through a full request, proving the earlier
    # click bought exactly one approved call and nothing standing.
    second_task, second_request = await _pending_request(broker, events, tool_name="Bash")
    await broker.decide(second_request.request_id, "deny", "no")
    second_result = await second_task
    _assert_deny(second_result)
    assert second_result.message == "no"


# ---------------------------------------------------------------------------
# viewer presence
# ---------------------------------------------------------------------------


async def test_no_viewer_connected_denies_without_ever_creating_a_request():
    broker, events = _broker(viewer_url="http://127.0.0.1:8765/?t=abc")
    result = await broker.ask("Write", {}, None)

    _assert_deny(result)
    assert "http://127.0.0.1:8765/?t=abc" in result.message
    assert events == []
    assert broker.pending_requests() == []


async def test_last_viewer_disconnecting_denies_every_outstanding_request():
    # A tiny grace rather than the real one: the denial is what this asserts,
    # and waiting out eight real seconds to see it would make the suite slow
    # without making the assertion stronger.
    broker, events = _broker(no_viewer_grace=0.01)
    broker.viewer_connected()
    broker.viewer_connected()  # a second tab, so the count does not hit zero yet
    task, request = await _pending_request(broker, events, tool_name="Write")

    broker.viewer_disconnected()
    await asyncio.sleep(0)
    assert not task.done()  # one viewer (the second tab) is still connected

    broker.viewer_disconnected()
    result = await task

    _assert_deny(result)
    assert "disconnected" in result.message


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


async def test_timeout_denies_when_no_decision_ever_arrives():
    broker, events = _broker(timeout=0.02)
    broker.viewer_connected()

    result = await broker.ask("Write", {}, None)

    _assert_deny(result)
    assert "0.02" in result.message
    assert broker.pending_requests() == []


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


async def test_shutdown_denies_a_request_already_in_flight():
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    broker.shutdown()
    result = await task

    _assert_deny(result)
    assert "shutting down" in result.message


async def test_shutdown_latches_every_later_ask_to_deny_immediately():
    broker, events = _broker()
    broker.viewer_connected()
    broker.shutdown()

    result = await broker.ask("Write", {}, None)

    _assert_deny(result)
    assert "shutting down" in result.message
    assert events == []  # never got as far as emitting a permission_request

    # Idempotent: a second shutdown with nothing pending changes nothing
    # observable about the next ask() either.
    broker.shutdown()
    result_again = await broker.ask("Write", {}, None)
    _assert_deny(result_again)


# ---------------------------------------------------------------------------
# decide() on an id that is not answerable
# ---------------------------------------------------------------------------


async def test_decide_on_an_unknown_id_raises():
    broker, _events = _broker()

    with pytest.raises(UnknownRequest):
        await broker.decide("pr_no_such_request", "allow")


async def test_decide_on_an_id_already_answered_raises_and_the_first_answer_stands():
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    await broker.decide(request.request_id, "allow")
    result = await task
    _assert_allow(result)

    with pytest.raises(UnknownRequest):
        await broker.decide(request.request_id, "deny", "too late")

    # The already-delivered result is untouched by the losing decide().
    _assert_allow(result)


async def test_two_concurrent_decides_on_one_id_leave_exactly_one_winner():
    """Two tabs holding the same permission card and both clicking is the
    race the module docstring names directly: scheduled as two tasks
    rather than two sequential ``await``s, so the event loop, not test
    ordering, decides who runs first."""
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    first = asyncio.ensure_future(broker.decide(request.request_id, "allow"))
    second = asyncio.ensure_future(broker.decide(request.request_id, "deny", "too late"))
    outcomes = await asyncio.gather(first, second, return_exceptions=True)

    successes = [o for o in outcomes if o is None]
    failures = [o for o in outcomes if isinstance(o, UnknownRequest)]
    assert len(successes) == 1
    assert len(failures) == 1

    result = await task
    _assert_allow(result)


async def test_replayed_request_is_pending_until_answered_then_not_answerable_twice():
    """``pending_requests()`` is what a reconnecting viewer replays a card
    from (Annealage Mesh's plan section 3.4). Two tabs shown the same replayed card must
    not both be able to answer it: the first ``decide`` wins, and the
    second is told, via ``UnknownRequest``, that it already lost."""
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events, tool_name="Write")

    assert broker.pending_requests() == [request]

    await broker.decide(request.request_id, "allow")
    await task

    assert broker.pending_requests() == []
    with pytest.raises(UnknownRequest):
        await broker.decide(request.request_id, "allow")


# ---------------------------------------------------------------------------
# on_event failing
# ---------------------------------------------------------------------------


async def test_on_event_raising_denies_and_leaves_nothing_pending(capsys):
    def _broken_on_event(event):
        raise RuntimeError("broadcast exploded")

    broker = PermissionBroker(_broken_on_event)
    broker.viewer_connected()

    result = await broker.ask("Write", {}, None)

    _assert_deny(result)
    assert broker.pending_requests() == []
    assert "broadcast exploded" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# permission_resolved: every request ends in exactly one announcement
# ---------------------------------------------------------------------------


def _resolutions(events):
    return [e for e in events if isinstance(e, PermissionResolved)]


async def test_a_decision_announces_the_resolution_with_its_outcome():
    """Without this event a browser that did not send the deciding frame has
    no way to learn the card is answered."""
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events)

    await broker.decide(request.request_id, "allow")
    await task

    assert [(r.request_id, r.outcome) for r in _resolutions(events)] == [
        (request.request_id, "allow")
    ]


@pytest.mark.parametrize("decision", ["allow", "allow_always", "deny"])
async def test_the_outcome_names_the_decision_that_applied(decision):
    """A pane that sent a different decision has to be able to tell that its
    own did not apply, which needs the outcome rather than a bare 'resolved'."""
    broker, events = _broker()
    broker.viewer_connected()
    task, request = await _pending_request(broker, events)

    await broker.decide(request.request_id, decision)
    await task

    assert _resolutions(events)[0].outcome == decision


async def test_a_timeout_announces_itself_as_a_timeout():
    broker, events = _broker(timeout=0.01)
    broker.viewer_connected()
    task, request = await _pending_request(broker, events)

    result = await task

    _assert_deny(result)
    assert _resolutions(events)[0].outcome == "timeout"


async def test_the_last_viewer_leaving_announces_no_viewer():
    broker, events = _broker(no_viewer_grace=0.01)
    broker.viewer_connected()
    task, _request = await _pending_request(broker, events)

    broker.viewer_disconnected()
    await task

    assert _resolutions(events)[0].outcome == "no_viewer"


async def test_shutdown_announces_shutdown():
    broker, events = _broker()
    broker.viewer_connected()
    task, _request = await _pending_request(broker, events)

    broker.shutdown()
    await task

    assert _resolutions(events)[0].outcome == "shutdown"


async def test_exactly_one_resolution_per_request_however_it_ended():
    """The announcement is emitted from the one place every resolution passes
    through, so no path can emit it twice and none can skip it."""
    broker, events = _broker()
    broker.viewer_connected()
    first_task, first = await _pending_request(broker, events, tool_name="Edit")
    second_task, _second = await _pending_request(broker, events, tool_name="Write")

    await broker.decide(first.request_id, "deny")
    await first_task
    broker.shutdown()
    await second_task

    assert len(_resolutions(events)) == 2
    assert len({r.request_id for r in _resolutions(events)}) == 2


async def test_the_resolution_is_announced_after_the_request_stops_being_pending():
    """A viewer that reconnects on the strength of this event and asks what is
    still outstanding must not be told about the request it just closed."""
    seen = []
    broker = PermissionBroker(
        lambda event: seen.append((event, [r.request_id for r in broker.pending_requests()]))
    )
    broker.viewer_connected()
    task = asyncio.ensure_future(broker.ask("Edit", {}, None))
    await asyncio.sleep(0)
    request_id = seen[0][0].request_id

    await broker.decide(request_id, "allow")
    await task

    resolution = [entry for entry in seen if isinstance(entry[0], PermissionResolved)][0]
    assert resolution[1] == []


async def test_a_failure_to_announce_does_not_break_the_decision(capsys):
    """The tool call is already answered by then, so a broadcast failure costs
    a card left on screen until the next reload, not a wrong decision."""
    events = []

    def _on_event(event):
        events.append(event)
        if isinstance(event, PermissionResolved):
            raise RuntimeError("broadcast exploded")

    broker = PermissionBroker(_on_event)
    broker.viewer_connected()
    task = asyncio.ensure_future(broker.ask("Edit", {}, None))
    await asyncio.sleep(0)

    await broker.decide(events[0].request_id, "allow")
    result = await task

    _assert_allow(result)
    assert broker.pending_requests() == []
    assert "broadcast exploded" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A reload is a disconnect followed by a connect
# ---------------------------------------------------------------------------


async def test_a_reload_keeps_an_outstanding_request_waiting():
    """The case the grace period exists for. With one browser open, which is the
    ordinary case, denying on the transition to zero viewers meant reloading the
    page cancelled whatever was being asked: the human came back to a pane with
    no card, and the model to a refusal it had done nothing to earn.
    """
    broker, events = _broker(no_viewer_grace=0.2)
    broker.viewer_connected()
    task, request = await _pending_request(broker, events)

    broker.viewer_disconnected()  # the page unloads
    await asyncio.sleep(0.05)
    assert not task.done(), "the request must survive the gap a reload leaves"

    broker.viewer_connected()  # the page comes back
    await asyncio.sleep(0.3)  # past what would have been the deadline
    assert not task.done(), "a viewer came back, so nothing should have expired"
    assert [r.request_id for r in broker.pending_requests()] == [request.request_id]

    # And it is still answerable, which is the whole point.
    await broker.decide(request.request_id, "allow")
    _assert_allow(await task)
    assert _resolutions(events)[0].outcome == "allow"


async def test_a_viewer_that_never_returns_is_denied_after_the_grace_period():
    broker, events = _broker(no_viewer_grace=0.05)
    broker.viewer_connected()
    task, _request = await _pending_request(broker, events)

    broker.viewer_disconnected()
    result = await task

    _assert_deny(result)
    assert "disconnected" in result.message
    assert _resolutions(events)[0].outcome == "no_viewer"


async def test_shutdown_during_the_grace_period_denies_at_once():
    """Nothing is coming back, so there is nothing left to wait for."""
    broker, events = _broker(no_viewer_grace=30.0)
    broker.viewer_connected()
    task, _request = await _pending_request(broker, events)

    broker.viewer_disconnected()
    broker.shutdown()
    result = await task

    _assert_deny(result)
    assert _resolutions(events)[0].outcome == "shutdown"


async def test_a_request_arriving_with_no_viewer_is_still_denied_immediately():
    """The grace period covers a request that already exists when the viewer
    goes; it does not make the broker hold a brand new one for a browser that is
    not there, which would leave the model waiting on nobody."""
    broker, _events = _broker(no_viewer_grace=30.0)
    result = await broker.ask("Write", {}, None)
    _assert_deny(result)
    assert "no browser viewer is connected" in result.message
    assert "open the toy viewer" in result.message
