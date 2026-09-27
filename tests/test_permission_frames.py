"""Tests for how ``http/ws.py`` answers a ``permission`` frame that decides nothing.

Two browser views can hold the same permission card, and both can be clicked.
One of them decides the request; the other's frame arrives for a request that is
already resolved. The property under test is that the loser is told, because a
swallowed refusal renders a human's Deny on an already-allowed request
identically to a Deny that took effect, and that is the one outcome this path
must never produce.

``_dispatch`` is called directly rather than through a real socket: the frame
has already been validated by the time it gets there, so a socket would add a
handshake and a protocol version exchange without adding anything to the
question of what the layer does with the exception.
"""

import json

import pytest

from annealage_agent.http import ws as ws_module
from annealage_agent.session.base import UnknownRequest
from annealage_agent.session.fake import FakeSession

pytestmark = pytest.mark.asyncio


class RecordingSocket:
    """Collects the frames ``_dispatch`` writes back to one connection."""

    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class StubRegistry:
    """Only what ``_dispatch`` reaches for on this path."""

    def __init__(self):
        self.touched = 0

    async def touch(self, conn):
        self.touched += 1


class _Conn:
    tab_id = "tab-1"
    human = None


def _frame(request_id="pr_1", decision="deny", message="not that file"):
    return {
        "v": 1,
        "type": "permission",
        "request_id": request_id,
        "decision": decision,
        "message": message,
    }


async def _dispatch(session, frame):
    sock = RecordingSocket()
    await ws_module._dispatch(sock, _Conn(), StubRegistry(), None, "tok", frame, session)
    return sock.sent


async def test_a_decision_that_lands_is_not_answered_with_a_refusal():
    session = FakeSession(lambda event: None)
    sent = await _dispatch(session, _frame())
    assert sent == []
    assert session.permission_decisions == [("pr_1", "deny", "not that file")]


async def test_a_second_decision_for_one_request_is_told_it_decided_nothing():
    session = FakeSession(lambda event: None)
    await _dispatch(session, _frame(decision="allow"))
    sent = await _dispatch(session, _frame(decision="deny"))

    assert len(sent) == 1
    assert sent[0]["type"] == "refused"
    reason = sent[0]["reason"]
    assert "already decided" in reason
    assert "not applied" in reason
    # The losing frame changed nothing, so only the first decision was recorded.
    assert session.permission_decisions == [("pr_1", "allow", "not that file")]


async def test_the_refusal_names_the_cause_rather_than_a_generic_failure():
    """The generic arm exists for a session that broke, and says to go and read
    the server output. Sending that for two tabs racing would describe an
    ordinary interaction as a fault and send the human somewhere useless."""
    session = FakeSession(lambda event: None)
    await _dispatch(session, _frame())
    sent = await _dispatch(session, _frame())

    reason = sent[0]["reason"]
    assert "server output" not in reason
    assert "could not handle" not in reason


async def test_a_session_that_actually_breaks_still_gets_the_generic_answer(capsys):
    class _BrokenSession(FakeSession):
        async def decide_permission(self, request_id, decision, message="", by=None):
            raise RuntimeError("broker exploded")

    sent = await _dispatch(_BrokenSession(lambda event: None), _frame())

    assert sent[0]["type"] == "refused"
    assert "could not handle" in sent[0]["reason"]
    assert "broker exploded" in capsys.readouterr().err


async def test_viewer_only_mode_answers_a_permission_frame_rather_than_dropping_it():
    sent = await _dispatch(None, _frame())
    assert sent[0]["type"] == "refused"
    assert "viewer-only" in sent[0]["reason"]


async def test_unknown_request_is_importable_without_the_agent_sdk():
    """``ws.py`` has to catch this exception and must keep working with no SDK
    installed, so the exception cannot live in the broker's module."""
    import annealage_agent.session.base as base

    assert base.UnknownRequest is UnknownRequest
    source = base.__file__ or ""
    assert source.endswith("base.py")
    import inspect

    assert "claude_agent_sdk" not in inspect.getsource(base)
