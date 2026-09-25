"""Tests for how ``http/ws.py`` dispatches a ``set_model`` frame to a session.

The regression this guards: ``http/ws.py``'s ``set_model`` branch
unconditionally calls ``session.set_model(...)``, but ``FakeSession`` -- the
``AgentSession`` stand-in used by ``app.py``/``ws.py`` tests, and by a
product's browser end-to-end harness (Annealage Mesh's ``test_viewer_e2e.py``)
-- had no ``set_model``
method at all. Every such call raised ``AttributeError``, which ``_dispatch``'s
generic exception arm swallows into an ordinary-looking ``refused`` frame:
sending a real ``set_model`` frame against any app built with ``FakeSession``
looked exactly like a legitimate refusal instead of a wiring bug.

``_dispatch`` is called directly rather than through a real socket, matching
``test_permission_frames.py``'s reasoning: the frame is already validated by
the time it reaches ``_dispatch``, so a socket would add a handshake without
adding anything to the question of what this layer does with the frame.
"""

import json

import pytest

from annealage_agent.http import ws as ws_module
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


def _frame(model="claude-haiku-5"):
    return {"v": 1, "type": "set_model", "model": model}


async def _dispatch(session, frame):
    sock = RecordingSocket()
    await ws_module._dispatch(sock, _Conn(), StubRegistry(), None, "tok", frame, session)
    return sock.sent


async def test_a_set_model_frame_is_recorded_rather_than_refused_with_an_attributeerror():
    session = FakeSession(lambda event: None)
    sent = await _dispatch(session, _frame())
    assert sent == []
    assert session.set_model_calls == ["claude-haiku-5"]


async def test_viewer_only_mode_answers_a_set_model_frame_rather_than_dropping_it():
    sent = await _dispatch(None, _frame())
    assert sent[0]["type"] == "refused"
    assert "viewer-only" in sent[0]["reason"]


async def test_a_session_that_actually_breaks_on_set_model_gets_the_generic_answer(capsys):
    class _BrokenSession(FakeSession):
        async def set_model(self, model):
            raise RuntimeError("model switch exploded")

    session = _BrokenSession(lambda event: None)
    sent = await _dispatch(session, _frame())
    assert sent[0]["type"] == "refused"
    assert "could not handle" in sent[0]["reason"]
    assert "model switch exploded" in capsys.readouterr().err
