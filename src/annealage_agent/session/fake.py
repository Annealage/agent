"""FakeSession: a scripted stand-in for AgentSession, with no SDK involved.

This is the first of the plan's two fake layers (section 3.11). It exists
so ``viewers.py`` and ``http/ws.py`` have something to broadcast in every
test, in this milestone and in M5's, without paying for
``claude-agent-sdk``'s 93 MB extra or a real conversation. The second
layer, a fake ``Transport`` under the real SDK client, is what actually
catches an SDK protocol break; this class never claims to.

``FakeSession`` records what is submitted to it and emits events only
when a test tells it to, via ``emit``. It does not attempt to simulate an
agent replying to a turn on its own: a test that wants a canned reply
asserts on ``submitted_turns`` and then calls ``emit`` itself, which
keeps the scripting in the test rather than hidden inside a second,
undocumented turn-behaviour language living in this file.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set, Tuple

from .base import AGENT_READY, AgentEvent, SandboxStatus, UnknownRequest


class FakeSession:
    """Implements the ``AgentSession`` Protocol from ``session/base.py``.

    ``on_event`` is called synchronously for every event ``emit`` is
    given; it carries no async contract of its own; a caller that needs
    the callback to run async work schedules that itself (e.g. by making
    ``on_event`` call ``loop.call_soon`` or an ``asyncio.Queue.put_nowait``,
    neither of which this class needs to know about).
    """

    def __init__(
        self,
        on_event: Callable[[AgentEvent], None],
        *,
        session_id: str = "fake-session",
        sdk_session_id: Optional[str] = None,
        cwd: str = ".",
    ):
        self.session_id = session_id
        self.sdk_session_id = sdk_session_id
        self.cwd = cwd
        self._on_event = on_event
        self._status = AGENT_READY

        # Recorded for tests to assert against; this class is a recorder
        # first and a scripting surface second.
        self.submitted_turns: List[Tuple[list, Optional[str]]] = []
        self.permission_decisions: List[Tuple[str, str, str]] = []
        self.decided_requests: Set[str] = set()
        self.interrupted = 0
        self.set_model_calls: List[str] = []
        self.started = 0
        self.closed = 0
        self.viewer_counts: List[int] = []
        # What backend_logs() lists; a test sets it to BackendLog entries.
        self.logs: list = []

    def agent_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        """Test hook: force ``agent_status()`` to a given value, for the
        connecting/unavailable states M4 has no real agent to produce."""
        self._status = status

    def emit(self, event: AgentEvent) -> None:
        """Push one scripted event through this session's sink, as if the
        agent this class stands in for had produced it on its own."""
        self._on_event(event)

    async def submit_turn(self, blocks: list, viewer: Optional[str] = None) -> None:
        self.submitted_turns.append((blocks, viewer))

    async def decide_permission(self, request_id: str, decision: str, message: str = "") -> None:
        """Record one decision, and raise ``UnknownRequest`` for a second one
        naming the same request.

        The repeat is modelled rather than exposed as a flag because it is the
        real broker's behaviour and the case worth testing: two views hold one
        card, both are clicked, and the second click decides nothing. A test
        that had to opt into that by setting a flag could pass while the
        ordinary path silently never raised at all.
        """
        if request_id in self.decided_requests:
            raise UnknownRequest("permission request %r was already decided" % (request_id,))
        self.decided_requests.add(request_id)
        self.permission_decisions.append((request_id, decision, message))

    async def interrupt(self) -> None:
        self.interrupted += 1

    async def set_model(self, model: str) -> None:
        self.set_model_calls.append(model)

    # -- lifecycle -----------------------------------------------------------
    # Recorded rather than merely accepted, so a test can assert that whatever
    # wired this session up actually drove it, and present at all because
    # ``app.serve`` calls start and close on any session it is given.

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    def on_viewer_presence(self, count: int) -> None:
        self.viewer_counts.append(count)

    def sandbox_status(self) -> SandboxStatus:
        """No shell, so nothing to contain and nothing requested. Reported as a
        real answer rather than omitted, so the banner has one shape to render
        for every session instead of a present case and an absent one."""
        return SandboxStatus(requested=False, active=False, missing=())

    def backend_logs(self) -> list:
        """Whatever a test put in ``logs``: a fake has no backend of its own."""
        return list(self.logs)
