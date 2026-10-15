"""``ExternalAgentSession``: the session of a run with no agent of its own,
whose tools an agent in another process calls through ``/mcp``.

A product whose tools a separately running agent may use (another Claude Code
session, attached through the stdio bridge,
``session/codex_mcp_stdio_bridge.py``, with the run's agent token in its
environment) builds its app with ``create_app(external_agents=True)``. In
agent mode that changes nothing, since ``/mcp`` is already mounted beside the
embedded agent. In viewer-only mode the app still builds the product's tools
and mounts ``/mcp``, and this is the session the rest of the app sees in
place of an agent's.

It has no conversation, so it reports the agent unavailable (the chat pane's
composer stays disabled) and refuses a turn. What it does have is the
``PermissionBroker`` that gates the external agent's calls: a write-grade tool
reaching ``/mcp`` and a tool that asks the human itself (resolving one of
their review comments) put the same permission card in the page as an
embedded agent's call would, the page's answer reaches the broker through
``decide_permission`` exactly as it does for one, and the pause control
refuses the external agent's view- and write-grade tools like an embedded
agent's. It keeps the broker's count of connected pages the way
``SdkSession`` does, so a call made while no page is open is refused at once
with the address to open, rather than left waiting for nobody.

**The cards hold only while the external agent lacks the browser token.** A
process that can read the product's startup banner (its ``#t=`` link), or
that runs as the same user and so can read the browser's history or this
process's memory, can act as the human: approve its own cards over ``/ws``
or change the review through the page's routes. The agent token keeps an
honest agent to its tools; it is not a boundary against one running as you.
"""

from __future__ import annotations

from typing import Optional

from .base import AGENT_UNAVAILABLE, SandboxStatus


class NoEmbeddedAgent(RuntimeError):
    """A turn or a model switch sent to a run that has no agent of its own."""


class ExternalAgentSession:
    """Implements the ``AgentSession`` Protocol from ``session/base.py`` for a
    run with no embedded agent; ``broker`` is the ``PermissionBroker`` the
    app's ``/mcp`` route and review tools ask through."""

    def __init__(self, on_event, broker):
        self._on_event = on_event
        self._broker = broker
        self._viewers_seen = 0

    def agent_status(self) -> str:
        return AGENT_UNAVAILABLE

    async def submit_turn(self, blocks: list, viewer: Optional[str] = None) -> None:
        raise NoEmbeddedAgent(
            "this run has no embedded agent; an agent in another process works through /mcp instead"
        )

    async def decide_permission(
        self, request_id: str, decision: str, message: str = "", by: Optional[str] = None
    ) -> None:
        await self._broker.decide(request_id, decision, message, by=by)

    async def interrupt(self) -> None:
        # Nothing of this process's is running a turn, so there is nothing to
        # stop; the external agent's own client is where it is interrupted.
        return None

    async def set_model(self, model: str) -> None:
        raise NoEmbeddedAgent("this run has no embedded agent whose model could be switched")

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        # Denies whatever the external agent is still waiting on, while there
        # is a socket left to carry the resolution to the page.
        self._broker.shutdown()

    def on_viewer_presence(self, count: int) -> None:
        """Drive the broker's count of connected pages to the registry's, as
        ``SdkSession.on_viewer_presence`` does and for the same reason."""
        while self._viewers_seen < count:
            self._broker.viewer_connected()
            self._viewers_seen += 1
        while self._viewers_seen > count:
            self._broker.viewer_disconnected()
            self._viewers_seen -= 1

    def sandbox_status(self) -> SandboxStatus:
        """No shell of this process's to contain: the external agent's own
        posture is its own client's to report."""
        return SandboxStatus(requested=False, active=False, missing=())

    def backend_logs(self) -> list:
        """None: the external agent's logs are its own client's, in another
        process this one knows nothing about."""
        return []
