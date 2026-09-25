"""On-demand live integration test for ``session/codex.py``'s ``CodexSession``
against the real ``openai_codex.client.CodexClient`` (plan:
Annealage Mesh's ``planning/tickets/phase7_live-integration-tests.md``).

Unlike ``tests/test_codex_session.py``, which drives ``CodexSession`` through
``FakeCodexClient``, this file constructs the real driver with no
``client_factory`` override at all: a real ``codex app-server`` subprocess
and one real (cheap, text-only) model turn. It proves nothing that the
fake-transport suite does not already prove about approval-gating or
protocol shape (out of scope here, see the ticket) -- it proves only that
the real SDK, given this package's actual construction of ``CodexConfig``/
``CodexClient``, can authenticate and complete a turn at all.

Gated behind the ``integration`` marker (``pyproject.toml``'s
``[tool.pytest.ini_options]``, excluded from every default run via
``addopts``) and a check that the ``codex`` CLI is on ``PATH``, so it skips
cleanly -- never errors, never runs silently -- wherever that binary is not
installed.

This test deliberately performs NO login and NO ``CODEX_HOME`` override:
``CodexSession.start()`` (``session/codex.py``) never touches ``CODEX_HOME``
or forces a login itself -- it calls ``account_read()`` and fails closed
only if no account is configured at all -- so it authenticates through
whatever real ``codex`` account is already logged in on this host, exactly
as a plain ``codex`` CLI invocation would. There is nothing for this package to
reconfigure here: this is the same account and the same default model
(``~/.codex/config.toml``'s own ``model =`` line) any other ``codex``
invocation on this host already uses.
"""

import asyncio
import os
import shutil

import pytest

from annealage_agent.session.base import AGENT_READY, AgentError, TextDelta, TurnEnd
from annealage_agent.session.codex import CodexSession
from annealage_agent.session.permissions import PermissionBroker

LIVE_PROMPT = "Reply with exactly the single word PONG and nothing else, no punctuation."

# The model this host's `codex` is already configured to use by default
# (~/.codex/config.toml's `model = "gpt-5.6-luna"`); passed explicitly so
# this test pins what it actually exercises rather than depending silently
# on whatever a config file elsewhere happens to say. Override with
# ANNEALAGE_AGENT_LIVE_CODEX_MODEL for a different account/config, never
# required.
DEFAULT_MODEL = "gpt-5.6-luna"


def _diagnose(events: list) -> str:
    """A short dump of every event's class name and key fields, attached to
    an assertion failure so a live-test failure is diagnosable from the
    assertion string in the first place."""
    lines = []
    for event in events:
        if isinstance(event, TextDelta):
            lines.append("TextDelta(turn=%d, text=%r)" % (event.turn, event.text))
        elif isinstance(event, AgentError):
            lines.append(
                "AgentError(stderr=%r, remediation=%r)" % (event.stderr, event.remediation)
            )
        elif isinstance(event, TurnEnd):
            lines.append("TurnEnd(turn=%d, stop_reason=%r)" % (event.turn, event.stop_reason))
        else:
            lines.append(type(event).__name__)
    return "\n".join(lines) if lines else "(no events recorded)"


async def _wait_for_turn_end(events: list, timeout: float) -> None:
    """Poll ``events`` until a ``TurnEnd`` appears, or fail fast (rather than
    hang the whole suite) once ``timeout`` elapses."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if any(isinstance(event, TurnEnd) for event in events):
            return
        if any(isinstance(event, AgentError) for event in events):
            raise AssertionError("agent reported an error:\n%s" % _diagnose(events))
        await asyncio.sleep(0.05)
    raise AssertionError("timed out waiting for turn_end:\n%s" % _diagnose(events))


def _reply_text(events: list) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


@pytest.mark.integration
@pytest.mark.skipif(
    not shutil.which("codex"),
    reason="the codex CLI is not on PATH; install and log it in to run the live Codex test",
)
@pytest.mark.asyncio
async def test_real_codex_backend_completes_a_turn(tmp_path):
    model = os.environ.get("ANNEALAGE_AGENT_LIVE_CODEX_MODEL") or DEFAULT_MODEL

    events = []
    session = CodexSession(
        events.append,
        cwd=str(tmp_path),
        session_id="live-codex",
        broker=PermissionBroker(events.append, timeout=30.0, no_viewer_grace=0.05),
        model=model,
    )
    session.on_viewer_presence(1)
    try:
        await session.start()
        assert session.agent_status() == AGENT_READY, _diagnose(events)
        await session.submit_turn([{"type": "text", "text": LIVE_PROMPT}])
        await _wait_for_turn_end(events, timeout=60.0)
    finally:
        await session.close()
    assert "pong" in _reply_text(events).lower(), _diagnose(events)
