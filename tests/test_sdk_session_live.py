"""One real, on-demand integration test for ``session/sdk.py``: proves the
real ``claude_agent_sdk`` package, given this package's actual generated
options and no ``transport=`` override, can authenticate against the live
Anthropic API and complete one trivial text-only turn.

``tests/test_sdk_session.py`` drives the same class through a hand-written
``FakeTransport`` and pins protocol shape; it proves nothing about whether a
real ``claude`` subprocess actually starts, authenticates, and replies. This
file closes that gap, gated behind the ``integration`` marker (opt-in only,
excluded from every default run by ``pyproject.toml``'s own `addopts`) and a
check that the ``claude`` CLI is on ``PATH``, so a default ``pytest``
invocation never depends on it and a host with no ``claude`` installed never
has it silently run.

No API key is set or required here: ``claude_agent_sdk``'s real subprocess
transport inherits ``os.environ`` wholesale (confirmed by reading the
installed ``_internal/transport/subprocess_cli.py`` before writing this),
and ``SdkSession`` never overrides it, so the spawned ``claude`` subprocess
authenticates through whatever this host's ``claude`` CLI is already logged
into (OAuth credentials at ``~/.claude/.credentials.json``, or an ambient
``ANTHROPIC_API_KEY``) -- the same account any other ``claude`` invocation
on this host already uses. ``tests/conftest.py``'s autouse
``isolated_user_config`` fixture only isolates ``XDG_CONFIG_HOME`` (the
product's own workspace-trust store), a path the real ``claude`` CLI's own credential
file does not live under, so it does not interfere with this.
"""

import asyncio
import os
import shutil

import pytest

from annealage_agent.session.base import AGENT_READY, AgentError, TextDelta, TurnEnd
from annealage_agent.session.permissions import PermissionBroker
from annealage_agent.session.sdk import SdkSession

LIVE_PROMPT = "Reply with exactly the single word PONG and nothing else, no punctuation."

# The model this test exercises. Override with
# ANNEALAGE_AGENT_LIVE_CLAUDE_MODEL for a different account/config, never
# required.
DEFAULT_MODEL = "haiku"


async def _wait_for_turn_end(events: list, timeout: float) -> None:
    """Poll ``events`` until a ``TurnEnd`` arrives, failing fast on an
    ``AgentError`` instead of waiting out the full timeout for one."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for event in events:
            if isinstance(event, TurnEnd):
                return
            if isinstance(event, AgentError):
                raise AssertionError("agent reported an error:\n%s" % _diagnose(events))
        await asyncio.sleep(0.05)
    raise AssertionError("timed out waiting for turn_end:\n%s" % _diagnose(events))


def _reply_text(events: list) -> str:
    """The model's full reply, as the concatenation of every ``TextDelta``."""
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def _diagnose(events: list) -> str:
    """A compact dump of every event seen so far, for an assertion failure
    message."""
    return "\n".join("%s: %r" % (type(event).__name__, event) for event in events)


@pytest.mark.integration
@pytest.mark.skipif(
    not shutil.which("claude"),
    reason="the claude CLI is not on PATH; install and log it in to run the live Claude test",
)
@pytest.mark.asyncio
async def test_real_claude_backend_completes_a_turn(tmp_path):
    model = os.environ.get("ANNEALAGE_AGENT_LIVE_CLAUDE_MODEL") or DEFAULT_MODEL
    events: list = []
    session = SdkSession(
        events.append,
        cwd=str(tmp_path),
        session_id="live-claude",
        broker=PermissionBroker(events.append, timeout=30.0, no_viewer_grace=0.05),
        model=model,
        sandbox=False,  # no bash/tool use in this prompt; do not require
        # bwrap/socat just to prove a live turn works
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
