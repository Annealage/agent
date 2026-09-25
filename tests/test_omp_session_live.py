"""On-demand live integration test for ``session/omp.py``: proves the real
``omp_rpc.RpcClient`` (not ``tests/test_omp_session.py``'s ``FakeRpcClient``)
can authenticate against a real, already-configured `omp` provider and
complete one real text-only turn (plan: Annealage Mesh's
``phase7_live-integration-tests.md``).

Skips at collection time, with a clear reason, unless the ``omp`` CLI is on
``PATH``. Excluded from every default ``pytest`` run by ``pyproject.toml``'s
``addopts = "-m 'not integration'"``; run this tier explicitly with
``pytest -m integration``.

**No ``omp_base_url``/config synthesis.** ``OmpSession`` supports two
modes (``session/omp.py``): given a ``base_url``, it synthesizes its own
throwaway custom-provider config for an arbitrary, self-hosted endpoint
`omp` does not already know about. Given no ``base_url``, it instead passes
``model`` straight through to `omp`'s own ``--model`` flag untouched, using
whatever providers this host's `omp` is already configured with -- the same
"provider/model" reference (``"titan/qwen3.8-27b"``) a human would type at
the CLI. This test deliberately exercises the second mode: it constructs no
custom provider, sets no ``PI_CODING_AGENT_DIR`` override, and needs no
base URL or API key of its own, because "titan" is already a provider this
host's `omp` knows about, with its own credentials.
"""

import asyncio
import os
import shutil

import pytest

from annealage_agent.session.base import AGENT_READY, AgentError, TextDelta, TurnEnd
from annealage_agent.session.omp import OmpSession
from annealage_agent.session.permissions import PermissionBroker

LIVE_PROMPT = "Reply with exactly the single word PONG and nothing else, no punctuation."

# The "provider/model" reference this host's omp is already configured
# with, exactly as typed at the CLI (`omp --model titan/qwen3.8-27b`).
# Override with ANNEALAGE_AGENT_LIVE_OMP_MODEL for a different
# account/config, never required -- same convention as the Claude/Codex live
# tests.
DEFAULT_MODEL = "titan/qwen3.8-27b"


def _reply_text(events):
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def _diagnose(events):
    """A human-readable dump of every event collected so far, for an
    assertion failure message -- a live backend's real stderr/remediation
    is what tells a developer whether a failure was auth, network, an
    unsupported model id, or an unexpected reply, not a bare `False`."""
    lines = ["%d event(s) collected:" % len(events)]
    for event in events:
        lines.append("  %r" % (event,))
    return "\n".join(lines)


async def _wait_for_turn_end(events, *, timeout):
    """Poll ``events`` for a ``TurnEnd``, failing fast on an ``AgentError``
    instead of waiting out the full timeout when the backend has already
    reported it cannot continue."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    seen = 0
    while True:
        while seen < len(events):
            event = events[seen]
            seen += 1
            if isinstance(event, AgentError):
                raise AssertionError("agent reported an error: %s" % _diagnose(events))
            if isinstance(event, TurnEnd):
                return event
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for turn_end: %s" % _diagnose(events))
        await asyncio.sleep(0.05)


@pytest.mark.integration
@pytest.mark.skipif(
    not shutil.which("omp"),
    reason="the omp CLI is not on PATH; install and configure it to run the live omp test",
)
@pytest.mark.asyncio
async def test_real_omp_backend_completes_a_turn(tmp_path):
    model = os.environ.get("ANNEALAGE_AGENT_LIVE_OMP_MODEL") or DEFAULT_MODEL
    events = []
    session = OmpSession(
        events.append,
        cwd=str(tmp_path),
        session_id="live-omp",
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
