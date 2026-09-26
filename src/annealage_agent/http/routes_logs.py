"""The page's view of the agent backend's own logs, for the Agent log section
of the settings window (``static/settings.js``).

Registers, for every product and whether or not a session exists:

    GET /agent/logs         the session's ``backend_logs()``: each entry's
                            name, kind, path and format, without its text
    GET /agent/logs/<name>  the entry called ``name`` with its text: at most
                            the last ``logfiles.TAIL_BYTES`` of it, from a
                            whole line

Both require the browser token and a permitted ``Origin``, and refuse with the
same opaque response ``/ws`` returns. The browser token only: these logs hold
the conversation, paths on this machine and whatever a provider said back, and
the agent token is handed to processes beside the agent's own shell (the Codex
stdio bridge), so it must not open them. No agent tool reads them either.

Nothing in a request names a file. ``name`` is looked up, exactly, among the
names in the list the session gives now, asked for again on every request, and
is never used as a path; a session lists only files it found where its backend
keeps them (``session/logfiles.py``), so there is nothing to traverse. The
name, not a position, is the key because the list grows while a page shows it
(omp's conversation file appears with the first message, the Claude transcript
and the Codex rollout likewise): a position the page learned earlier could name
a different log by the time it is asked for. A session's names are unique. A
run with no session (viewer-only) lists nothing.
"""

import asyncio
import sys
from urllib.parse import unquote

from ..session import logfiles
from .ws import _origin_is_allowed, _token_is_allowed, refusal


def register_log_routes(app, *, session, token, allowed_origins=()):
    """Register ``GET /agent/logs`` and ``GET /agent/logs/<name>`` on ``app``
    over ``session`` (an ``AgentSession``, or ``None`` for a run without one)."""

    async def _entries():
        # backend_logs looks at files, so it runs off the loop; it must not
        # raise, and a session that does anyway lists nothing rather than
        # failing the window it is shown in.
        if session is None:
            return []
        loop = asyncio.get_running_loop()
        try:
            return list(await loop.run_in_executor(None, session.backend_logs))
        except Exception as exc:
            sys.stderr.write("warning: could not list the agent backend's logs: %r\n" % (exc,))
            return []

    @app.get("/agent/logs")
    async def list_logs(req):
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()
        entries = await _entries()
        return {
            "ok": True,
            "tail_bytes": logfiles.TAIL_BYTES,
            "logs": [entry.to_wire() for entry in entries],
        }, 200

    @app.get("/agent/logs/<path:name>")
    async def show_log(req, name):
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()
        # microdot hands the segment over still percent-encoded.
        name = unquote(name)
        entry = next((entry for entry in await _entries() if entry.name == name), None)
        if entry is None:
            return {"ok": False, "error": "there is no agent log called %r" % name}, 404
        loop = asyncio.get_running_loop()
        try:
            tail = await loop.run_in_executor(None, logfiles.read_tail, entry)
        except OSError as exc:
            return {
                "ok": False,
                "error": "%s could not be read: %s" % (entry.name, exc.strerror or exc),
            }, 404
        body = entry.to_wire()
        body.update(tail)
        body["ok"] = True
        return body, 200
