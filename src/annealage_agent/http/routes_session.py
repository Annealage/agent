"""The hosted front's control of one worker's agent session.

Registered only in hosted agent mode (``create_app(hosted_mode=True, ...)``),
where the front launches the worker for one delegated project and needs to
start, observe and end the agent session in it without a browser socket:

    GET  /agent/session          the session's report (``AgentHolder.report``)
    POST /agent/session          launch: open the session if none is live
    POST /agent/session/cancel   cancel: end it now, and keep it ended

Each goes through the app's ``identity.BrowserAuth`` with a verified delegation
naming the operation: ``project.read`` for the report, ``project.exec`` for a
launch or a cancel (the grade of a ``turn`` or ``interrupt`` frame). They
refuse with the same opaque response ``/ws`` returns. The worker is bound to
one project by the job that started it, so no request names a project or a
session; the launch is for the delegated project's session, and the front's
verifier is what ties the delegation to this job.

Launch is idempotent: with a session already live it opens nothing and says so
(``"launched": false``). A cancel is not undone by a page connecting or by a
tool call, as an idle close is; only a launch reopens a cancelled session. The
report carries no backend, model or credential detail, which hosted settings
hide. A launch and a cancel are recorded as ``session_event`` records with the
requesting principal when a state sink is given (``hosted_state.SessionEvents``).

Hosted agent mode is limited to the omp backend (``create_app`` refuses any
other), so there is no backend to choose here.
"""

from .ws import refusal


def register_session_routes(app, *, auth, holder, events=None):
    """Register the three routes on ``app`` over ``holder`` (the app's
    ``AgentHolder``), gated by ``auth``. ``events`` is the app's
    ``hosted_state.SessionEvents``, or ``None`` to record nothing."""

    @app.get("/agent/session")
    async def session_status(req):
        if auth.authenticate(req, "project.read") is None:
            return refusal()
        return {"ok": True, "session": holder.report()}, 200

    @app.post("/agent/session")
    async def session_launch(req):
        human = auth.authenticate(req, "project.exec")
        if human is None:
            return refusal()
        launched = await holder.launch()
        if events is not None:
            events.note(human)
            if launched:
                await events.emit_async("launch", human=human)
        if holder.session is None:
            # A factory that raises is reported on stderr; the front retries.
            return {
                "ok": False,
                "error": "the agent session could not be opened; see the worker's output",
                "session": holder.report(),
            }, 503
        return {"ok": True, "launched": launched, "session": holder.report()}, 200

    @app.post("/agent/session/cancel")
    async def session_cancel(req):
        human = auth.authenticate(req, "project.exec")
        if human is None:
            return refusal()
        interrupted = await holder.cancel()
        was_live = interrupted is not None
        if events is not None:
            events.note(human)
            if was_live:
                await events.emit_async("cancel", {"interrupted_turns": interrupted}, human=human)
        return {
            "ok": True,
            "cancelled": was_live,
            "interrupted_turns": interrupted or [],
            "session": holder.report(),
        }, 200
