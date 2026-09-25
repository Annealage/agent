"""Construct the agent session for one run: the backend switch.

A product's CLI resolves everything a run needs (which backend, the settings,
the session id, the bind, the two tokens) and then, inside the
``build_session(on_event, *, bus)`` factory the app calls, hands them to
``build_session`` here. Everything from that point on is the same for every
product: one ``PermissionBroker`` shared with ``/mcp``, and the one session
class the backend names, wired to the product's tool server.

Each backend's session module is imported only inside its own branch, so a
``claude``-backend run never pays for ``openai_codex`` or ``omp_rpc``, both
optional dependencies, and nothing here is imported at all by a viewer-only
run.
"""

from . import sessions
from . import settings as settings_module


def _resumable_sdk_id(serve_dir, session_id):
    """The SDK conversation id recorded for ``session_id``, or None.

    None is an ordinary outcome, not an error: a session whose client never
    connected has no conversation to resume, and it is still resumable as a
    session, just as a fresh conversation in the same folder.
    """
    info = sessions.get_session_info(serve_dir, session_id)
    return info.sdk_session_id if info is not None else None


def build_session(
    backend,
    on_event,
    *,
    bus,
    serve_dir,
    session_id,
    resumed,
    settings,
    mcp_host,
    mcp_port,
    agent_token,
    trusted_config_digest=None,
):
    """The session ``backend`` names, constructed and not yet started.

    ``on_event`` and ``bus`` are what ``create_app`` handed the product's
    factory. ``bus.tools`` is the product's tool server, built once by
    ``create_app`` and read here, never rebuilt; ``bus.broker`` is set here,
    to the broker this session's approvals go through, so ``create_app`` can
    gate a write-class call arriving through ``/mcp`` with the very same
    instance.

    ``settings`` is the run's ``settings.Resolved``. ``resumed`` says whether
    ``session_id`` was resolved from ``-c``/``-r`` rather than created fresh,
    which is when the backend is asked to resume its own conversation.
    ``mcp_host``/``mcp_port`` are where ``/mcp`` listens and ``agent_token``
    the only token it accepts, for the Codex bridge; the browser token is not
    a parameter here, because no backend is ever given it, and the address a
    refusal names when no browser is attached to decide a permission is
    ``bus.url``, which carries no token either.
    ``trusted_config_digest`` is the Claude configuration digest the startup
    trust gate accepted, if it ran.
    """
    if backend not in settings_module.BACKENDS:
        raise AssertionError("unreachable: settings.py validates backend's choices")

    from .session.permissions import PermissionBroker

    broker = PermissionBroker(
        on_event,
        permissions_path=sessions.state_dir(serve_dir) / "permissions.toml",
        viewer_url=bus.url,
    )
    # app.py reads this back once build_session returns, to gate a
    # write-class tool call arriving through /mcp with the exact same
    # broker instance this session's own approval flow uses (its own
    # comment on the bus.tools/bus.broker wiring seam explains why
    # bus, not a new parameter here: build_session's (on_event, *, bus)
    # signature is the shape every existing test fixture already
    # assumes, and widening it would break all of them for a value only
    # this real closure needs to hand back out).
    bus.broker = broker

    def _record_sdk_id(sdk_id):
        sessions.set_sdk_session_id(serve_dir, session_id, sdk_id)

    # The backend resumes only a conversation it already knows; a freshly
    # created session has no backend id to resume yet.
    resume = _resumable_sdk_id(serve_dir, session_id) if resumed else None

    if backend == "codex":
        # Imported only in this branch, per the module docstring's own
        # "keep the claude backend free of an unnecessary dependency
        # import" intent: openai-codex is an optional extra, and a
        # claude-backend run must not require it to be installed.
        from .session.codex import CodexSession

        return CodexSession(
            on_event,
            cwd=serve_dir,
            session_id=session_id,
            broker=broker,
            model=settings["model"],
            effort=settings["effort"],
            resume=resume,
            on_sdk_session_id=_record_sdk_id,
            # The host's own /mcp endpoint (phase3_codex-tool-mcp-bridge.md):
            # host is the bind this run resolved, never a hardcoded
            # loopback, since app.py's allowed_hosts check only accepts
            # the exact bind address a non-loopback run chose (a
            # tailnet-bound server does not also accept 127.0.0.1).
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            mcp_token=agent_token,
        )

    if backend == "omp":
        # Imported only in this branch, per the module docstring's own
        # "keep the claude backend free of an unnecessary dependency
        # import" intent: omp_rpc is a separately installed package
        # (see session/omp.py), and a claude-backend run must not
        # require it to be installed.
        from .session.omp import OmpSession

        return OmpSession(
            on_event,
            cwd=serve_dir,
            session_id=session_id,
            broker=broker,
            model=settings["model"],
            base_url=settings["omp_base_url"],
            api_key=settings["omp_api_key"],
            # Mirrors SdkSession's mcp_servers=bus.tools.mcp_servers:
            # a snapshot taken once, at construction, rather than a live
            # reference to the tool server this run already built.
            tool_table=bus.tools.tool_table(),
            on_sdk_session_id=_record_sdk_id,
        )

    from .session.sdk import SdkSession

    return SdkSession(
        on_event,
        cwd=serve_dir,
        session_id=session_id,
        broker=broker,
        # The product's tool server, built once by create_app and shared
        # through bus.tools (see app.py's own comment on that channel)
        # rather than built again here. Its read- and view-grade tools are
        # the session's allow list, so they never prompt; the write-grade
        # ones are absent from every allow list, which is what makes them
        # reach the broker above and therefore the human.
        mcp_servers=bus.tools.mcp_servers,
        allowed_tools=bus.tools.pre_allowed,
        model=settings["model"],
        effort=settings["effort"],
        permission_mode=settings["permission_mode"],
        resume=resume,
        on_sdk_session_id=_record_sdk_id,
        # What the CLI's trust gate accepted, so the session can refuse tool
        # calls if it stops being true while the run is in progress.
        trusted_config_digest=trusted_config_digest,
    )
