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

from . import product, sessions
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
    omp_agent_dir=None,
    omp_binary=None,
    omp_config_dir=None,
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

    ``omp_agent_dir``, ``omp_config_dir`` and ``omp_binary`` are the omp
    backend's agent directory (its ``PI_CODING_AGENT_DIR``: its own auth,
    settings and models), its config root (``PI_CONFIG_DIR``, in place of
    ``~/.omp`` with the user's ``APPEND_SYSTEM.md``, plugins and ``.env``;
    it must lie under ``$HOME``) and an absolute path to the omp executable;
    ``None`` uses omp as installed on ``PATH``, with the user's own
    directories. A service running under its own account passes all three;
    the agent directory alone still leaves the user's config root in effect.
    A project's own context files are read either way. They are arguments
    rather than settings keys because a settings key is writable from the
    page (``PUT /settings``), and neither an executable nor a profile
    directory is something a browser should choose. The omp conversation is
    kept under ``<state dir>/omp`` and resumed from the file recorded in the
    session's ``meta.json``.

    The session's write-protected patterns are ``bus.write_protected``, the
    app's (``create_app``); a stand-in bus without them takes the product's.
    """
    if backend not in settings_module.BACKENDS:
        raise AssertionError("unreachable: settings.py validates backend's choices")

    from .session.permissions import PermissionBroker

    broker = PermissionBroker(
        on_event,
        permissions_path=sessions.state_dir(serve_dir) / "permissions.toml",
        viewer_url=bus.url,
        timeout=float(settings["approval_timeout"]),
        # The tools that ask the human themselves (tools.asks_the_human):
        # their requests are never remembered, and a grant for one of their
        # names already in permissions.toml is ignored. A bus with no tool
        # server has none.
        never_remembered=bus.tools.never_remembered if bus.tools is not None else (),
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

    # What the product says this run is about (Product.session_context), added
    # to the backend's system prompt; None adds nothing. Followed by what each
    # remote MCP server the tool server reached said about itself
    # (ToolServer.remote_instructions), which every backend gets the same way.
    context_hook = product.current().session_context
    context = context_hook(bus, serve_dir) if context_hook is not None else None
    remote_instructions = bus.tools.remote_instructions if bus.tools is not None else None
    instructions = "\n\n".join(p for p in (context, remote_instructions) if p) or None

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
            # One more bridge per remote MCP server, each at /mcp/<remote>.
            mcp_remotes=tuple(r.name for r in bus.tools.remotes) if bus.tools is not None else (),
            instructions=instructions,
            turn=getattr(bus, "turn", 0),
        )

    if backend == "omp":
        # Imported only in this branch, per the module docstring's own
        # "keep the claude backend free of an unnecessary dependency
        # import" intent: omp_rpc is a separately installed package
        # (see session/omp.py), and a claude-backend run must not
        # require it to be installed.
        from .session.omp import OmpSession

        def _record_session_file(path):
            sessions.set_omp_session_file(serve_dir, session_id, path)

        info = sessions.get_session_info(serve_dir, session_id) if resumed else None
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
            # reference to the tool server this run already built. The
            # product's tools by their own names, each remote's as
            # <remote>__<tool>.
            tool_table=bus.tools.host_tool_table(),
            on_sdk_session_id=_record_sdk_id,
            instructions=instructions,
            turn=getattr(bus, "turn", 0),
            agent_dir=omp_agent_dir,
            config_dir=omp_config_dir,
            binary=omp_binary,
            session_dir=sessions.state_dir(serve_dir) / "omp",
            resume=info.omp_session_file if info is not None else None,
            on_session_file=_record_session_file,
        )

    from .session.sdk import SdkSession

    return SdkSession(
        on_event,
        cwd=serve_dir,
        session_id=session_id,
        broker=broker,
        # The product's tool server, built once by create_app and shared
        # through bus.tools (see app.py's own comment on that channel)
        # rather than built again here: the product's own in-process server
        # and one per remote MCP server. Their read- and view-grade tools are
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
        instructions=instructions,
        # A resumed session continues its history's turn numbering.
        turn=getattr(bus, "turn", 0),
        write_protected=getattr(bus, "write_protected", None),
    )
