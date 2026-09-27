"""Builds the microdot application for any product and owns its asyncio
startup and shutdown.

``create_app`` wires the agent layer's routes (``/ws``, the chat, settings,
agent log and ``/mcp`` routes, and ``/agent/static/`` for its own front end) and whatever
routes the product registers onto a fresh ``Microdot`` instance, together with
the pieces every product shares: the Host check, the event log, the viewer
registry and bus, the product's tool server, the session, and the security
headers every response carries. A fresh instance per call means independent
served directories never share route state, which matters for tests.
``serve`` binds the socket, hands control back
to the caller once it is actually listening, then serves until interrupted and
closes the listener before returning.

Everything that belongs to one app's run rather than to its socket lives on
the app itself, in its ``AgentHolder`` (``app.agent_holder``): the live
session, which an idle app closes and a returning page resumes as a new one,
the lifecycle ``serve`` drives (``app.agent_start``/``app.agent_stop``), and
the status summary a front door shows (``app.agent_status``). That is what
lets several apps share one server under URL prefixes
(``frontdoor.FrontDoor``), each started and stopped on its own.

A product's own application module (Mesh's ``app.py``, for one) calls both,
adding its routes and the background tasks that watch its files.
"""

import asyncio
import base64
import hashlib
import inspect
import re
import sys
import time

from microdot import Microdot, Request

from . import net, product, protocol, sessions
from . import settings as settings_module
from .http.routes_chat import register_chat_routes
from .http.routes_login import LoginNonces, register_login_routes, register_whoami_route
from .http.routes_logs import register_log_routes
from .http.routes_mcp import register_mcp_routes
from .http.routes_review import register_review_routes
from .http.routes_settings import register_settings_routes
from .http.static import register_agent_static_routes
from .http.ws import host_is_allowed, ping_forever, refusal, register_ws
from .identity import BrowserAuth, check_bind
from .review.watcher import ReviewWatcher
from .session import secret_paths
from .session.base import (
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentModelChanged,
    AgentStatus,
    PermissionResolved,
    TurnEnd,
)
from .session.events import EventLog
from .session.permissions import OUTCOME_SHUTDOWN
from .viewers import ViewerBus, ViewerRegistry

# Upper bound on how long shutdown waits for in-flight requests to drain
# once the listening socket has stopped accepting new connections. Past
# this, the process exits with those requests abandoned rather than blocking
# until they finish: an interrupt during a large in-progress model transfer
# must return control in a couple of seconds, not stall until that transfer
# completes.
SHUTDOWN_DRAIN_TIMEOUT = 2.0

#: What ``app.agent_status()`` reports as ``agent`` for an app whose session
#: is closed: by its idle timer (the next page to connect resumes it) or by
#: ``agent_stop``. The other three values are the session's own.
AGENT_CLOSED = "closed"

#: The longest an app with an idle timeout waits between two looks at whether
#: it has been idle long enough; a shorter timeout is looked at that often.
IDLE_SWEEP_INTERVAL = 5.0

#: A ``url_prefix``: empty, or one or more path segments each introduced by a
#: ``/``, with no trailing ``/``, and nothing microdot would read as a
#: placeholder (``<``) or a URL would read as a query or fragment. No ``.``
#: either: microdot joins a route's static segments into its regex unescaped,
#: so a prefix ``/p/a.b`` would also match ``/p/axb`` and take another app's
#: requests.
_URL_PREFIX_RE = re.compile(r"(?:/[A-Za-z0-9_~-]+)*")

# microdot's request body limits are class attributes on ``Request`` and
# therefore process-global: raising them (done in configure_request_limits,
# called from create_app) affects every route this process ever serves, not
# only /upload and a product's own body-reading routes, and every other
# microdot app or test sharing this interpreter. 8 MiB comfortably covers
# the largest body a product page sends today, Mesh's full-page pin review
# on its /submit (microdot's 16 KiB default caps a submission at roughly 68
# pins of the {id, part, label, point, normal, faceIndex, comment} shape its
# viewer sends): a 400-pin submission measures about 84 KB, so 8 MiB is far more
# headroom than the real payload needs, and it matches files.MAX_IMAGE_BYTES,
# the separate cap /upload enforces on itself. The actual exposure of that
# headroom is per in-flight request, not aggregate: microdot imposes no cap
# on concurrent connections and no read timeout, so N slow or stalled
# clients each declaring a large Content-Length can together hold open N
# times this many bytes' worth of allowance for as long as they keep their
# sockets open. That aggregate exposure is unbounded; capping it needs a
# connection limit or a read timeout, and this module has neither.
#
# configure_request_limits sets max_body_length to 0, microdot's documented
# "always access the body as a stream" value: no route is ever handed a
# buffered req.body. Every route that reads a body (/submit, /upload)
# checks Content-Length itself first, refusing a missing or zero value,
# then reads exactly that many bytes off req.stream in bounded chunks. A
# route that read the stream without that check would, on a real
# connection with no declared length, read forever: req.stream is then the
# raw client reader, and nothing ever makes such a read return.
MAX_REQUEST_BODY = 8 * 1024 * 1024


def configure_request_limits():
    """Raise microdot's process-global request body limits to MAX_REQUEST_BODY.

    Called from create_app rather than run at import time, so the global
    mutation happens as a deliberate step tied to building an app, not as a
    side effect of merely importing this module.
    """
    Request.max_content_length = MAX_REQUEST_BODY
    Request.max_body_length = 0


def inline_script_hashes(html_path):
    """Base64 sha256 hashes of every inline ``<script>`` body in ``html_path``.

    Computed from the packaged file at startup rather than written down as a
    constant, so editing the import map cannot leave a policy that blocks the
    page it is meant to allow. A file that cannot be read yields nothing, which
    produces a policy that refuses the inline script: failing closed is right
    here, and the product's CLI has already refused to start if its page is
    missing.
    """
    try:
        html = html_path.read_text(encoding="utf-8")
    except OSError:
        return ()
    hashes = []
    for match in re.finditer(r"<script\b[^>]*>(.*?)</script>", html, re.DOTALL):
        body = match.group(1)
        if not body.strip():
            continue
        digest = hashlib.sha256(body.encode("utf-8")).digest()
        hashes.append("'sha256-%s'" % base64.b64encode(digest).decode("ascii"))
    return tuple(hashes)


def content_security_policy(html_path):
    """The one policy every response carries.

    ``default-src 'none'`` is the point of it: every kind of fetch this page can
    make has to be named, so a source introduced later fails visibly rather than
    working quietly. The rest is the smallest set that lets the page work, and
    each entry has a reason at its use site in ``create_app``.
    """
    script_src = " ".join(("'self'",) + inline_script_hashes(html_path))
    return "; ".join(
        (
            "default-src 'none'",
            "script-src %s" % script_src,
            "style-src 'self'",
            "img-src 'self' data:",
            "connect-src 'self' ws: wss:",
            "font-src 'self'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
        )
    )


def create_app(
    serve_dir,
    *,
    page_html,
    port,
    token=None,
    agent_token=None,
    host=net.DEFAULT_HOST,
    extra_origins=(),
    extra_hosts=(),
    write_protected=None,
    session_id=None,
    build_session=None,
    register_routes=None,
    settings=None,
    login=None,
    review_store=None,
    external_agents=False,
    url_prefix="",
    resume_session=None,
    idle_timeout=None,
    identity=None,
):
    """Build a Microdot app serving ``serve_dir``, routes registered, not started.

    ``page_html`` is the path of the product's page, whose inline scripts the
    Content-Security-Policy hashes. ``register_routes(app, allowed_origins)``
    registers the product's own routes, and is called before the agent
    layer's. The product's tool server comes from the installed product's
    ``build_tools`` (``product.py``), called here once in agent mode; an
    agent-mode app for a product with no tool builder is refused here, at
    startup, rather than failing on the first tool call.

    ``login`` is the run's ``LoginNonces`` (``http/routes_login.py``), from
    which the CLI issues the nonce in the URL it opens a browser on; ``None``
    gives the app a fresh one of its own.

    ``host`` is an address already resolved (``net.resolve_bind`` does the
    resolving, in the product's CLI), and together with ``port``,
    ``extra_origins`` and ``extra_hosts`` it decides the exact ``Origin`` and
    ``Host`` values this app accepts. Computing those here, from the bind, is
    what lets a remote viewer work at all: an allowlist hardcoded to
    localhost would refuse every tailnet client. ``extra_origins`` and
    ``extra_hosts`` add the name a proxy or ``tailscale serve`` fronts the
    server under (``https://box.tailnet.ts.net`` and ``box.tailnet.ts.net``).

    ``write_protected`` is this app's write-protected patterns (glob patterns
    relative to ``serve_dir``, as ``Product.write_protected`` documents),
    checked here and handed to the session through ``bus.write_protected``;
    ``None`` takes the product's.

    ``token`` is the browser token: with an allowed tailnet login
    (``identity``, below), the only credential ``/ws``, the chat routes,
    ``/settings`` and every product route that asks for one accept. ``None``
    means no token was configured, and with no identity either ``/ws`` then
    refuses every request. ``agent_token`` is the separate per-run agent
    token, the only credential ``/mcp`` accepts (no identity header opens
    it); ``None`` means ``/mcp`` refuses every request. The two are kept
    apart because the agent token is handed to a
    process beside the agent's own shell (the Codex stdio bridge) and the
    browser token authorises permission decisions, so a run whose two tokens
    are equal is refused here rather than built. ``net.load_token`` keeps a
    browser token in a file for a service whose link must survive restarts.

    ``identity`` is an ``identity.TailscaleIdentity``, the tailnet logins
    allowed to act as the human when the server sits behind ``tailscale
    serve``, beside the token (``None``: the token alone). An app with one
    must be bound to loopback, and is refused otherwise: serve's identity
    headers are trustworthy only on a port nothing but serve and this host
    can reach. ``app.agent_auth`` is the app's one ``identity.BrowserAuth``,
    the check every browser route of the agent layer makes, and the one a
    product's own routes should make (``app.agent_auth.authenticate(req)``,
    a ``Human`` or ``None`` to refuse); it is set before ``register_routes``
    is called.

    ``session_id`` is the id the CLI resolved for this run (fresh or resumed,
    per plan section 3.4), or None for viewer-only; it is reported in the
    ``hello`` frame's ``session`` object and names the conversation the tool
    server writes out.

    ``review_store`` is the product's ``review.ReviewStore``, or ``None`` for a
    product that keeps no review. With one, ``GET``/``POST /review`` serve it
    to the page, the product's tool builder finds it on ``bus.review_store``
    (the same instance, so a change made by a tool reaches the watcher at
    once), and ``app.agent_review_watcher`` is the ``ReviewWatcher`` that
    ``serve`` runs beside the server to publish ``review_changed``.

    ``external_agents`` serves ``/mcp`` in viewer-only mode too, for an agent
    in another process (attached through the stdio bridge with
    ``agent_token``): the product's tools are built anyway, and the session
    the app runs with is an ``ExternalAgentSession`` (``session/external.py``)
    owning the ``PermissionBroker`` that puts that agent's write-grade calls
    in front of the human as permission cards. In agent mode ``/mcp`` is
    served beside the embedded agent whatever this says. A viewer-only run
    that sets it imports the agent SDK, which the product's tools are
    declared with.

    ``settings`` is the ``settings.Resolved`` this run started with, which
    the CLI builds because only it knows which flags were given. Passing
    ``None`` resolves the file and default layers here instead, so a caller
    with no flags to declare, which is every test, needs to know nothing about
    settings at all.

    ``url_prefix`` is the path this app is mounted under in a front door
    (``"/p/demo"``: no trailing slash; ``""``, the default, is an app served
    at the root). It goes into every address this app gives out itself:
    ``bus.url`` (the no-viewer message and the broker's ``viewer_url``) and,
    through ``bus.url_prefix``, the Codex bridge's ``--path`` for ``/mcp``
    and each ``/mcp/<remote>``. A URL a route builds for the page (``/upload``'s
    ``url``) comes from the request instead (microdot's ``req.url_prefix``),
    so an app at the root answers exactly as it did before prefixes existed.
    A mounted app refuses to save ``host`` and ``port`` from its settings
    window, since the front door owns the bind (``http/routes_settings.py``).

    ``resume_session`` is a factory shaped like ``build_session``
    (``(on_event, *, bus) -> session``) for the session an idle-closed app
    reopens with (a product's passes ``resumed=True`` to
    ``launch.build_session``, so the backend resumes the conversation).
    ``idle_timeout`` is how many seconds (more than zero) an app that has one
    waits, with no page connected, no turn running and no permission request
    open, before it closes its session and that session's broker; the next
    page to connect (or a turn, or a call through ``/mcp``) builds and starts
    a new one from ``resume_session`` (``AgentHolder``), switched to the model
    the human last chose if that is not the run's. Without both, the app never
    closes for being idle.

    Building an app blocks: the product's tool server connects to its remote
    MCP servers (up to ``remote.DISCOVERY_TIMEOUT`` each) and the event log is
    read back from disk. Nothing here needs a running event loop, so a
    process that builds apps while it serves others (a front door adding a
    workspace) calls this through ``asyncio.to_thread`` rather than on the
    loop every other app is served from.
    """
    if agent_token is not None and agent_token == token:
        raise ValueError(
            "the agent token must differ from the browser token: it is handed to "
            "processes beside the agent's shell, and the browser token approves "
            "permission requests"
        )
    check_url_prefix(url_prefix)
    if idle_timeout is not None and not idle_timeout > 0:
        # Zero would make the idle sweep spin the event loop every other app
        # of a front door is served from.
        raise ValueError("idle_timeout must be a positive number of seconds: %r" % (idle_timeout,))
    configure_request_limits()
    if write_protected is None:
        write_protected = product.current().write_protected
    else:
        # Checked before tuple(): a bare string would otherwise become one
        # pattern per character, each of which passes the check.
        secret_paths.check_write_protected(write_protected)
        write_protected = tuple(write_protected)
    csp_value = content_security_policy(page_html)
    bind = net.bind_from_address(host)
    check_bind(identity, bind)
    allowed_origins = net.allowed_origins(bind, port, extra_origins)
    allowed_hosts = net.allowed_hosts(bind, port, extra_hosts)
    auth = BrowserAuth(token, identity, allowed_origins=allowed_origins)
    installed = product.current()
    server_header = installed.server_header
    if (session_id is not None or external_agents) and installed.build_tools is None:
        raise RuntimeError(
            "%s has no tool builder (Product.build_tools), so an app serving its "
            "tools to an agent cannot be built for it" % installed.distribution
        )

    app = Microdot()
    app.agent_url_prefix = url_prefix
    app.agent_auth = auth
    install_host_check(app, allowed_hosts)

    if settings is None:
        settings = settings_module.resolve(serve_dir)
    app.agent_settings = settings

    if register_routes is not None:
        register_routes(app, allowed_origins)
    register_chat_routes(app, serve_dir, auth=auth)
    register_agent_static_routes(app)
    app.agent_login = login if login is not None else LoginNonces()
    register_login_routes(app, token=token, nonces=app.agent_login, allowed_origins=allowed_origins)
    register_whoami_route(app, auth=auth)
    register_settings_routes(
        app,
        serve_dir,
        auth=auth,
        settings=settings,
        session_id=session_id,
        bind=bind.address,
        port=port,
        mounted=bool(url_prefix),
    )
    register_review_routes(app, store=review_store, auth=auth)

    # The registry reports presence before the holder exists to take it (the
    # holder is built from the registry); the name is bound by the time a
    # connection can arrive.
    def _presence(count):
        holder.on_presence(count)

    # Given a path in agent mode, so the conversation survives the process. The
    # 500-event ring alone covers a browser reconnecting; it is this file that
    # lets `-c` resume a session and `-r` report what a session cost, both of
    # which read it back off disk. Viewer-only mode has no session and no
    # conversation, so it gets a ring and nothing on disk.
    event_log = EventLog(
        str(sessions.events_path(serve_dir, session_id)) if session_id is not None else None
    )
    # A resumed session's history already holds turns 1..N. The next turn is
    # N+1 on every backend (launch.build_session passes bus.turn on), so the
    # page never merges a new turn into an old one of the same number. A turn
    # an earlier process never finished (killed mid-turn) is closed here, so
    # the page does not show a finished history as a running turn, and so is
    # a permission request it asked and nobody answered, first, as the broker
    # would have at shutdown: its card would come back with the history, and
    # no one is waiting on the answer any more.
    for request_id in event_log.unresolved_requests:
        event_log.append(PermissionResolved(request_id=request_id, outcome=OUTCOME_SHUTDOWN))
    for turn in event_log.unfinished_turns:
        event_log.append(TurnEnd(turn=turn, stop_reason="interrupted", cost_usd=0.0))
    registry = ViewerRegistry(event_log=event_log, on_presence=_presence)
    # The tool layer's view of the browser, and the holder of the human's pause
    # switch. Built here rather than by the session factory because both halves
    # of it are properties of this app: the registry it calls through, and the
    # URL a tool has to name when it reports that no viewer is attached. It
    # imports no SDK, so a viewer-only run pays nothing for it, and ``ws.py``
    # needs it whether or not a session exists in order to answer a browser's
    # pause control.
    # The tokenless address: ViewerBus's docstring says why the browser token
    # must not appear in anything a tool or the broker says to the model.
    bus = ViewerBus(
        registry,
        url=net.server_url(bind, port, url_prefix),
        url_prefix=url_prefix,
        publish=_event_publisher(registry, event_log),
        turn=event_log.last_turn,
    )
    bus.write_protected = write_protected
    # The product's tool server, built once, here, whether or not this backend
    # is Claude: both the in-process driver's own ``.mcp_servers`` (Claude,
    # read out of ``bus.tools`` by ``build_session``'s own closure - see
    # the comment on that channel below) and the Codex tool-exposure bridge's
    # ``/mcp`` route (mounted below, whether or not anything ever calls it)
    # need the exact *same* already-``_wrap``-gated handler set
    # (``tools.py``'s own design constraint: one place builds it, no
    # transport re-derives it independently). Built only when
    # ``session_id is not None`` (agent mode) so a viewer-only run still
    # imports no SDK at all: the product's ``build_tools`` imports the SDK
    # when called, never before. This must stay gated on the id, not on
    # ``build_session is not None``: the CLI's real ``build_session`` closure
    # is always a real callable (its mode check happens inside the closure
    # body when called, returning ``None`` for viewer-only), so gating on the
    # callable's mere presence would import `claude_agent_sdk` and build an
    # unused tool server on every viewer-only run - a real regression a prior
    # version of this comment introduced and then reverted (see git log). A
    # caller that supplies its own ``build_session`` factory returning a real
    # session without a real ``session_id`` (a test fixture's scripted
    # ``FakeSession``, say) must pass a ``session_id`` too - that is the
    # actual contract this function relies on, not something to work around
    # here.
    tools = None
    # The product's review store reaches its tool builder the same way the
    # tool server and the broker cross between this function and the
    # session factory (see below): on the bus, set before ``build_tools``
    # reads it, ``None`` when the product keeps no review. One store instance
    # for the tools, the routes and the watcher is what makes a tool's write
    # notify the watcher directly rather than wait for its next sample.
    bus.review_store = review_store
    # The run's settings reach it the same way, so a product key can shape
    # the tools (Annealage Loom's remote MCP server URL, say).
    bus.settings = settings
    if session_id is not None or external_agents:
        tools = installed.build_tools(bus, serve_dir, session_id)
    # ``bus`` is the one object both this function and the CLI's
    # ``build_session`` closure already share, so it doubles as the wiring
    # seam between them in both directions without widening
    # ``build_session``'s own ``(on_event, *, bus)`` signature - the shape
    # every existing ``build_session`` fixture across the test suite already
    # assumes. ``tools`` flows this function -> the closure (read for
    # Claude's ``.mcp_servers`` and pre-allowed list, omp's tool table,
    # never rebuilt); ``broker`` flows the other way, set by that closure at
    # the exact point it already constructs ``PermissionBroker``, read below
    # once ``build_session`` has returned, so the same broker instance gates
    # the session's own approval flow, a write-class call arriving through
    # ``/mcp``, and the one approval a review tool asks for itself (resolving
    # a human's comment, ``review/tools.py``), which reads it at call time.
    bus.tools = tools
    bus.broker = None
    session_info = {
        "id": session_id if session_id is not None else "viewer-only",
        "sdk_session_id": None,
        "cwd": str(serve_dir),
        "agent": "unavailable",
        # The effective starting model this run's agent session was (or
        # will be) constructed with (settings.py's "model" key, Phase 2's
        # per-project default). Kept live afterwards: `_event_publisher`,
        # given this same dict, updates this key in place whenever an
        # `AgentModelChanged` event is published, so a later `hello` reads
        # whatever the session is actually running rather than only this
        # startup snapshot (see protocol.build_hello's docstring on why the
        # field itself is documented as a snapshot -- the corrected value
        # written back here is what makes it stay one that is current).
        "model": settings.get("model"),
        "steers": False,
    }
    app.agent_registry = registry
    app.agent_event_log = event_log
    app.agent_bus = bus
    app.agent_tools = tools
    app.agent_review_store = review_store
    # Its own publisher rather than the session's: the watcher runs whether
    # or not a session exists (a viewer-only run has a review too), and it
    # has no reason to see the session's hello-frame bookkeeping.
    app.agent_review_watcher = (
        ReviewWatcher(review_store, _event_publisher(registry, event_log))
        if review_store is not None
        else None
    )

    # ``build_session`` is called with the callback a session must use to
    # publish an event, plus the bus its tools drive the browser through, and
    # returns the session or None for viewer-only. It is a factory rather than a
    # constructed object so that both of those, which need the registry and the
    # log built above, exist before the session that will use them, without
    # either module importing the other. ``resume_session`` goes through the
    # same path when an idle-closed app reopens (``AgentHolder``).
    def _make_session(factory):
        # A broker belongs to one session: the factory sets the new one, and
        # a factory that sets none leaves /mcp failing closed rather than
        # gating with the shut-down broker of the session before.
        bus.broker = None
        session = None
        if factory is not None:
            session = factory(_event_publisher(registry, event_log, session_info), bus=bus)
        if session is None and external_agents:
            # Imported here, like the backends' own sessions: a product that
            # never asks for this pays nothing for it.
            from .session.external import ExternalAgentSession
            from .session.permissions import PermissionBroker

            publish = _event_publisher(registry, event_log, session_info)
            # The broker a real session's factory would have built
            # (launch.py), over the same grants file, and set on the bus for
            # the same reason: /mcp below and a review tool that asks the
            # human read it there.
            bus.broker = PermissionBroker(
                publish,
                permissions_path=sessions.state_dir(serve_dir) / "permissions.toml",
                viewer_url=bus.url,
                timeout=float(settings["approval_timeout"]),
                never_remembered=tools.never_remembered,
            )
            session = ExternalAgentSession(publish, bus.broker)
        return session

    holder = AgentHolder(
        app,
        bus=bus,
        registry=registry,
        event_log=event_log,
        session_info=session_info,
        tools=tools,
        review_watcher=app.agent_review_watcher,
        make_session=_make_session,
        resume_session=resume_session,
        idle_timeout=idle_timeout,
    )
    app.agent_holder = holder
    app.agent_start = holder.start
    app.agent_stop = holder.stop
    app.agent_on_stop = holder.on_stop
    app.agent_status = holder.status
    app.agent_status_listeners = holder.status_listeners
    # A factory returning None is the ordinary viewer-only case, not a
    # failure: it leaves an app that serves the product's page with no agent
    # attached rather than one that could not be built.
    session = _make_session(build_session)
    holder.install(session)
    # A tool result's end_turn (tools.ok) reaches whichever session is live
    # then; a session that cannot stop a turn has nothing to call.
    bus.end_turn_handler = holder.end_turn
    if session is not None:
        # /mcp is mounted only when this app has a session, for the same
        # reason /ws answers a turn only then: a viewer-only app has no
        # tools and no broker to gate them, so there is nothing for this
        # route to serve. tools is never None here (built above whenever
        # session_id is not None, which every real caller - the CLI's
        # build_session, and every test fixture that wants a real session -
        # must set for exactly this reason). The broker is the live
        # session's, read per call: bus.broker, set by the session factory
        # while constructing PermissionBroker, and None only if a factory
        # never sets it, in which case register_mcp_routes fails closed on
        # every write-class call rather than gating with no broker at all.
        async def _current_broker():
            await holder.ensure()
            holder.note_activity()
            return bus.broker

        register_mcp_routes(
            app,
            tools=tools,
            current_broker=_current_broker,
            agent_token=agent_token,
            allowed_origins=allowed_origins,
        )

    # Listed from whatever session is live; a viewer-only run still answers,
    # with nothing to list.
    register_log_routes(app, current_session=lambda: holder.session, auth=auth)

    register_ws(
        app,
        auth=auth,
        token=token,
        allowed_hosts=allowed_hosts,
        registry=registry,
        event_log=event_log,
        session_info=session_info,
        holder=holder,
    )

    install_response_handlers(app, csp_value, server_header)
    return app


def check_url_prefix(url_prefix):
    """Refuse a ``url_prefix`` that is not ``""`` or ``/segment[/segment...]``
    with no trailing slash: microdot mounts by concatenation, so ``/p/x/``
    would put every route at ``/p/x//...``, a ``<`` would be read as a
    placeholder handing every route an argument it does not take, and a
    ``.`` would match any character (``_URL_PREFIX_RE``)."""
    if not isinstance(url_prefix, str) or not _URL_PREFIX_RE.fullmatch(url_prefix):
        raise ValueError(
            "url_prefix must be empty or like /p/demo (path segments of letters, digits, "
            "'_', '~' and '-', no trailing slash): %r" % (url_prefix,)
        )


def install_host_check(app, allowed_hosts):
    """Refuse, on every route of ``app``, a request whose ``Host`` is not in
    ``allowed_hosts``."""

    @app.before_request
    async def _check_host(req):
        # Every route, not only /ws: a rebound DNS name can read /manifest
        # and write through /submit as readily as it can open a socket. A
        # before_request handler that returns a value short-circuits the
        # route entirely, so a refused request never reaches a handler.
        if not host_is_allowed(req, allowed_hosts):
            return refusal()
        # Explicit: microdot treats any returned value as a short-circuit, so
        # "carry on to the route" is expressed by returning nothing at all.
        return None


def install_response_handlers(app, csp_value, server_header, *, front_door=False):
    """The JSON 413, the access log and the headers every response of ``app``
    carries: ``csp_value`` as its Content-Security-Policy, ``no-store``, the
    ``Server`` header and the rest.

    ``front_door`` is for the parent app apps are mounted in
    (``frontdoor.FrontDoor``). A mounted app's own handlers run first on its
    routes and set every header, which the parent's then leave alone (each
    only fills in a header that is missing); the parent's access log skips
    those routes, which the mounted app has logged already, so each request
    is one line. Its 413 handler is the one that answers, even for a mounted
    route: microdot checks the body limit before it looks the route up.
    """

    @app.errorhandler(413)
    async def _payload_too_large(req):
        # microdot's own 413 (request body over Request.max_content_length)
        # is a bare text/plain response; every other failure on /submit
        # returns {"ok": false, "error": ...}, so this keeps that contract
        # for the one failure mode the route handler itself never sees.
        return {
            "ok": False,
            "error": "request body exceeds the %d byte limit" % Request.max_content_length,
        }, 413

    async def _access_log(req, res):
        # One line per request to stderr, independent of stdout (used for
        # the startup banner and the /submit summary), so a server reachable
        # from a remote or Tailscale-bound address gives visible evidence
        # that traffic is arriving even when the human never opens a
        # browser tab locally. ``req`` is None when the request line itself
        # could not be parsed, in which case there is nothing to report but
        # the failure.
        if req is None:
            sys.stderr.write('  ? - "?" %s -\n' % res.status_code)
            return res
        if front_door and req.subapp is not None:
            return res
        addr = req.client_addr[0] if req.client_addr else "-"
        sys.stderr.write(
            '  %s - "%s %s HTTP/%s" %s -\n'
            % (addr, req.method, req.path, req.http_version, res.status_code)
        )
        return res

    async def _security_headers(req, res):
        # A policy rather than a default, because this origin holds an agent
        # with a shell and serves files out of a directory whose contents came
        # from somewhere else. `default-src 'none'` means every fetch a page can
        # make has to be named below, so a source added later fails loudly here
        # rather than working quietly.
        #
        # `script-src` carries a hash rather than 'unsafe-inline' because a
        # product's page has its inline scripts (Mesh's has one, the import
        # map) packaged rather than generated: the hashes are computed at
        # startup from the file that will actually be served, so the two
        # cannot drift.
        #
        # `img-src` allows `data:` because a product page may composite onto a
        # canvas snapshot through an Image whose src is a data URL (Mesh's
        # sketch overlay does), and `connect-src` allows the WebSocket scheme
        # because /ws is the transport. Everything else is same-origin.
        # `frame-ancestors` and `base-uri` are not about this page's own
        # fetches: they stop the page being framed by another origin and stop
        # injected markup relocating every relative URL on the page.
        if "Content-Security-Policy" not in res.headers:
            res.headers["Content-Security-Policy"] = csp_value
        if "Referrer-Policy" not in res.headers:
            # The URL carries the per-run token in its fragment, which is never
            # sent anywhere; this covers the path and query as well.
            res.headers["Referrer-Policy"] = "no-referrer"
        return res

    async def _no_store(req, res):
        # Every response here is either live data or a file that may change
        # between requests (a model regenerated on disk, say); nothing served
        # should ever be cached by the browser. This runs as both
        # after_request and after_error_request, since microdot only routes a
        # response through the first of those two lists, never both,
        # depending on whether the route raised.
        if "Cache-Control" not in res.headers:
            res.headers["Cache-Control"] = "no-store"
        if "Server" not in res.headers:
            res.headers["Server"] = server_header
        # Without this, a browser may sniff a mislabelled asset's bytes and
        # render it as HTML or SVG regardless of the Content-Type this
        # process sent, which is exactly the content-type restriction the
        # /asset route relies on to keep uploaded images from executing as
        # script on this origin.
        if "X-Content-Type-Options" not in res.headers:
            res.headers["X-Content-Type-Options"] = "nosniff"
        return res

    app.after_request(_access_log)
    app.after_error_request(_access_log)
    app.after_request(_no_store)
    app.after_error_request(_no_store)
    app.after_request(_security_headers)
    app.after_error_request(_security_headers)


def _event_publisher(registry, event_log, session_info=None):
    """Return the ``on_event`` callback a session publishes through.

    Appending to the log and broadcasting are one action, not two, and the
    order matters: the seq comes from the log, so it has to be assigned before
    the frame carrying it can be built. A session calls this synchronously from
    its message pump, which is not a place that can await, so the broadcast is
    scheduled as a task rather than awaited here.

    A broadcast that fails must not stop the log from having recorded the
    event: the log is what a reconnecting browser replays from, so an event
    that reached the log but no live socket is recoverable, while the reverse
    is a hole in the history.

    ``session_info``, when given, is the same mutable dict ``register_ws``'s
    ``_greet`` reads fresh on every ``hello`` (``http/ws.py``): an
    ``AgentModelChanged`` here means a live ``set_model`` actually took
    effect, so ``session_info["model"]`` is updated in place before the event
    reaches the log. Without this, a browser tab connecting after the switch
    only recovers the running model while the event that announced it is
    still inside the replay ring buffer; once evicted, a fresh ``hello``
    would otherwise fall back to permanently showing the CLI-configured
    starting model instead of what the session is actually running. The
    latest ``AgentError`` is kept the same way, as
    ``session_info["agent_error"]``, and dropped once an ``AgentStatus``
    says the agent is ready: a page opened while the agent is down learns
    why from the ``hello``, since it raises no banner from replayed history.

    Every ``AgentError`` is also written to this process's stderr
    (``_journal_agent_error``), so a service's journal says why its agent is
    down without anyone opening the page. It is written here, the one place
    every session's events pass through, rather than by each backend.
    """

    # The last AgentError written to stderr, as (remediation, stderr).
    journaled = [None]

    def publish(event):
        if isinstance(event, AgentError):
            _journal_agent_error(event, journaled)
        if session_info is not None and isinstance(event, AgentModelChanged):
            session_info["model"] = event.model
        if session_info is not None and isinstance(event, AgentError):
            session_info["agent_error"] = {
                "remediation": event.remediation,
                "stderr": event.stderr,
            }
        if session_info is not None and isinstance(event, AgentStatus):
            if event.status == AGENT_READY:
                session_info["agent_error"] = None
        seq = event_log.append(event)
        frame = protocol.build_event(seq, event.to_wire())
        asyncio.ensure_future(registry.broadcast(frame))

    return publish


#: How much of a backend's own text one journal entry carries: its end, which
#: is where the reason usually is. The page's banner shows all of it.
JOURNAL_STDERR_LIMIT = 4000


def _journal_agent_error(event, journaled):
    """Write ``event`` to stderr as ``agent error: <remediation>: <stderr>``,
    the backend's own text trimmed and its later lines indented under the
    first, unless it repeats the error written last (``journaled[0]``).

    A burst of identical errors is one fact: each turn sent to an agent that
    never started, or a backend reporting the same failure again, would
    otherwise bury the one line worth reading. A different error in between
    lets the same one be written again, since it then says something new.
    """
    key = (event.remediation, event.stderr)
    if journaled[0] == key:
        return
    journaled[0] = key
    detail = (event.stderr or "").strip()
    if len(detail) > JOURNAL_STDERR_LIMIT:
        detail = "..." + detail[-JOURNAL_STDERR_LIMIT:]
    line = "agent error: %s" % (event.remediation or "the agent reported an error")
    if detail:
        first, *rest = detail.splitlines()
        line += ": " + "\n".join([first] + [("  " + more) if more.strip() else "" for more in rest])
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


#: How often ``serve`` tries a remote MCP server that could not be reached
#: again, after the one immediate retry the session's first turn triggers.
REMOTE_RETRY_INTERVAL = 60.0


async def retry_remotes(tools, bus, session, interval=REMOTE_RETRY_INTERVAL):
    """Try the tool server's unreached remotes again until none is left: once
    as soon as the session's first turn arrives (``bus.first_turn``), and
    every ``interval`` seconds. A remote reached is handed to the session
    through its ``set_tool_table``, and, when the session took it, the model
    is told through a note on its next turn (with what the remote says about
    its own tools, which the system prompt could not carry). Only omp can take
    new tools mid-session: a Claude session's SDK servers and allow list, and
    Codex's bridge entries and tool list, are fixed when it starts, so there
    the remote's tools arrive with the next session start (a restart, which
    ``-c`` resumes). An app passes its ``AgentHolder`` as ``session``, which
    hands the table to whichever session is live when a remote is reached."""
    # Imported here: the tool module loads the agent SDK, which a viewer-only
    # run never does, and this runs only for an agent session with remotes.
    from .tools import instructions_of

    first_turn_seen = False
    # Reached, but not yet handed to the session: a failed set_tool_table is
    # tried again next tick rather than lost, since reconnect has already
    # moved these out of tools.unreached.
    pending = ()
    while tools.unreached or pending:
        if first_turn_seen:
            await asyncio.sleep(interval)
        else:
            try:
                await asyncio.wait_for(bus.first_turn.wait(), interval)
                first_turn_seen = True
            except asyncio.TimeoutError:
                pass
        if tools.unreached:
            try:
                pending += await tools.reconnect()
            except Exception as exc:
                sys.stderr.write("warning: retrying the remote MCP servers failed: %r\n" % (exc,))
        if not pending:
            continue
        names = ", ".join(r.name for r in pending)
        set_tool_table = getattr(session, "set_tool_table", None)
        try:
            taken = set_tool_table is not None and await set_tool_table(tools.host_tool_table())
        except Exception as exc:
            sys.stderr.write(
                "warning: could not give the agent %s's tools yet, trying again: %r\n"
                % (names, exc)
            )
            continue
        reached, pending = pending, ()
        if not taken:
            sys.stderr.write(
                "warning: %s can be reached now, but this agent backend cannot take new tools "
                "mid-session; they arrive when the session is next started\n" % names
            )
            continue
        note = (
            "The %s MCP server, which could not be reached when this session started, "
            "has been reached: its tools are available to you now." % names
        )
        instructions = instructions_of(reached)
        bus.queue_note(note + ("\n\n" + instructions if instructions else ""))


class AgentHolder:
    """One app's live agent session, and the lifecycle and status built
    around it (``app.agent_holder``; ``create_app`` builds it).

    **Why a holder.** A session and its ``PermissionBroker`` run once:
    ``close()`` shuts the broker down for good, after which every ``ask``
    denies. So an app idle past its ``idle_timeout`` closes its session and,
    when a page next connects (``/ws``), a turn arrives or an agent in another
    process calls ``/mcp``, builds a new one from ``resume_session`` instead
    of restarting the old one (``ensure``). Nothing that needs the session
    keeps it: ``/ws``, ``/mcp`` and ``/mcp/<remote>``, ``/agent/logs``, the
    registry's presence listener, the bus's ``end_turn_handler`` and
    ``retry_remotes`` all read this object when they run. ``session`` is the
    live session (``None`` while closed, or for a viewer-only app), and
    ``app.agent_session`` is kept equal to it; ``bus.broker`` is always the
    live session's broker, set by the factory that built it.

    **Lifecycle** (``app.agent_start``/``app.agent_stop``): ``start``
    starts the session once the socket listens and the app's tasks
    (the product's ``background``, the pings, the review watcher, the remote
    retry, the idle sweep); ``stop`` cancels those, closes the session, tells
    every viewer to go away and then runs ``on_stop`` (``app.agent_on_stop``),
    the product's own teardown, sync or async, each one's failure reported
    and the rest still run.

    **Status** (``app.agent_status``): a plain dict for a front door's list of
    apps, with ``status_listeners`` (``app.agent_status_listeners``) called
    with no arguments whenever it changes. It is kept from the events the
    app's event log records (``EventLog.observers``), whichever publisher
    sent them: ``user_turn`` starts a turn and ``turn_end`` ends it, a
    ``permission_request`` waits until its ``permission_resolved``, and an
    ``attention`` waits until the human's next turn.
    """

    def __init__(
        self,
        app,
        *,
        bus,
        registry,
        event_log,
        session_info,
        tools,
        review_watcher,
        make_session,
        resume_session=None,
        idle_timeout=None,
    ):
        self._app = app
        self.bus = bus
        self._registry = registry
        self._session_info = session_info
        self._tools = tools
        self._review_watcher = review_watcher
        self._make_session = make_session
        self._resume_session = resume_session
        self._idle_timeout = (
            float(idle_timeout) if idle_timeout is not None and resume_session is not None else None
        )
        self.session = None
        #: Closed by the idle sweep; the next ``ensure`` resumes it.
        self.closed = False
        self._started = False
        self._stopped = False
        self._tasks = []
        self._start_task = None
        # The idle close in flight or last done, shielded from cancellation
        # so a stop arriving meanwhile waits for it rather than abandoning it.
        self._closing = None
        # The one teardown, shielded likewise: a cancelled agent_stop leaves
        # it running, and a second call waits for the same one.
        self._stop_task = None
        # The model every session this app builds starts on (the run's
        # settings); session_info["model"] follows live switches.
        self._startup_model = session_info.get("model")
        # Held across every change of session (idle close, resume, stop), so
        # a page arriving while one closes waits for it and then resumes.
        self._lock = asyncio.Lock()
        self.on_stop = []
        self.status_listeners = []
        self._viewers = 0
        self._turns = set()
        self._requests = set()
        self._attention = None
        self._last_activity = time.time()
        self._last_active = time.monotonic()
        event_log.observers.append(self._observe)

    # -- the session -----------------------------------------------------------

    def install(self, session):
        """Make ``session`` (or ``None``) the live session."""
        self.session = session
        self._app.agent_session = session
        if session is None:
            self._session_info["agent"] = AGENT_UNAVAILABLE
            return
        # The hello frame publishes whatever the session currently knows, so a
        # tab that connects later sees a ready agent rather than the
        # connecting state session_info was built with.
        self._session_info["agent"] = session.agent_status()
        self._session_info["steers"] = bool(getattr(session, "steers", False))

    async def ensure(self):
        """The live session, after resuming one if the idle sweep closed the
        last: a new session from ``resume_session``, started in the
        background (the page learns when it is ready from its
        ``agent_status`` events, as it does at startup). A factory that
        raises leaves the app closed, reported, for the next caller to try
        again."""
        if self.closed and not self._stopped:
            async with self._lock:
                if self.closed and not self._stopped:
                    self._resume()
        return self.session

    def _resume(self):
        # The model the human last switched to, which the new session does not
        # start on: every factory builds from the run's settings.
        wanted = self._session_info.get("model")
        try:
            session = self._make_session(self._resume_session)
        except Exception as exc:
            sys.stderr.write("error: could not resume the agent session: %r\n" % (exc,))
            return
        self.closed = False
        self.install(session)
        if session is not None:
            # What the new session runs until the switch is made again: the
            # hello says so, and the switch's own event puts it back.
            self._session_info["model"] = self._startup_model
            if self._viewers:
                # A turn frame resuming on a connection already open: the new
                # session's broker has to know a page is there to answer it.
                session.on_viewer_presence(self._viewers)
            self._start_task = asyncio.ensure_future(
                _start_session(session, model=wanted if wanted != self._startup_model else None)
            )
        self._note_change()

    async def _close_idle(self):
        async with self._lock:
            remaining = self._idle_remaining()
            if remaining is None or remaining > 0:
                return
            session, start = self.session, self._start_task
            self._start_task = None
            # Closed before the session is: a page arriving meanwhile waits on
            # the lock and then resumes, rather than getting the one closing.
            self.closed = True
            self.install(None)
            self._turns.clear()
            # Shielded: a stop cancelling this sweep waits for the close
            # rather than abandoning it halfway (a backend's child left behind).
            self._closing = asyncio.ensure_future(_end_session(start, session))
            await asyncio.shield(self._closing)
        self._note_change()

    # -- what reads the session at call time -------------------------------------

    def on_presence(self, count):
        """The registry's presence listener: ``count`` pages are connected."""
        self._viewers = count
        try:
            if self.session is not None:
                self.session.on_viewer_presence(count)
        finally:
            self._note_change()

    def end_turn(self):
        """``bus.end_turn_handler``: a tool result asked to end the turn."""
        handler = getattr(self.session, "end_turn_after_tool", None)
        if handler is not None:
            handler()

    async def set_tool_table(self, table):
        """What ``retry_remotes`` hands a reached remote's tools to: the live
        session's ``set_tool_table``, or False (not taken) for a session that
        cannot take tools mid-session or while the app is closed, since the
        next session is built with every remote reached by then."""
        set_table = getattr(self.session, "set_tool_table", None)
        if set_table is None:
            return False
        return await set_table(table)

    def note_activity(self):
        """Something used this app without changing its status (a call through
        ``/mcp``): it is not idle."""
        self._note_change()

    # -- lifecycle ---------------------------------------------------------------

    async def start(self, background=()):
        """Start the session and this app's tasks; once only, and only once
        the socket the app is served on is listening.

        Listening first matters: a Codex backend's start() launches
        session/codex.py's stdio-to-HTTP MCP proxy (codex_mcp_stdio_bridge.py)
        as a subprocess of the app-server it also launches, and that proxy's
        first tools/list call reaches this app's own /mcp route while Codex's
        own session startup is still in progress. Starting the session before
        the socket accepts connections would point that first call at a port
        nothing is listening on yet, which Codex treats as the MCP server
        having failed, leaving every product tool unavailable for the rest of
        the session. A failure inside start() is reported as an event and
        never raised, so this cannot stop the page from being served either
        way.

        ``background`` is the product's own long-running coroutine functions,
        each called and started as a task once the session has started, and
        cancelled by ``stop``.
        """
        if self._started:
            raise RuntimeError("this app has already been started")
        self._started = True
        if self.session is not None:
            await self.session.start()
        # Started here rather than in create_app, because create_app is called
        # by tests that have no running loop to own a background task and no
        # interest in one; a task per constructed app would leak a task per
        # test.
        tasks = [asyncio.ensure_future(run()) for run in background]
        tasks.append(asyncio.ensure_future(ping_forever(self._registry)))
        if self._review_watcher is not None:
            tasks.append(asyncio.ensure_future(self._review_watcher.run()))
        if self.session is not None and self._tools is not None and self._tools.unreached:
            tasks.append(asyncio.ensure_future(retry_remotes(self._tools, self.bus, self)))
        if self._idle_timeout is not None:
            tasks.append(asyncio.ensure_future(self._sweep_idle()))
        self._tasks = tasks

    async def stop(self):
        """Cancel the app's tasks, close its session, close every viewer's
        socket, then run ``on_stop``. Idempotent: the teardown runs once, as
        a task of its own that cancelling a caller does not cancel, so a
        second call waits for the same one; and the ``on_stop`` hooks run
        however the rest of it ends, so a product's lock is released even
        then."""
        if self._stop_task is None:
            self._stop_task = asyncio.ensure_future(self._stop())
        await asyncio.shield(self._stop_task)

    async def _stop(self):
        self._stopped = True
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        try:
            async with self._lock:
                # Before the viewers are told to go away, so a permission
                # request still open is denied and its event reaches the
                # browser on the socket that is about to close, rather than
                # vanishing with it. An idle close under way (the sweep just
                # cancelled) is finished, not abandoned.
                if self.closed:
                    if self._closing is not None:
                        await asyncio.shield(self._closing)
                else:
                    start, self._start_task = self._start_task, None
                    await _end_session(start, self.session)
                self._turns.clear()
            # Viewers are told before the listener closes, so a browser
            # reconnects or falls back at once instead of waiting out its
            # liveness timeout, and so the bounded drain of the server is not
            # spent waiting on WebSocket handlers that would never return on
            # their own.
            await self._registry.close_all()
        finally:
            self._note_change()
            await self._run_on_stop()

    async def _run_on_stop(self):
        """Every ``on_stop`` hook, each one's failure reported and the rest
        still run, a cancellation meanwhile included (re-raised after)."""
        cancelled = None
        for hook in list(self.on_stop):
            try:
                result = hook()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError as exc:
                cancelled = exc
            except Exception as exc:
                sys.stderr.write("warning: a stop hook failed: %r\n" % (exc,))
        if cancelled is not None:
            raise cancelled

    async def _sweep_idle(self):
        interval = min(self._idle_timeout, IDLE_SWEEP_INTERVAL)
        while True:
            remaining = self._idle_remaining()
            if remaining is not None and remaining <= 0:
                await self._close_idle()
                remaining = None
            await asyncio.sleep(interval if remaining is None else min(interval, remaining))

    def _idle_remaining(self):
        """Seconds until the idle sweep closes this app, or ``None`` while
        something keeps it open: a page, a turn, a permission request, or
        there being no open session to close."""
        if self._idle_timeout is None or self.closed or self._stopped or self.session is None:
            return None
        if self._viewers or self._turns or self._requests:
            return None
        return self._idle_timeout - (time.monotonic() - self._last_active)

    # -- status ------------------------------------------------------------------

    def status(self):
        """``{"agent", "turn_running", "waiting", "attention",
        "last_activity", "viewers"}``: the agent's status (connecting, ready,
        unavailable, or ``AGENT_CLOSED``), whether a turn is running, whether
        the agent waits on the human (a permission request open, or an
        ``attention`` raised since their last turn, whose ``title: body`` is
        ``attention``), when anything last changed (epoch seconds), and how
        many pages are connected."""
        if self._stopped or self.closed:
            agent = AGENT_CLOSED
        elif self.session is None:
            agent = AGENT_UNAVAILABLE
        else:
            agent = self.session.agent_status()
        return {
            "agent": agent,
            "turn_running": bool(self._turns),
            "waiting": bool(self._requests) or self._attention is not None,
            "attention": self._attention,
            "last_activity": self._last_activity,
            "viewers": self._viewers,
        }

    def _observe(self, wire):
        kind = wire.get("kind")
        if kind == "user_turn":
            self._turns.add(wire.get("turn"))
            self._attention = None
        elif kind == "turn_end":
            self._turns.discard(wire.get("turn"))
        elif kind == "permission_request":
            self._requests.add(wire.get("request_id"))
        elif kind == "permission_resolved":
            self._requests.discard(wire.get("request_id"))
        elif kind == "attention":
            self._attention = "%s: %s" % (wire.get("title", ""), wire.get("body", ""))
        elif kind != "agent_status":
            return
        self._note_change()

    def _note_change(self):
        self._last_activity = time.time()
        self._last_active = time.monotonic()
        for listen in list(self.status_listeners):
            try:
                listen()
            except Exception as exc:
                sys.stderr.write("warning: an app status listener failed: %r\n" % (exc,))


async def _start_session(session, model=None):
    """Start a resumed ``session`` and, when ``model`` is given, switch it
    back to that model: the one the human chose before the app closed, which
    its factory knew nothing of. The switch publishes ``agent_model_changed``
    like one made from the page."""
    try:
        await session.start()
    except Exception as exc:
        sys.stderr.write("warning: the agent session did not start: %r\n" % (exc,))
        return
    if model is None:
        return
    if session.agent_status() != AGENT_READY:
        sys.stderr.write(
            "warning: the resumed agent session is not ready, so it stays on its "
            "starting model rather than %s\n" % model
        )
        return
    try:
        await session.set_model(model)
    except Exception as exc:
        sys.stderr.write(
            "warning: could not switch the resumed agent session back to %s: %r\n" % (model, exc)
        )


async def _end_session(start, session):
    """Cancel ``start`` (a session's start task, or None) if still running,
    then close ``session`` (or nothing, for None)."""
    if start is not None and not start.done():
        start.cancel()
        try:
            await start
        except asyncio.CancelledError:
            pass
    if session is None:
        return
    try:
        await session.close()
    except Exception as exc:
        sys.stderr.write("warning: the agent session did not close cleanly: %r\n" % (exc,))


async def serve(app, host, port, on_ready=None, background=()):
    """Serve ``app`` on ``host``:``port`` until interrupted.

    Binds with ``start_serving=False`` and then explicitly awaits
    ``Server.start_serving()`` before calling ``on_ready``: binding alone
    only creates the socket, the kernel does not call ``listen()`` on it
    until serving actually starts, so a caller connecting between bind and
    that call would see a refused connection. ``on_ready`` therefore
    describes a socket that is already accepting connections, not one that
    merely will be. ``on_ready`` may be a plain function or a coroutine
    function; if calling it returns an awaitable, that awaitable is awaited
    before this function proceeds, which lets a caller offload blocking
    work (such as opening a browser) onto the event loop's executor instead
    of running it inline on the loop that is meant to already be serving.

    Between the two, the app is started (``app.agent_start``, with
    ``background``: the product's own long-running coroutine functions, Mesh's
    models watcher's ``run`` for one), and on the way out it is stopped
    (``app.agent_stop``) before the listener closes. Functions rather than
    coroutines, so nothing is created that a failed bind would leave never
    awaited. The review watcher (``app.agent_review_watcher``) is started
    the same way without being listed, since ``create_app`` built it.

    ``start_serving()`` alone is enough to keep the server accepting
    connections; nothing further needs to run for that to continue, so this
    then simply blocks until the caller cancels the task (``KeyboardInterrupt``
    in the CLI). ``asyncio.Server.serve_forever()`` is deliberately not used
    for that block: its own ``CancelledError`` handler runs an *unbounded*
    ``close()`` plus ``wait_closed()`` internally before re-raising, which
    would defeat ``SHUTDOWN_DRAIN_TIMEOUT`` below by blocking on any
    still-open connection before this function's own bounded wait ever gets
    a chance to run. Shutdown, on ``KeyboardInterrupt`` or task cancellation,
    stops accepting new connections and waits up to
    ``SHUTDOWN_DRAIN_TIMEOUT`` for in-flight requests to finish; past that
    bound it returns anyway, since microdot has no way to cut an in-flight
    request off short of dropping the connection.

    An app built with an identity (``create_app(identity=...)``) is refused
    here too (``ValueError``) when ``host`` is not loopback: this is the
    address actually bound, and ``create_app``'s ``host`` only decided its
    allowlists.
    """
    auth = getattr(app, "agent_auth", None)
    check_bind(getattr(auth, "identity", None), net.bind_from_address(host))
    server = await app.start_server(host=host, port=port, start_serving=False)
    await server.start_serving()
    try:
        await app.agent_start(background)
        if on_ready is not None:
            result = on_ready()
            if inspect.isawaitable(result):
                await result
        await asyncio.Event().wait()
    finally:
        await app.agent_stop()
        await close_server(server)


async def close_server(server):
    """Stop ``server`` accepting connections and wait up to
    ``SHUTDOWN_DRAIN_TIMEOUT`` for the requests in flight."""
    server.close()
    try:
        await asyncio.wait_for(server.wait_closed(), timeout=SHUTDOWN_DRAIN_TIMEOUT)
    except asyncio.TimeoutError:
        pass
