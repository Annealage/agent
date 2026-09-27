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

A product's own application module (Mesh's ``app.py``, for one) calls both,
adding its routes and the background tasks that watch its files.
"""

import asyncio
import base64
import hashlib
import inspect
import re
import sys

from microdot import Microdot, Request

from . import net, product, protocol, sessions
from . import settings as settings_module
from .http.routes_chat import register_chat_routes
from .http.routes_login import LoginNonces, register_login_routes
from .http.routes_logs import register_log_routes
from .http.routes_mcp import register_mcp_routes
from .http.routes_review import register_review_routes
from .http.routes_settings import register_settings_routes
from .http.static import register_agent_static_routes
from .http.ws import host_is_allowed, ping_forever, refusal, register_ws
from .review.watcher import ReviewWatcher
from .session import secret_paths
from .session.base import (
    AGENT_READY,
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

    ``token`` is the browser token: the only credential ``/ws``, the chat
    routes, ``/settings`` and every product route that asks for one accept.
    ``None`` means no token was configured, and ``/ws`` then refuses every
    request. ``agent_token`` is the separate per-run agent token, the only
    credential ``/mcp`` accepts; ``None`` means ``/mcp`` refuses every
    request. The two are kept apart because the agent token is handed to a
    process beside the agent's own shell (the Codex stdio bridge) and the
    browser token authorises permission decisions, so a run whose two tokens
    are equal is refused here rather than built. ``net.load_token`` keeps a
    browser token in a file for a service whose link must survive restarts.

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
    """
    if agent_token is not None and agent_token == token:
        raise ValueError(
            "the agent token must differ from the browser token: it is handed to "
            "processes beside the agent's shell, and the browser token approves "
            "permission requests"
        )
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
    allowed_origins = net.allowed_origins(bind, port, extra_origins)
    allowed_hosts = net.allowed_hosts(bind, port, extra_hosts)
    installed = product.current()
    server_header = installed.server_header
    if (session_id is not None or external_agents) and installed.build_tools is None:
        raise RuntimeError(
            "%s has no tool builder (Product.build_tools), so an app serving its "
            "tools to an agent cannot be built for it" % installed.distribution
        )

    app = Microdot()

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

    if settings is None:
        settings = settings_module.resolve(serve_dir)
    app.agent_settings = settings

    if register_routes is not None:
        register_routes(app, allowed_origins)
    register_chat_routes(app, serve_dir, token=token, allowed_origins=allowed_origins)
    register_agent_static_routes(app)
    app.agent_login = login if login is not None else LoginNonces()
    register_login_routes(app, token=token, nonces=app.agent_login, allowed_origins=allowed_origins)
    register_settings_routes(
        app,
        serve_dir,
        token=token,
        allowed_origins=allowed_origins,
        settings=settings,
        session_id=session_id,
        bind=bind.address,
        port=port,
    )
    register_review_routes(app, store=review_store, token=token, allowed_origins=allowed_origins)

    # Set after the session exists, since the broker it belongs to is built by
    # the session factory below; a list with one slot rather than a nonlocal so
    # the closure reads the current value instead of capturing None.
    presence_listener = []

    def _presence(count):
        for listen in presence_listener:
            listen(count)

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
        url=net.server_url(bind, port),
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
    # either module importing the other.
    session = None
    if build_session is not None:
        session = build_session(_event_publisher(registry, event_log, session_info), bus=bus)
    app.agent_session = session
    if session is None and external_agents:
        # Imported here, like the backends' own sessions: a product that never
        # asks for this pays nothing for it.
        from .session.external import ExternalAgentSession
        from .session.permissions import PermissionBroker

        publish = _event_publisher(registry, event_log, session_info)
        # The broker a real session's factory would have built (launch.py),
        # over the same grants file, and set on the bus for the same reason:
        # /mcp below and a review tool that asks the human read it there.
        bus.broker = PermissionBroker(
            publish,
            permissions_path=sessions.state_dir(serve_dir) / "permissions.toml",
            viewer_url=bus.url,
            timeout=float(settings["approval_timeout"]),
            never_remembered=tools.never_remembered,
        )
        session = app.agent_session = ExternalAgentSession(publish, bus.broker)
    if session is not None:
        # Both of these read the session, so both are inside this guard: a
        # factory returning None is the ordinary viewer-only case, not a
        # failure, and it must leave an app that serves the product's page
        # with no agent attached rather than one that could not be built.
        presence_listener.append(session.on_viewer_presence)
        # The hello frame publishes whatever the session currently knows, so a
        # tab that connects later sees a ready agent rather than the
        # connecting state this dict was built with.
        session_info["agent"] = session.agent_status()
        session_info["steers"] = bool(getattr(session, "steers", False))
        # A tool result's end_turn (tools.ok) reaches the session through
        # the bus; a session that cannot stop a turn has no handler.
        bus.end_turn_handler = getattr(session, "end_turn_after_tool", None)
        # /mcp is mounted here, not unconditionally above, for the same
        # reason register_ws's own bus= is None until a session exists: a
        # viewer-only app has no tools and no broker to gate them, so there
        # is nothing for this route to serve. tools is never None here
        # (built above whenever session_id is not None, which every real
        # caller - the CLI's build_session, and every test fixture that
        # wants a real session - must set for exactly this reason); broker
        # is bus.broker, set by build_session's own closure while
        # constructing PermissionBroker - None only if a caller supplied a
        # build_session that never sets it, in which case register_mcp_routes
        # fails closed on every write-class call rather than gating with no
        # broker at all.
        register_mcp_routes(
            app,
            tools=tools,
            broker=bus.broker,
            agent_token=agent_token,
            allowed_origins=allowed_origins,
        )

    # Listed from whatever session this run ended up with; a viewer-only run
    # still answers, with nothing to list.
    register_log_routes(app, session=session, token=token, allowed_origins=allowed_origins)

    register_ws(
        app,
        token=token,
        allowed_origins=allowed_origins,
        allowed_hosts=allowed_hosts,
        registry=registry,
        event_log=event_log,
        session_info=session_info,
        session=session,
        bus=bus if session is not None else None,
    )

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

    return app


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
    ``-c`` resumes)."""
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

    ``background`` is the product's own long-running coroutine functions
    (Mesh: its models watcher's ``run``), each called and started as a task
    once the session has started, and cancelled on the way out. Functions
    rather than coroutines, so nothing is created that a failed bind would
    leave never awaited. The review watcher (``app.agent_review_watcher``)
    is started the same way without being listed, since ``create_app``
    built it.

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
    """
    server = await app.start_server(host=host, port=port, start_serving=False)
    await server.start_serving()
    # Started after the listening socket is already accepting connections,
    # not before: a Codex backend's start() launches
    # session/codex.py's stdio-to-HTTP MCP proxy
    # (codex_mcp_stdio_bridge.py) as a subprocess of the app-server it also
    # launches, and that proxy's first tools/list call reaches this
    # process's own /mcp route while Codex's own session startup is still
    # in progress. Every route is already registered by create_app, above,
    # well before this point, but the socket itself only starts accepting
    # connections here; starting the session before this would point that
    # first call at a port nothing is listening on yet, which Codex treats
    # as the MCP server having failed, leaving every product tool unavailable
    # for the rest of the session. A failure inside start() is reported as
    # an event and never raised, so this cannot stop the page from being
    # served either way.
    if app.agent_session is not None:
        await app.agent_session.start()
    # Started here rather than in create_app, because create_app is called by
    # tests that have no running loop to own a background task and no interest
    # in one; a task per constructed app would leak a task per test.
    tasks = [asyncio.ensure_future(start()) for start in background]
    tasks.append(asyncio.ensure_future(ping_forever(app.agent_registry)))
    if app.agent_review_watcher is not None:
        tasks.append(asyncio.ensure_future(app.agent_review_watcher.run()))
    if app.agent_session is not None and app.agent_tools is not None and app.agent_tools.unreached:
        tasks.append(
            asyncio.ensure_future(retry_remotes(app.agent_tools, app.agent_bus, app.agent_session))
        )
    if on_ready is not None:
        result = on_ready()
        if inspect.isawaitable(result):
            await result
    try:
        await asyncio.Event().wait()
    finally:
        for task in tasks:
            task.cancel()
        if app.agent_session is not None:
            # Before the viewers are told to go away, so a permission request
            # still open is denied and its event reaches the browser on the
            # socket that is about to close, rather than vanishing with it.
            await app.agent_session.close()
        # Viewers are told before the listener closes, so a browser reconnects
        # or falls back at once instead of waiting out its liveness timeout,
        # and so the bounded drain below is not spent waiting on WebSocket
        # handlers that would never return on their own.
        await app.agent_registry.close_all()
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=SHUTDOWN_DRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            pass
