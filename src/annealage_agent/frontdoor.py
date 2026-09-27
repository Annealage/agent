"""One server for many agent apps: a front door that mounts each under its own
URL prefix, ``/p/<id>/``, beside a page of the product's own listing them.

Each mounted app is a whole ``app.create_app`` app, built with
``url_prefix="/p/<id>"`` and ``login=front.login``, with its own served
directory, session, event log, review store, tool server and background
tasks. What the apps share is what belongs to the process: the one listening
socket (so a mounted app's own host and port settings are refused,
``http/routes_settings.py``), the browser and agent tokens, and one
``LoginNonces``, so a nonce issued anywhere opens any app. microdot's
``mount(app, prefix, local=True)`` keeps each app's own handlers (its Host
check, headers and access log) on its own routes; the front door has the same
set of its own for everything else, a path under a prefix that names no route
included, and its 413 handler answers for every app, since microdot refuses an
oversized body before it looks the route up.

The front door starts every app it holds once it is listening (``serve``),
and one mounted while it serves is started then (``mount``, or
``await front.start_app(id)`` to wait for that); each app's idle timeout,
where it has one, runs on its own (``app.AgentHolder``); on the way out every
app is stopped, then the server. Building an app blocks for as long as its
remote MCP servers take to answer (up to ``remote.DISCOVERY_TIMEOUT``), so a
product that adds one while others are served calls ``create_app`` through
``asyncio.to_thread`` and then ``mount`` on the event loop.

A product's page served under a prefix reaches the agent layer's modules
through the import map entry ``"agent/": "./agent/static/"`` (relative, so it
resolves under the prefix; the policy's ``base-uri 'none'`` rules out a
``<base>``), and names every route of its own relative to the page, through
``agent/url.js``'s ``appUrl``, as the agent layer's modules do.
"""

import asyncio
import inspect
import re
import sys
from pathlib import Path

from microdot import Microdot, Response

from . import app as agent_app
from . import net, product
from .http.routes_login import LoginNonces, register_login_routes, register_whoami_route
from .http.static import register_agent_static_routes
from .http.ws import refusal
from .identity import BrowserAuth, check_bind

#: An app id: one path segment, used literally in a URL and in a route. No
#: ``.``: microdot puts a route's static text into its regex unescaped, so an
#: app ``a.b`` would also answer for ``/p/axb/...``.
_APP_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def mount_prefix(app_id):
    """Where ``FrontDoor.mount`` puts the app ``app_id``: the ``url_prefix``
    to build it with."""
    return "/p/%s" % app_id


class FrontDoor:
    """The parent app (``self.app``, a ``Microdot``) apps are mounted in, and
    the server they are all served from.

    ``page_html`` is the product's front page, served at ``/``, whose inline
    scripts the Content-Security-Policy hashes as ``create_app``'s does.
    ``token`` is the browser token every mounted app was built with, which
    ``GET /apps`` accepts too, and ``agent_token`` their agent token.
    ``identity`` is the ``identity.TailscaleIdentity`` every mounted app was
    built with (``None``: none), refused with a bind that is not loopback as
    ``create_app`` refuses it. ``self.app.agent_auth`` is the front door's
    ``identity.BrowserAuth``, set before ``register_routes`` is called, for
    the product's own routes on it.
    ``host``, ``port``, ``extra_origins`` and ``extra_hosts`` are the bind
    and the names a proxy fronts it under, exactly as each app was given
    them. ``login`` is the ``LoginNonces`` every app shares (``front.login``;
    a fresh one when ``None``), which also answers ``POST /login`` at the
    root. ``register_routes(app, allowed_origins)`` registers the product's
    own routes on the front door (a "new workspace" form's POST, say).

    Routes of its own: ``GET /`` (the page), ``GET /apps`` (``apps()`` as
    JSON, for the page to poll), ``GET /whoami``, ``POST /login``,
    ``/agent/static/`` (the agent layer's front end, for the page), and
    ``/p/<id>``, which redirects to ``/p/<id>/`` so the page's relative URLs
    resolve under its app.
    """

    def __init__(
        self,
        page_html,
        *,
        token,
        agent_token,
        host,
        port,
        extra_origins=(),
        extra_hosts=(),
        login=None,
        register_routes=None,
        identity=None,
    ):
        if agent_token is not None and agent_token == token:
            raise ValueError("the agent token must differ from the browser token")
        agent_app.configure_request_limits()
        page_html = Path(page_html)
        self.host = host
        self.port = port
        self.login = login if login is not None else LoginNonces()
        self.identity = identity
        bind = net.bind_from_address(host)
        check_bind(identity, bind)
        allowed_origins = net.allowed_origins(bind, port, extra_origins)
        auth = BrowserAuth(token, identity, allowed_origins=allowed_origins)
        self.app = Microdot()
        self.app.agent_auth = auth
        self._apps = {}
        self._background = {}
        self._starts = {}
        self._serving = False
        agent_app.install_host_check(self.app, net.allowed_hosts(bind, port, extra_hosts))

        if register_routes is not None:
            register_routes(self.app, allowed_origins)
        page = page_html.read_bytes()

        @self.app.get("/")
        async def front_page(req):
            return Response(page, headers={"Content-Type": "text/html; charset=utf-8"})

        @self.app.get("/apps")
        async def list_apps(req):
            if auth.authenticate(req) is None:
                return refusal()
            return self.apps(), 200

        register_agent_static_routes(self.app)
        register_login_routes(
            self.app, token=token, nonces=self.login, allowed_origins=allowed_origins
        )
        register_whoami_route(self.app, auth=auth)
        agent_app.install_response_handlers(
            self.app,
            agent_app.content_security_policy(page_html),
            product.current().server_header,
            front_door=True,
        )

    def mount(self, app_id, app, *, background=()):
        """Serve ``app`` under ``/p/<app_id>/``; call on the event loop.

        ``app`` must have been built with ``url_prefix=mount_prefix(app_id)``
        (every address it gives out assumes it), ``login=self.login`` (a
        nonce the front door issues must open it) and ``identity=`` the front
        door's (a login the front page signs in must be the one the app takes).
        ``background`` is the
        product's own long-running coroutine functions for this app, started
        with it (``app.agent_start``). An id already mounted is refused: there
        is no unmounting, microdot copies an app's routes in when it is
        mounted. Mounted while the front door serves, the app is started at
        once, as a task; ``await start_app(app_id)`` waits for it.
        """
        if not isinstance(app_id, str) or not _APP_ID_RE.fullmatch(app_id):
            raise ValueError(
                "an app id is one path segment of letters, digits, '_' and '-': %r" % (app_id,)
            )
        if app_id in self._apps:
            raise ValueError("an app is already mounted as %r" % app_id)
        prefix = mount_prefix(app_id)
        if getattr(app, "agent_url_prefix", None) != prefix:
            raise ValueError(
                "the app for %r must be built with url_prefix=%r, not %r"
                % (app_id, prefix, getattr(app, "agent_url_prefix", None))
            )
        if getattr(app, "agent_login", None) is not self.login:
            raise ValueError(
                "the app for %r must be built with login=front.login, so the front "
                "door's nonces open it" % app_id
            )
        if getattr(getattr(app, "agent_auth", None), "identity", None) is not self.identity:
            raise ValueError(
                "the app for %r must be built with the front door's identity, so the "
                "logins that sign in to the front page are the ones it takes" % app_id
            )
        self.app.mount(app, prefix, local=True)

        async def to_directory(req):
            # The page's own URLs are relative to it, so it is only ever
            # served with the trailing slash.
            location = prefix + "/"
            if req.query_string:
                location += "?" + req.query_string
            return Response.redirect(location)

        self.app.get(prefix)(to_directory)
        self._apps[app_id] = app
        self._background[app_id] = tuple(background)
        if self._serving:
            task = self._starts[app_id] = asyncio.ensure_future(
                app.agent_start(self._background[app_id])
            )
            task.add_done_callback(lambda done: _report_start(app_id, done))

    async def start_app(self, app_id):
        """Start the app mounted as ``app_id``, or wait for the start already
        under way (``serve`` and ``mount`` start them; this is for a caller that
        needs the app started before it goes on)."""
        task = self._starts.get(app_id)
        if task is None:
            app = self._apps[app_id]
            task = self._starts[app_id] = asyncio.ensure_future(
                app.agent_start(self._background[app_id])
            )
        await task

    def apps(self):
        """``{app id: app.agent_status()}`` for every app mounted."""
        return {app_id: app.agent_status() for app_id, app in self._apps.items()}

    async def serve(self, on_ready=None):
        """Serve every mounted app until interrupted, as ``app.serve`` serves
        one: listening before ``on_ready`` (a function or coroutine function)
        is called and before any app is started; on the way out, every app
        stopped (``app.agent_stop``, which runs its ``agent_on_stop``), then
        the listener closed, draining for at most
        ``app.SHUTDOWN_DRAIN_TIMEOUT``. The apps are stopped even when the
        bind fails, so their own teardown (a workspace lock) still runs."""
        server = None
        try:
            server = await self.app.start_server(
                host=self.host, port=self.port, start_serving=False
            )
            await server.start_serving()
            self._serving = True
            await self._start_all()
            if on_ready is not None:
                result = on_ready()
                if inspect.isawaitable(result):
                    await result
            await asyncio.Event().wait()
        finally:
            self._serving = False
            await self._stop_all()
            if server is not None:
                await agent_app.close_server(server)

    async def _start_all(self):
        ids = list(self._apps)
        results = await asyncio.gather(
            *(self.start_app(app_id) for app_id in ids), return_exceptions=True
        )
        for app_id, result in zip(ids, results, strict=True):
            if isinstance(result, Exception):
                sys.stderr.write("error: the app %s did not start: %r\n" % (app_id, result))

    async def _stop_all(self):
        starting = [task for task in self._starts.values() if not task.done()]
        for task in starting:
            task.cancel()
        await asyncio.gather(*starting, return_exceptions=True)
        ids = list(self._apps)
        results = await asyncio.gather(
            *(self._apps[app_id].agent_stop() for app_id in ids), return_exceptions=True
        )
        for app_id, result in zip(ids, results, strict=True):
            if isinstance(result, Exception):
                sys.stderr.write(
                    "warning: the app %s did not stop cleanly: %r\n" % (app_id, result)
                )


def _report_start(app_id, task):
    """Write why the app mounted as ``app_id`` did not start, if it did not:
    one mounted while serving is started as a task nobody may await."""
    if not task.cancelled() and task.exception() is not None:
        sys.stderr.write("error: the app %s did not start: %r\n" % (app_id, task.exception()))
