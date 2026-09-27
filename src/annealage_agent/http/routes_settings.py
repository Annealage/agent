"""Routes behind the settings window: what this run is using, what is saved,
and the diagnostics beside them.

Registers, against one served directory:

    GET  /settings   every setting's effective value and provenance, what is
                      saved on disk but not in effect, how the window lays the
                      keys out, and the diagnostics block
    PUT  /settings   validate a batch of changes and write each to its own layer

Both go through the app's ``identity.BrowserAuth`` (a permitted ``Origin``,
then a tailnet login or the browser token), and refuse with the same
opaque response ``/ws`` returns, so neither tells an unauthenticated caller
which check it failed. ``PUT`` is privilege-relevant beyond the obvious: it can
change the bind address for the next run, so it validates and rejects rather
than merging what it is handed, and refuses outright to write a token or
``bypassPermissions`` (``settings.apply`` owns both refusals).

**Effective and saved are different questions, and this route answers both.**
For a restart-effect key, what is in effect is what this process resolved at
startup: reporting a freshly read file instead would claim a port the server is
not listening on. A settings window that showed only that could not confirm a
write it had just made, so anything now on disk that differs is reported under
``pending``, which is how the window says "9000 saved, this run is still on
8765" rather than silently discarding the write or lying about it. A file edited
by hand while the server runs surfaces the same way.

A load-effect key works the other way round, because each page applies it as it
loads: for a browser asking now, the value on disk is the one in effect, so
those keys report the fresh resolution and never appear in ``pending``.
Reporting the startup value for them would leave every later page applying a
preference the human had already changed.

Diagnostics duplicates the product's ``doctor`` command (Mesh:
``annealage-mesh doctor``) on purpose: a remote user
looking at a phone has no terminal to run it in, which is exactly when knowing
whether the ``claude`` binary was found matters. Both read one collector, so
they cannot drift.
"""

import asyncio
import functools
import sys

from .. import diagnostics, product
from .. import settings as settings_module
from . import read_json_body
from .ws import refusal

#: The settings a mounted app neither applies nor saves: the front door it is
#: served under owns the process's one listening socket.
BIND_KEYS = ("host", "port")

#: Why, as the settings window shows it and a refused write says it.
MOUNTED_BIND_REASON = (
    "this workspace is served by a front door that owns the address and port "
    "for every workspace, so it has no host or port of its own"
)


def register_settings_routes(
    app,
    serve_dir,
    *,
    auth,
    settings=None,
    session_id=None,
    bind=None,
    port=None,
    mounted=False,
):
    """Register ``GET`` and ``PUT /settings`` on ``app``, gated by ``auth``
    (``identity.BrowserAuth``).

    ``settings`` is the ``settings.Resolved`` this run started with, and the
    flags that produced it travel on it, so a re-resolution here reports the
    same layer as being in effect rather than promoting a file's value over a
    flag that still outranks it.

    ``session_id``, ``bind`` and ``port`` are passed straight through to the
    diagnostics collector, which is the only consumer of them here, and so is
    the running session's ``backend_logs``, read off ``app.agent_session`` per
    request because the session is built after these routes are registered.

    ``mounted`` is an app served under a front door's URL prefix
    (``create_app(url_prefix=...)``). Its ``host`` and ``port`` (``BIND_KEYS``)
    are reported as not in effect and not editable (``"in_effect": false``
    with the reason beside it, and never ``pending``), and a ``PUT`` naming
    either is refused whole, saying why.
    """
    if settings is None:
        settings = settings_module.resolve(serve_dir)

    async def _payload():
        """The response body both routes return, built off the event loop.

        ``diagnostics.collect`` runs ``git --version`` and ``claude
        --version`` and reads the lock file, so it goes through the executor
        like every other blocking call in this package. Re-resolving the
        settings reads two small TOML files and goes with it.
        """
        loop = asyncio.get_running_loop()
        session = getattr(app, "agent_session", None)
        collect = functools.partial(
            diagnostics.collect,
            serve_dir,
            session_id=session_id,
            bind=bind,
            port=port,
            backend=settings["backend"],
            omp_base_url=settings["omp_base_url"],
            omp_api_key=settings["omp_api_key"],
            backend_logs=getattr(session, "backend_logs", None),
        )
        facts = await loop.run_in_executor(None, collect)
        wire = settings.to_wire()
        if mounted:
            for name in BIND_KEYS:
                wire[name].update(editable=False, in_effect=False, reason=MOUNTED_BIND_REASON)
        installed = product.current()
        body = {
            "ok": True,
            "settings": wire,
            # How the window lays the keys out, from each key's own section
            # (``settings.sections``), so a product's key is shown where the
            # product declared it without the window naming it.
            "sections": settings_module.sections(),
            # The names the window's prose needs and cannot know on its own:
            # which product's version the diagnostics block reports, and the
            # state directory the project config file lives in.
            "product": {
                "title": installed.title,
                "version": installed.version,
                "state_dirname": installed.state_dirname,
            },
            "diagnostics": facts,
        }

        # What is on disk now, which after a write is not what this run started
        # with. A hand-edited file that no longer parses must not take the whole
        # window down with it, since the window is where someone would go to
        # find out what is wrong, so the failure is reported as a field rather
        # than raised.
        try:
            saved = await loop.run_in_executor(
                None, functools.partial(settings_module.resolve, serve_dir, flags=settings.flags)
            )
        except settings_module.SettingsError as exc:
            body["pending"] = {}
            body["saved_error"] = str(exc)
            return body

        # "In effect" means something different per key, and conflating the two
        # is what would make this route lie. A restart-effect key is in effect
        # as this process resolved it at startup, whatever the file says now. A
        # load-effect key is applied by each page as it loads, so for a page
        # asking right now the value on disk *is* the one in effect, and
        # reporting the startup value would leave the browser applying a stale
        # preference forever.
        pending = {}
        for name, entry in wire.items():
            if entry.get("in_effect") is False:
                continue
            if entry["effect"] == "load":
                entry["value"] = saved[name]
                entry["from"] = saved.provenance(name)
            elif saved[name] != entry["value"]:
                pending[name] = {"value": saved[name], "from": saved.provenance(name)}
        body["pending"] = pending
        return body

    @app.get("/settings")
    async def get_settings(req):
        if auth.authenticate(req) is None:
            return refusal()
        return await _payload(), 200

    @app.put("/settings")
    async def put_settings(req):
        if auth.authenticate(req) is None:
            return refusal()

        data, error = await read_json_body(req)
        if error is not None:
            return error
        if not isinstance(data, dict):
            return {"ok": False, "error": "body must be a JSON object"}, 400
        unknown = sorted(set(data) - {"changes"})
        if unknown:
            return {"ok": False, "error": "unknown body field: %s" % ", ".join(unknown)}, 400
        changes = data.get("changes")
        if not isinstance(changes, dict):
            return {
                "ok": False,
                "error": 'body must be {"changes": {...}} '
                "with an object of setting names to values",
            }, 400
        if not changes:
            return {"ok": False, "error": "changes is empty; nothing to write"}, 400
        refused = sorted(name for name in BIND_KEYS if name in changes) if mounted else ()
        if refused:
            return {
                "ok": False,
                "error": "%s cannot be set here: %s" % (" and ".join(refused), MOUNTED_BIND_REASON),
            }, 400

        loop = asyncio.get_running_loop()
        try:
            _resolved, written = await loop.run_in_executor(
                None,
                functools.partial(settings_module.apply, serve_dir, changes, flags=settings.flags),
            )
        except settings_module.SettingsError as exc:
            # Every refusal this route makes past the body's shape comes from
            # here, in one message written for whoever is reading the window.
            return {"ok": False, "error": str(exc)}, 400
        except OSError as exc:
            sys.stderr.write("error: could not write settings: %s\n" % exc)
            return {"ok": False, "error": "could not write the settings file: %s" % exc}, 500

        body = await _payload()
        body["written"] = written
        return body, 200
