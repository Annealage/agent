"""Serving files through a cached, scanned index: the agent layer's own front
end, a product's page and modules, and anything else a product indexes.

Registers, on every app the agent layer builds:

    GET  /agent/static/<path:rel>   the agent layer's own front end (the chat
                                    pane, settings window, socket client and
                                    the store they share), from ``static/``

``/agent/static/`` is a prefix of its own, outside a product's ``/static/``,
so the two trees can never answer for each other: each is a separate
``files.StaticIndex`` with its own cache, and neither route can resolve a
name the other's scan found. A product's page imports these modules through
its import map (Mesh maps the bare prefix ``agent/`` to this route), which is
what lets the prefix move without a product module changing.

The rest of this module is the machinery every indexed route shares, whether
it serves a packaged asset or a product's own index of a served directory
(Mesh's models): ``make_index_cache`` scans off the event loop and reuses the
result for a short window, and ``serve_indexed`` resolves one file through it,
rescanning once when the file on disk is not the one the scan validated.
"""

import asyncio
import os
import stat
import time
from pathlib import Path
from urllib.parse import unquote

from .. import files
from . import file_response, not_modified

#: The agent layer's own static tree, and the URL prefix it is served under.
AGENT_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
AGENT_STATIC_PREFIX = "/agent/static/"

# How long a scanned index (a static tree's, or a product's index of the
# served directory) is reused before the next request triggers a fresh scan.
# One page load fetches the shell, the stylesheets and every script the shell
# imports, plus whatever the product then loads; without this, that single
# page load would pay for one full walk per file instead of one walk shared
# across all of them. The trade this makes: a file added, removed or changed
# on disk, or a static file edited during development, can take up to this
# long to be reflected, rather than appearing on the very next request.
INDEX_CACHE_TTL = 1.0


def make_index_cache(build, target):
    """Return a ``(get, invalidate)`` pair caching one ``build(target)`` scan.

    ``build`` is ``files.build_static_index`` or a product's own index builder
    (Mesh: ``paths.build_model_index``), called with ``target`` (a static tree
    or a served directory) off the event loop, its result reused for up to
    ``INDEX_CACHE_TTL`` seconds. This is the one caching implementation every
    indexed route goes through, each with its own call to this factory rather
    than a second copy of the logic, since the shape (scan off-loop, cache
    with a TTL, share one in-flight scan, allow an explicit invalidate) is
    identical whatever is scanned. A separate call is a separate cache, which
    is what keeps two trees from ever answering for each other.

    Only one scan is ever in flight at a time. A cache miss stores the
    scan's ``Task`` before awaiting it, so every request that arrives while
    that scan is still running awaits the same task instead of launching its
    own; without this, a page load's simultaneous requests would each pay for
    a full scan during every cold window, and a burst of concurrent requests
    during one cold window would each launch a separate walk, all competing
    for the same default executor that file reads and writes also depend on.
    """
    cache = {"index": None, "at": 0.0, "pending": None}

    async def _scan_and_cache():
        loop = asyncio.get_running_loop()
        idx = await loop.run_in_executor(None, build, target)
        cache["index"] = idx
        cache["at"] = time.monotonic()
        return idx

    async def get():
        now = time.monotonic()
        cached = cache["index"]
        if cached is not None and now - cache["at"] < INDEX_CACHE_TTL:
            return cached
        pending = cache["pending"]
        if pending is not None:
            return await pending
        loop = asyncio.get_running_loop()
        task = loop.create_task(_scan_and_cache())
        cache["pending"] = task
        try:
            return await task
        finally:
            cache["pending"] = None

    def invalidate():
        """Drop the cached scan so the next request rescans.

        Called when a file's bytes turn out not to be the ones the scan
        validated, which is what a regenerated file looks like: a build step
        or a script writing a new file and renaming it over the old one leaves
        a different inode behind. Rescanning immediately keeps the workflow of
        regenerating a file, or editing a script during development, and
        reloading the page, rather than making the page wait out the cache
        window on a stale identity.
        """
        cache["index"] = None
        cache["at"] = 0.0

    return get, invalidate


def _lstat_or_none(path):
    """``os.lstat`` returning None instead of raising, for a file that may have
    gone between an index scan and now."""
    try:
        return os.lstat(path)
    except OSError:
        return None


async def maybe_not_modified(req, target):
    """Return a 304 if the client's ``If-None-Match`` matches ``target``, else None.

    The tag is computed from a fresh ``lstat`` of the indexed path rather than
    from whatever the cached scan recorded, so a file edited within the index
    cache window is never affirmed as unchanged. That costs one stat per
    conditional request, which is what a megabyte of vendored JavaScript is
    worth avoiding on a phone over a tailnet. Because the tag describes the
    file itself (its inode, size and modification time), two trees serving a
    file under the same relative name never share one.

    ``lstat`` rather than ``stat``: a symlink appearing at an indexed name is
    refused here for the same reason the open path refuses it, so this cannot
    become a way to have a link's target validated. A tag mismatch simply
    falls through to the ordinary serve path, which applies the full open-time
    identity check, so a wrong or stale client tag costs a transfer and never
    a wrong answer.
    """
    header = req.headers.get("If-None-Match")
    if not header:
        return None
    loop = asyncio.get_running_loop()
    st = await loop.run_in_executor(None, _lstat_or_none, target)
    if st is None or not stat.S_ISREG(st.st_mode):
        return None
    tags = [t.strip() for t in header.split(",")]
    # "*" asks for a 304 if any representation exists at all, which one
    # confirmed here does.
    if "*" in tags or files.file_validator(st) in tags:
        return not_modified(st)
    return None


async def serve_indexed(
    get_idx, invalidate, req, request_key, lookup, content_type_for, revalidatable=False
):
    """Serve one file through a cached index, rescanning once on a mismatch.

    ``get_idx`` and ``invalidate`` come from ``make_index_cache``, so this
    function has no idea what kind of file it is serving; ``lookup(idx)``
    returns the ``(target, identity)`` pair for whichever index it was given,
    and everything below is the retry policy every indexed route shares.

    The scan pins each file's inode and the open refuses anything else,
    which is what stops a name being relinked to a file outside the scanned
    directory between the two. A regenerated file is indistinguishable from
    that at the moment of the open, so one retry after a fresh scan
    separates them: a real regeneration resolves to the new file, while a
    relinked outside file fails the scan's own containment and link checks
    and stays unreachable.

    ``revalidatable`` is set by the routes serving packaged assets, which are
    large, change only when the package does, and are fetched again on every
    page load. It is not set for a product's served files; see
    ``file_response``.
    """
    for attempt in (0, 1):
        idx = await get_idx()
        target, identity = lookup(idx)
        if target is None:
            if attempt == 0:
                invalidate()
                continue
            return "not found: %s" % request_key, 404
        if revalidatable:
            unchanged = await maybe_not_modified(req, target)
            if unchanged is not None:
                return unchanged
        res = await file_response(
            target, content_type_for(target), req.method, identity, revalidatable
        )
        if res.status_code != 404:
            return res
        if attempt == 0:
            invalidate()
    return "not found: %s" % request_key, 404


def static_tree(static_dir):
    """Return ``serve(req, rel, request_key=None)`` for one packaged static tree.

    Each call builds its own ``files.StaticIndex`` cache over ``static_dir``,
    so a product's tree and the agent layer's are separate scans, separate
    caches and separate namespaces. ``rel`` is the already-decoded relative
    path looked up in the index; ``request_key`` is what a 404 names, which a
    route passes as the raw path it was given (default: ``rel``). Every file
    served this way carries a validator and asks to be revalidated.
    """
    get_idx, invalidate = make_index_cache(files.build_static_index, static_dir)

    async def serve(req, rel, request_key=None):
        return await serve_indexed(
            get_idx,
            invalidate,
            req,
            rel if request_key is None else request_key,
            lambda idx: (idx.by_rel(rel), idx.identity_of(rel)),
            lambda target: files.StaticIndex.content_type_of(rel),
            revalidatable=True,
        )

    return serve


def register_agent_static_routes(app):
    """Register ``GET /agent/static/<path:rel>`` on ``app``."""
    serve = static_tree(AGENT_STATIC_DIR)

    @app.get(AGENT_STATIC_PREFIX + "<path:rel>")
    async def agent_static(req, rel):
        return await serve(req, unquote(rel), rel)
