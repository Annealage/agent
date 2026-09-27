"""Microdot route modules for the agent layer, and the helpers they share.

Two things live here because more than one route module needs them: reading a
JSON request body off the unbuffered stream, and returning file bytes without
blocking the event loop.

``microdot.Response.send_file`` and its own file-object and sync-generator
body handling both read the file synchronously on the event loop (see
``Response.body_iter`` in microdot's source: a file object is pumped with
blocking ``read()`` calls, not scheduled through an executor). A large file
(a product's model, say) served that way stalls every other request, and later, every other in-flight
tool call and the token stream, since they all share this one loop. Every
route in this package that returns file bytes therefore builds its response
through ``file_response`` below instead of ``send_file``.
"""

import asyncio
import errno
import fcntl
import json
import os
import stat
import sys

from microdot import Response

from .. import files

# Read chunk size for file responses. Large enough that a multi-megabyte
# file needs only a few dozen executor round trips rather than thousands, and
# small enough that no single in-flight response holds more than a quarter
# megabyte of file content in memory at once.
CHUNK_SIZE = 256 * 1024


# Ceiling on a JSON request body. Every JSON route in this package carries
# either a settings change, an export request or a product's own form (Mesh's
# page of pin comments), none of which is large; the cap exists so a body with a declared
# length is refused before it is read rather than after, since a request that
# holds the token can otherwise make this process buffer whatever it declares.
MAX_JSON_BODY = 256 * 1024


async def read_json_body(req, *, max_bytes=MAX_JSON_BODY, allow_empty=False):
    """Read and parse one JSON request body. Returns ``(data, None)`` or
    ``(None, (payload, status))`` for the caller to return unchanged.

    The body is read off ``req.stream`` rather than ``req.body`` because
    ``app.py`` sets ``Request.max_body_length`` to 0, so microdot buffers
    nothing (``req.body``, ``req.json`` and ``req.form`` raise on a request
    with a body: ``app.StreamedRequest``). The declared length is checked
    first and unconditionally: on a real connection ``req.stream`` is the raw
    client reader, and reading it with no declared length never returns.

    ``allow_empty`` accepts a request with no body at all and yields ``{}``,
    for a route whose every field has a default.
    """
    content_length = req.content_length
    if content_length <= 0:
        if allow_empty and content_length == 0:
            return {}, None
        return None, (
            {"ok": False, "error": "Content-Length is required and must be greater than zero"},
            411,
        )
    if content_length > max_bytes:
        return None, (
            {
                "ok": False,
                "error": "body is %d bytes, over the %d byte "
                "limit for a JSON request" % (content_length, max_bytes),
            },
            413,
        )
    chunks = []
    total = 0
    while total < content_length:
        chunk = await req.stream.read(min(CHUNK_SIZE, content_length - total))
        if not chunk:
            return None, (
                {"ok": False, "error": "body ended after %d of %d bytes" % (total, content_length)},
                400,
            )
        chunks.append(chunk)
        total += len(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8")), None
    except (ValueError, UnicodeDecodeError) as exc:
        return None, ({"ok": False, "error": "invalid JSON: %s" % exc}, 400)


async def _iter_file_chunks(fh, total, chunk_size=CHUNK_SIZE):
    """Async generator yielding up to ``total`` bytes from open file ``fh``.

    Each read is a blocking filesystem call, run through the default
    executor so the event loop keeps servicing other connections while this
    file is read. ``total`` is the size observed by ``file_response`` at
    open time; the generator never yields more than that many bytes even if
    the file grows while being read, so the declared ``Content-Length`` and
    the delivered body always agree regardless of what happens to the file
    on disk after the response has started.
    """
    loop = asyncio.get_running_loop()
    remaining = total
    try:
        while remaining > 0:
            chunk = await loop.run_in_executor(None, fh.read, min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        await loop.run_in_executor(None, fh.close)


async def file_response(
    path, content_type, method="GET", expect_identity=None, revalidatable=False
):
    """Build a Response that streams ``path``'s bytes without blocking the loop.

    The file is opened, and its size taken from that open descriptor via
    ``fstat``, before any header is constructed, so a file that vanishes or
    becomes unreadable between the caller's existence check and this call
    produces a 404 with no headers sent yet, rather than a committed 200
    whose promised ``Content-Length`` the body then fails to deliver.

    ``content_type`` is supplied by the caller (from ``paths.CONTENT_TYPES``)
    rather than guessed here, since the extension-to-type mapping is
    path-safety-adjacent policy that belongs with the rest of that logic.

    ``method`` distinguishes a HEAD request, which microdot's own
    ``Response.write`` never iterates or closes the body of (it sends
    headers only). For HEAD, the file is stat'd rather than opened, so no
    descriptor is left for garbage collection to eventually close; a caller
    that always opened the file regardless of method would leak one file
    descriptor per HEAD request until the interpreter got around to
    finalising the unread file object.

    ``revalidatable`` adds an ``ETag`` and ``Cache-Control: no-cache`` so a
    client can skip the transfer next time by asking. It is opt-in per route
    rather than the default, and only the packaged static assets set it. The
    tag is computed from the stat of the descriptor actually being read, so it
    describes exactly the bytes this response delivers rather than what a
    scan saw earlier. ``no-cache`` means "store it, but ask me before reusing
    it", which is what keeps a module edited during development from being
    served out of a browser cache while still saving the megabyte of vendored
    JavaScript on every reload. Model bytes deliberately do not set this: a
    model is the artefact under review and is regenerated constantly, so a
    stale one is the failure this tool exists to avoid.
    """
    loop = asyncio.get_running_loop()
    if method == "HEAD":
        try:
            st = await loop.run_in_executor(None, os.stat, path)
        except OSError as exc:
            sys.stderr.write("warning: could not stat %s for a HEAD response: %s\n" % (path, exc))
            return Response("not found", 404)
        headers = {"Content-Type": content_type, "Content-Length": str(st.st_size)}
        if revalidatable:
            headers.update(_revalidation_headers(st))
        return Response(body=b"", headers=headers)
    try:
        fh = await loop.run_in_executor(None, _open_regular_file, path, expect_identity)
    except OSError as exc:
        # The body carries neither the resolved path nor a distinction
        # between "does not exist" and "exists but could not be opened"
        # (permissions, a broken symlink target): either detail would let
        # an unauthenticated client probe the filesystem through this
        # route's own error message. The real path and reason go to stderr
        # instead, for whoever is running the server to see.
        sys.stderr.write("warning: could not open %s for a file response: %s\n" % (path, exc))
        return Response("not found", 404)
    st = os.fstat(fh.fileno())
    headers = {"Content-Type": content_type, "Content-Length": str(st.st_size)}
    if revalidatable:
        headers.update(_revalidation_headers(st))
    return Response(body=_iter_file_chunks(fh, st.st_size), headers=headers)


def _revalidation_headers(st):
    """Validator and caching headers for a response a client may revalidate."""
    return {
        "ETag": files.file_validator(st),
        "Cache-Control": "no-cache",
    }


def not_modified(st):
    """A bodyless 304 for a file a client's cached copy already matches.

    Carries the same validator and caching headers a 200 for that file would,
    because a client that revalidates once has to be able to revalidate again:
    a 304 that dropped them would leave the cached entry with no way to be
    checked a third time, and the next load would transfer the whole file.
    """
    return Response(body=b"", status_code=304, headers=_revalidation_headers(st))


def _open_regular_file(path, expect_identity=None):
    """Open ``path`` for reading, refusing anything that is not a plain file.

    ``expect_identity`` is the ``(st_dev, st_ino)`` a caller already validated
    for this path. When given, the opened descriptor must match it, which is
    what refuses a file relinked into place after that validation and before
    this open.

    A route decides a path is servable and this opens it a moment later, on a
    different thread, so whatever the name refers to can change in between.
    Two shapes matter. A symlink appearing there would redirect the read, and
    ``O_NOFOLLOW`` refuses it. A FIFO would block the executor thread inside
    the kernel until a writer appeared, which for a served directory nobody is
    writing to means forever, and enough of those exhaust the pool and stop
    every route that needs it; ``O_NONBLOCK`` lets the open return so the
    ``S_ISREG`` check below can reject it.

    ``O_NONBLOCK`` is cleared afterwards because it is a property of the open
    file description, and leaving it set would let a read on a slow
    filesystem return short or fail with EAGAIN rather than blocking, which
    the chunked body iterator does not expect. It is inert on a regular file
    once the file is known to be one.
    """
    fd = os.open(path, os.O_RDONLY | files.OPEN_GUARD_FLAGS)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "not a regular file: %s" % path)
        if expect_identity is not None and (st.st_dev, st.st_ino) != expect_identity:
            raise OSError(errno.EINVAL, "changed since it was checked: %s" % path)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb")
