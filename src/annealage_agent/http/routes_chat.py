"""Route handlers for what the chat pane sends the server, an image or a
request to write the conversation out, and for reading those images back.

Registers, against one served directory:

    POST /upload                  accepts one raw image (PNG, JPEG or WEBP)
                                   into images/, named by this process alone
    GET  /asset/<path:rel>        image bytes, restricted to an images/ subtree
                                   (uploads, captured views, and anything a
                                   project already keeps there)
    POST /session/<sid>/export    renders that session's event log into review/

The export route deliberately does not go through the permission broker, while
the ``export_transcript`` tool the model calls does. That asymmetry is the
point: a human pressing their own Export button has already decided, and a card
asking them to approve what they just clicked teaches them to click cards
without reading. The model asking for the same write has decided nothing on
the human's behalf, so it asks.

Refuses, in the order checked: an absent or wrong token, or ``t`` given more
than once (the same ``ws.refusal()`` response ``/ws`` returns, so an
unauthenticated caller cannot tell this route apart from any other); a
disallowed ``Origin``, checked here as well as on ``/ws`` because this is the
one route in the process that writes into the human's project and a POST
carrying a raw body is a CORS simple request, so no preflight stands in front
of it; an unknown query key, ``kind`` given twice, or ``kind`` outside its
whitelist; a missing or zero Content-Length; a Content-Length over
``files.MAX_IMAGE_BYTES``; a body whose bytes are not a PNG, JPEG or WEBP;
and, once past every check above, whatever ``files.create_unique_image_file``
itself refuses (an ``images/`` entry that cannot be written into, or every
generated name already taken), plus a write that fails partway or a body
shorter than its declared length.

Nothing is left on disk by a refusal, including one decided after the file
exists and including a caller that simply disappears: every exit past the
open goes through ``_abandon``.

The body is raw bytes, not multipart: microdot ships no multipart parser, and
the one thing a client chooses about an upload, its kind (``upload``, or one
the installed product adds through ``Product.upload_kinds``, e.g. Mesh's
``sketch``), fits in a query parameter. The written file's name, including its
extension, never comes from the client; see ``files.create_unique_image_file``
and ``files.sniff_image``.

``/asset`` asks for no token, as it never has: an image under images/ is the
project's own, git-tracked file, fetched by the page's ``<img>`` tags, which
cannot carry one. What it does enforce is containment, entirely in
``files.resolve_asset``, and a content type from the narrow
``files.ASSET_CONTENT_TYPES`` so nothing under images/ is ever labelled as
active content on this origin.
"""

import asyncio
import functools
import os
import re
import sys
from urllib.parse import unquote

from .. import files, product, sessions
from ..session import events
from . import CHUNK_SIZE, file_response, read_json_body
from .ws import _origin_is_allowed, _token_is_allowed, refusal


class _Refused(Exception):
    """A refusal decided after the file exists, carrying its own response.

    Raised rather than returned so that one handler, the one that also closes
    the descriptor and removes the file, decides every exit past the open.
    """

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# The kind every product accepts: an image the human attached in the
# composer. It is also what an absent "kind" means.
GENERIC_UPLOAD_KIND = "upload"

# What a product's own upload kind may look like. The kind becomes the first
# component of the written file's name (``files.create_unique_image_file``),
# so it is held to a short lowercase slug that keeps every generated name
# inside ``files._IMAGE_NAME_RE``; checked once, when the product is
# installed, rather than trusted because a product wrote it. Applied with
# ``fullmatch``: ``$`` would let a trailing newline through.
_PRODUCT_KIND_RE = re.compile(r"[a-z][a-z0-9-]{0,31}")


def check_product_upload_kinds(kinds):
    """Raise ``ValueError`` if ``kinds`` cannot be a product's upload kinds: a
    bare string rather than a tuple of them (``("sketch")`` without its comma
    would otherwise register one kind per letter), the generic kind again, a
    kind given twice, or one that is not wholly a short lowercase slug.
    Changes nothing; called by ``product.install``."""
    if isinstance(kinds, str):
        raise ValueError("product upload kinds must be a tuple of kinds, not the string %r" % kinds)
    seen = set()
    for kind in kinds:
        if not isinstance(kind, str) or not _PRODUCT_KIND_RE.fullmatch(kind):
            raise ValueError("product upload kind %r is not a short lowercase slug" % (kind,))
        if kind == GENERIC_UPLOAD_KIND or kind in seen:
            raise ValueError("product upload kind %r collides with another kind" % kind)
        seen.add(kind)


def upload_kinds():
    """The values the "kind" query parameter may take, the generic one first.

    Held here, not in a product's own files, because it is this route's own
    query parameter being validated, not a property of a file under images/;
    the product only names the extra kinds its page sends (Mesh: ``sketch``).
    """
    return (GENERIC_UPLOAD_KIND,) + tuple(product.current().upload_kinds)


_ALLOWED_QUERY_KEYS = frozenset(("t", "kind"))


def _query_values(req, key):
    """Every value ``key`` was given in ``req``'s query string, in order.

    ``req.args`` is a plain ``{}`` when the request had no query string at
    all, and a ``MultiDict`` (which has ``getlist``) whenever it did, even if
    the string was empty. The same two-shape handling ``ws.py``'s own token
    check already does for ``t``.
    """
    if hasattr(req.args, "getlist"):
        return req.args.getlist(key)
    return [req.args[key]] if key in req.args else []


def _upload_kind(req):
    """Return ``(kind, None)`` or ``(None, error)`` for this request's query string.

    Only ``t`` and ``kind`` may appear, ``kind`` at most once and only from
    ``upload_kinds()``; anything else is refused without reading any of the
    body. Called only once the token has already passed, so this whitelist
    is never a way to probe the route unauthenticated.
    """
    keys = set(req.args.keys()) if hasattr(req.args, "keys") else set()
    unknown = sorted(keys - _ALLOWED_QUERY_KEYS)
    if unknown:
        return None, "unknown query parameter: %s" % ", ".join(unknown)
    kind_values = _query_values(req, "kind")
    if len(kind_values) > 1:
        return None, "kind must not be given more than once"
    allowed = upload_kinds()
    kind = kind_values[0] if kind_values else GENERIC_UPLOAD_KIND
    if kind not in allowed:
        return None, "kind must be one of: %s" % ", ".join(allowed)
    return kind, None


def _write_all(fd, data):
    """Write every byte of ``data`` to ``fd``, looping past a short write.

    ``os.write`` is not guaranteed to write its whole argument in one call.
    Run through an executor by the caller, like every other blocking call
    this route makes.
    """
    written = 0
    while written < len(data):
        written += os.write(fd, data[written:])


def _finish(fd):
    """Flush ``fd`` to disk and close it, once every byte has been written."""
    os.fsync(fd)
    os.close(fd)


def _abandon(fd, target):
    """Close ``fd`` and remove ``target``, best-effort, after a failed upload.

    Called once the file already exists and something after that point
    refused the upload, so a rejected request leaves nothing on disk for
    ``/asset`` to serve. Both steps are best-effort: a failure closing or
    removing a file this process just created and is walking away from is
    not a new error worth surfacing over whatever refusal is already being
    returned to the client.
    """
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.unlink(target)
    except OSError:
        pass


async def _receive(req, loop, fd, first, content_length):
    """Stream the rest of the body onto ``fd`` and return the total written.

    Raises ``_Refused`` for a body that stops short of its declared length or
    a write that fails, and propagates anything the stream itself raises. The
    caller owns ``fd``, so nothing here closes it: a peer that vanishes
    mid-upload makes ``read`` raise, and that has to reach the one place that
    knows there is a file to remove.
    """
    total = len(first)
    try:
        await loop.run_in_executor(None, _write_all, fd, first)
    except OSError as exc:
        raise _Refused(500, "write failed: %s" % exc) from exc

    while total < content_length:
        chunk = await req.stream.read(min(CHUNK_SIZE, content_length - total))
        if not chunk:
            raise _Refused(400, "body ended after %d of %d bytes" % (total, content_length))
        try:
            await loop.run_in_executor(None, _write_all, fd, chunk)
        except OSError as exc:
            raise _Refused(500, "write failed: %s" % exc) from exc
        total += len(chunk)

    try:
        await loop.run_in_executor(None, _finish, fd)
    except OSError as exc:
        raise _Refused(500, "write failed: %s" % exc) from exc
    return total


def _export_options(data):
    """Return ``(fmt, include, None)`` or ``(None, None, error)`` for an export
    request body.

    Both fields are optional and both are checked against the renderer's own
    tuples, with the allowed values named in the refusal: a client that asked
    for ``"md"`` needs telling that the word is ``"markdown"``, not that its
    request was invalid.
    """
    unknown = sorted(set(data) - {"format", "include"})
    if unknown:
        return None, None, "unknown body field: %s" % ", ".join(unknown)
    fmt = data.get("format", events.TRANSCRIPT_FORMATS[0])
    include = data.get("include", events.TRANSCRIPT_INCLUDE[0])
    if fmt not in events.TRANSCRIPT_FORMATS:
        return None, None, ("format must be one of: %s" % ", ".join(events.TRANSCRIPT_FORMATS))
    if include not in events.TRANSCRIPT_INCLUDE:
        return None, None, ("include must be one of: %s" % ", ".join(events.TRANSCRIPT_INCLUDE))
    return fmt, include, None


async def _file_or_same_404(target, ctype, method, request_key, expect_identity=None):
    """Stream ``target``, or fall back to the caller's own miss message.

    The caller has already confirmed ``target`` exists as a file; the only
    way ``file_response`` still 404s from here is an open failure (a
    permission error, or the file vanishing in the race between that check
    and this call). That failure must read exactly like the caller's own
    "the path never resolved to anything" 404 for ``request_key`` - not
    file_response's own generic body - so a client cannot use a difference
    in wording to tell "does not exist" apart from "exists but is
    unreadable" for a path it does not have permission to see the real
    answer to.
    """
    res = await file_response(target, ctype, method, expect_identity)
    if res.status_code == 404:
        return "not found: %s" % request_key, 404
    return res


def register_chat_routes(app, serve_dir, *, token, allowed_origins=()):
    """Register ``POST /upload``, ``GET /asset/<rel>`` and
    ``POST /session/<sid>/export`` on ``app``."""
    serve_dir = files.resolve_serve_dir(serve_dir)

    @app.post("/upload")
    async def upload(req):
        # Token then Origin, the same order and the same opaque refusal /ws
        # uses, so neither check tells an unauthenticated caller which one it
        # failed.
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()

        kind, error = _upload_kind(req)
        if error is not None:
            return {"ok": False, "error": error}, 400

        content_length = req.content_length
        if content_length <= 0:
            return {
                "ok": False,
                "error": "Content-Length is required and must be greater than zero",
            }, 411
        if content_length > files.MAX_IMAGE_BYTES:
            return {
                "ok": False,
                "error": "body is %d bytes, over the %d "
                "byte image limit" % (content_length, files.MAX_IMAGE_BYTES),
            }, 413

        loop = asyncio.get_running_loop()

        # Read only as far as a verdict needs, looping rather than taking one
        # read as the answer because a real socket's read(n) can return far
        # fewer than n bytes even mid-transfer, and treating a short first read
        # as the whole head would refuse a perfectly good, merely slow, upload.
        # The loop stops at files.SNIFF_MIN_BYTES because the sniffer looks no
        # further: reading a declared 8 MiB to the end before refusing its
        # first twelve bytes would let any holder of the token make this
        # process buffer the whole cap per connection, and nothing here limits
        # concurrent connections or times out a slow read.
        want = min(files.SNIFF_MIN_BYTES, content_length)
        first = b""
        while len(first) < want:
            chunk = await req.stream.read(min(CHUNK_SIZE, content_length - len(first)))
            if not chunk:
                break
            first += chunk
        sniff = files.sniff_image(first)
        if sniff is None:
            return {
                "ok": False,
                "error": "not a recognised image; only PNG, JPEG and WEBP are accepted",
            }, 415
        media_type, suffix = sniff

        try:
            created = await loop.run_in_executor(
                None, files.create_unique_image_file, serve_dir, kind, suffix
            )
        except FileExistsError:
            return {
                "ok": False,
                "error": "could not find a free name in "
                "images/ after %d attempts" % files.UNIQUE_IMAGE_NAME_ATTEMPTS,
            }, 500
        except OSError as exc:
            # A directory this process cannot write into, a full filesystem, a
            # quota. Answered in the same shape as every other refusal here,
            # because a caller parsing JSON gets nothing it can show a human
            # out of microdot's plain-text 500.
            return {"ok": False, "error": "could not create the file: %s" % exc}, 500
        if created is None:
            return {
                "ok": False,
                "error": "refusing to write: %s/ must be a "
                "real directory this process can create a file in" % files.IMAGES_DIRNAME,
            }, 500
        fd, target = created

        # One exit for everything past the open. The stream reads inside
        # _receive raise ConnectionResetError when a peer disappears
        # mid-upload, which a closed tab, an aborted fetch or a dropped link
        # all produce routinely, and a cancelled task raises too. Either one
        # escaping would leave the descriptor open until the process hits
        # EMFILE and no upload ever succeeds again, and would leave a
        # truncated image in a git-tracked directory that /asset serves and
        # the model can attach later.
        try:
            total = await _receive(req, loop, fd, first, content_length)
        except BaseException as exc:
            await loop.run_in_executor(None, _abandon, fd, target)
            if isinstance(exc, _Refused):
                return {"ok": False, "error": exc.message}, exc.status
            raise

        return {
            "ok": True,
            "path": "images/%s" % target.name,
            "url": "/asset/%s" % target.name,
            "bytes": total,
            "media_type": media_type,
        }, 200

    @app.get("/asset/<path:rel>")
    async def asset(req, rel):
        found = files.resolve_asset(serve_dir, unquote(rel))
        if found is None:
            return "not found: %s" % rel, 404
        target, identity = found
        ctype = files.ASSET_CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        return await _file_or_same_404(target, ctype, req.method, rel, identity)

    @app.post("/session/<sid>/export")
    async def export_session(req, sid):
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()

        loop = asyncio.get_running_loop()
        # Checked before the body is read, and against this project's own
        # records rather than the filesystem, so an id that names a session
        # belonging to some other directory is a 404 here rather than a path
        # this route then tries to read.
        known = await loop.run_in_executor(None, sessions.get_session_info, serve_dir, sid)
        if known is None:
            return {"ok": False, "error": "no session %r in this project" % sid}, 404

        data, error = await read_json_body(req, allow_empty=True)
        if error is not None:
            return error
        if not isinstance(data, dict):
            return {"ok": False, "error": "body must be a JSON object"}, 400
        fmt, include, message = _export_options(data)
        if message is not None:
            return {"ok": False, "error": message}, 400

        try:
            target = await loop.run_in_executor(
                None,
                functools.partial(
                    events.export_transcript, serve_dir, sid, fmt=fmt, include=include
                ),
            )
        except OSError as exc:
            # A review/ that is a symlink or a regular file, a full disk, a
            # directory this process cannot write into. The reason reaches the
            # human, since they are the only one who can fix any of them.
            sys.stderr.write("error: could not export transcript: %s\n" % exc)
            return {"ok": False, "error": "could not write the transcript: %s" % exc}, 500

        written = await loop.run_in_executor(None, target.stat)
        return {
            "ok": True,
            "path": "%s/%s" % (files.REVIEW_DIRNAME, target.name),
            "bytes": written.st_size,
            "format": fmt,
            "include": include,
        }, 200
