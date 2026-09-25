"""Turning an ``image_path`` block into something the model can see.

An ``image_path`` block (``{"type": "image_path", "path": "images/<name>"}``)
is this project's own invention, checked by ``protocol.py``'s
``_BLOCK_SPECS`` on the way in but meaningless to the CLI, which speaks only
the Anthropic content-block shapes. ``expand_turn_blocks`` is what turns
such a block into something the model can actually see: an inline base64
``image`` block, standing next to a ``text`` block naming the same
``images/<name>`` string, so the model can later hand that path to its own
tools. One without the other is half of fact 6's dual delivery, not a
smaller version of it: pixels with no path leave the model unable to name
what it saw, and a path with no pixels is not an attachment at all.

Ordering is not a preference. Fact 6 requires every image block to precede
whatever text refers to it; the observed failure when that is violated is
the model ignoring the attached image outright and going hunting the
filesystem for a file matching words in the trailing text instead of
looking at what it was handed. ``expand_turn_blocks`` therefore reorders
unconditionally: every block this expansion produces for an ``image_path``
(an accepted pair, or a rejection note standing in for one) is placed before
every ``text`` block the caller supplied, regardless of how the two were
interleaved on the wire.

The entry point is ``expand_turn_blocks``. It is a pure function over a block
list and a directory, which is why it lives here rather than in ``sdk.py``:
that module owns the client, the options and the message pump, and none of the
reasoning below is about any of the three.
"""

import base64
import os
from typing import Any, Tuple

from .. import files

# Per-turn cap on how many image_path blocks are expanded. store.js's own
# MAX_CHAT_ATTACHMENTS is 4, so a turn built by the shipped composer never
# reaches this; it is what still refuses a turn frame assembled by hand over
# the ``/ws`` protocol, or a future caller of ``submit_turn`` that does not
# share that policy. A block beyond the cap is dropped, not silently: one
# text note names every path dropped this way.
MAX_TURN_IMAGES = 4

# Cap on the sum of decoded (pre-base64) bytes this expansion will inline in
# one turn, checked against a running total as attachments are accepted, in
# encounter order, so a later small attachment can still fit after an earlier
# large one has used most of the budget rather than one oversized image
# starving every attachment behind it.
#
# Twice ``files.MAX_INLINE_IMAGE_BYTES``, which is deliberately less than
# ``MAX_TURN_IMAGES`` times it: four attachments are allowed, but four at the
# per-image ceiling would put about 19 MiB of base64 into the single JSON line
# ``submit_turn`` writes and the CLI child must read whole before it can act on
# any of it, so this budget trims such a turn well before the count cap alone
# would.
MAX_TURN_IMAGE_BYTES = 2 * files.MAX_INLINE_IMAGE_BYTES

# An ``image_path`` block's ``path`` is always "images/<name>" (plan section
# D): project-relative, one directory component, so the model can hand this
# exact string to its own tools later. Stripping this prefix is the first of
# two containment checks, and the only one that runs with no filesystem
# access at all: anything not shaped this way is refused on the string alone.
_IMAGES_PREFIX = files.IMAGES_DIRNAME + "/"


class _ImageRejected(Exception):
    """One ``image_path`` block could not be expanded into pixels.

    Carries the sentence that becomes this attachment's text note, so a human
    who attached a file and gets no image back still learns why, rather than
    the attachment simply vanishing from the turn.
    """


# Longest a rejected attachment's path is quoted at inside a note, and the most
# paths the overflow note names before it switches to a count. Both bound the
# notes themselves: a ``turn`` frame is only checked for block shape, so its
# paths are arbitrary strings of any length up to the frame ceiling, and a note
# quoting them whole would turn the caps that exist to bound one JSON line into
# a way to grow it. It is also the one place an inbound string is copied into
# the model's context, which is reason enough to keep it short.
_PATH_IN_NOTE_CHARS = 120
_PATHS_NAMED_IN_NOTE = 3


def _show_path(path: Any) -> str:
    """``path`` as a short, quoted, single-line string safe to put in a note."""
    shown = path if isinstance(path, str) else repr(path)
    if len(shown) > _PATH_IN_NOTE_CHARS:
        shown = shown[:_PATH_IN_NOTE_CHARS] + "..."
    return repr(shown)


def _images_rel(path: Any) -> str:
    """The single path component after ``images/``, or raise ``_ImageRejected``.

    Refuses, without touching the filesystem: anything that is not a string,
    does not begin with ``images/``, names nothing after that prefix, or
    names more than one component (a second ``/``, which is what a traversal
    attempt such as ``images/../../etc/passwd`` produces once the prefix is
    stripped). ``resolve_asset`` below re-refuses a single-component name that
    still climbs out via a leading dot or a symlink; this check exists so the
    plainly malformed case never reaches it.
    """
    if not isinstance(path, str) or not path.startswith(_IMAGES_PREFIX):
        raise _ImageRejected(
            "an attachment was not an images/<name> reference and was "
            "dropped: %s" % (_show_path(path),)
        )
    rel = path[len(_IMAGES_PREFIX) :]
    if not rel or "/" in rel:
        raise _ImageRejected(
            "%s does not name a single file directly under images/ and was "
            "dropped" % (_show_path(path),)
        )
    return rel


def _read_turn_image(serve_dir: str, path: Any) -> Tuple[str, bytes]:
    """Read and sniff one ``image_path`` block's file, or raise ``_ImageRejected``.

    ``files.resolve_asset`` is the second and only filesystem-aware
    containment check: it refuses a name that resolves outside
    ``serve_dir``/images (including through a symlink), a symlinked
    ``images/`` itself, and anything that is not a regular file, returning the
    resolved path and the identity it validated. That identity is reasserted
    against the freshly opened descriptor below, the same re-check
    ``files.read_fixed_file`` performs for the same reason: resolution and
    open are separate operations, and a served directory's ``images/`` is not
    a tree only this process writes to, so the name can be relinked in
    between.

    The size cap is ``files.MAX_INLINE_IMAGE_BYTES``, what may be inlined,
    which is well below what ``/upload`` will store: a photograph a human
    attached is kept at full resolution as evidence, and a file above the
    inline cap is refused as a picture while still being named to the model as
    a path it can read. The check runs before the bytes are read, not after,
    because a file placed under ``images/`` by hand rather than through
    ``/upload`` is bounded by nothing else, and reading an oversized file just
    to reject it would spend the memory and IO this check exists to avoid.

    The media type is decided by ``files.sniff_image`` against these bytes,
    never against ``path``'s own suffix: that suffix was chosen by this
    project from a file's bytes for anything ``/upload`` wrote, or was not
    chosen by anyone for a file a human dropped into ``images/`` directly,
    and either way it is the weaker source of the two.
    """
    rel = _images_rel(path)
    resolved = files.resolve_asset(serve_dir, rel)
    if resolved is None:
        raise _ImageRejected(
            "%s does not resolve inside images/ and was dropped" % (_show_path(path),)
        )
    target, identity = resolved
    try:
        fd = os.open(str(target), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise _ImageRejected(
            "%s could not be read (%s) and was dropped" % (_show_path(path), exc)
        ) from exc
    try:
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino) != identity:
            raise _ImageRejected(
                "%s changed while it was being attached and was dropped" % (_show_path(path),)
            )
        if st.st_size > files.MAX_INLINE_IMAGE_BYTES:
            # Neither an error nor a drop. The file is on disk, is served by
            # /asset and is named to the model in this very note, so only the
            # inline copy is withheld. Inlining it anyway would put a base64
            # block past what the API accepts into the turn, which fails the
            # whole turn rather than one attachment.
            raise _ImageRejected(
                "%s is %d bytes, over the %d byte limit for an image sent "
                "inline, so it was not attached as a picture. It is on disk at "
                "that path and can be read with the Read tool."
                % (_show_path(path), st.st_size, files.MAX_INLINE_IMAGE_BYTES)
            )
        chunks = []
        remaining = st.st_size
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    except OSError as exc:
        raise _ImageRejected(
            "%s could not be read (%s) and was dropped" % (_show_path(path), exc)
        ) from exc
    finally:
        os.close(fd)
    sniff = files.sniff_image(data)
    if sniff is None:
        raise _ImageRejected(
            "%s is not a recognised image (PNG, JPEG or WEBP) and was dropped" % (_show_path(path),)
        )
    media_type, _suffix = sniff
    return media_type, data


def _image_pair(path: str, media_type: str, data: bytes) -> list:
    """The two blocks one accepted attachment expands to: the pixels, then
    text naming the path the model can hand to its own tools afterwards. See
    this section's header comment for why neither block is sent alone."""
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(data).decode("ascii"),
            },
        },
        {"type": "text", "text": "attached image: %s" % path},
    ]


def expand_turn_blocks(blocks: list, serve_dir: str) -> list:
    """Expand every ``image_path`` block in ``blocks`` into what the CLI
    understands, reordered so every image precedes every text block (see
    this section's header comment for why, citing fact 6).

    Runs on an executor thread (``submit_turn`` is the only caller): every
    check below except the pure string check in ``_images_rel`` touches the
    filesystem.

    Never raises for a bad, unreadable or oversized attachment, and never
    drops the turn: a rejected ``image_path`` becomes one ``text`` block
    stating why, so the human who attached a file that did not make it still
    learns something instead of the attachment silently vanishing. A
    text-only ``blocks`` (no ``image_path`` among them, the common case)
    returns with its text blocks in their original order and nothing else
    changed, since there is nothing here for this function to do.
    """
    images = [b for b in blocks if isinstance(b, dict) and b.get("type") == "image_path"]
    texts = [
        b
        for b in blocks
        if not (isinstance(b, dict) and b.get("type") == "image_path") and not _is_blank_text(b)
    ]

    considered = images[:MAX_TURN_IMAGES]
    overflow = images[MAX_TURN_IMAGES:]

    expanded = []
    budget = MAX_TURN_IMAGE_BYTES
    for block in considered:
        path = block.get("path")
        try:
            media_type, data = _read_turn_image(serve_dir, path)
        except _ImageRejected as exc:
            expanded.append({"type": "text", "text": str(exc)})
            continue
        if len(data) > budget:
            expanded.append(
                {
                    "type": "text",
                    "text": (
                        "%s was not attached: this turn's %d byte image budget is "
                        "already spent" % (path, MAX_TURN_IMAGE_BYTES)
                    ),
                }
            )
            continue
        budget -= len(data)
        expanded.extend(_image_pair(path, media_type, data))

    if overflow:
        named = ", ".join(_show_path(b.get("path")) for b in overflow[:_PATHS_NAMED_IN_NOTE])
        rest = len(overflow) - min(len(overflow), _PATHS_NAMED_IN_NOTE)
        expanded.append(
            {
                "type": "text",
                "text": (
                    "%d more attachment(s) were dropped, over this turn's %d-attachment "
                    "limit: %s%s"
                    % (
                        len(overflow),
                        MAX_TURN_IMAGES,
                        named,
                        ", and %d more" % rest if rest else "",
                    )
                ),
            }
        )

    out = expanded + texts
    # An empty content array is refused by the API exactly as an empty text
    # block is, so a frame whose every block was blank cannot be forwarded as
    # it stands. The composer will not send one (it requires text or an
    # attachment), which leaves a hand-assembled frame, and answering it with a
    # statement of what arrived is more use to both sides than a failed request.
    if not out:
        return [{"type": "text", "text": "(this message arrived empty)"}]
    return out


def _is_blank_text(block: Any) -> bool:
    """Whether ``block`` is a text block carrying nothing but whitespace.

    The API refuses an empty or whitespace-only text block outright and fails
    the request that carried it, so one reaching the transport costs the human
    the whole turn, image included. The composer sends an attachment with no
    typed message as exactly that shape, and a hand-assembled ``turn`` frame
    can too, which is why the check is here rather than only in the browser.
    """
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and not str(block.get("text") or "").strip()
    )
