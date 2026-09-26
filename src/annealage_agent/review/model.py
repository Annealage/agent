"""The review model: comments, anchors, stores, and what each store can do.

See this package's ``__init__`` for how the pieces fit together. Nothing here
imports an agent SDK, so a viewer-only run can build a store, serve it and
watch it without paying for one.
"""

import dataclasses
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

HUMAN = "human"
MODEL = "model"
#: Who wrote a comment: the person at the page, or the agent.
AUTHORS = (HUMAN, MODEL)

OPEN = "open"
RESOLVED = "resolved"
#: A comment's status, in a store that keeps one (``Capabilities.can_resolve``).
STATUSES = (OPEN, RESOLVED)


class ReviewError(ValueError):
    """A review file cannot be read as comments, or a change to it is refused.

    A ``ValueError`` on purpose: a tool handler that lets one propagate reaches
    the model through ``tools._wrap``, which passes a ``ValueError``'s message
    through verbatim, so every message raised as one is written for whoever
    reads it (the model, or the human through a route's error field) and says
    what to do next, not only what went wrong.
    """


@dataclasses.dataclass(frozen=True)
class Comment:
    """One comment on the product's view: the human's, or the model's callout.

    ``id`` is the store's number for it and is never reused, so a reply naming
    #4 cannot land on a different comment after #4 is deleted. It is ``None``
    only for a record a published format let someone write by hand without one
    (a hand-written Mesh callout), which no tool can then address.

    ``anchor`` is where the comment is, in the product's ``AnchorSpace`` (a
    sheet and a point in millimetres, a point on a 3D part). ``ref`` names the
    thing at that point when the product has such a thing (the part under the
    pin on a schematic). ``status`` and ``resolution`` are ``None`` in a store
    that keeps no status.

    ``extra`` carries the product's own fields the model does not interpret
    (Mesh's face ``label``), so they survive a round trip through a store.
    ``record`` is the comment exactly as its store's file holds it, which is
    what a list tool shows the model: for a store whose file format is a
    published contract (Mesh's), that is the verbatim record, unknown keys and
    key order included, so reading a file through this model changes nothing
    the model sees. It takes no part in equality.
    """

    id: Optional[int]
    anchor: Mapping[str, Any]
    text: str
    author: str
    ref: Optional[str] = None
    status: Optional[str] = None
    resolution: Optional[str] = None
    extra: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    record: Optional[Mapping[str, Any]] = dataclasses.field(default=None, compare=False, repr=False)

    def to_wire(self):
        """The product-neutral JSON shape ``GET /review`` serves the page.

        Fields that are ``None`` or empty are left out rather than written as
        null, the same convention every server event follows, so a page can
        tell "this store keeps no status" from a status it did not expect.
        """
        data = {"id": self.id, "anchor": dict(self.anchor)}
        if self.ref is not None:
            data["ref"] = self.ref
        data["text"] = self.text
        data["author"] = self.author
        if self.status is not None:
            data["status"] = self.status
        if self.resolution is not None:
            data["resolution"] = self.resolution
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    def shown(self):
        """What a tool shows the model for this comment: its store's record
        when the store keeps one, else the product-neutral shape."""
        return dict(self.record) if self.record is not None else self.to_wire()


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """What a store supports, which decides which tools and routes exist.

    ``can_resolve``: comments carry a status, and the model may resolve one
    (``resolve_comment``). ``can_delete_own``: the model may delete its own
    callouts (``delete_callout``); nobody's tool deletes a human's comment.
    ``human_adds_via_api``: the page adds a human comment one at a time
    through ``POST /review``, rather than through a flow of the product's own
    (Mesh's pins are drafts in the page until the human submits them all).
    ``human_sets_status``: the page resolves or reopens any comment through
    ``POST /review/<id>``, which is how the human says a resolved comment was
    not addressed after all (a model reading the list sees it open again).
    ``max_open_model_callouts``: how many open callouts the model may have at
    once, ``None`` for no limit. Every one is a marker the human has to read,
    so a model that pins a note per feature makes the page unusable.
    """

    can_resolve: bool = False
    can_delete_own: bool = False
    human_adds_via_api: bool = False
    human_sets_status: bool = False
    max_open_model_callouts: Optional[int] = None

    def to_wire(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Listing:
    """What one read of a store found: its ``comments``, and ``meta``, facts
    the store's own format records about the collection as a whole in the
    same read (Mesh: when the human last submitted), for a presenter that
    shows them. Read together so the two cannot disagree."""

    comments: Tuple[Comment, ...]
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Written:
    """The result of one change to a store: the ``comment`` it changed (as it
    now stands, or as it was before a delete), ``count``, how many comments
    by that comment's author the store holds afterwards, and ``path``, the
    file the change was written to (``None`` when nothing needed writing)."""

    comment: Comment
    count: int
    path: Optional[Path] = None


class AnchorSpace:
    """Where a comment can be, in one product's terms.

    A product subclasses this once: Loom's anchor is a sheet name and a point
    in sheet millimetres, Mesh's a point on a 3D part. The agent layer never
    interprets an anchor; it asks the product's space to check one and stores
    what comes back.

    ``name`` identifies the space to the page (``GET /review`` reports it).
    ``schema`` is the anchor's fields as tool input, ``{"properties": {...},
    "required": [...]}``: ``add_callout`` offers exactly these properties to
    the model and hands exactly these keys of its arguments to ``validate``.
    """

    name = ""
    schema: Mapping[str, Any] = {"properties": {}, "required": []}

    def validate(self, anchor):
        """Return a normalised copy of ``anchor``, or raise ``ReviewError``
        with a message the model (or the page) can act on.

        Called on every anchor a tool or the page supplies, and again by the
        store before it writes, so it must be idempotent: a normalised anchor
        validates to itself. Stored anchors are not re-validated on read,
        because a product's space can change under them (a sheet renamed by a
        rebuild) and a human's comment must stay readable when it does.
        """
        raise NotImplementedError

    def ref_at(self, anchor):
        """The name of the thing at ``anchor`` (the part under a pin), or
        ``None``. The store records it on a comment that names none itself."""

    def check_ref(self, anchor, ref):
        """Return ``ref`` if it names something the model may point at near
        ``anchor``, or raise ``ReviewError`` saying what exists instead."""
        return ref


class ReviewStore:
    """Where a product's comments live, and the one path every change takes.

    A store is synchronous and file-backed: every call reads its file afresh,
    so a hand edit, a ``git checkout`` or an external agent writing the file
    directly is seen by the next call, and every write replaces the file
    atomically under ``file_lock``. Callers on an event loop run its methods
    in an executor.

    ``anchor_space`` validates every anchor the store is given, and
    ``capabilities`` says which of the operations below it supports; the
    ones it does not raise ``ReviewError``. A subclass calls
    ``super().__init__()`` and ``self._changed()`` after each write, which is
    the change notification: ``add_listener(fn)`` registers ``fn`` to be
    called, with no arguments and on whatever thread made the change, once a
    write has landed. The app's ``ReviewWatcher`` listens this way, so a
    change through the store is announced at once rather than on the
    watcher's next poll; the poll remains what sees changes made around the
    store.
    """

    capabilities = Capabilities()
    anchor_space = AnchorSpace()

    def __init__(self):
        self._listeners: List[Callable[[], None]] = []

    # -- change notification ------------------------------------------------

    def add_listener(self, listener):
        self._listeners.append(listener)

    def remove_listener(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _changed(self):
        # A listener that fails must not turn a write that has already landed
        # into a reported failure: the change is on disk and the watcher's
        # poll will still announce it.
        for listener in list(self._listeners):
            try:
                listener()
            except Exception as exc:
                sys.stderr.write("warning: a review change listener failed: %r\n" % (exc,))

    # -- the operations -------------------------------------------------------

    def list_comments(self, author=None, status=None):
        """``Listing`` of the comments by ``author`` (``None``: everyone's)
        whose status is ``status`` (``None``: any). Raises ``ReviewError``
        when the file exists but cannot be read as comments."""
        raise NotImplementedError

    def get_comment(self, comment_id):
        """The comment ``comment_id``, or ``ReviewError`` saying it is absent."""
        raise ReviewError("this review store cannot look a comment up by id")

    def add_comment(self, *, anchor, text, author, ref=None, extra=None):
        """Add a comment and return ``Written``. The anchor is validated by
        ``anchor_space``; ``ref`` defaults to ``anchor_space.ref_at(anchor)``;
        a model's callout past ``max_open_model_callouts`` is refused."""
        raise NotImplementedError

    def resolve_comment(self, comment_id, resolution=None):
        """Mark ``comment_id`` resolved with ``resolution`` (the note on what
        was changed) and return ``Written``. Resolving a resolved comment
        changes nothing. Whether the human must approve is the caller's
        policy (``review.tools``), not the store's."""
        raise ReviewError(
            "this review keeps no status, so comments cannot be resolved; say what "
            "you changed instead"
        )

    def reopen_comment(self, comment_id):
        """Mark ``comment_id`` open again and return ``Written``. The human's
        operation (``POST /review/<id>``), never a tool's: it says a resolved
        comment was not addressed after all. A resolution already recorded is
        kept, so the model reading the list can tell a comment reopened after
        it resolved it from one never resolved. Reopening an open comment
        changes nothing."""
        raise ReviewError("this review keeps no status, so comments cannot be reopened")

    def delete_callout(self, comment_id):
        """Delete the model's own callout ``comment_id`` and return
        ``Written`` with the deleted comment. A human's comment is never
        deleted here."""
        raise ReviewError("this review does not let callouts be deleted")

    def state(self):
        """``(digest, settled)`` for what the store's files hold right now,
        for ``ReviewWatcher``: a digest of the bytes (``None`` when there is
        nothing), and whether they parse, which is how a write observed
        half-way is told from a finished one. Never raises for a missing or
        refused file; either reads as absent."""
        raise NotImplementedError


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def file_lock(path):
    """The process-wide lock serialising read-modify-write of the file at
    ``path``.

    One lock per file rather than per store, so two stores over the same file
    (a tool server's and a route's, or two tool servers in one test) still
    cannot interleave their reads and writes and lose a comment or hand out
    one id twice. A ``threading`` lock, because store calls run in executor
    threads. It covers this process only: another process writing the same
    file (an external agent editing it directly) can still race a write here,
    which is why every write is an atomic replace, so the worst case is one
    writer's change replacing the other's, never a corrupt file.
    """
    path = Path(path)
    key = os.path.join(os.path.realpath(path.parent), path.name)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
    return lock


def bytes_state(*raws):
    """``(digest, settled)`` over the bytes of one or more files, each
    ``None`` when absent: the ``ReviewStore.state`` most stores need.

    Each file's bytes are length-prefixed into one digest, so moving bytes
    from one file to the next is still a change. ``settled`` is whether every
    present file parses as JSON: a writer the store does not control (an
    external agent, a hand edit) need not write atomically, and a digest that
    changed says nothing about whether the write has finished, while bytes
    that parse do, for every incomplete write that is not itself
    coincidentally valid JSON.
    """
    digest = hashlib.sha256()
    present = False
    settled = True
    for raw in raws:
        if raw is None:
            digest.update(b"-")
            continue
        present = True
        digest.update(b"+%d:" % len(raw))
        digest.update(raw)
        try:
            json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            settled = False
    return (digest.hexdigest() if present else None), settled
