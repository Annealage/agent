"""``JsonReviewStore``: the review's own file, for a product with no published
format of its own to keep (Annealage Loom's ``designs/src/<id>.review.json``).

The file format, version 2::

    {
      "version": 2,
      "next_id": 4,
      "comments": [
        {
          "id": 3,
          "anchor": {"sheet": "root", "x_mm": 81.28, "y_mm": 45.72},
          "ref": "R1",
          "text": "is this the footprint we agreed?",
          "author": "human",
          "status": "resolved",
          "resolution": "changed R1 to 0603 and moved it clear of U2"
        }
      ]
    }

Per comment: ``id`` (an integer, unique in the file), ``anchor`` (an object
in the product's anchor space), ``text``, ``author`` (``"human"`` or
``"model"``) and ``status`` (``"open"`` or ``"resolved"``) are always present;
``ref`` (what is at the anchor), ``resolution`` (how a resolved comment was
addressed), ``by`` (the tailnet login of the human who wrote a human comment)
and ``status_by`` (the login of the human who last resolved or reopened it
from the page) only when set. ``by`` and ``status_by`` are only ever a
signed-in human's: a comment written or a status set through the browser
token, or by the model, carries neither, and the model setting a status
clears ``status_by``. Any other key on a comment is the product's
(``Comment.extra``) and is kept as written; so is any other top-level key.

**The anchor is nested, not flattened into the comment.** Loom's version 1
file wrote ``sheet``, ``x_mm`` and ``y_mm`` beside ``id`` and ``text``. Nesting
them costs that literal compatibility, and Loom has no committed review file to
carry forward, so what it buys decides it: the comment's own keys and an
anchor space's keys can never collide, so no product has to avoid naming an
anchor field ``id``, ``text``, ``ref`` or ``status`` (or whatever this format
adds next), and the store can read and check every record the same way
without knowing which keys belong to the anchor. The format is version 2 so a
version 1 file is refused with a message that says so, rather than misread.

**The rules**, which are Loom's version 1 rules unchanged:

- The file is the store. Every call reads it afresh, so a hand edit, a ``git
  checkout`` or an external agent's direct write is seen by the next call.
- Every write replaces the file atomically (``files.atomic_replace``), under
  ``file_lock`` so two writers in this process cannot lose each other's
  comment or hand out one id twice. **The lock is per process**: two
  processes serving one file (two ``loom`` runs of one design, or a
  separately running agent editing the file directly) can still race, so one
  writer's change can replace the other's. The atomic replace keeps that to a
  lost change, never a corrupt file; running one server per review file is
  what rules it out.
- Ids are never reused: ``next_id`` only grows, so deleting the newest
  comment does not free its number, and a reply naming #4 cannot land on a
  different comment.
- A file that exists but does not parse, does not have this shape, or is not
  a plain single-linked file is refused, and never overwritten: it holds a
  human's words, and fixing it is theirs to do.
- A comment the human reopens (``reopen_comment``) keeps the ``resolution``
  it was resolved with, so it reads as "resolved once, then reopened" rather
  than as never addressed; resolving it again replaces the note, and
  resolving with no note keeps the one it has.
"""

import dataclasses
import json
import os
from pathlib import Path

from .. import files
from .model import (
    AUTHORS,
    MODEL,
    OPEN,
    RESOLVED,
    STATUSES,
    Capabilities,
    Comment,
    Listing,
    ReviewError,
    ReviewStore,
    Written,
    bytes_state,
    file_lock,
)

FORMAT_VERSION = 2

#: Open model callouts allowed at once unless the product says otherwise:
#: Loom's version 1 limit.
DEFAULT_MAX_OPEN_CALLOUTS = 50

# The keys a comment record's own fields occupy; anything else on a record is
# the product's (``Comment.extra``).
_RECORD_KEYS = (
    "id",
    "anchor",
    "ref",
    "text",
    "author",
    "by",
    "status",
    "resolution",
    "status_by",
)


class JsonReviewStore(ReviewStore):
    """The native store over the file at ``path``, anchors checked by
    ``anchor_space``. Comments carry a status: the model resolves them, the
    page adds the human's comments through ``POST /review`` and resolves or
    reopens any comment through ``POST /review/<id>``. The product configures
    the rest: ``max_open_model_callouts`` (``None``: no limit) and
    ``can_delete_own``, whether the model may delete its own callouts (off
    for a product that keeps answered callouts as a record, resolved)."""

    def __init__(
        self,
        path,
        anchor_space,
        *,
        max_open_model_callouts=DEFAULT_MAX_OPEN_CALLOUTS,
        can_delete_own=True,
    ):
        super().__init__()
        self.path = Path(path)
        self.anchor_space = anchor_space
        self.capabilities = Capabilities(
            can_resolve=True,
            can_delete_own=can_delete_own,
            human_adds_via_api=True,
            human_sets_status=True,
            max_open_model_callouts=max_open_model_callouts,
        )

    # -- reading ---------------------------------------------------------------

    def _read_bytes(self):
        """The file's bytes, ``None`` when it does not exist, or
        ``ReviewError`` when something is at its name that this store will
        not read (and therefore will not replace either)."""
        try:
            os.lstat(self.path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ReviewError("%s cannot be read (%s)" % (self.path.name, exc)) from None
        raw = files.read_fixed_file(self.path.parent, self.path.name)
        if raw is None:
            raise ReviewError(
                "%s is not a plain, single-linked file of at most a few megabytes (a "
                "link, a FIFO, or too large), so it is neither read nor replaced; "
                "fix it by hand" % self.path.name
            )
        return raw

    def _load(self):
        """``(document, next_id, comments)`` as the file holds them now."""
        raw = self._read_bytes()
        if raw is None:
            return {}, 1, []
        name = self.path.name
        try:
            document = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ReviewError("%s does not parse (%s); fix it by hand" % (name, exc)) from None
        if not isinstance(document, dict) or not isinstance(document.get("comments"), list):
            raise ReviewError('%s has no "comments" list; fix it by hand' % name)
        version = document.get("version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ReviewError(
                "%s is review format version %r, and this build reads version %d, "
                'whose comments carry their position as an "anchor" object; convert '
                "it by hand" % (name, version, FORMAT_VERSION)
            )
        comments = [_comment_from_record(record, name) for record in document["comments"]]
        ids = [c.id for c in comments]
        if len(set(ids)) != len(ids):
            raise ReviewError("%s gives two comments the same id; fix it by hand" % name)
        stored_next = document.get("next_id")
        if not isinstance(stored_next, int) or isinstance(stored_next, bool):
            stored_next = 1
        next_id = max([stored_next, 1] + [i + 1 for i in ids])
        return document, next_id, comments

    def _save(self, document, next_id, comments):
        # Checked per write, like every fixed-name file this package writes:
        # os.replace acts on the directory entry and cannot be redirected, but
        # a link or a FIFO at the name is not a file this store read, so
        # replacing it would discard whatever it pointed at unread.
        if files.safe_fixed_file(self.path.parent, self.path.name) is None:
            raise ReviewError(
                "refusing to write %s: it is not a plain, single-linked file" % self.path.name
            )
        out = dict(document)
        out["version"] = FORMAT_VERSION
        out["next_id"] = next_id
        out["comments"] = [c.record for c in comments]
        payload = (json.dumps(out, indent=2) + "\n").encode("utf-8")
        try:
            files.atomic_replace(self.path, payload)
        except OSError as exc:
            raise ReviewError("could not write %s (%s)" % (self.path.name, exc)) from None
        return self.path

    # -- the operations -------------------------------------------------------

    # Reads take the file's lock too. read_fixed_file opens the file and then
    # checks the name still resolves to that inode; an in-process atomic
    # replace landing between the two fails that check, and a healthy file
    # would be reported as one to fix by hand. Writers in other processes can
    # still race a read, but only a writer this process runs does so often.

    def list_comments(self, author=None, status=None):
        with file_lock(self.path):
            _document, _next_id, comments = self._load()
        return Listing(
            tuple(
                c
                for c in comments
                if (author is None or c.author == author) and (status is None or c.status == status)
            )
        )

    def get_comment(self, comment_id):
        with file_lock(self.path):
            comments = self._load()[2]
        return comments[self._index(comments, comment_id)]

    def _index(self, comments, comment_id):
        for index, comment in enumerate(comments):
            if comment.id == comment_id:
                return index
        raise ReviewError("there is no comment #%s in %s" % (comment_id, self.path.name))

    def add_comment(self, *, anchor, text, author, ref=None, extra=None, by=None):
        anchor = self.anchor_space.validate(anchor)
        if not isinstance(text, str) or not text.strip():
            raise ReviewError("a comment needs text saying what it is about")
        if author not in AUTHORS:
            raise ReviewError("author must be one of %s" % ", ".join(AUTHORS))
        _check_login(by)
        extra = dict(extra or {})
        clashing = sorted(set(extra) & set(_RECORD_KEYS))
        if clashing:
            raise ReviewError(
                "%s cannot be extra fields: a comment has its own" % ", ".join(clashing)
            )
        if ref is None:
            ref = self.anchor_space.ref_at(anchor)
        with file_lock(self.path):
            document, next_id, comments = self._load()
            limit = self.capabilities.max_open_model_callouts
            if author == MODEL and limit is not None:
                open_own = sum(1 for c in comments if c.author == MODEL and c.status == OPEN)
                if open_own >= limit:
                    raise ReviewError(
                        "%d of your callouts are open, which is the limit; resolve or "
                        "delete the ones that have been answered before adding more" % open_own
                    )
            comment = _comment(
                next_id, anchor, text.strip(), author, ref or None, OPEN, None, extra, by=by
            )
            comments.append(comment)
            path = self._save(document, next_id + 1, comments)
        self._changed()
        return Written(comment, _count(comments, author), path)

    def resolve_comment(self, comment_id, resolution=None, by=None):
        if isinstance(resolution, str):
            resolution = resolution.strip() or None
        return self._set_status(comment_id, RESOLVED, resolution, by)

    def reopen_comment(self, comment_id, by=None):
        return self._set_status(comment_id, OPEN, None, by)

    def _set_status(self, comment_id, status, resolution, by):
        """``comment_id`` with ``status``, and ``resolution`` when one is
        given (the one it has otherwise), set by ``by`` (``status_by``, which
        ``None`` clears); nothing is written when the status is already
        that."""
        _check_login(by)
        with file_lock(self.path):
            document, next_id, comments = self._load()
            index = self._index(comments, comment_id)
            comment = comments[index]
            if comment.status == status:
                return Written(comment, _count(comments, comment.author))
            comment = _comment(
                comment.id,
                comment.anchor,
                comment.text,
                comment.author,
                comment.ref,
                status,
                resolution if resolution is not None else comment.resolution,
                comment.extra,
                by=comment.by,
                status_by=by,
            )
            comments[index] = comment
            path = self._save(document, next_id, comments)
        self._changed()
        return Written(comment, _count(comments, comment.author), path)

    def delete_callout(self, comment_id):
        if not self.capabilities.can_delete_own:
            raise ReviewError(
                "this review keeps answered callouts; resolve one rather than delete it"
            )
        with file_lock(self.path):
            document, next_id, comments = self._load()
            index = self._index(comments, comment_id)
            comment = comments[index]
            if comment.author != MODEL:
                raise ReviewError(
                    "comment #%s is the human's; only your own callouts can be deleted, "
                    "and a human's comment is resolved rather than removed" % comment_id
                )
            del comments[index]
            # next_id is written back unchanged, which is what keeps the
            # deleted comment's number from ever being handed out again.
            path = self._save(document, next_id, comments)
        self._changed()
        return Written(comment, _count(comments, MODEL), path)

    def state(self):
        try:
            with file_lock(self.path):
                raw = self._read_bytes()
        except ReviewError:
            raw = None
        return bytes_state(raw)


def _count(comments, author):
    return sum(1 for c in comments if c.author == author)


def _check_login(by):
    if by is not None and not (isinstance(by, str) and by):
        raise ReviewError("by must be a human's login, or absent")


def _comment(
    comment_id, anchor, text, author, ref, status, resolution, extra, by=None, status_by=None
):
    """A ``Comment`` together with its native record, keys in the documented
    order and the product's extra keys after them."""
    record = {"id": comment_id, "anchor": dict(anchor)}
    if ref is not None:
        record["ref"] = ref
    record["text"] = text
    record["author"] = author
    if by is not None:
        record["by"] = by
    record["status"] = status
    if resolution is not None:
        record["resolution"] = resolution
    if status_by is not None:
        record["status_by"] = status_by
    record.update(extra)
    return Comment(
        id=comment_id,
        anchor=dict(anchor),
        text=text,
        author=author,
        ref=ref,
        status=status,
        resolution=resolution,
        by=by,
        status_by=status_by,
        extra=dict(extra),
        record=record,
    )


def _comment_from_record(record, name):
    """One stored record as a ``Comment``, checked strictly: a record this
    format did not write is a file somebody edited, and misreading it would
    put words on the wrong comment."""
    if not isinstance(record, dict):
        raise ReviewError("%s holds a comment that is not an object: %r" % (name, record))
    label = "%s: comment #%s" % (name, record.get("id", "?"))
    comment_id = record.get("id")
    if not isinstance(comment_id, int) or isinstance(comment_id, bool):
        raise ReviewError("%s has no integer id; fix it by hand" % label)
    anchor = record.get("anchor")
    if not isinstance(anchor, dict):
        raise ReviewError('%s has no "anchor" object; fix it by hand' % label)
    text = record.get("text")
    if not isinstance(text, str):
        raise ReviewError("%s has no text; fix it by hand" % label)
    author = record.get("author")
    if author not in AUTHORS:
        raise ReviewError("%s: author must be one of %s" % (label, ", ".join(AUTHORS)))
    status = record.get("status", OPEN)
    if status not in STATUSES:
        raise ReviewError("%s: status must be one of %s" % (label, ", ".join(STATUSES)))
    ref = record.get("ref")
    resolution = record.get("resolution")
    by = record.get("by")
    status_by = record.get("status_by")
    if any(value is not None and not isinstance(value, str) for value in (ref, resolution)):
        raise ReviewError("%s: ref and resolution must be strings when present" % label)
    if any(value is not None and not isinstance(value, str) for value in (by, status_by)):
        raise ReviewError("%s: by and status_by must be logins when present" % label)
    extra = {k: v for k, v in record.items() if k not in _RECORD_KEYS}
    comment = _comment(
        comment_id, anchor, text, author, ref, status, resolution, extra, by, status_by
    )
    # The record as read, not as this store would have written it: the two
    # differ only in key order, and a presenter shows the file's.
    return dataclasses.replace(comment, record=dict(record))
