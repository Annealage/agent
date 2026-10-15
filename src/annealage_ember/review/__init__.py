"""The shared review model: comments pinned on a product's view, the model's
callouts beside them, and who may change which.

A product's page lets the human pin a comment at a point on what it shows (a
3D part, a schematic sheet), and the model points back with callouts of its
own, so neither side describes a location in words. This package owns that
model for every product; the product supplies only what is its own:

- ``model.py``: ``Comment`` (id never reused, anchor, ``ref``, text, author
  human or model, status and resolution in a store that keeps them, the
  product's extra fields), ``AnchorSpace`` (the product's idea of where a
  comment can be: its tool-input schema, ``validate`` and ``ref_at``),
  ``ReviewStore`` (where comments live, with its ``Capabilities`` and change
  listeners) and ``file_lock``.
- ``native.py``: ``JsonReviewStore``, the review's own JSON file, whose
  format that module documents. A product with no file format of its own to
  keep uses it (Annealage Loom).
- ``watcher.py``: ``ReviewWatcher``, which publishes the generic
  ``review_changed`` event whenever the store's files change, through the
  store or around it.
- ``tools.py``: the model's review tools and their approval policy (resolving
  a human's comment is always a permission card). It imports the agent SDK,
  so it is imported by name, in agent mode only, rather than from here.

A product whose files are already a published contract keeps them behind its
own ``ReviewStore`` (Annealage Mesh's ``mesh-comments.json`` and
``mesh-callouts.json``), so there is one source of truth and no migration.

The app side: ``app.create_app(review_store=...)`` serves the store at
``GET``/``POST /review`` (``http/routes_review.py``, browser token), hands it
to the product's tool builder as ``bus.review_store``, and runs the watcher
beside the server; ``static/review.js`` is the page's client (fetch and
subscribe, no rendering: every product draws anchors its own way).
"""

from .model import (
    AUTHORS,
    HUMAN,
    MODEL,
    OPEN,
    RESOLVED,
    STATUSES,
    AnchorSpace,
    Capabilities,
    Comment,
    Listing,
    ReviewError,
    ReviewStore,
    Written,
    bytes_state,
    file_lock,
)
from .native import JsonReviewStore
from .watcher import ReviewWatcher

__all__ = [
    "AUTHORS",
    "HUMAN",
    "MODEL",
    "OPEN",
    "RESOLVED",
    "STATUSES",
    "AnchorSpace",
    "Capabilities",
    "Comment",
    "JsonReviewStore",
    "Listing",
    "ReviewError",
    "ReviewStore",
    "ReviewWatcher",
    "Written",
    "bytes_state",
    "file_lock",
]
