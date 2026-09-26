"""The page's side of the review: what ``static/review.js`` fetches and posts.

Registers, whether or not the product keeps a review:

    GET  /review    every comment in the product-neutral shape
                    (``Comment.to_wire``), with the store's capabilities and
                    anchor space name
    POST /review    add one of the human's comments, ``{"anchor": {...},
                    "text": "..."}``, when the store's capabilities say the
                    page adds them this way (``human_adds_via_api``)

Both require the browser token and a permitted ``Origin``, and refuse with the
same opaque response ``/ws`` returns, so neither tells an unauthenticated
caller which check it failed. The review is the human's words about their
project, so it is not readable without the token either, unlike a product's
own file routes (Mesh's ``/callouts``) whose contract predates this one.

A product with no review store still has both routes, answering 404: the
route list is the agent layer's own, the same for every product, which is also
what lets the page client name ``/review`` without naming a product route.

A POST changes nothing but the store: the store notifies its listeners, the
app's ``ReviewWatcher`` publishes ``review_changed``, and every page, the one
that posted included, refetches through ``GET``. The POST's own answer carries
the new comment so the page that posted it can show it before that.
"""

import asyncio
import functools

from .. import product
from ..review.model import HUMAN, ReviewError
from . import read_json_body
from .ws import _origin_is_allowed, _token_is_allowed, refusal


def register_review_routes(app, *, store, token, allowed_origins=()):
    """Register ``GET`` and ``POST /review`` on ``app`` over ``store`` (a
    ``ReviewStore``, or ``None`` when the product keeps no review)."""

    def _no_review():
        return {
            "ok": False,
            "error": "%s keeps no review in this run" % product.current().title,
        }, 404

    @app.get("/review")
    async def get_review(req):
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()
        if store is None:
            return _no_review()
        loop = asyncio.get_running_loop()
        try:
            listing = await loop.run_in_executor(None, store.list_comments)
        except ReviewError as exc:
            # The file is there and unreadable: a fault the human fixes by
            # hand, reported as such rather than as an empty review, which a
            # page would render as "no comments" over the top of theirs.
            return {"ok": False, "error": str(exc)}, 409
        return {
            "ok": True,
            "anchor_space": store.anchor_space.name,
            "capabilities": store.capabilities.to_wire(),
            "comments": [comment.to_wire() for comment in listing.comments],
        }, 200

    @app.post("/review")
    async def add_review_comment(req):
        if not _token_is_allowed(req, token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()
        if store is None:
            return _no_review()
        if not store.capabilities.human_adds_via_api:
            return {
                "ok": False,
                "error": "%s adds the human's comments its own way, not through this route"
                % product.current().title,
            }, 405
        data, error = await read_json_body(req)
        if error is not None:
            return error
        if not isinstance(data, dict):
            return {"ok": False, "error": "body must be a JSON object"}, 400
        unknown = sorted(set(data) - {"anchor", "text"})
        if unknown:
            return {"ok": False, "error": "unknown body field: %s" % ", ".join(unknown)}, 400
        anchor = data.get("anchor")
        text = data.get("text")
        if not isinstance(anchor, dict) or not isinstance(text, str):
            return {
                "ok": False,
                "error": 'body must be {"anchor": {...}, "text": "..."}',
            }, 400
        loop = asyncio.get_running_loop()
        try:
            written = await loop.run_in_executor(
                None,
                functools.partial(store.add_comment, anchor=anchor, text=text, author=HUMAN),
            )
        except ReviewError as exc:
            return {"ok": False, "error": str(exc)}, 400
        return {"ok": True, "comment": written.comment.to_wire()}, 200
