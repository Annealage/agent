"""The review tools: the model's side of a review, over any ``ReviewStore``.

``review_tools(store, bus=...)`` returns ``@tool`` definitions a product adds
to the list it builds its ``ToolServer`` from, next to its own tools, and
grades in its own ``Grading`` like any other. Four kinds exist,
each described by a small dataclass a product may configure:

``ListComments`` (``list_comments``)
    The comments, optionally only one author's, and, in a store that keeps
    status, filtered by ``status`` (open by default, resolved, or all).
``AddCallout`` (``add_callout``)
    A callout of the model's own at an anchor the product's ``AnchorSpace``
    validates, with its text, the thing at that point (``ref``, when the
    product has such a thing) and any of the product's own extra fields. The
    store refuses one past its ``max_open_model_callouts``.
``ResolveComment`` (``resolve_comment``)
    Only for a store that ``can_resolve``: marks a comment resolved with a
    note on what was changed.
``DeleteCallout`` (``delete_callout``)
    Only for a store that ``can_delete_own``: removes one of the model's own
    callouts. A human's comment is never deleted by a tool.

With no configuration, a product gets exactly the tools its store supports,
named as above. A product whose tools are already a published surface (Mesh:
``list_comments`` for the human's comments and ``list_callouts`` for the
model's, a text field called ``comment``, a face ``label``) passes its own
``tools=``: every name, description, input schema and output is a field there,
so the model sees the product's surface unchanged while the validation, the
limits and the approval policy stay here.

**The approval policy.** Each product grades these tools like its own (Mesh
grades adding and deleting a callout write-grade, so each is a card; Loom will
pre-allow ``add_callout`` and rely on the open-callout limit). One rule is not
the product's to grade: **resolving a human's comment always reaches the
human as a permission card, and every time**, because it changes their
review. So the resolve tool asks the human itself (``tools.asks_the_human``):
its handler asks the session's broker (``bus.broker``), before resolving an
open comment of the human's, with a request that carries the comment and the
model's note, so the card says which comment and what was done about it. The
model's own callouts, and a comment already resolved, resolve with no card.
Because the handler asks, no transport may ask first: ``ToolServer`` treats
the tool as read-grade whatever grade the product gave it, so Claude
pre-allows it and neither ``/mcp`` nor omp gates it, and each human-comment
resolve is asked exactly once on every backend. For the same reason the
pause switch does not refuse it. The broker never remembers an answer to it
(``ToolServer.never_remembered``): the pane offers no "always allow", a
grant sent anyway is a one-time allow, and one already in ``permissions.toml``
is ignored. With no broker, a human's comment is not resolved.
"""

import asyncio
import dataclasses
import functools
from typing import Any, Callable, Mapping, Optional

from claude_agent_sdk import tool

from .. import product
from ..tools import _tool_json_schema, asks_the_human, fail, namespaced, ok
from .model import HUMAN, MODEL, OPEN, RESOLVED, ReviewError

#: What ``list_comments``' ``status`` accepts, in a store that keeps status.
LIST_STATUSES = (OPEN, RESOLVED, "all")


def present_listing(store, listing):
    """The default ``list_comments`` result: every comment as its store
    records it."""
    return ok({"count": len(listing.comments), "comments": [c.shown() for c in listing.comments]})


def present_added(store, written):
    return ok({"added": written.comment.shown()})


def present_resolved(store, written):
    return ok({"resolved": written.comment.shown()})


def present_deleted(store, written):
    return ok({"deleted": written.comment.id})


_LIST_DESCRIPTION = (
    "Read the comments on this review: the human's, pinned on the view, and "
    "your own callouts. Each has its id, where it is (anchor), what is there "
    "(ref, when known), what it says and who wrote it (human or model). Read "
    "them before acting on feedback, and before add_callout to avoid "
    "repeating a note."
)
_LIST_STATUS_SENTENCE = (
    " status picks open comments (the default), resolved ones, or all; a "
    "resolved comment says how it was addressed, and a human's comment that "
    "is open again after you resolved it was reopened by the human because "
    "the change did not satisfy them."
)
_ADD_DESCRIPTION = (
    "Pin a note of your own at a point on the view, which the human sees as a "
    "marker beside their own comments. This is how to point at a location "
    "instead of describing it: put the callout on the thing you are asking "
    "about or reporting on, and say what you mean in the text."
)
_RESOLVE_DESCRIPTION = (
    "Mark a comment resolved, with a note on what you changed. Resolving one "
    "of the human's comments changes their review, so it waits for them to "
    "approve it in the page and is refused if they decline or nobody is there "
    "to answer. Resolve your own callouts once they have been answered."
)
_DELETE_DESCRIPTION = (
    "Remove one of your own callouts by its id once it has served its "
    "purpose, so it stops cluttering the view. A human's comment cannot be "
    "deleted."
)


@dataclasses.dataclass(frozen=True)
class ListComments:
    """A tool listing comments: ``author`` restricts it to one author's
    (``None``: everyone's). ``schema`` replaces the input schema built here
    (an empty one, or ``status`` in a store that keeps status). ``present``
    turns ``(store, Listing)`` into the tool's result."""

    name: str = "list_comments"
    description: str = _LIST_DESCRIPTION
    author: Optional[str] = None
    schema: Optional[Mapping[str, Any]] = None
    present: Callable = present_listing


@dataclasses.dataclass(frozen=True)
class AddCallout:
    """The tool adding the model's callout. Its input is the anchor space's
    properties, ``text_field`` (the text), ``ref_field`` (``None``: not
    offered; the store records what is at the anchor anyway) and
    ``extra_fields`` (``{name: JSON schema property}``: the product's own
    optional text fields, kept on the comment stripped, a value that is not
    non-empty text being left out). ``schema`` replaces the input schema built
    from those, and must declare exactly those properties. ``present`` turns
    ``(store, Written)`` into the tool's result."""

    name: str = "add_callout"
    description: str = _ADD_DESCRIPTION
    text_field: str = "text"
    text_description: str = "what you are saying about that point"
    ref_field: Optional[str] = "ref"
    extra_fields: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    schema: Optional[Mapping[str, Any]] = None
    present: Callable = present_added


@dataclasses.dataclass(frozen=True)
class ResolveComment:
    """The tool resolving a comment; its input is ``id`` and ``note``."""

    name: str = "resolve_comment"
    description: str = _RESOLVE_DESCRIPTION
    schema: Optional[Mapping[str, Any]] = None
    present: Callable = present_resolved


@dataclasses.dataclass(frozen=True)
class DeleteCallout:
    """The tool deleting one of the model's callouts; its input is ``id``."""

    name: str = "delete_callout"
    description: str = _DELETE_DESCRIPTION
    schema: Optional[Mapping[str, Any]] = None
    present: Callable = present_deleted


def default_tools(capabilities):
    """The review tools a store with ``capabilities`` supports, with their
    default names, descriptions, schemas and results."""
    listing = ListComments()
    if capabilities.can_resolve:
        listing = dataclasses.replace(
            listing, description=_LIST_DESCRIPTION + _LIST_STATUS_SENTENCE
        )
    tools = [listing, AddCallout()]
    if capabilities.can_resolve:
        tools.append(ResolveComment())
    if capabilities.can_delete_own:
        tools.append(DeleteCallout())
    return tuple(tools)


def review_tools(store, *, bus, tools=None):
    """``@tool`` definitions for ``tools`` (default: ``default_tools``) over
    ``store``.

    ``bus`` is the app's ``ViewerBus``; the resolve tool reads the session's
    ``PermissionBroker`` off it at call time (``bus.broker``, which the
    session factory sets after the tools are built).

    Raises ``ValueError`` for a configuration the store cannot serve (a
    resolve tool over a store that keeps no status, a delete tool over one
    that does not allow it) or a schema that disagrees with what its handler
    reads, at startup rather than on the first call.
    """
    specs = default_tools(store.capabilities) if tools is None else tuple(tools)
    lists = [spec for spec in specs if isinstance(spec, ListComments)]
    # The list tools the other tools' messages send the model to: the one
    # that shows the model's own callouts, and the one that shows every
    # comment (a product that splits the two, as Mesh does, has no second).
    callouts_list = next((s.name for s in lists if s.author in (None, MODEL)), "the list tool")
    comments_list = next((s.name for s in lists if s.author is None), callouts_list)
    built = []
    for spec in specs:
        if isinstance(spec, ListComments):
            built.append(_list_tool(spec, store))
        elif isinstance(spec, AddCallout):
            built.append(_add_tool(spec, store))
        elif isinstance(spec, ResolveComment):
            if not store.capabilities.can_resolve:
                raise ValueError(
                    "%s is configured over a review store that keeps no status" % spec.name
                )
            built.append(asks_the_human(_resolve_tool(spec, store, bus, comments_list)))
        elif isinstance(spec, DeleteCallout):
            if not store.capabilities.can_delete_own:
                raise ValueError(
                    "%s is configured over a review store that does not let callouts be "
                    "deleted" % spec.name
                )
            built.append(_delete_tool(spec, store, callouts_list))
        else:
            raise ValueError("not a review tool: %r" % (spec,))
    return built


async def _blocking(fn, *args, **kwargs):
    """Run one store call off the event loop: every call reads a file."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _declared(schema):
    return set(_tool_json_schema(schema)["properties"])


def _require_declared(spec_name, schema, names):
    missing = sorted(set(names) - _declared(schema))
    if missing:
        raise ValueError("%s's schema does not declare %s" % (spec_name, ", ".join(missing)))


def _comment_id(args, message):
    value = args.get("id")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReviewError(message)
    return value


def _list_tool(spec, store):
    keeps_status = store.capabilities.can_resolve
    schema = spec.schema
    if schema is None:
        schema = {}
        if keeps_status:
            schema = {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": list(LIST_STATUSES),
                        "description": "open (the default), resolved, or all",
                    }
                },
            }

    @tool(spec.name, spec.description, schema)
    async def list_handler(args):
        status = None
        if keeps_status:
            status = args.get("status", OPEN)
            if status not in LIST_STATUSES:
                raise ReviewError("status must be one of %s" % ", ".join(LIST_STATUSES))
            if status == "all":
                status = None
        listing = await _blocking(store.list_comments, author=spec.author, status=status)
        return spec.present(store, listing)

    return list_handler


def _add_tool(spec, store):
    space = store.anchor_space
    anchor_keys = tuple(space.schema.get("properties", {}))
    accepted = anchor_keys + (spec.text_field,)
    if spec.ref_field is not None:
        accepted += (spec.ref_field,)
    accepted += tuple(spec.extra_fields)
    if len(set(accepted)) != len(accepted):
        raise ValueError("%s names one input field twice: %s" % (spec.name, ", ".join(accepted)))
    schema = spec.schema
    if schema is None:
        properties = dict(space.schema.get("properties", {}))
        properties[spec.text_field] = {"type": "string", "description": spec.text_description}
        if spec.ref_field is not None:
            properties[spec.ref_field] = {
                "type": "string",
                "description": "what the note is about, when it is not simply what "
                "is at that point; left out, the thing at the point is used",
            }
        properties.update(spec.extra_fields)
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(space.schema.get("required", ())) + [spec.text_field],
        }
    elif _declared(schema) != set(accepted):
        # A verbatim schema that offered a field the handler ignores, or hid
        # one it reads, would be a tool that silently drops what the model
        # said, so the two are made to agree at startup.
        raise ValueError(
            "%s's schema declares %s but its handler reads %s"
            % (spec.name, ", ".join(sorted(_declared(schema))), ", ".join(sorted(accepted)))
        )
    empty_text = (
        "%s must say what you mean about that point; a callout with no comment is "
        "a marker the human cannot interpret" % spec.text_field
    )

    @tool(spec.name, spec.description, schema)
    async def add_handler(args):
        # The anchor first, then the text: the order a model's mistakes are
        # reported in, one at a time.
        anchor = space.validate({key: args[key] for key in anchor_keys if key in args})
        text = args.get(spec.text_field)
        if not isinstance(text, str) or not text.strip():
            raise ReviewError(empty_text)
        ref = None
        if spec.ref_field is not None and args.get(spec.ref_field) is not None:
            value = args[spec.ref_field]
            if not isinstance(value, str):
                raise ReviewError("%s must be text, or left out" % spec.ref_field)
            if value.strip():
                ref = space.check_ref(anchor, value.strip())
        extra = {}
        for key in spec.extra_fields:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                extra[key] = value.strip()
        written = await _blocking(
            store.add_comment, anchor=anchor, text=text.strip(), author=MODEL, ref=ref, extra=extra
        )
        return spec.present(store, written)

    return add_handler


def _resolve_tool(spec, store, bus, comments_list):
    schema = spec.schema
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "the comment's id, as %s reports it" % comments_list,
                },
                "note": {
                    "type": "string",
                    "description": "what you changed to address it, which the human "
                    "reads before approving",
                },
            },
            "required": ["id"],
        }
    else:
        _require_declared(spec.name, schema, ("id",))
    bad_id = "id must be a comment id, as reported by %s" % comments_list

    @tool(spec.name, spec.description, schema)
    async def resolve_handler(args):
        comment_id = _comment_id(args, bad_id)
        note = args.get("note")
        if note is not None and not isinstance(note, str):
            raise ReviewError("note must be text saying what you changed")
        note = (note or "").strip()
        comment = await _blocking(store.get_comment, comment_id)
        if comment.author == HUMAN and comment.status != RESOLVED:
            denial = await _ask_human(spec.name, bus, comment, note)
            if denial is not None:
                return fail(denial)
        written = await _blocking(store.resolve_comment, comment_id, note or None)
        return spec.present(store, written)

    return resolve_handler


async def _ask_human(tool_name, bus, comment, note):
    """Ask the human, through the session's broker, whether the model may
    resolve their ``comment``; ``None`` if they allowed it, else the reason it
    was refused, for the model.

    Asked under the tool's model-visible name, which the broker never
    remembers an answer for (``ToolServer.never_remembered``). The input
    carries the comment itself and the model's note, so the card shows the
    human which of their comments this is and what the model says it did.
    """
    broker = getattr(bus, "broker", None)
    if broker is None:
        return (
            "resolving the human's comment needs their approval, and this session has "
            "no permission broker to ask them through, so it was not resolved; tell "
            "the human what you changed instead"
        )
    name = namespaced(product.current().mcp_server_name, tool_name)
    decision = await broker.ask(
        name, {"id": comment.id, "note": note, "comment": comment.to_wire()}, None
    )
    if decision.allow:
        return None
    return decision.message or "the human did not approve resolving their comment"


def _delete_tool(spec, store, callouts_list):
    schema = spec.schema
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "the callout's id, as %s reports it" % callouts_list,
                }
            },
            "required": ["id"],
        }
    else:
        _require_declared(spec.name, schema, ("id",))
    bad_id = "id must be a callout id, as reported by %s" % callouts_list

    @tool(spec.name, spec.description, schema)
    async def delete_handler(args):
        callout_id = _comment_id(args, bad_id)
        written = await _blocking(store.delete_callout, callout_id)
        return spec.present(store, written)

    return delete_handler
