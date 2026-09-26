"""Assembly of a product's tool server, and the two policies every tool obeys.

A product supplies its tools as ``claude_agent_sdk`` ``@tool`` definitions
(its ``Product.build_tools``) together with a ``Grading`` that sorts every one
of them into three grades by **what a mistake would cost**. This module turns
the two into one in-process MCP server, ``ToolServer``, and owns everything
about that server that is not the product's own: the refusal to build a set
whose grading does not match it, the pause gate, the mapping of viewer
failures to messages a model can act on, the pre-allowed name list every
backend receives, and the transport-neutral ``tool_table`` the non-Claude
backends build their own servers from.

``read`` changes nothing. Reading the view, a list of the project's parts, the
comments or a screenshot (Mesh's read tools) leaves the project and the view
exactly as they were, so these are
pre-allowed and never interrupt anyone.

``view`` changes only what is on the screen, and does so in front of the
human, who is looking at that screen. These are pre-allowed too. The
reasoning is not that they are harmless in the abstract, it is that an
approval card is the wrong control for them: the loop a product's tools exist
for has the model reframing something it has just changed, several times a
turn, and a card per view change (in Mesh, per camera move) would either be
clicked without reading or
turned off with one standing grant, which is worse than not asking. The
control that fits is the pause switch, which refuses all of them at once for
as long as the human wants the view to hold still, and that is what it is
for.

``write`` leaves something behind after the page is closed: in Mesh, a
callout in a file the human's own tooling reads, an image in the project directory, or a
transcript that carries whatever was said about the hardware under review.
These are deliberately **absent** from every allow list, which is what makes
them reach the broker and therefore the human as a card. Adding one of these
names to a pre-allowed list anywhere would silently remove that card.

So two derived sets follow, and they are not the same set:
``Grading.pre_allowed`` is read plus view, and ``Grading.pause_gated`` is view
plus write. The product's grading is the only place any of this is written
down, and ``_verify`` refuses to build a server whose tools do not match it
exactly, so a tool added to a product without being graded fails at startup
rather than defaulting into a posture nobody chose.

One kind of tool is graded by the agent layer instead: one whose handler asks
the human itself, per call, with a question only it can phrase (resolving one
of the human's review comments). ``asks_the_human`` marks it, and the server
treats it as read-grade whatever the product said, so no transport asks a
vaguer question first; the product must still grade it, like any tool.

Every handler returns one of ``ok`` or ``fail`` and nothing else. A tool
result reaches the model as text, so what a handler returns is prose it will
read: ``ok`` renders a payload as indented JSON, because coordinates and part
names are what these tools are for and JSON is the shape a model reads them
out of most reliably, and ``fail`` returns a sentence saying what went wrong
and what to do instead. That second half matters more than it looks: a deny's
message reaches the model verbatim (plan section 2a, fact 15), so "no viewer
connected; ask the human to open <url>" is worth more than a status code.

Importing this module imports ``claude_agent_sdk``, so only agent-mode code
paths import it; a viewer-only run never does.
"""

import asyncio
import dataclasses
import json
import sys
from collections import namedtuple

from claude_agent_sdk import create_sdk_mcp_server

from . import product
from .viewers import CallError, NoViewerConnected, ViewerGone


def namespaced(server_name, name):
    """The model-visible name of tool ``name`` on the MCP server
    ``server_name``: ``mcp__<server>__<tool>``. A bare name in an allow list,
    a deny list or a hook matcher silently matches nothing (plan section 2,
    fact 1), so nothing writes one by hand: it goes through here."""
    return "mcp__%s__%s" % (server_name, name)


def ok(payload=None, *, text=None):
    """A successful tool result: ``payload`` as JSON, or ``text`` verbatim."""
    if text is None:
        text = json.dumps(payload, indent=2, default=str)
    return {"content": [{"type": "text", "text": text}]}


def fail(message):
    """A failed tool result, whose ``message`` the model reads as the reason.

    ``is_error`` is what the SDK turns into a tool result the model is told
    failed; without it a refusal reads as a successful call that happened to
    return the word "refused", which a model will act on as if it had worked.
    """
    return {"content": [{"type": "text", "text": message}], "is_error": True}


class Grading(namedtuple("Grading", "read view write")):
    """A product's tool names in their three grades, each a tuple of bare
    names (see this module's docstring for what each grade means).

    The tuples keep the product's own order, so the allow list derived from
    them can be read against whatever plan lists the tools line by line.
    """

    __slots__ = ()

    @property
    def pre_allowed(self):
        """What never prompts, as the model sees it: read plus view."""
        return self.read + self.view

    @property
    def pause_gated(self):
        """What the pause switch refuses: everything that changes anything,
        whether the change is to the screen or to the project. Deliberately not
        the same set as what prompts, because the two questions are different:
        a card asks "may this happen at all", and the pause switch says "not
        right now, I am working"."""
        return self.view + self.write


#: One already-``_wrap``-gated tool, in the shape a non-Claude MCP bridge
#: (``http/routes_mcp.py``) or host-tool driver (``session/omp.py``) needs to
#: build its own server from - ``ToolServer.tool_table()``'s values. ``write``
#: folds in write-grade membership so ``routes_mcp`` can gate a write-grade
#: call through ``PermissionBroker`` (Codex's app-server has no equivalent of
#: Claude's own ``can_use_tool`` hook for a generic MCP tool call - see that
#: module's docstring).
ToolSpec = namedtuple("ToolSpec", "schema description handler write")

#: JSON Schema type words for the plain ``{param: python_type}`` shorthand a
#: handful of tools use (Mesh's ``set_visibility``, ``set_up_axis``,
#: ``select_pin``, ``measure``); every python type any of them actually uses
#: is a key here, and anything else falls back to ``"string"``, matching
#: ``claude_agent_sdk``'s own private ``_python_type_to_json_schema``'s final
#: fallback for a type it does not recognise either.
_JSON_SCHEMA_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _tool_json_schema(input_schema):
    """The JSON Schema ``inputSchema`` a non-Claude MCP transport needs for
    one tool, from the exact ``input_schema`` its own ``@tool(...)`` call
    declared.

    Every tool declares one of two shapes: already a full JSON Schema object
    (has both ``"type"`` and ``"properties"``), or the ``{param: python_type}``
    shorthand ``claude_agent_sdk``'s own ``@tool`` also accepts and expands
    itself, only for Claude (``create_sdk_mcp_server``'s private
    ``_build_schema``, not reused here: that function is unstable,
    single-underscore SDK-internal API, and this mirrors only the two shapes
    tools actually declare, never that function's third, TypedDict, branch,
    which nothing uses). ``{}`` - every no-argument tool's declared schema -
    falls into the second branch and comes out as an object schema with no
    properties, the same shape ``create_sdk_mcp_server`` would build for it.
    """
    if "type" in input_schema and "properties" in input_schema:
        return input_schema
    properties = {
        name: {"type": _JSON_SCHEMA_TYPES.get(py_type, "string")}
        for name, py_type in input_schema.items()
    }
    return {"type": "object", "properties": properties, "required": list(properties)}


def _verify(tools, grading):
    """Refuse a tool set that does not match ``grading`` exactly."""
    name = product.current().name
    built = [t.name for t in tools]
    duplicated = sorted({n for n in built if built.count(n) > 1})
    if duplicated:
        raise RuntimeError("%s tools declared twice: %s" % (name, ", ".join(duplicated)))
    grades = {"read": grading.read, "view": grading.view, "write": grading.write}
    for first, second in (("read", "view"), ("read", "write"), ("view", "write")):
        overlap = sorted(set(grades[first]) & set(grades[second]))
        if overlap:
            raise RuntimeError(
                "%s tool(s) %s are classified both %s and %s"
                % (name, ", ".join(overlap), first, second)
            )
    classified = set(grading.read) | set(grading.view) | set(grading.write)
    unclassified = sorted(set(built) - classified)
    if unclassified:
        raise RuntimeError(
            "%s tool(s) %s are built but not classified read, view or write in "
            "the product's tool grading; every default is wrong for something, so "
            "there is no default" % (name, ", ".join(unclassified))
        )
    missing = sorted(classified - set(built))
    if missing:
        raise RuntimeError(
            "%s tool(s) %s are classified in the product's tool grading but not "
            "built, so their names are pre-allowed and match nothing" % (name, ", ".join(missing))
        )


# The attribute ``asks_the_human`` sets on a tool's handler function.
_ASKS_THE_HUMAN = "_annealage_asks_the_human"


def asks_the_human(tool_def):
    """Mark ``tool_def`` as a tool whose handler asks the human itself, and
    return it.

    Such a tool decides per call whether a call needs a card (resolving one
    of the human's review comments does, resolving the model's own callout
    does not; ``review/tools.py``), and asks the ``PermissionBroker`` itself
    with a request that says what it is about. So ``ToolServer`` treats it as
    read-grade whatever grade the product gave it: pre-allowed, never marked
    write in ``tool_table``, so no transport asks a second, less specific,
    question first, and not pause-gated, since every call that changes
    anything the human cares about asks them anyway. Its model-visible and
    bare names are the server's ``never_remembered``: no "always allow" may
    stand in for the question it asks.
    """
    setattr(tool_def.handler, _ASKS_THE_HUMAN, True)
    return tool_def


def _wrap(tool_def, *, bus, gated, paused_message):
    """Apply the pause gate and the failure mapping to one tool.

    Both live here rather than in each handler, which is what lets the handler
    modules be plain argument-validate-call-return code with no ``try`` blocks
    of their own. The mapping is not cosmetic: the four ways a viewer call can
    fail mean four different things to a model, and flattening them to one
    message would leave it retrying a call that will never work or giving up on
    one that would work on the next attempt.

    ``ValueError`` is the handler modules' way of rejecting an argument, and
    its message is written for the model, so it is passed through verbatim.
    Anything else reaching here is a bug in the product: it is logged for the
    human with the tool's name and reported as a failed call, rather than
    raised into the MCP layer, where it would reach the model as an
    infrastructure error that says nothing about which tool broke.

    ``paused_message`` is what a gated tool says while the human has paused
    the view; the product writes it, because what "still possible" means is
    the product's to say.
    """
    name = product.current().name

    async def handler(args):
        if gated and bus.paused:
            return fail(paused_message)
        try:
            return await tool_def.handler(args)
        except ValueError as exc:
            return fail(str(exc))
        except NoViewerConnected as exc:
            # Carries plan section 3.3's exact wording, including the URL to
            # ask the human to open, so it is passed through unedited.
            return fail(str(exc))
        except ViewerGone:
            return fail(
                "the view this went to closed before it answered, so %s "
                "did not happen; ask the human whether the page is still "
                "open, then try again" % tool_def.name
            )
        except asyncio.TimeoutError:
            return fail(
                "the viewer did not answer %s in time, so it may or may "
                "not have happened; the page may be busy or in a "
                "background tab. Read the state back before assuming "
                "either way" % tool_def.name
            )
        except CallError as exc:
            error = exc.error or {}
            return fail(
                "the viewer refused %s: %s (%s)"
                % (
                    tool_def.name,
                    error.get("message", "no reason given"),
                    error.get("code", "no code"),
                )
            )
        except Exception as exc:
            sys.stderr.write("error: %s tool %s failed: %r\n" % (name, tool_def.name, exc))
            return fail(
                "%s failed inside %s itself (%s), which is a bug rather "
                "than anything you did; tell the human and carry on "
                "without it" % (tool_def.name, name, type(exc).__name__)
            )

    return dataclasses.replace(tool_def, handler=handler)


class ToolServer:
    """A product's tool server for one session.

    Built per session rather than at import, because every handler closes over
    the ``ViewerBus`` and the served directory of the run it belongs to.
    ``tools`` are the product's ``@tool`` definitions, ``grading`` their
    grades, ``bus`` the ``ViewerBus`` holding the pause switch, and
    ``paused_message`` what a gated tool answers while it is on.

    A tool marked ``asks_the_human`` is moved to the read grade whatever
    ``grading`` says (see that function); ``self.grading`` is the grading
    after that move, the one every derived list is built from.

    The server is named after the installed product's ``mcp_server_name``,
    and so is ``mcp_servers``' one key, deliberately: the key is what the
    model-visible ``mcp__<key>__<tool>`` name is built from, so a key that
    disagreed with the name ``pre_allowed`` is built from would leave every
    pre-allowed name matching nothing and every one of those tools prompting.
    """

    def __init__(self, tools, *, grading, bus, paused_message):
        _verify(tools, grading)
        installed = product.current()
        self.name = installed.mcp_server_name
        self.asks_the_human = tuple(
            t.name for t in tools if getattr(t.handler, _ASKS_THE_HUMAN, False)
        )
        moved = set(self.asks_the_human)
        self.grading = Grading(
            read=tuple(n for n in grading.read if n not in moved) + self.asks_the_human,
            view=tuple(n for n in grading.view if n not in moved),
            write=tuple(n for n in grading.write if n not in moved),
        )
        gated = set(self.grading.pause_gated)
        self.tools = tuple(
            _wrap(tool_def, bus=bus, gated=tool_def.name in gated, paused_message=paused_message)
            for tool_def in tools
        )
        self.server = create_sdk_mcp_server(
            self.name, version=installed.version, tools=list(self.tools)
        )

    @property
    def mcp_servers(self):
        return {self.name: self.server}

    @property
    def pre_allowed(self):
        """Every read- and view-grade tool, namespaced, in grading order: the
        ``allowed_tools`` a Claude session is built with. Write-grade tools are
        absent, which is what makes each of them reach the broker."""
        return tuple(namespaced(self.name, tool) for tool in self.grading.pre_allowed)

    @property
    def never_remembered(self):
        """The names no "always allow" may cover: every ``asks_the_human``
        tool, namespaced (as Claude, ``/mcp`` and the handler itself ask the
        broker) and bare (as omp's host-tool gate does), for the session's
        ``PermissionBroker`` (``launch.py``)."""
        return tuple(namespaced(self.name, n) for n in self.asks_the_human) + self.asks_the_human

    def tool_table(self):
        """``{name: ToolSpec(schema, description, handler, write)}`` off the
        same already-``_wrap``-gated handlers ``.mcp_servers`` is built from
        - the transport-neutral shape a non-Claude driver's own MCP bridge
        (``http/routes_mcp.py``) builds its own server from, without
        duplicating the pause-gate/failure-mapping ``_wrap`` already applied
        above. Description travels alongside the schema, not only the name:
        it is the search surface a model picks a tool from, so a transport
        that dropped it would leave a non-Claude backend calling these tools
        blind to what each one is for.
        """
        write = set(self.grading.write)
        return {
            tool_def.name: ToolSpec(
                schema=_tool_json_schema(tool_def.input_schema),
                description=tool_def.description,
                handler=tool_def.handler,
                write=tool_def.name in write,
            )
            for tool_def in self.tools
        }
