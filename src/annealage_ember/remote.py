"""Remote MCP servers a product's tool server proxies, beside its own tools.

A product declares them in its ``build_tools``, next to its own tools:
``ToolServer(tools, grading=..., bus=..., paused_message=...,
remote=(RemoteServer(name, url, grading),))``. Each one reaches every backend
as a server namespace of its own (``tools.ToolServer`` says how each backend
names it), through the same grading, pause gate and permission broker as the
product's own tools.

**Why a proxy and not the backend's own MCP client.** A Claude session runs
with ``strict_mcp_config`` and only the servers this package gives it, omp
gets only host tools, and Codex only the stdio bridge, so a server a product
wants its agent to reach has to come through here anyway. Passing it on as a
native ``{"type": "http"}`` entry would reach Claude alone, and there the
backend's own client would decide what runs without a card. Proxying keeps
every call on the path the product's tools take: ``tools._wrap`` (the pause
gate on view and write, the failure mapping) and, for a write-grade tool, the
broker. Streamable HTTP only: no command or stdio entry, so nothing a
declaration names is ever spawned, the property ``strict_mcp_config`` exists
for.

**Discovery, when the tool server is built.** For each remote:
connect, ``initialize``, make the optional ``prime`` call (a server that
advertises more tools once a first tool has been called, as some do until
their getting-started tool is called, lists only a few before it), then
``list_tools``. Each listed tool the remote's grading names becomes a proxied
tool with the remote's own schema and description, and one the remote's
``excluded`` names (a tool the product leaves out on purpose) is left out
silently. Unlike the product's own tools, which ``tools._verify`` holds to
their grading exactly, a remote's tool set changes without the product
changing, so a listed tool neither names is left out and a graded one it does
not list is skipped, each with a warning, and neither stops the session. Nor
does a remote that cannot be reached within ``DISCOVERY_TIMEOUT``: the
session starts without it, and
``ToolServer.reconnect`` tries it again later (``app.serve`` does, on a timer
and on the first turn). A reached remote's tool set is fixed from then on; its
later ``tools/list_changed`` is not followed.

``ToolServer`` is built synchronously, inside ``create_app``, which products
call from within their running event loop. Discovery therefore runs on a
worker thread with an event loop of its own and is waited for there: nothing
is served yet while the tool server is being built, and the MCP client's
connections never touch the caller's loop, whether or not it has one.

**Each call opens its own connection.** A proxied call connects,
initializes, calls and disconnects, all inside the handler's own task.
Handlers run on whichever loop the backend calls them on (Claude's SDK
server, Codex's ``/mcp`` route and omp's host-tool bridge all post onto the
app's loop, a test's may be another), and an MCP client session's streams
and task group belong to the loop and task that opened them. A connection per
call is correct on any loop by construction, needs no lifecycle of its own at
shutdown, and is unaffected by a remote restarting or expiring its sessions
between calls. The cost is an ``initialize`` round trip per call, small next
to a model turn. No ``prime`` call per connection: a tool is callable by name
whether or not the session it arrives on has advertised it.

**Results.** Text and image content passes through; any other kind is left
out, with a line saying so. A remote ``isError`` is a failed call (``fail``'s
shape), and a remote that cannot be reached, answers with an error or does
not answer in time gives a failed call whose message says which, and what to
do. Everything a remote returns is untrusted input, like any tool result:
nothing here reads it as an instruction. A remote's ``initialize``
instructions are passed to the model (``ToolServer.remote_instructions``),
because a remote's own advice on using its tools is what they are for.
"""

import asyncio
import contextlib
import json
import re
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import anyio
import httpx
import httpx2
import mcp.types as types
from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client

from . import product, remote_actions
from .tools import Grading, _wrap, fail, namespaced

#: How long connecting, ``initialize``, the ``prime`` call and ``list_tools``
#: may take together for one remote at startup. Remotes are reached
#: concurrently, so this is also the longest a startup waits for them all.
DISCOVERY_TIMEOUT = 10.0

#: How long one proxied call may take, connecting included. The Codex stdio
#: bridge sets no read timeout of its own, so this is what bounds it.
CALL_TIMEOUT = 60.0

# A remote's name is its server namespace on every backend: a TOML bare key
# in Codex's ``mcp_servers.<name>`` override, a path segment of ``/mcp/<name>``
# and part of every tool name a model sees, so it is held to the characters
# all of those accept, and ``__`` (the separator in those tool names) is
# refused.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class RemoteServer(
    namedtuple(
        "RemoteServer",
        "name url grading prime headers excluded annotation_grading product_only read_tools denied annotation_default_view",
        defaults=(None, None, (), False, (), (), (), False),
    )
):
    """A remote MCP server, reached over streamable HTTP at ``url``.

    ``name`` is its server namespace (``mcp__<name>__<tool>`` to Claude and
    Codex, ``<name>__<tool>`` to omp). ``grading`` is a ``tools.Grading`` of
    its tool names, which decides what is proxied at all and how, exactly as
    a product's grading does for its own tools. ``prime`` is ``(tool,
    arguments)``, called once before listing, for a server that advertises
    more tools after a first call. ``headers`` are sent with every request.
    ``excluded`` names are silently withheld. ``annotation_grading`` adds
    read-annotated tools at read grade; ``product_only`` withholds named
    tools unless explicitly graded. ``annotation_default_view`` admits other
    discovered tools at view grade when annotation grading is enabled.
    ``read_tools`` and ``denied`` govern direct product ``call`` access.
    """

    __slots__ = ()


#: A remote this session reached at startup: its name, its grading narrowed
#: to the tools it listed, those tools (``tools._wrap``-gated proxies), its
#: in-process SDK server for Claude, and its ``initialize`` instructions.
Connected = namedtuple("Connected", "name grading tools server instructions")


def connect(servers, *, product_server, bus, paused_message, retry=False):
    """``(connected, unreached)``: the ``Connected`` remotes of ``servers``,
    in order, and the ``RemoteServer``s that could not be reached. Refuses,
    before connecting to anything, a remote named like ``product_server``, two
    remotes with one name, a name no backend can carry, a grading naming a
    tool twice, or a tool both graded and excluded.

    ``retry`` is a second attempt at remotes already reported unreachable
    (``ToolServer.reconnect``): one still unreachable is not reported again,
    and one reached now is."""
    _check(servers, product_server)
    listings = _discover(servers)
    version = product.current().version
    connected = []
    unreached = []
    for server, listing in zip(servers, listings, strict=True):
        if isinstance(listing, BaseException):
            unreached.append(server)
            if not retry:
                _warn(
                    "the %s MCP server could not be reached (%s), so this session starts "
                    "without its tools; it is tried again while the session runs"
                    % (server.name, _reason(listing, DISCOVERY_TIMEOUT) or repr(_leaf(listing)))
                )
            continue
        if retry:
            _warn("the %s MCP server, unreachable until now, has been reached" % server.name)
        instructions, listed = listing
        listed = {tool.name: tool for tool in listed}
        explicit_read = set(server.grading.read)
        explicit_view = set(server.grading.view)
        explicit_write = set(server.grading.write)
        product_only = set(server.product_only)
        denied = set(server.denied)
        if server.annotation_grading:
            for name, tool in listed.items():
                annotations = getattr(tool, "annotations", None)
                if getattr(annotations, "read_only_hint", None) is True and (
                    name not in product_only
                    and name not in denied
                    and name not in explicit_view
                    and name not in explicit_write
                ):
                    explicit_read.add(name)
                elif (
                    server.annotation_default_view
                    and name not in denied
                    and name not in product_only
                    and name not in server.excluded
                    and name not in explicit_read
                    and name not in explicit_write
                ):
                    explicit_view.add(name)
        explicit_read.difference_update(denied | product_only)
        explicit_view.difference_update(denied)
        explicit_write.difference_update(denied)
        graded = explicit_read | explicit_view | explicit_write
        ungraded = sorted(set(listed) - graded - set(server.excluded) - product_only)
        if ungraded:
            _warn(
                "the %s MCP server lists %s, which its grading does not name, so the "
                "agent does not get them" % (server.name, ", ".join(ungraded))
            )
        unlisted = sorted(graded - set(listed))
        if unlisted:
            _warn(
                "the %s MCP server does not list %s, which its grading names, so the "
                "agent does not get them" % (server.name, ", ".join(unlisted))
            )
        grading = Grading(
            tuple(n for n in server.grading.read if n in listed and n not in denied)
            + tuple(sorted(n for n in explicit_read if n in listed and n not in server.grading.read)),
            tuple(n for n in server.grading.view if n in listed and n not in denied)
            + tuple(sorted(n for n in explicit_view if n in listed and n not in server.grading.view)),
            tuple(n for n in server.grading.write if n in listed and n not in denied),
        )
        gated = set(grading.pause_gated)
        tools = tuple(
            _wrap(
                _proxied(server._replace(grading=grading), listed[name], bus),
                bus=bus,
                gated=name in gated,
                paused_message=paused_message,
                hosted_key=namespaced(server.name, name),
            )
            for name in grading.read + grading.view + grading.write
        )
        connected.append(
            Connected(
                name=server.name,
                grading=grading,
                tools=tools,
                server=create_sdk_mcp_server(server.name, version=version, tools=list(tools)),
                instructions=(instructions or "").strip() or None,
            )
        )
    return tuple(connected), tuple(unreached)


def _check(servers, product_server):
    seen = set()
    for server in servers:
        name = server.name
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or "__" in name:
            raise RuntimeError(
                "remote MCP server name %r is not letters, digits, - and _ (without __)" % (name,)
            )
        if name == product_server:
            raise RuntimeError(
                "remote MCP server %r has the product's own server name, so its tools and "
                "the product's would share one namespace" % name
            )
        if name in seen:
            raise RuntimeError("remote MCP server %r is declared twice" % name)
        seen.add(name)
        for first, second in (("read", "view"), ("read", "write"), ("view", "write")):
            overlap = sorted(
                set(getattr(server.grading, first)) & set(getattr(server.grading, second))
            )
            if overlap:
                raise RuntimeError(
                    "%s tool(s) %s are classified both %s and %s"
                    % (name, ", ".join(overlap), first, second)
                )
        graded = set(server.grading.read) | set(server.grading.view) | set(server.grading.write)
        excluded = sorted(graded & set(server.excluded))
        if excluded:
            raise RuntimeError(
                "%s tool(s) %s are both graded and excluded" % (name, ", ".join(excluded))
            )


def _warn(message):
    sys.stderr.write("warning: %s\n" % message)


# -- the MCP client ----------------------------------------------------------


@contextlib.asynccontextmanager
async def _session(server):
    """An initialized ``ClientSession`` on ``server``, and its
    ``InitializeResult``, closed (and the remote's session ended) on exit."""
    async with httpx2.AsyncClient(
        headers=dict(server.headers or {}), timeout=httpx2.Timeout(CALL_TIMEOUT)
    ) as client:
        async with streamable_http_client(server.url, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                yield session, initialized


async def _call(session, name, arguments):
    """``tools/call`` ``name`` with ``arguments``, the result unvalidated.

    Sent as a plain request rather than through ``ClientSession.call_tool``,
    which lists the server's tools after every call to check structured
    content against an output schema, and logs a warning for a tool the
    connection has not had advertised: every call here would pay a second
    round trip and, on a server that advertises progressively, print that
    warning. Nothing here reads structured content as more than text to pass
    on, so there is nothing for the check to protect.
    """
    request = types.CallToolRequest(
        params=types.CallToolRequestParams(name=name, arguments=arguments)
    )
    return await session.send_request(request, types.CallToolResult)


async def _listing(server):
    """``(instructions, [types.Tool])`` for ``server``, primed first."""
    with anyio.fail_after(DISCOVERY_TIMEOUT):
        async with _session(server) as (session, initialized):
            if server.prime is not None:
                tool, arguments = server.prime
                try:
                    failed = (await _call(session, tool, arguments)).is_error
                except MCPError:
                    failed = True
                if failed:
                    _warn(
                        "the %s MCP server answered its prime call, %s, with an error, so "
                        "it may list fewer tools" % (server.name, tool)
                    )
            listed = []
            params = None
            while True:
                page = await session.list_tools(params=params)
                listed.extend(page.tools)
                if not page.next_cursor:
                    break
                params = types.PaginatedRequestParams(cursor=page.next_cursor)
    return initialized.instructions, listed


def _discover(servers):
    """``_listing`` for every one of ``servers`` at once, on a worker thread
    with a loop of its own (see the module docstring); a remote that failed
    has the exception in its place."""

    async def gather():
        return await asyncio.gather(*(_listing(s) for s in servers), return_exceptions=True)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="remote-mcp") as pool:
        return pool.submit(asyncio.run, gather()).result()


def _leaf(exc):
    """The first exception inside ``exc``'s exception groups, or ``exc``."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


def _reason(exc, timeout):
    """Why a remote could not be reached, in a few words; ``None`` for an
    exception that is not about reaching it (a bug here)."""
    leaf = _leaf(exc)
    if isinstance(leaf, TimeoutError):
        return "no answer within %g s" % timeout
    if isinstance(leaf, (httpx.HTTPStatusError, httpx2.HTTPStatusError)):
        return "HTTP %d" % leaf.response.status_code
    if isinstance(leaf, MCPError):
        return "it answered with an error: %s" % leaf.error.message
    if isinstance(leaf, (httpx.TransportError, httpx2.TransportError, OSError)):
        return "%s: %s" % (type(leaf).__name__, leaf) if str(leaf) else type(leaf).__name__
    return None
def _action_context(bus):
    return _read_context(getattr(bus, "remote_action_context", None))


def _read_context(callback, *args):
    if not callable(callback):
        return None
    try:
        value = callback(*args)
    except Exception:
        return None
    return value if isinstance(value, dict) else None



# -- proxied tools -------------------------------------------------------------


def _proxied(server, tool, bus=None):
    """An unwrapped ``SdkMcpTool`` calling ``tool`` on ``server``.

    Its schema is the remote's own, given an explicit ``properties`` so that
    both the SDK and ``tools._tool_json_schema`` pass it on unchanged rather
    than reading it as the ``{param: type}`` shorthand. omp refuses a tool
    with no description, so one without gets a plain one.
    """
    schema = dict(tool.input_schema or {})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    description = (tool.description or "").strip() or "%s, on the %s MCP server" % (
        tool.name, server.name,
    )
    name = tool.name

    async def handler(args):
        context = _action_context(bus)
        expected_context = context
        if name in server.grading.write:
            approval_context = getattr(bus, "remote_approval_context", None)
            if callable(approval_context):
                expected_context = _read_context(
                    approval_context, server.name + "__" + name, args
                )
                store = getattr(bus, "remote_action_store", None)
                if expected_context is None or store is None or not remote_actions.safe_context(
                    expected_context, store.project
                )["context_verified"]:
                    return fail("remote write refused: its approved source context is unavailable or invalid")
        return await call(
            server, name, args, action_store=getattr(bus, "remote_action_store", None),
            source="agent", read_only=name in server.grading.read,
            allowed_reads=(name,) if name in server.grading.read else (),
            allowed_writes=() if name in server.grading.read else (name,),
            context=context, expected_context=expected_context,
            context_callback=getattr(bus, "remote_action_context", None),
        )

    return SdkMcpTool(name=name, description=description, input_schema=schema, handler=handler)


async def call(
    server, name, arguments, *, action_store=None, source="agent", read_only=False,
    approval=None, context=None, expected_context=None, context_callback=None,
    allowed_reads=(), allowed_writes=(), timeout=None,  # noqa: ASYNC109
):
    """Call a remote tool, enforcing its read policy and recording writes."""
    timeout = CALL_TIMEOUT if timeout is None else timeout
    if name in server.denied:
        return fail("remote call refused: %s is denied" % name)
    if (
        server.read_tools
        and (read_only or not server.annotation_default_view)
        and name not in server.read_tools
        and name not in allowed_reads
        and name not in allowed_writes
    ):
        return fail(
            "remote call refused: %s is not read-only; name it in allowed_reads "
            "or allowed_writes to permit it" % name
        )
    invocation = None
    if not read_only:
        if action_store is None:
            return fail("remote write refused: durable action ledger is unavailable")
        if expected_context is not None:
            before = remote_actions.safe_context(expected_context, action_store.project)
            current = remote_actions.safe_context(context, action_store.project)
            if before != current:
                return fail("remote write refused because its approved authority context changed")
        try:
            invocation, blocked = action_store.begin(
                server.name, name, arguments, source=source, approval=approval, context=context
            )
        except Exception:
            return fail(
                "remote write refused: durable action ledger could not reserve it; nothing was sent"
            )
        if blocked:
            return fail(
                "%s on %s has an unresolved prior invocation (%s); it was not sent again. "
                "Ask the human to inspect and explicitly reconcile action %s."
                % (name, server.name, blocked["outcome"], blocked["id"])
            )
    sent = False
    stale_context = False
    result = None
    try:
        with anyio.fail_after(timeout):
            async with _session(server) as (session, _):
                current_context = _read_context(context_callback) if callable(context_callback) else context
                if invocation and expected_context is not None:
                    stale_context = (
                        remote_actions.safe_context(expected_context, action_store.project)
                        != remote_actions.safe_context(current_context, action_store.project)
                    )
                if not stale_context:
                    if invocation:
                        action_store.dispatching(invocation)
                    sent = True
                    result = await _call(session, name, arguments)
    except BaseException as exc:
        if invocation:
            action_store.outcome(invocation, "uncertain" if sent else "not_sent")
        if isinstance(exc, asyncio.CancelledError):
            raise
        if result is not None:
            if invocation:
                action_store.outcome(invocation, "uncertain" if result.is_error else "reported_result")
            return _result(server.name, name, result)
        leaf = _leaf(exc)
        if sent and isinstance(leaf, MCPError):
            return fail(
                "the %s MCP server refused %s: %s" % (server.name, name, leaf.error.message)
            )
        reason = _reason(leaf, timeout)
        if reason is None:
            raise
        if sent:
            return fail(
                "%s was sent to the %s MCP server but no answer came back (%s), so it "
                "may or may not have happened; check before retrying. Invocation %s is blocked."
                % (name, server.name, reason, invocation["id"] if invocation else "unrecorded")
            )
        return fail(
            "%s did not run: the %s MCP server could not be reached (%s). Try again shortly."
            % (name, server.name, reason)
        )
    if stale_context:
        action_store.outcome(invocation, "not_sent")
        return fail("remote write refused because its approved authority context changed")
    if invocation:
        action_store.outcome(invocation, "uncertain" if result.is_error else "reported_result")
    return _result(server.name, name, result)


def _result(server_name, tool_name, result):
    content = []
    omitted = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, types.ImageContent):
            content.append({"type": "image", "data": block.data, "mimeType": block.mime_type})
        else:
            omitted.append(block.type)
    if result.is_error:
        text = "\n".join(item["text"] for item in content if item["type"] == "text")
        return fail(
            text or "%s on the %s MCP server failed without saying why" % (tool_name, server_name)
        )
    if not content and result.structured_content is not None:
        content.append({"type": "text", "text": json.dumps(result.structured_content, indent=2)})
    if omitted:
        content.append(
            {
                "type": "text",
                "text": "(%s also returned %s content, which is not passed on here)"
                % (tool_name, ", ".join(sorted(set(omitted)))),
            }
        )
    return {"content": content}
