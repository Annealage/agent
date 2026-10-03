"""``POST /mcp``: the HTTP/authority side of the Codex tool-exposure bridge.

Annealage Mesh's ``planning/tickets/phase3_codex-tool-mcp-bridge.md``
(design) and ``planning/20260919_codex-mcp-bridge-finding.md`` (why there
are two hops, not one): Codex's app-server only ever launches an MCP server as a stdio
subprocess, never registers one over HTTP directly. So this route is not
reached by Codex itself - it is reached by the agent layer's own stdio-to-HTTP proxy
(``session/codex_mcp_stdio_bridge.py``), which Codex's app-server launches
as a subprocess per thread (``session/codex.py``'s ``config_overrides``) and
which speaks real MCP over stdio to Codex on one side and this route on the
other.

Because this endpoint's only legitimate caller is that proxy - a process
the product's own run launches and that never leaves this machine - it
speaks a small, internal JSON contract of this package's own rather than the official MCP SDK's full
Streamable HTTP transport (session ids, SSE, resumability): that machinery
exists for a general-purpose MCP client reachable from anywhere, which this
is not. The protocol conformance that actually matters - the wire format
Codex's own real MCP client sees - is owned entirely by the proxy's stdio
side, built on the official ``mcp`` SDK's own ``Server``/``stdio_server``.
This route's request and response bodies are still built from
``mcp.types.Tool``/``CallToolResult`` (via ``model_dump(by_alias=True)``),
so the JSON shape matches the real MCP wire format field-for-field
(``inputSchema``, not ``input_schema``) and the proxy can parse it back with
the same types on its own side - the two ends cannot drift apart
independently, even though the transport between them is not itself
Streamable HTTP.

Request body: ``{"method": "tools/list"}`` or ``{"method": "tools/call",
"params": {"name": ..., "arguments": {...}}}``. Response body on success:
``{"result": {...}}``, the ``ListToolsResult``/``CallToolResult`` shape for
the method called. A malformed request is a plain ``{"ok": false, "error":
...}`` 400, matching every other JSON route in this package
(``read_json_body``).

Token- and Origin-gated like ``/ws``/``/settings`` (``ws.py``'s
``_token_is_allowed``/``_origin_is_allowed``, reused rather than reinvented),
but against a different token: the run's *agent* token, never the browser
token. A tool-execution surface reachable from wherever the app-server
subprocess runs must never be unauthenticated, even bound to loopback only,
since the subprocess is not necessarily co-located with a human who already
passed the browser's own token check. And it must not be the browser token,
because the bridge that calls this route runs beside the agent's own shell,
and the browser token is what approves a permission card over ``/ws`` (see
``session/codex_mcp_stdio_bridge.py`` for how the agent token reaches it).

**Where the write-class approval gate lives, and why here.** Every write-class
product tool call must reach ``PermissionBroker`` exactly once
(``phase3_codex-tool-mcp-bridge.md``'s own acceptance criteria). For Claude,
that already happens entirely outside ``tools.py``: the Claude Agent
SDK calls ``session/sdk.py``'s own ``can_use_tool`` for every tool not in
``allowed_tools`` (every write-class one), before the handler ever runs.
Codex's app-server has no equivalent hook for a generic external MCP tool
call - its own ``approval_handler`` fires only for its two native actions,
``item/commandExecution/requestApproval`` and
``item/fileChange/requestApproval`` (``session/codex.py``'s
``_APPROVAL_METHOD_TOOL``), never for ``tools/call`` on an MCP server it has
attached. Without a gate somewhere in this bridge, a write-class tool routed
through it would reach the broker zero times, not two - the failure mode
this route exists to close. ``tools.py``'s own ``_wrap`` deliberately
does not call the broker (see its docstring), so this is not a second place
the READ/VIEW/WRITE classification gets decided: it is the one place this
particular transport connects the classification ``tool_table()`` already
carries (``ToolSpec.write``) to this particular driver's approval mechanism,
exactly as ``session/sdk.py`` does for its own.
"""

import hmac
import time

import mcp.types as types

from ..tools import namespaced
from . import read_json_body
from .ws import _origin_is_allowed, _token_is_allowed, refusal


def _tool_list_result(tool_table):
    """A ``ListToolsResult``, JSON-ready, for every tool ``tool_table`` has."""
    tools = [
        types.Tool(name=name, description=spec.description, inputSchema=spec.schema)
        for name, spec in tool_table.items()
    ]
    return types.ListToolsResult(tools=tools).model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def _content_blocks(result):
    """``tools/__init__.py``'s ``{"content": [...]}`` shape to MCP content
    blocks - the same two kinds ``create_sdk_mcp_server`` itself translates
    (``ok``/``fail`` only ever produce ``text``; ``capture_view`` is the one
    handler in this package that also produces ``image``). Nothing else is
    built anywhere under ``tools/``, so nothing else is handled here.
    """
    blocks = []
    for item in result.get("content", []):
        kind = item.get("type")
        if kind == "text":
            blocks.append(types.TextContent(type="text", text=item["text"]))
        elif kind == "image":
            blocks.append(
                types.ImageContent(type="image", data=item["data"], mimeType=item["mimeType"])
            )
    return blocks


async def _call_tool_result(tool_table, broker, name, arguments, *, server_name):
    """A ``CallToolResult`` for one ``tools/call``, the broker consulted
    first and exactly once when ``name`` is write-class. See this module's
    docstring for why that gate lives here rather than in ``tool_table()``'s
    own already-``_wrap``-gated handler. ``server_name`` is the name of the
    server ``tool_table`` belongs to (the product's, or a remote's), which the
    broker is asked under (``mcp__<server>__<tool>``), the same name Claude's
    own ``can_use_tool`` hook would see for this call.
    """
    spec = tool_table.get(name)
    if spec is None:
        return types.CallToolResult(
            isError=True,
            content=[
                types.TextContent(type="text", text="no such %s tool: %r" % (server_name, name))
            ],
        )
    if spec.write:
        if broker is None:
            # Fail closed: a session with no broker configured is not a
            # session a write-class tool should ever run unsupervised in,
            # whatever the reason it is missing.
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(
                        type="text",
                        text="no permission broker is configured for this session, so "
                        "%s cannot run" % name,
                    )
                ],
            )
        decision = await broker.ask(namespaced(server_name, name), arguments, None)
        if not decision.allow:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=decision.message)]
            )
    result = await spec.handler(arguments)
    return types.CallToolResult(
        content=_content_blocks(result), isError=bool(result.get("is_error", False))
    )


def register_mcp_routes(
    app,
    *,
    tools,
    current_broker,
    agent_token,
    allowed_origins=(),
    hosted_mode=False,
    hosted_bus=None,
    hosted_tool_ops=None,
):
    """Register ``POST /mcp``, and ``POST /mcp/<remote>`` for each remote MCP
    server the tool server reached, on ``app``.

    ``tools`` is the product's ``ToolServer`` ``create_app`` builds once
    and shares with ``build_session`` (see its own comment for why one
    instance, not two): its already-``_wrap``-gated handlers are read once,
    here, into ``tool_table()``'s transport-neutral shape, at registration
    time rather than per request, since the tool set is fixed for the life
    of one served directory's app. A remote's tools (``remote_tables()``) are
    served on a route of their own, the path the Codex bridge registered for
    that remote is pointed at (``session/codex.py``), under bare names, with
    the broker asked under ``mcp__<remote>__<tool>``. Those are looked up
    when a call arrives, not here: a remote first reached after startup
    (``app.retry_remotes``) has a bridge of its own in the next session an
    app resumes with, which must find its route.

    ``agent_token`` is the run's agent token, and the only credential these
    routes accept. It is deliberately not the browser token: this route's
    caller is a subprocess the agent's own backend launches, whose command
    line and environment the agent's shell may be able to read, and the
    browser token is what authorises a permission decision over ``/ws``.
    Holding the agent token lets a caller do what the agent can already do
    through its own tools, with every write-grade call still reaching the
    human; it opens no browser route (``create_app`` refuses a run whose two
    tokens are equal).

    ``current_broker`` is an async function returning the broker a
    ``tools/call`` is gated by, awaited once per call: ``create_app``'s
    returns whatever the live session's factory attached to ``bus.broker``
    (``app.py``'s own comment on that channel), after resuming a session its
    idle timer closed, so a call never meets the broker of a session that
    has gone. ``create_app`` registers these routes only once a real
    session exists; a viewer-only app has no ``/mcp`` at all.
    """
    if hosted_mode:
        if hosted_bus is None:
            raise ValueError("hosted MCP routes require the hosted turn bus")
        hosted_bus.hosted_tool_ops = dict(hosted_tool_ops or {})
    tool_table = tools.tool_table()
    # Rebuilt only when the reached remotes change (``ToolServer.reconnect``
    # replaces the tuple).
    remote_cache = {"remotes": None, "tables": {}}

    def remote_table(name):
        if remote_cache["remotes"] is not tools.remotes:
            remote_cache["remotes"] = tools.remotes
            remote_cache["tables"] = tools.remote_tables()
        return remote_cache["tables"].get(name)

    async def serve(req, table, server_name):
        if hosted_mode:
            expected = getattr(hosted_bus, "hosted_turn_secret", None)
            if (
                expected is None
                or not hosted_bus.hosted_turn_live
                or time.time() >= hosted_bus.hosted_turn_exp
                or not hmac.compare_digest(
                    req.headers.get("Authorization", ""), "Bearer " + expected
                )
            ):
                return refusal()
        elif not _token_is_allowed(req, agent_token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()
        if table is None:
            return {"ok": False, "error": "no such MCP server"}, 404

        data, error = await read_json_body(req)
        if error is not None:
            return error
        if not isinstance(data, dict) or not isinstance(data.get("method"), str):
            return {
                "ok": False,
                "error": 'body must be {"method": "tools/list" | "tools/call", ...}',
            }, 400

        method = data["method"]
        params = data.get("params") or {}
        if not isinstance(params, dict):
            return {"ok": False, "error": '"params" must be an object'}, 400

        if method == "tools/list":
            return {"result": _tool_list_result(table)}, 200

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") if params.get("arguments") is not None else {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return {
                    "ok": False,
                    "error": '"params" must be {"name": str, "arguments"?: object}',
                }, 400
            call_result = await _call_tool_result(
                table, await current_broker(), name, arguments, server_name=server_name
            )
            return {
                "result": call_result.model_dump(mode="json", by_alias=True, exclude_none=True)
            }, 200

        return {"ok": False, "error": "unknown method: %r" % method}, 400

    @app.post("/mcp")
    async def mcp_route(req):
        return await serve(req, tool_table, tools.name)

    @app.post("/mcp/<remote>")
    async def remote_mcp_route(req, remote):
        return await serve(req, remote_table(remote), remote)
