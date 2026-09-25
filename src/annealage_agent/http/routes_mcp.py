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

import mcp.types as types

from .. import product
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
    own already-``_wrap``-gated handler. ``server_name`` is the tool server's
    own name, which the broker is asked under (``mcp__<server>__<tool>``), the
    same name Claude's own ``can_use_tool`` hook would see for this call.
    """
    spec = tool_table.get(name)
    if spec is None:
        return types.CallToolResult(
            isError=True,
            content=[
                types.TextContent(
                    type="text", text="no such %s tool: %r" % (product.current().name, name)
                )
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


def register_mcp_routes(app, *, tools, broker, agent_token, allowed_origins=()):
    """Register ``POST /mcp`` on ``app``.

    ``tools`` is the product's ``ToolServer`` ``create_app`` builds once
    and shares with ``build_session`` (see its own comment for why one
    instance, not two): its already-``_wrap``-gated handlers are read once,
    here, into ``tool_table()``'s transport-neutral shape, at registration
    time rather than per request, since the tool set is fixed for the life
    of one served directory's app.

    ``agent_token`` is the run's agent token, and the only credential this
    route accepts. It is deliberately not the browser token: this route's
    caller is a subprocess the agent's own backend launches, whose command
    line and environment the agent's shell may be able to read, and the
    browser token is what authorises a permission decision over ``/ws``.
    Holding the agent token lets a caller do what the agent can already do
    through its own tools, with every write-grade call still reaching the
    human; it opens no browser route (``create_app`` refuses a run whose two
    tokens are equal).

    ``broker`` is whatever ``build_session`` attached to ``bus.broker``
    while constructing the session (``app.py``'s own comment on that
    channel); ``None`` for a viewer-only app, which never reaches this
    function at all (``create_app`` only calls it once a real session
    exists).
    """
    tool_table = tools.tool_table()
    server_name = tools.name

    @app.post("/mcp")
    async def mcp_route(req):
        if not _token_is_allowed(req, agent_token):
            return refusal()
        if not _origin_is_allowed(req, allowed_origins):
            return refusal()

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
            return {"result": _tool_list_result(tool_table)}, 200

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") if params.get("arguments") is not None else {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return {
                    "ok": False,
                    "error": '"params" must be {"name": str, "arguments"?: object}',
                }, 400
            call_result = await _call_tool_result(
                tool_table, broker, name, arguments, server_name=server_name
            )
            return {
                "result": call_result.model_dump(mode="json", by_alias=True, exclude_none=True)
            }, 200

        return {"ok": False, "error": "unknown method: %r" % method}, 400
