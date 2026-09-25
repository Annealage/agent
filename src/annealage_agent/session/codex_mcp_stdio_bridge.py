"""The stdio-to-HTTP proxy Codex launches per thread to reach a product's tools.

``planning/tickets/phase3_codex-tool-mcp-bridge.md`` and
``planning/20260919_codex-mcp-bridge-finding.md``: Codex's app-server only
ever attaches an MCP server as a stdio subprocess, never registers one over
HTTP directly. ``session/codex.py``'s ``CodexSession.start()`` registers
*this* module (``python -m <the product's codex_bridge_module>``, via
``CodexConfig.config_overrides``) as that subprocess. It is a thin,
short-lived translation shim with no tool logic of its own - it does not
know what a callout is, what write-class means, or what the human sees; it
only forwards ``tools/list``/``tools/call`` to the host process's own
``/mcp`` endpoint and hands back whatever comes back.

**How it authenticates, and why not on the command line.** ``/mcp`` accepts
only the per-run *agent* token, never the browser token that authorises
``/ws`` (see ``http/routes_mcp.py``). That token reaches this process through
its environment, as ``ANNEALAGE_AGENT_TOKEN``: ``CodexSession`` puts it in the
app-server's environment (``CodexConfig.env``) and names it in the server's
``env_vars``, which is Codex's own allowlist of variables it copies into an
MCP child's otherwise scrubbed environment. A command line is readable by
every user on the machine through ``ps``; an environment is readable only by
the same user (and root), through ``/proc/<pid>/environ``. The same user
includes the agent's own shell, which is why the split matters more than the
hiding: this token grants exactly what the agent can already do through this
bridge - list and call its own tools, with every write-grade call still going
to the human - and nothing a browser session can, so reading it gets an agent
nothing it did not have. An inherited file descriptor or a 0600 file would
hide it no better from the same user and would need plumbing Codex does not
offer for an MCP child; the environment is the channel Codex supports.

The stdio side is the one that actually has to be a fully conformant MCP
server, since Codex's real, unmodified client is on the other end of it and
this project does not control that side. So it is built entirely on the
official ``mcp`` SDK's own ``Server``/``stdio_server`` - the same officially
tested stdio-server path every other MCP stdio server uses, which handles
the ``initialize`` handshake, capability negotiation and JSON-RPC framing
without a line of protocol code here. The one thing that *is* this module's
own responsibility is what the registered ``list_tools``/``call_tool``
handlers do when the SDK calls them: an authenticated HTTP POST to the host's
``/mcp`` (``http/routes_mcp.py``), which is a small internal JSON contract
between this process and that endpoint, not itself a second MCP transport -
see that module's docstring for why the two ends do not need to renegotiate
MCP a second time on the inner hop.

No search was found for an existing MIT/Apache-licensed, dependency-light
stdio-to-HTTP MCP proxy worth vendoring instead of this file: the ecosystem's
best-known example of the pattern, ``mcp-remote``, is an npm package with no
Python equivalent, and reaching for it would mean shipping a Node runtime
dependency this project does not otherwise have, for a translation loop this file
implements in well under 100 lines using a dependency (``mcp``, ``httpx``)
this project already carries transitively through ``claude-agent-sdk`` (a
base dependency, not the optional ``codex`` extra) and now declares
directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

import anyio
import httpx
import mcp.server.stdio
import mcp.types as types
from mcp.server import InitializationOptions, NotificationOptions, Server

#: The environment variable the agent token arrives in. Codex does not keep
#: ``*TOKEN*`` variables out of the shells it runs for the model by default,
#: so ``CodexSession`` also excludes this name through
#: ``shell_environment_policy.exclude``; that policy governs shells only, and
#: the ``env_vars`` forwarding to this bridge is unaffected by it.
AGENT_TOKEN_ENV = "ANNEALAGE_AGENT_TOKEN"

#: How long one forwarded call is given to reach the host process and come
#: back. Generous: the host's own ``ViewerBus.CALL_TIMEOUT`` (10s) already bounds
#: how long a viewer-facing tool call can take, and a write-class call adds
#: however long the human takes to answer an approval card
#: (``PermissionBroker.DEFAULT_TIMEOUT``, five minutes) - this must clear
#: both comfortably, since a timeout here reaches Codex as a failed tool call
#: with no way to tell "the host is slow" from "the human has not answered yet".
CALL_TIMEOUT = 310.0


class AuthorityError(RuntimeError):
    """Raised when the host's own ``/mcp`` endpoint refuses or fails a call.
    Left to propagate out of a registered handler: the ``mcp`` SDK's own
    ``Server`` turns an uncaught exception into a JSON-RPC error response
    (``call_tool``) or an ``isError`` result (``list_tools``'s caller sees a
    failed listing) without crashing the stdio session - see
    ``mcp.server.lowlevel.server.Server._handle_request``."""


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=__spec__.name if __spec__ is not None else "codex_mcp_stdio_bridge",
        description="stdio<->HTTP MCP proxy: speaks MCP over stdio to a "
        "codex app-server subprocess and MCP-shaped JSON over HTTP to the "
        "host process's own /mcp endpoint. Launched by CodexSession, never by "
        "hand. The agent token is read from the %s environment variable." % AGENT_TOKEN_ENV,
    )
    parser.add_argument("--host", required=True, help="the host process's bind address")
    parser.add_argument("--port", required=True, type=int, help="the host process's port")
    parser.add_argument(
        "--server-name", required=True, help="the MCP server name to report to Codex"
    )
    parser.add_argument(
        "--server-version", required=True, help="the MCP server version to report to Codex"
    )
    parser.add_argument("--path", default="/mcp", help="the host's MCP route (default /mcp)")
    return parser.parse_args(argv)


async def _call_authority(
    client: httpx.AsyncClient, url: httpx.URL, method: str, params: dict
) -> Any:
    """POST one ``{"method": ..., "params": ...}`` request to the host's ``/mcp``
    and return its ``"result"``, or raise ``AuthorityError``."""
    try:
        response = await client.post(url, json={"method": method, "params": params})
    except httpx.HTTPError as exc:
        raise AuthorityError("could not reach the host's own /mcp endpoint: %r" % exc) from exc
    if response.status_code != 200:
        raise AuthorityError(
            "the host's /mcp endpoint refused %s: HTTP %d" % (method, response.status_code)
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthorityError("the host's /mcp endpoint returned invalid JSON: %r" % exc) from exc
    if "error" in payload:
        raise AuthorityError(str(payload["error"]))
    return payload.get("result")


def build_server(client: httpx.AsyncClient, url: httpx.URL, *, name: str, version: str) -> Server:
    """A low-level ``Server`` named ``name``/``version`` whose
    ``list_tools``/``call_tool`` handlers forward to the host's own ``/mcp``
    endpoint - the whole of this proxy's own logic, everything else being the
    official SDK's stdio machinery."""
    server: Server = Server(name, version=version)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        result = await _call_authority(client, url, "tools/list", {})
        return [types.Tool.model_validate(t) for t in (result or {}).get("tools", [])]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
        result = await _call_authority(
            client, url, "tools/call", {"name": name, "arguments": arguments}
        )
        return types.CallToolResult.model_validate(result or {})

    return server


def authority_url(host: str, port: int, path: str) -> httpx.URL:
    """The URL to POST to on the host process. Built from separate
    ``host``/``port`` argv fields, not a pre-joined string, so IPv6 bracketing
    (``httpx.URL``'s own job) never has to be done twice by two different
    pieces of code agreeing on the same escaping."""
    return httpx.URL(scheme="http", host=host, port=port, path=path)


async def run(args: argparse.Namespace, token: str) -> None:
    url = authority_url(args.host, args.port, args.path).copy_merge_params({"t": token})
    try:
        async with httpx.AsyncClient(timeout=CALL_TIMEOUT) as client:
            server = build_server(client, url, name=args.server_name, version=args.server_version)
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=args.server_name,
                        server_version=args.server_version,
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
    except anyio.get_cancelled_exc_class():
        # Resolved here, while the anyio backend anyio.run() (in main(),
        # below) started is still active. Codex's app-server tears this
        # subprocess down by closing its stdin (stdio_server's read loop
        # then ends and server.run() returns normally) or by killing it
        # outright; this only catches a Ctrl-C given to the subprocess
        # directly during manual testing, on a backend that surfaces it as
        # a cancellation rather than KeyboardInterrupt, so exit quietly
        # rather than propagating a traceback for an interruption that is
        # not a bug. Deliberately not caught in main()'s except clause
        # below instead: by the time anyio.run() has returned or raised
        # there, no async backend is active, and calling
        # anyio.get_cancelled_exc_class() then raises
        # anyio.NoEventLoopError rather than matching anything, replacing
        # the exception this was meant to catch with a different one and
        # still exiting through a traceback.
        pass


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get(AGENT_TOKEN_ENV)
    if not token:
        # Refused before speaking MCP at all, so Codex reports the server as
        # failed to start rather than listing no tools: an unauthenticated
        # bridge could only ever be refused by /mcp anyway.
        sys.stderr.write(
            "error: %s is not set; this bridge is launched by CodexSession\n" % AGENT_TOKEN_ENV
        )
        return 2
    try:
        anyio.run(run, args, token)
    except KeyboardInterrupt:
        # See run()'s own except clause above for the cancellation case;
        # this catches the same manual-Ctrl-C interruption on a backend
        # (asyncio, the default anyio.run() uses here with no ``backend=``
        # given) that surfaces it as KeyboardInterrupt instead.
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
