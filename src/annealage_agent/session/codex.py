"""The real agent session for the Codex backend: a low-level ``CodexClient``
behind the ``AgentSession`` seam.

**Why this file drives ``openai_codex.client.CodexClient`` directly, never
``Codex``/``AsyncCodex``/``AsyncCodexClient``.** Read
``planning/20260919_codex-approval-handler-finding.md`` for the full source
trail; the short version is that the public wrapper classes have no
constructor parameter for ``approval_handler`` at all, and silently
auto-accept every ``item/commandExecution/requestApproval`` request via
``CodexClient._default_approval_handler`` when none is given. For a project
whose whole approval-broker design exists to put a human between the model
and a destructive command, that is not a partial gap, it is a total bypass.
Constructing ``CodexClient`` directly and always passing ``approval_handler``
is the only way to keep a Codex-backed session honest.

**Why every call into the client crosses a thread bridge, not only the
approval callback.** ``CodexClient``'s own docstring: "Synchronous typed
JSON-RPC client for `codex app-server` over stdio." There is no working
async variant that also supports ``approval_handler``. This class therefore
owns a dedicated single-worker ``ThreadPoolExecutor`` (``self._executor``)
that every one-shot blocking call (``start``, ``initialize``,
``account_read``, ``thread_start``/``thread_resume``, ``turn_start``,
``turn_interrupt``, ``close``, the login calls) runs on via
``self._run_blocking``, the one helper this file uses instead of repeating
``loop.run_in_executor`` at each call site.

**Why turn-notification draining gets its OWN executor, not the one above.**
This is a correction to the ticket's original approach sketch, found by
reading ``client.py``'s ``_message_router.py`` internals rather than
assuming: ``CodexClient`` is explicitly designed for concurrent access from
multiple threads (each JSON-RPC request gets its own response queue, each
turn gets its own notification queue, and only a lock around writing to the
child's stdin is shared), specifically so that a caller waiting on
``next_turn_notification`` for a long-running turn does not block a
*different* caller wanting to send ``turn/interrupt`` at the same time.
Putting the notification-drain loop (which blocks for the whole duration of
a turn, including however long a pending approval takes) on the SAME
single-worker executor as ``turn_interrupt`` would queue every interrupt
request behind the very turn it is meant to cut short, and it would never
run until the turn ended on its own -- silently defeating ``interrupt()``.
``self._drain_executor`` is therefore a second, separate single-worker pool
used for nothing but draining one turn's notifications at a time.

**Where the approval callback actually runs.** ``client.py``'s single
reader thread (``_reader_loop``) reads one JSON-RPC message at a time; when
that message is a server request (an approval), it calls
``self._approval_handler`` synchronously and cannot read the next message
until that call returns. So the approval bridge below
(``_approval_handler``, ``asyncio.run_coroutine_threadsafe(broker.ask(...),
loop).result()``) blocks *that* thread, a third one distinct from both
``self._executor`` and ``self._drain_executor`` -- CodexClient's own
internal reader thread, spawned inside ``client.start()``. Blocking it is
correct and intentional (it is exactly ``can_use_tool``'s "the turn is
paused until answered" semantics, mirrored from ``session/sdk.py``), not a
bug to route around.

Everything else about this file's shape mirrors ``session/sdk.py``
deliberately: the same ``_set_status``/``AGENT_CONNECTING``/``AGENT_READY``/
``AGENT_UNAVAILABLE`` transitions, the same ``_emit`` helper, the same
viewer-presence counting feeding ``PermissionBroker``, and the same "a
session owns its turn and keeps producing events regardless of whether a
browser is attached" pump discipline -- except the pump here is a background
OS thread per turn (``_drain_turn``), because ``next_turn_notification`` is
blocking, not an async generator the event loop can iterate directly.
"""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from openai_codex.client import CodexClient, CodexConfig
from openai_codex.generated.v2_all import (
    AgentMessageDeltaNotification,
    ApiKeyLoginAccountParams,
    ApprovalsReviewer,
    AskForApproval,
    AskForApprovalValue,
    ChatgptLoginAccountParams,
    CommandExecutionStatus,
    CommandExecutionThreadItem,
    ErrorNotification,
    FileChangeThreadItem,
    ItemCompletedNotification,
    ItemStartedNotification,
    LoginAccountParams,
    McpToolCallThreadItem,
    PatchApplyStatus,
    SandboxMode,
    ThreadResumeParams,
    ThreadStartParams,
    TurnCompletedNotification,
)
from openai_codex.models import Notification

from .. import product
from . import logfiles, turn_images
from .base import (
    AGENT_CONNECTING,
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentModelChanged,
    AgentStatus,
    SandboxStatus,
    SessionReset,
    TextDelta,
    ToolResult,
    ToolUse,
    TurnEnd,
    UnknownRequest,
    turn_not_sent,
)
from .codex_mcp_stdio_bridge import AGENT_TOKEN_ENV
from .permissions import Decision, _toml_string

# The tool name a commandExecution approval is reported under. Deliberately
# the same string Claude's own Bash tool uses (session/permissions.py's
# NEVER_REMEMBERED), not a Codex-specific name: the
# broker's NEVER_REMEMBERED set (session/permissions.py) refuses to persist a
# standing "always allow" grant for unrestricted shell access keyed on this
# exact string, and that safety invariant must hold for every backend, not
# only the one it was written against.
_TOOL_COMMAND_EXECUTION = "Bash"

# The tool name a fileChange (patch-apply) approval is reported under. Not
# NEVER_REMEMBERED: an allow-always grant for file edits is the same
# standing trust a human can already give Claude's Edit/Write tools.
_TOOL_FILE_CHANGE = "FileChange"

_APPROVAL_METHOD_TOOL = {
    "item/commandExecution/requestApproval": _TOOL_COMMAND_EXECUTION,
    "item/fileChange/requestApproval": _TOOL_FILE_CHANGE,
}


class CodexSession:
    """An ``AgentSession`` driving a real ``openai_codex.client.CodexClient``.

    ``on_event`` is called with every ``AgentEvent`` this session produces,
    exactly as ``session/sdk.py``'s ``SdkSession`` documents; this class
    never touches a socket, an ``EventLog`` or a ``ViewerRegistry`` either.

    ``client_factory`` is this class's injection seam for tests, the Codex
    equivalent of ``SdkSession``'s ``transport=``. ``CodexClient`` has no
    public ``Transport`` abstraction of its own to substitute (it hardcodes
    a real subprocess and a real reader thread internally), so the seam here
    is one level up: a callable taking the same ``config``/``approval_handler``
    keywords ``CodexClient()`` does and returning anything with its public
    method surface. Defaults to ``CodexClient`` itself.
    """

    def __init__(
        self,
        on_event,
        *,
        cwd,
        session_id,
        broker=None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        resume: Optional[str] = None,
        on_sdk_session_id=None,
        client_factory: Optional[Callable[..., Any]] = None,
        mcp_host: Optional[str] = None,
        mcp_port: Optional[int] = None,
        mcp_token: Optional[str] = None,
        mcp_remotes: tuple = (),
        instructions: Optional[str] = None,
        turn: int = 0,
    ):
        self._on_event = on_event
        self.cwd = str(cwd)
        self.session_id = session_id
        self.sdk_session_id = None
        self._broker = broker
        self._model = model
        self._effort = effort
        self._resume = resume
        self._on_sdk_session_id = on_sdk_session_id
        self._client_factory = client_factory or CodexClient
        # The host's own /mcp endpoint (http/routes_mcp.py), for the
        # config_overrides this class registers its stdio-proxy subprocess
        # with in start(); see _mcp_config_overrides. ``mcp_token`` is the
        # run's agent token, the only one /mcp accepts, and never the browser
        # token that authorises /ws: it reaches the bridge through the
        # app-server's environment (_mcp_env), not a command line. All three
        # None (the default) means "no product tools attached to this
        # session" - every fake-transport test that does not care about the
        # tool-exposure bridge leaves them unset and gets an empty
        # config_overrides, deliberately, rather than a bridge pointed at
        # nothing.
        self._mcp_host = mcp_host
        self._mcp_port = mcp_port
        self._mcp_token = mcp_token
        # The remote MCP servers the product's tool server reached
        # (ToolServer.remotes), by name: each gets a bridge of its own,
        # pointed at /mcp/<name> (_mcp_config_overrides).
        self._mcp_remotes = tuple(mcp_remotes)
        # The product's session context (Product.session_context), given to
        # the thread as its developer instructions.
        self._instructions = instructions or None

        self._status = AGENT_CONNECTING
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread_id: Optional[str] = None
        self._active_turn_id: Optional[str] = None
        # The last turn of a resumed session's history (launch passes
        # bus.turn), so the next one continues its numbering.
        self._turn = int(turn)
        self._closing = False
        self._viewers_seen = 0
        # Set by end_turn_after_tool: the next MCP tool call Codex reports
        # complete interrupts the turn, which then ends as ended_by_tool.
        self._end_turn_pending = False
        self._stop_reason: Optional[str] = None
        # The client's own bounded stderr tail (see backend_logs), kept here
        # so it outlives a client that failed to start and was dropped.
        self._stderr_lines = None

        # See this module's docstring: one pool for one-shot control calls,
        # a second, separate one for the long-lived per-turn drain, so an
        # interrupt is never queued behind the very turn it means to cut
        # short.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codex-session")
        self._drain_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codex-drain")

    # -- AgentSession surface ------------------------------------------------

    def agent_status(self) -> str:
        return self._status

    def _set_status(self, status: str) -> None:
        """Record a status and announce the change. See ``SdkSession``'s
        identical method for why: announced only on a change, and after the
        field is set, so anything the callback does synchronously sees it."""
        if status == self._status:
            return
        self._status = status
        self._emit(AgentStatus(status=status))

    def sandbox_status(self) -> SandboxStatus:
        """The posture in effect, for the banner and ``doctor``.

        Unlike Claude's sandbox (``session/sdk.py``'s ``SANDBOX_SETTINGS``),
        which depends on ``bwrap``/``socat`` being present on the host,
        Codex's sandbox is built into the bundled ``codex`` binary itself
        (landlock/seccomp on Linux, Seatbelt on macOS) with no external
        dependency this process can check or that the app-server has been
        observed to report failing. ``thread_start`` always requests
        ``SandboxMode.workspace_write``, so this reports that request as
        both requested and active unconditionally, matching the "Codex uses
        its own Sandbox enum, neither needs bwrap/socat on the host" note in
        the product CLI's startup gate (Mesh's ``cli.py``).
        """
        return SandboxStatus(requested=True, active=True, missing=())

    def backend_logs(self) -> list:
        """The thread's rollout (the one being resumed until the app-server
        reports its own id) and the app-server's stderr. ``CodexClient`` keeps
        its last 400 stderr lines only on its private ``_stderr_lines`` deque,
        which ``start`` holds on to, read defensively since a test's fake client
        has none."""
        lines = self._stderr_lines
        return logfiles.codex_logs(
            self.sdk_session_id or self._resume,
            "\n".join(list(lines)) if lines is not None else None,
        )

    def on_viewer_presence(self, count: int) -> None:
        """Keep the broker's view of viewer count in step with the registry's.
        Identical in intent to ``SdkSession.on_viewer_presence``; see its
        docstring."""
        if self._broker is None:
            return
        while self._viewers_seen < count:
            self._broker.viewer_connected()
            self._viewers_seen += 1
        while self._viewers_seen > count:
            self._broker.viewer_disconnected()
            self._viewers_seen -= 1

    async def submit_turn(self, blocks: list, viewer: Optional[str] = None) -> None:
        """Start one Codex turn from ``blocks`` and let a background thread
        drain its notifications into events until ``turn/completed``.

        Unlike ``SdkSession.submit_turn``, which sends the turn and lets an
        already-running pump task consume the rest of the (single, whole-
        session) message stream, Codex's notification model is scoped per
        turn: a fresh subscription is registered for each ``turn/start``
        response and closed once that turn completes. This method therefore
        starts the turn, then hands the new turn id to ``_drain_turn`` on
        ``self._drain_executor`` and returns without waiting for it, which
        is what keeps events flowing regardless of whether a browser is
        attached (the same invariant ``SdkSession``'s pump keeps).
        """
        if self._client is None or self._status != AGENT_READY or self._thread_id is None:
            self._emit(
                AgentError(stderr="", remediation=turn_not_sent(self._status), viewer=viewer)
            )
            return
        self._turn += 1
        try:
            loop = asyncio.get_running_loop()
            expanded = await loop.run_in_executor(
                None, turn_images.expand_turn_blocks, blocks, self.cwd
            )
            items = _to_codex_input_items(expanded)
            # A per-turn override, not the thread-level default set at
            # thread_start/thread_resume time (_start_or_resume_thread):
            # TurnStartParams.model genuinely overrides "for this turn and
            # subsequent turns" within the same thread (confirmed via
            # openai_codex's generated/v2_all.py), so reading self._model
            # fresh here, on every call, is what makes a live set_model
            # actually take effect on the next turn sent -- the thread-start
            # value alone would only ever reflect whatever model was current
            # when the thread was created.
            params = {}
            if self._effort:
                params["effort"] = self._effort
            if self._model:
                params["model"] = self._model
            params = params or None
            started = await self._run_blocking(
                self._client.turn_start, self._thread_id, items, params
            )
        except Exception as exc:
            self._fail(exc, viewer=viewer)
            return
        turn_id = started.turn.id
        self._active_turn_id = turn_id
        future = self._loop.run_in_executor(self._drain_executor, self._drain_turn, turn_id, viewer)
        future.add_done_callback(self._on_drain_future_done)

    async def decide_permission(self, request_id: str, decision: str, message: str = "") -> None:
        """Route a human's decision to the broker. Identical to
        ``SdkSession.decide_permission``; see its docstring for why
        ``UnknownRequest`` propagates deliberately."""
        if self._broker is None:
            return
        try:
            await self._broker.decide(request_id, decision, message)
        except UnknownRequest:
            raise
        except Exception as exc:
            sys.stderr.write(
                "warning: permission decision %s was not applied: %r\n" % (request_id, exc)
            )

    async def interrupt(self) -> None:
        if self._client is None or self._thread_id is None or self._active_turn_id is None:
            return
        # Deny every approval still awaiting a decision before sending
        # turn/interrupt, not after: a native approval blocks CodexClient's
        # sole reader thread synchronously inside _approval_handler's
        # broker.ask() call (see that method's docstring below), and
        # turn/interrupt's own response can only ever be routed by that
        # same thread. Waiting on turn_interrupt first, as this method used
        # to, hangs until the human separately answers the pending
        # approval or the broker's own five-minute timeout expires.
        # Resolving the broker's pending future here unblocks the reader
        # thread instead, exactly as broker.shutdown() already does for
        # close() -- decide() rather than the private _deny_all_pending
        # close() uses, since this must not also stop future asks the way
        # shutdown() does; the session may still take another turn.
        if self._broker is not None:
            for request in list(self._broker.pending_requests()):
                try:
                    await self._broker.decide(request.request_id, "deny", "turn interrupted")
                except UnknownRequest:
                    # Already decided or timed out between the snapshot
                    # above and this call; nothing left to unblock.
                    pass
        try:
            await self._run_blocking(
                self._client.turn_interrupt, self._thread_id, self._active_turn_id
            )
        except Exception as exc:
            # Best-effort, like SdkSession's interrupt: the turn is either
            # already finished or the child is gone, and both surface
            # through the drain loop's own failure handling.
            sys.stderr.write("warning: interrupt failed: %r\n" % (exc,))

    def end_turn_after_tool(self) -> None:
        """A tool result asked to end the turn (``tools.ok``'s ``end_turn``).
        Called from the tool's handler behind ``/mcp``, before Codex has its
        result; the turn is interrupted once Codex reports that MCP tool call
        complete (``_handle_item_completed``)."""
        self._end_turn_pending = True

    async def set_model(self, model: str) -> None:
        """Store the new model; the next ``TurnStartParams`` construction in
        ``submit_turn`` reads ``self._model`` fresh and passes it as a
        per-turn override, which Codex documents as overriding "for this
        turn and subsequent turns" within the same thread -- no new thread
        or reconnect is needed."""
        if model == self._model:
            return
        self._model = model
        self._emit(AgentModelChanged(model=model))

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Connect the client, verify an account is configured, and start
        (or resume) a thread; never raise. See ``SdkSession.start``'s
        docstring for why: the HTTP server starts independently and must
        keep serving the viewer whatever the agent does.
        """
        self._loop = asyncio.get_running_loop()
        try:
            self._client = self._client_factory(
                config=CodexConfig(
                    cwd=self.cwd,
                    client_name=product.current().module,
                    client_title=product.current().display_name,
                    config_overrides=self._mcp_config_overrides(),
                    env=self._mcp_env(),
                ),
                approval_handler=self._approval_handler if self._broker is not None else None,
            )
            self._stderr_lines = getattr(self._client, "_stderr_lines", None)
            await self._run_blocking(self._client.start)
            await self._run_blocking(self._client.initialize)
            account = await self._run_blocking(self._client.account_read)
        except Exception as exc:
            # start() may already have spawned the app-server subprocess
            # and its reader/stderr threads, and registered the MCP proxy
            # (self._mcp_config_overrides()), by the time initialize() or
            # account_read() raises; dropping the only reference to
            # self._client without closing it first would leak all of that
            # for the life of this still-serving process, since close()
            # below can no longer reach it once self._client is None. Same
            # try/except/warn pattern close() itself uses.
            if self._client is not None:
                try:
                    await self._run_blocking(self._client.close)
                except Exception as close_exc:
                    sys.stderr.write(
                        "warning: agent client did not close cleanly: %r\n" % (close_exc,)
                    )
                self._client = None
            self._fail(exc)
            return
        if account.account is None:
            self._fail_not_authenticated()
            return
        try:
            thread_id = await self._start_or_resume_thread()
        except Exception as exc:
            self._fail(exc)
            return
        self._remember_thread_id(thread_id)
        self._set_status(AGENT_READY)

    async def close(self) -> None:
        self._closing = True
        if self._broker is not None:
            # Before the client goes, while there is still a socket to carry
            # the denial event and while the RPC it belongs to can still get
            # a result: see SdkSession.close's identical ordering.
            self._broker.shutdown()
        if self._client is not None:
            try:
                await self._run_blocking(self._client.close)
            except Exception as exc:
                sys.stderr.write("warning: agent client did not close cleanly: %r\n" % (exc,))
        self._client = None
        self._executor.shutdown(wait=True)
        self._drain_executor.shutdown(wait=True)
        self._set_status(AGENT_UNAVAILABLE)

    def _mcp_config_overrides(self) -> tuple:
        """``--config`` overrides registering the agent layer's own
        tool-exposure bridge (``planning/tickets/phase3_codex-tool-mcp-bridge.md``)
        as an MCP server, named after the product's ``mcp_server_name``, scoped
        to this one launched app-server process - never written to the human's
        real ``~/.codex/config.toml``. Each remote MCP server the product's
        tool server reached (``mcp_remotes``) gets one more, named after it
        and pointed at ``/mcp/<remote>`` (``--path``), so the model sees that
        remote's tools as a server of their own.

        Empty when this session was constructed with no ``/mcp`` endpoint to
        point at (``mcp_host``/``_port``/``_token`` all ``None``, the
        constructor's default): every fake-transport test that is not
        exercising the tool-exposure bridge leaves them unset and gets Codex
        launched with nothing extra to prove wrong, rather than a proxy
        pointed at a server that was never given to it.

        Each entry is one ``--config key=value`` CLI flag
        (``client.py``'s own launch-argument construction, confirmed by
        reading it directly rather than assumed -
        ``planning/20260919_codex-mcp-bridge-finding.md``); Codex parses
        ``value`` as a TOML literal, so ``command`` is a quoted TOML string
        and ``args`` is a TOML array-of-strings literal, both built through
        ``_toml_string`` (``session/permissions.py``'s own escaper for
        exactly this purpose elsewhere in this project, reused rather than
        duplicated: the escaping a value needs to be a safe TOML string is
        the same whichever file the string ends up written into). The
        launched proxy is always ``sys.executable -m`` the product's
        ``codex_bridge_module`` (the agent layer's own bridge) - the same
        interpreter and installed package running this process, guaranteed
        to have that module and its own dependencies (``mcp``, ``httpx``)
        importable regardless of whether the optional ``codex`` extra is
        installed, since both are already transitive dependencies of
        ``claude-agent-sdk``, a base dependency, and are now declared
        directly.

        The agent token is not among these: every entry here becomes part of
        the app-server's command line. ``env_vars`` (Codex's allowlist of
        variables copied into an MCP child's scrubbed environment) names the
        variable ``_mcp_env`` sets instead; checked against codex-cli 0.154.0,
        which forwards a variable so named and drops it when it is not named.

        The dotted ``key=value`` form below was later confirmed against a
        live ``codex app-server`` (0.154.0) for ``command``, ``args`` and
        ``env_vars``: a bare ``command="..."`` scalar assignment, and
        ``args=[...]``/``env_vars=[...]`` as TOML array-of-strings literals
        assigned the same dotted-path way.

        The last entry keeps the token out of the shells Codex runs for the
        model. Codex does not drop ``*TOKEN*`` variables by default (checked
        against 0.154.0: ``codex sandbox -- env`` shows a ``*_TOKEN``
        variable unless ``shell_environment_policy`` excludes it), and the
        app-server's environment is what those shells inherit. The exclude
        applies only to shells; the MCP bridge still receives the variable
        through ``env_vars`` (both checked against the real binary). A dotted
        override replaces the whole list, so the user's own top-level
        ``exclude`` is read back and kept (``_shell_excludes``).
        """
        if self._mcp_host is None or self._mcp_port is None or self._mcp_token is None:
            return ()
        installed = product.current()

        # No token here: this list becomes the app-server's own command line,
        # and so does anything in an ``env`` table, both readable through
        # ``ps``. The token travels in the app-server's environment instead
        # (``_mcp_env``), and ``env_vars`` names it as the one variable Codex
        # copies from there into the bridge's otherwise scrubbed environment.
        def bridge(key, server_name, path_args=()):
            proxy_args = [
                "-m",
                installed.codex_bridge_module,
                "--host",
                self._mcp_host,
                "--port",
                str(self._mcp_port),
                "--server-name",
                server_name,
                "--server-version",
                installed.version,
                *path_args,
            ]
            key = "mcp_servers.%s" % key
            return (
                "%s.command=%s" % (key, _toml_string(sys.executable)),
                "%s.args=[%s]" % (key, ", ".join(_toml_string(arg) for arg in proxy_args)),
                "%s.env_vars=[%s]" % (key, _toml_string(AGENT_TOKEN_ENV)),
            )

        overrides = bridge(installed.mcp_server_name, installed.distribution)
        for name in self._mcp_remotes:
            overrides += bridge(name, name, ("--path", "/mcp/%s" % name))
        return overrides + (
            "shell_environment_policy.exclude=[%s]"
            % ", ".join(_toml_string(name) for name in _shell_excludes()),
        )

    def _mcp_env(self) -> Optional[dict]:
        """The app-server's extra environment: the agent token, for
        ``_mcp_config_overrides``' ``env_vars`` to hand on to the bridge, or
        None when no bridge is registered. See
        ``session/codex_mcp_stdio_bridge.py`` for why the environment and not
        the command line."""
        if not self._mcp_config_overrides():
            return None
        return {AGENT_TOKEN_ENV: self._mcp_token}

    # -- OAuth: an explicit fallback, never the default path -----------------
    #
    # Neither method is wired to any UI in this ticket's scope (the tool/MCP
    # bridge and any future settings-window login button are later phases);
    # they exist so a caller has a real, working, non-stub entry point for
    # both paths, per the design constraint that login_api_key() must exist
    # but must not be the only one documented to a human. The startup
    # failure path (_fail_not_authenticated) points a human at the bundled
    # `codex login` binary directly instead, which is something a human at a
    # terminal can actually run; these two methods are for a future in-app
    # trigger that can poll `wait_for_login_completed` afterward.

    async def login_chatgpt(self):
        """Start browser-based ChatGPT OAuth login. Returns the app-server's
        own response, which carries the login id and URL; this class does
        not open a browser or block on completion itself."""
        if self._client is None:
            raise RuntimeError("the codex client has not been started")
        return await self._run_blocking(
            self._client.account_login_start,
            LoginAccountParams(root=ChatgptLoginAccountParams(type="chatgpt")),
        )

    async def login_api_key(self, api_key: str):
        """Authenticate with an API key: the pay-per-token fallback this
        backend documents as secondary, never the default."""
        if self._client is None:
            raise RuntimeError("the codex client has not been started")
        return await self._run_blocking(
            self._client.account_login_start,
            LoginAccountParams(root=ApiKeyLoginAccountParams(type="apiKey", api_key=api_key)),
        )

    async def wait_for_chatgpt_login(self, login_id: str):
        """Block (on the executor, not the event loop) until the login
        started by ``login_chatgpt`` completes."""
        if self._client is None:
            raise RuntimeError("the codex client has not been started")
        return await self._run_blocking(self._client.wait_for_login_completed, login_id)

    # -- the executor bridge --------------------------------------------------

    async def _run_blocking(self, func, *args, **kwargs):
        """Run one blocking ``CodexClient`` call on ``self._executor``.

        The one helper every one-shot call in this file uses instead of
        repeating ``loop.run_in_executor`` at each site: ``start``,
        ``initialize``, ``account_read``, ``thread_start``/``thread_resume``,
        ``turn_start``, ``turn_interrupt``, ``close`` and the login calls.
        Turn-notification draining deliberately does NOT use this helper;
        see the module docstring for why it has its own executor instead.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

    # -- thread lifecycle ------------------------------------------------------

    async def _start_or_resume_thread(self) -> str:
        """``thread_start``, or ``thread_resume`` when a prior Codex thread
        id was given. A resume that fails falls back to starting fresh and
        announces ``SessionReset``, mirroring ``SdkSession``'s handling of a
        resume the CLI could not honour.

        ``approvals_reviewer=ApprovalsReviewer.user`` and the explicit
        ``AskForApproval(root=AskForApprovalValue.on_request)`` policy are
        constructed directly rather than through the curated ``ApprovalMode``
        enum, which maps its ``auto_review`` member to an AI reviewer
        (``ApprovalsReviewer.auto_review``), not the human path this session
        exists to provide. See this module's docstring and
        ``planning/20260919_codex-approval-handler-finding.md``.
        """
        approval_policy = AskForApproval(root=AskForApprovalValue.on_request)
        if self._resume:
            try:
                resumed = await self._run_blocking(
                    self._client.thread_resume,
                    self._resume,
                    ThreadResumeParams(
                        thread_id=self._resume,
                        cwd=self.cwd,
                        model=self._model,
                        approval_policy=approval_policy,
                        approvals_reviewer=ApprovalsReviewer.user,
                        sandbox=SandboxMode.workspace_write,
                        developer_instructions=self._instructions,
                    ),
                )
                return resumed.thread.id
            except Exception as exc:
                self._emit(
                    SessionReset(
                        reason="asked to resume %s and it could not be resumed (%s: %s); "
                        "this is a new conversation" % (self._resume, type(exc).__name__, exc)
                    )
                )
        started = await self._run_blocking(
            self._client.thread_start,
            ThreadStartParams(
                cwd=self.cwd,
                model=self._model,
                approval_policy=approval_policy,
                approvals_reviewer=ApprovalsReviewer.user,
                sandbox=SandboxMode.workspace_write,
                developer_instructions=self._instructions,
            ),
        )
        return started.thread.id

    def _remember_thread_id(self, thread_id: str) -> None:
        """Record the Codex thread id, which is never fabricated: it stays
        ``None`` until the app-server reports one. Identical in intent to
        ``SdkSession._remember_sdk_session``."""
        if thread_id and thread_id != self.sdk_session_id:
            self.sdk_session_id = thread_id
            self._thread_id = thread_id
            if self._on_sdk_session_id is not None:
                try:
                    self._on_sdk_session_id(thread_id)
                except Exception as exc:
                    sys.stderr.write("warning: could not record the codex thread id: %r\n" % (exc,))
        else:
            self._thread_id = thread_id

    # -- the approval bridge ---------------------------------------------------

    def _approval_handler(self, method: str, params) -> dict:
        """The ``ApprovalHandler`` passed to ``CodexClient``.

        Runs on ``CodexClient``'s own internal reader thread
        (``client.py``'s ``_reader_loop`` -> ``_handle_server_request``), a
        third thread distinct from both ``self._executor`` and
        ``self._drain_executor``. See this module's docstring for why
        blocking it is correct and intentional.

        ``params`` has no typed model anywhere in the installed SDK (checked
        directly: neither ``client.py`` nor ``generated/v2_all.py`` declares
        one for either ``requestApproval`` method), so it is forwarded to the
        broker verbatim as the request's ``input_data`` -- the same
        provider-neutral contract ``PermissionRequest.input`` already has for
        the Claude backend's raw tool-call arguments.
        """
        tool_name = _APPROVAL_METHOD_TOOL.get(method)
        if tool_name is None or self._broker is None or self._loop is None:
            return {"decision": "accept"}
        input_data = params if isinstance(params, dict) else {}
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._broker.ask(tool_name, input_data, None), self._loop
            )
            decision = future.result()
        except Exception as exc:
            sys.stderr.write(
                "warning: could not reach the permission broker for %s: %r\n" % (method, exc)
            )
            return {"decision": "decline"}
        return _to_codex_decision(decision)

    # -- turn-notification draining --------------------------------------------

    def _drain_turn(self, turn_id: str, viewer: Optional[str]) -> None:
        """Consume ``turn_id``'s notifications until ``turn/completed``,
        marshalling each into an event via ``loop.call_soon_threadsafe``.

        Runs on ``self._drain_executor``, never on ``self._executor`` (see
        module docstring). Hand-rolled off ``CodexClient``'s own
        ``register_turn_notifications``/``next_turn_notification``/
        ``unregister_turn_notifications`` -- the same low-level primitives
        ``wait_for_turn_completed`` and ``stream_text`` use internally --
        because the convenience ``TurnHandle.stream()`` wrapper is reachable
        only through the public ``Codex``/``AsyncCodex`` classes this design
        avoids. Never raises: an exception here is reported through
        ``_drain_failed`` on the event loop instead.
        """
        try:
            self._client.register_turn_notifications(turn_id)
            while True:
                notification = self._client.next_turn_notification(turn_id)
                self._loop.call_soon_threadsafe(self._handle_notification, notification, viewer)
                if notification.method == "turn/completed":
                    return
        except Exception as exc:
            self._loop.call_soon_threadsafe(self._drain_failed, turn_id, exc, viewer)
        finally:
            try:
                self._client.unregister_turn_notifications(turn_id)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._clear_active_turn, turn_id)

    def _on_drain_future_done(self, future) -> None:
        """Safety net: ``_drain_turn`` already catches everything it can, so
        this should never fire, but a fire-and-forgotten
        ``run_in_executor`` future that raises is otherwise only ever
        reported to a log asyncio itself decides to write to."""
        exc = future.exception()
        if exc is not None:
            sys.stderr.write("warning: turn notification drain crashed: %r\n" % (exc,))

    def _drain_failed(self, turn_id: str, exc: BaseException, viewer: Optional[str]) -> None:
        if self._closing:
            return
        self._fail(exc, viewer=viewer)

    def _clear_active_turn(self, turn_id: str) -> None:
        if self._active_turn_id == turn_id:
            self._active_turn_id = None

    def _handle_notification(self, notification: Notification, viewer: Optional[str]) -> None:
        """Turn one ``Notification`` into an ``AgentEvent``, or ignore it.

        Only the kinds this file maps to something the chat pane renders are
        handled; every other notification kind (reasoning deltas, plan
        updates, token-usage updates, ...) has no agent event to become and is
        silently ignored, the same tolerance ``SdkSession._handle`` shows for
        SDK message kinds it does not render.
        """
        method = notification.method
        payload = notification.payload
        if method == "item/agentMessage/delta" and isinstance(
            payload, AgentMessageDeltaNotification
        ):
            if payload.delta:
                self._emit(TextDelta(turn=self._turn, text=payload.delta, viewer=viewer))
            return
        if method == "item/started" and isinstance(payload, ItemStartedNotification):
            self._handle_item_started(payload, viewer)
            return
        if method == "item/completed" and isinstance(payload, ItemCompletedNotification):
            self._handle_item_completed(payload, viewer)
            return
        if method == "turn/completed" and isinstance(payload, TurnCompletedNotification):
            self._handle_turn_completed(payload, viewer)
            return
        if method == "error" and isinstance(payload, ErrorNotification):
            self._emit(
                AgentError(
                    stderr=payload.error.message,
                    remediation="the Codex turn reported an error"
                    + (" and will retry" if payload.will_retry else ""),
                    viewer=viewer,
                )
            )
            return

    def _handle_item_started(self, payload: ItemStartedNotification, viewer: Optional[str]) -> None:
        item = payload.item.root
        if isinstance(item, CommandExecutionThreadItem):
            self._emit(
                ToolUse(
                    turn=self._turn,
                    tool_use_id=item.id,
                    name=_TOOL_COMMAND_EXECUTION,
                    # item.cwd is a LegacyAppPathString (a RootModel wrapping
                    # a plain str, generated/v2_all.py:2165), not a str
                    # itself; unwrapped the same way client.py's own
                    # `_run.py` unwraps a ThreadItem's root, or this dict
                    # embeds a pydantic object where AgentEvent.to_wire()
                    # needs a JSON-able value.
                    input={"command": item.command, "cwd": _unwrap(item.cwd)},
                    viewer=viewer,
                )
            )
        elif isinstance(item, FileChangeThreadItem):
            self._emit(
                ToolUse(
                    turn=self._turn,
                    tool_use_id=item.id,
                    name=_TOOL_FILE_CHANGE,
                    input={"paths": [change.path for change in item.changes]},
                    viewer=viewer,
                )
            )
        # An agentMessage item's text arrives token by token through
        # item/agentMessage/delta above; emitting it again here would
        # double every reply in the pane, the same reasoning
        # SdkSession._handle applies to AssistantMessage's TextBlock.
        # Every other item kind (reasoning, plan, mcpToolCall, ...) has no
        # agent event to become and is left unhandled.

    def _handle_item_completed(
        self, payload: ItemCompletedNotification, viewer: Optional[str]
    ) -> None:
        item = payload.item.root
        if isinstance(item, McpToolCallThreadItem) and self._end_turn_pending:
            self._end_turn_pending = False
            self._stop_reason = "ended_by_tool"
            asyncio.ensure_future(self.interrupt())
            return
        if isinstance(item, CommandExecutionThreadItem):
            self._emit(
                ToolResult(
                    tool_use_id=item.id,
                    is_error=item.status == CommandExecutionStatus.failed,
                    text=item.aggregated_output or "",
                    viewer=viewer,
                )
            )
        elif isinstance(item, FileChangeThreadItem):
            self._emit(
                ToolResult(
                    tool_use_id=item.id,
                    is_error=item.status == PatchApplyStatus.failed,
                    text=", ".join(change.path for change in item.changes),
                    viewer=viewer,
                )
            )

    def _handle_turn_completed(
        self, payload: TurnCompletedNotification, viewer: Optional[str]
    ) -> None:
        turn = payload.turn
        # Codex has no per-turn cost figure comparable to the Claude API's
        # total_cost_usd (subscription billing does not meter a turn this
        # way); 0.0 is an honest "not applicable" rather than a guess.
        stop_reason, self._stop_reason = self._stop_reason or turn.status.value, None
        self._emit(TurnEnd(turn=self._turn, stop_reason=stop_reason, cost_usd=0.0, viewer=viewer))

    # -- failure ---------------------------------------------------------------

    def _fail(self, exc: BaseException, viewer: Optional[str] = None) -> None:
        """Report a failure as an event and mark the session unavailable.
        Never raises. Identical in intent to ``SdkSession._fail``."""
        self._set_status(AGENT_UNAVAILABLE)
        self._emit(
            AgentError(
                stderr="%s: %s" % (type(exc).__name__, exc),
                remediation=_remediation_for(exc),
                viewer=viewer,
            )
        )

    def _fail_not_authenticated(self) -> None:
        """No Codex account is configured. Unlike every other failure in
        this file, this is not an exception -- ``account_read`` answered
        successfully with ``account: null`` -- so it gets its own
        remediation naming the bundled binary directly, the one thing a
        human at a terminal can actually run (verified against the
        installed 0.154.0 binary's own ``--help`` output, not guessed:
        ``codex login`` opens a browser for the ChatGPT-subscription path,
        ``codex login --with-api-key`` reads a key from stdin for the
        pay-per-token fallback). ``login_chatgpt``/``login_api_key`` above
        exist for a future in-app trigger; a human reading this message has
        no such button yet, so it points at the CLI instead.
        """
        self._set_status(AGENT_UNAVAILABLE)
        self._emit(
            AgentError(
                stderr="account/read reported no configured Codex account",
                remediation="the Codex account is not configured: run `codex login` "
                "(opens a browser for your ChatGPT subscription) or, to pay per "
                "token instead, `printenv OPENAI_API_KEY | codex login "
                "--with-api-key`%s" % _bundled_codex_binary_hint(),
            )
        )

    def _emit(self, event) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            sys.stderr.write("warning: could not deliver %s: %r\n" % (type(event).__name__, exc))


# ---------------------------------------------------------------------------
# Decision application: session/permissions.py's provider-neutral Decision to
# the app-server's own accept/acceptForSession/decline vocabulary.
# ---------------------------------------------------------------------------


def _to_codex_decision(decision: Decision) -> dict:
    """The ``{"decision": ...}`` shape ``item/commandExecution/requestApproval``
    and ``item/fileChange/requestApproval`` both require, confirmed from the
    app-server protocol docs (the ticket's design constraints):
    ``Decision(allow=True, remember_tool=None)`` -> one-time ``"accept"``;
    ``Decision(allow=True, remember_tool=<tool>)`` -> ``"acceptForSession"``,
    scoped to this app-server process only; any deny -> ``"decline"``.
    ``acceptForSession`` is a convenience layered on top of the broker's own
    ``<state dir>/permissions.toml`` grant, not a replacement for it: that
    file is what survives a restart.

    Never raises, mirroring ``session/sdk.py``'s ``_to_claude_result``: a
    returned dict or a raised exception here would both read to the model as
    the permission system erroring rather than a decision (fact 14).
    """
    if not decision.allow:
        return {"decision": "decline"}
    if decision.remember_tool is not None:
        return {"decision": "acceptForSession"}
    return {"decision": "accept"}


def _unwrap(value):
    """A generated ``RootModel``'s plain value, or ``value`` itself.

    Several typed fields this file reads (``CommandExecutionThreadItem.cwd``
    is ``LegacyAppPathString``, a ``RootModel[str]``) wrap a JSON scalar in a
    pydantic object rather than exposing it directly; an event carrying one
    of those objects verbatim would break ``AgentEvent.to_wire()``'s JSON
    encoding once it reached the WebSocket. The same unwrap
    ``openai_codex._run.py`` itself uses for a ``ThreadItem``'s root
    (``item.root if hasattr(item, "root") else item``).
    """
    return value.root if hasattr(value, "root") else value


# ---------------------------------------------------------------------------
# Content-block translation: the agent layer's Anthropic-shaped turn blocks (what
# turn_images.expand_turn_blocks produces, the same shape SdkSession sends
# untouched) to Codex's own UserInput wire items.
# ---------------------------------------------------------------------------


def _to_codex_input_items(blocks: list) -> list:
    """``blocks`` (already expanded by ``turn_images.expand_turn_blocks``,
    Anthropic-shaped: ``{"type": "text", ...}`` and ``{"type": "image",
    "source": {"type": "base64", ...}}``) as Codex ``UserInput`` wire items.

    Codex's schema (``generated/v2_all.py``'s ``UserInput`` union) has no
    inline-base64 image variant; ``ImageUserInput.url`` is a plain string,
    so an expanded attachment's already-decoded bytes are carried across as
    a ``data:`` URI, the same encoding a base64 image already is, wrapped in
    the media-type-prefixed form most multipart-input APIs accept.
    """
    items = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            if text:
                items.append({"type": "text", "text": text})
        elif block_type == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64" and source.get("data"):
                media_type = source.get("media_type") or "application/octet-stream"
                items.append(
                    {"type": "image", "url": "data:%s;base64,%s" % (media_type, source["data"])}
                )
    if not items:
        items.append({"type": "text", "text": "(this message arrived empty)"})
    return items


def _shell_excludes() -> list:
    """The ``shell_environment_policy.exclude`` list this app-server runs
    with: the user's own top-level list from ``$CODEX_HOME/config.toml``
    (``~/.codex`` by default), kept because a dotted ``--config`` override
    replaces the list rather than extending it, plus the agent token's
    variable.

    A file that is missing, unreadable, not TOML, or holds something other
    than a list of strings contributes nothing: Codex itself reports a broken
    config, and the token's own exclusion must not depend on the user's file
    parsing. A per-profile or project-level list is not read; it would be
    replaced for this run the same way.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: tomllib is stdlib only from 3.11.
        import tomli as tomllib

    home = os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    try:
        with open(os.path.join(home, "config.toml"), "rb") as fh:
            config = tomllib.load(fh)
        existing = config.get("shell_environment_policy", {}).get("exclude", [])
    except (OSError, ValueError, AttributeError):
        existing = []
    if not isinstance(existing, list) or not all(isinstance(n, str) for n in existing):
        existing = []
    return existing + ([AGENT_TOKEN_ENV] if AGENT_TOKEN_ENV not in existing else [])


# ---------------------------------------------------------------------------
# Remediation text, matched by exception class name for the same reason
# session/sdk.py's _remediation_for is: a renamed or added SDK error
# degrades to the generic message instead of an ImportError at startup.
# ---------------------------------------------------------------------------


def _remediation_for(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "FileNotFoundError":
        return (
            "the bundled codex runtime could not be located; reinstall with the "
            "codex extra (`uv sync --extra codex`, or `%s`), or set "
            "CodexConfig.codex_bin explicitly" % product.current().codex_install
        )
    if name == "TransportClosedError":
        doctor = product.current().doctor_hint
        return "the codex app-server exited or closed its connection; %s" % (
            ("run %s, and " % doctor if doctor else "")
            + "check that it is authenticated (run `codex login`)"
        )
    if name in ("InvalidRequestError", "InvalidParamsError", "MethodNotFoundError"):
        return (
            "the codex app-server rejected a request this build sent, which "
            "usually means a version mismatch; check the pinned SDK range"
        )
    return "the agent is unavailable; its captured output says why"


def _bundled_codex_binary_hint() -> str:
    """ " (the bundled binary <product> uses is at <path>)", or "" if it cannot be
    located -- the same defensive lookup diagnostics.py's _codex_cli_info
    makes, reused here so the remediation message names the exact binary
    this process would itself run rather than leaving a human to guess
    whether a `codex` on PATH is the same one."""
    try:
        from codex_cli_bin import bundled_codex_path

        return " (the bundled binary %s uses is at %s)" % (
            product.current().name,
            bundled_codex_path(),
        )
    except Exception:
        return ""
