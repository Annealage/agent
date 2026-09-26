"""The real agent session for the omp (Oh My Pi) backend: an
``omp_rpc.RpcClient`` behind the ``AgentSession`` seam.

**Where ``omp_rpc`` actually comes from.** It is not on PyPI under
``omp-rpc``/``omp_rpc`` (`pip install omp-rpc` and `uv pip install omp-rpc`
both fail with "not found in the package registry"; already checked and
documented in `planning/tickets/phase4_omp-session.md`'s revalidated stamp).
The package's real, MIT-licensed source lives in the upstream ``oh-my-pi``
monorepo at ``python/omp-rpc`` (github.com/can1357/oh-my-pi), which
``omp://rpc.md`` itself names as "the bundled `omp-rpc` distribution", and
installs cleanly as an ordinary wheel build via
``pip install "omp-rpc @ git+https://github.com/can1357/oh-my-pi.git@<sha>
#subdirectory=python/omp-rpc"`` (verified against commit
``71c5eec978b0e7ce9ff057eb4e311f67f4f03eb9``). This is deliberately *not*
declared as a `pyproject.toml` optional extra the way `codex` is: hatchling
refuses to build metadata for a direct (VCS) reference at all unless
``tool.hatch.metadata.allow-direct-references`` is set, and even with that
opt-in, PyPI's own upload validation rejects a package whose metadata
carries a direct URL dependency -- declaring one here would break this
project's real publish workflow (`.github/workflows/publish.yml`), not
merely be inconvenient. Until `omp-rpc` publishes real PyPI releases (at
which point this becomes a normal `"omp-rpc>=X,<Y"` extra, mirroring
`codex`'s), a `backend = "omp"` deployment installs it with the command
above as a manual, documented prerequisite. This module imports the real
package rather than re-implementing the wire protocol, because a maintained,
MIT-licensed client that already handles v2 chunk reassembly, message
pagination and host-tool/host-URI dispatch exists and is safe to embed; the
protocol doc (`omp://rpc.md`) remains the wire contract of record, not a
fallback this module has to also speak by hand.

**Concurrency model -- resolved by reading `omp_rpc/client.py` directly, not
assumed.** ``RpcClient`` is not asyncio-native. It is a synchronous,
thread-based client with the same shape ``openai_codex.client.CodexClient``
has (`session/codex.py`'s own module docstring): ``start()`` spawns a real
subprocess and a dedicated ``stdout`` reader thread (plus a separate
``stderr`` thread), and every registered event listener
(``on_message_update``, ``on_tool_execution_start``, ``on_ui_request``, ...)
is invoked *synchronously on that reader thread* as frames arrive. Every
``RpcClient`` method that sends a command and waits for its response
(``start``, ``stop``, ``prompt``, ``abort``, ``get_state``, ...) blocks the
calling thread. This session therefore owns one small ``ThreadPoolExecutor``
(``self._executor``) that every such one-shot blocking call runs on via
``self._run_blocking``, exactly mirroring `session/codex.py`'s
``_run_blocking``/``self._executor`` -- except this file needs no second
"drain" executor the way Codex's does. Codex's notification stream is scoped
per turn (a fresh subscription registered for each ``turn_start``, drained by
a dedicated background thread `session/codex.py` spawns itself); RPC's event
stream is one persistent subscription for the life of the process, delivered
by ``RpcClient``'s own reader thread with no polling required from this file
at all, which is structurally closer to `session/sdk.py`'s single long-lived
pump than to Codex's per-turn drain. Host-tool calls get an even better
guarantee the wire protocol doc does not mention: ``RpcClient`` spawns a
*fresh daemon thread per host-tool call* (`client.py`'s
``_handle_host_tool_call``), so blocking one tool's ``execute`` callback on a
human decision never blocks the shared reader thread or any other concurrent
tool call the way it would if execution ran inline.

**Permission design -- two independent, broker-backed gates, not one.** The
ticket's approach sketch flags ``extension_ui_request{method:"confirm"}`` as
the write-class approval surface; reading the actual bundled
``@oh-my-pi/pi-coding-agent`` TypeScript source (this workstation's
installed `omp` build) shows the live per-tool approval gate
(``ExtensionToolWrapper``, wrapped around every registered tool, including
RPC host tools) actually resolves to a two-option ``extension_ui_request
{method:"select", options:["Approve","Deny"]}``, not ``confirm`` -- a detail
the wire-protocol doc does not surface and that could plausibly change
between builds either way. Relying on that exact shape (or reverse-engineering
a tool-name correlation out of free-text approval prompts) would make this
class only as correct as one build's private implementation. Instead:

1. This session launches `omp` with ``--auto-approve``, which neutralises
   `omp`'s own native approval gate entirely (confirmed via `omp --help`:
   "Auto-approve all tool calls (skip approval prompts)"), with
   ``--no-tools`` (``tools=()``), so the model's only capabilities are the
   host tools this session registers from `tool_table()`, and with
   ``--no-extensions`` (confirmed via `omp --help`: "Disable extension
   discovery (explicit -e paths still work)"). This last flag matters on
   its own: `omp` 18.1.22 unconditionally discovers ``.omp``/``.pi``
   extension files from ``cwd`` and loads them as trusted code with no
   trust prompt; without ``--no-extensions``, a project-local extension
   combined with ``--auto-approve`` could register its own native tools
   and execute at session start entirely outside this file's own
   ``host_tool_call`` broker gate -- the "real gate" claim below would be
   false the moment a session's ``cwd`` contained one. With extension
   discovery disabled, nothing but the host tools registered below can
   ever run. The *real* gate is therefore inside this file's own
   ``host_tool_call`` adapter: a write-class tool's ``execute`` callback
   calls ``broker.ask(...)`` itself, exactly the role
   `session/codex.py`'s ``_approval_handler`` and `session/sdk.py`'s
   ``_can_use_tool`` play for their own backends, before ever running the
   real handler. This is unaffected by whichever UI-request shape a given
   `omp` build happens to use for its own native gate, because that gate
   never fires.
2. `extension_ui_request{method:"confirm"}` is still handled, exactly as
   the ticket specifies, as a defensive second layer: `omp` may still
   raise a confirm for something unrelated to tool execution (a login
   flow, a provider-tier notice), and a host must answer every request it
   receives or risk stalling the run. When one arrives, this session
   looks up which tool call is currently in flight (tracked from
   ``tool_execution_start``/``tool_execution_end``, in order on the same
   reader thread that delivers the confirm, so there is no race between
   "note which tool started" and "a confirm arrives for it") and asks the
   broker on that tool's behalf, *before* answering -- but only when
   exactly one tool call is in flight. `omp`'s default parallel-tool-
   execution mode runs sibling tool calls from the same assistant turn
   concurrently (``RpcClient`` spawns a fresh daemon thread per host-tool
   call precisely because more than one may be running at once -- see
   this module's concurrency-model note above), so a ``confirm`` frame
   carries no tool-call id on the wire and cannot be safely attributed to
   "whichever tool call is currently in flight" once two or more are
   pending: the wrong tool's ``execute`` callback could silently receive
   another tool's allow/deny decision. When zero or more than one tool
   call is in flight, this session declines the confirm outright rather
   than guess. ``PermissionBroker.ask()`` already checks its own
   granted-tools set and shuts the door on a no-viewer session before it
   ever creates a request or emits anything a human would see (see
   `permissions.py`'s ``ask()`` docstring) -- this session never
   duplicates that check itself, it just always calls into the one
   function that owns it, for both gates. A confirm this session cannot
   attribute to exactly one in-flight tool is declined outright, never
   approved blind.

**Custom-provider injection -- resolved by reading `omp`'s own settings and
provider docs, not assumed.** ``RpcClient`` has no ``base_url``/``api_key``
constructor knobs of its own; a custom OpenAI-compatible provider is
`omp`-level config (`omp://providers.md`'s "Custom providers in
`models.yml`"), loaded only from ``<agent dir>/models.yml``/``.yaml``
(`omp://models.md`'s "Config file location") -- there is no `--config`
overlay path or per-project file for it. ``PI_CODING_AGENT_DIR`` relocates
that entire agent directory (auth store, cache, and the model config
alongside it), so this session gives every launch its own throwaway temp
directory via that env var, writes a ``models.yml`` there, and tears the
directory down in ``close()``. This keeps the human's own `omp` install
(auth, other providers, saved sessions) completely untouched by a
product-launched local-backend run. The conversation itself is never kept
there: it lives in ``session_dir`` (``--session-dir``), below. The file is
written as ``json.dumps(...)`` rather than through a YAML library: JSON is
valid YAML, `omp`'s config loader accepts a `.yml` path with JSON content
without complaint, and this avoids adding a YAML dependency for a one-off
machine-generated file no human ever hand-edits.

**A profile of its own.** `omp` reads two directories of the user's:
its config root (``~/.omp``: ``agent/APPEND_SYSTEM.md``, ``plugins/``,
``.env`` and the rest) and, inside it by default, its agent directory (auth,
settings, ``models.yml``, sessions). ``agent_dir`` points
``PI_CODING_AGENT_DIR`` at a directory the product names (a service's own
auth, settings and ``models.yml``); it and ``omp_base_url`` are exclusive,
since both would own that variable (a profile names its custom endpoint in
its own ``models.yml``). ``agent_dir`` alone still leaves the user's config
root in effect; ``config_dir`` replaces that too, through ``PI_CONFIG_DIR``,
which `omp` joins to ``$HOME``, so it must lie under ``$HOME`` and is passed
relative to it. A project's own context files (``AGENTS.md`` and the like in
the served directory and above it) are read whatever these say. ``binary``
is an absolute path to the `omp` executable, for a service whose ``PATH``
does not have it; never relative, which would resolve inside the served
directory.

**Steering and resuming.** Every human message is sent as ``prompt`` with
``streamingBehavior: "steer"``: an idle `omp` starts a turn with it, a busy
one queues it to redirect the running turn (``omp://rpc.md``, "While
streaming"; a prompt with no streaming behaviour is refused while a turn
runs). Sending the steer form every time rather than choosing from this
session's own idea of whether a turn is running means a race between the
two can never turn a message into a refusal. The pane shows a steer as the
next turn: the running turn ends as ``steered`` and the agent's further
output carries the new turn number. A message `omp` refuses is reported in
the chat and leaves the session ready; only a dead `omp` process marks it
unavailable. With ``session_dir`` given, `omp` keeps its conversation there
(``--session-dir``, in the workspace's state directory, never in the
throwaway agent directory), the conversation file it reports
(``get_state``'s ``sessionFile``) is recorded through ``on_session_file``,
and ``resume`` names such a file to switch to at start (``switch_session``).
Tokens and cost per turn are the difference between two
``get_session_stats`` readings, one at each turn's end.

``omp_api_key``, when set, is never written into ``models.yml`` as a
literal string: `omp`'s own ``apiKey`` resolution (`omp://providers.md`)
treats that field as an environment-variable name first and a leading
``!`` as a shell command to run, so a literal secret that happens to
collide with a real env var name would be silently replaced by that
variable's value, and one that happens to start with ``!`` would be
executed. This session instead generates a per-session environment
variable name (``_api_key_env_name``), sets it on the launched
subprocess's own environment (never the human's), and writes that name
into ``apiKey`` -- using `omp`'s primary resolution path deliberately,
not fighting it.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from omp_rpc import RpcClient, host_tool

from .. import product
from ..tools import host_tool_name
from . import turn_images
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
)


def _provider_id() -> str:
    """The provider id this session's generated `models.yml` registers its
    custom endpoint under, ``<product>-local`` (Mesh: ``mesh-local``).
    Arbitrary and internal: nothing outside this file ever needs to know it,
    since the `--model` flag this session builds always carries the full
    "provider/modelId" reference."""
    return "%s-local" % product.current().name


# `settings.py`'s `model` key is nullable ("unset falls back to the backend's
# own default"); a local/arbitrary endpoint has no real "default model" the
# way a hosted provider does, so this is a permissive placeholder rather
# than a guess at a real model id. Many single-model local servers
# (llama.cpp serving one loaded model, most `vllm` deployments) ignore the
# `model` field on an OpenAI-compatible request entirely; a multi-model
# server (Ollama) requires a real match, so a human running one of those
# must set `model` explicitly -- this only makes that an explicit setting
# rather than a required-but-undocumented one.
_DEFAULT_MODEL_ID = "default"

# Startup and per-request timeouts long enough for a slow local model on
# modest hardware to answer a "start"/"prompt" round trip. `RpcClient`'s own
# defaults (30s startup, 30s request) are already generous enough for
# ordinary use, so kept as-is rather than overridden here; recorded as a
# comment because the next reader of `start()` should not have to check
# `omp_rpc.client` to learn there is no override happening.


class OmpSession:
    """An ``AgentSession`` driving a real ``omp_rpc.RpcClient``.

    ``on_event`` is called with every ``AgentEvent`` this session produces,
    exactly as ``session/sdk.py`` and ``session/codex.py`` document; this
    class never touches a socket, an ``EventLog`` or a ``ViewerRegistry``
    either.

    ``client_factory`` is this class's injection seam for tests, the same
    role ``CodexSession.client_factory`` plays: a callable taking the same
    keywords ``RpcClient()`` does and returning anything with its public
    method surface. Defaults to ``RpcClient`` itself.

    ``tool_table`` is the product tool server's ``ToolServer.host_tool_table()``
    snapshot (``{name: ToolSpec(schema, description, handler, write)}``: the
    product's tools by their own names, each remote MCP server's as
    ``<remote>__<tool>``), taken once at construction the same way
    ``SdkSession`` is handed ``bus.tools.mcp_servers`` once: every tool this
    session ever exposes to `omp` comes from this snapshot, registered as
    `omp` host tools rather than through a second transport (unlike Codex,
    which needs its own stdio-to-HTTP MCP bridge -- see this file's module
    docstring). A write-grade call asks the broker under that same name.
    ``set_tool_table`` replaces it mid-session (a remote reached late).

    ``agent_dir``, ``config_dir``, ``binary``, ``session_dir``, ``resume`` and
    ``on_session_file`` are this module's docstring's profile (agent and
    config directories), executable,
    conversation directory, conversation file to resume, and the callback
    that records the conversation file `omp` reports. Without
    ``session_dir`` the conversation is not kept (``--no-session``).
    """

    #: A message sent while a turn runs redirects it (``hello``'s ``steers``).
    steers = True

    def __init__(
        self,
        on_event,
        *,
        cwd,
        session_id,
        broker=None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tool_table: Optional[dict] = None,
        on_sdk_session_id=None,
        client_factory: Optional[Callable[..., Any]] = None,
        instructions: Optional[str] = None,
        agent_dir=None,
        config_dir=None,
        binary: Optional[str] = None,
        session_dir=None,
        resume: Optional[str] = None,
        on_session_file=None,
        turn: int = 0,
    ):
        self._on_event = on_event
        self.cwd = str(cwd)
        self.session_id = session_id
        self.sdk_session_id = None
        self._broker = broker
        self._model = model
        # The product's session context (Product.session_context), passed as
        # --append-system-prompt; only when set, so the client is built with
        # exactly today's keywords otherwise.
        self._instructions = instructions or None
        self._base_url = base_url
        self._api_key = api_key
        self._tool_table = dict(tool_table or {})
        self._on_sdk_session_id = on_sdk_session_id
        self._client_factory = client_factory or RpcClient
        self._profile_dir = (
            os.path.abspath(os.path.expanduser(str(agent_dir))) if agent_dir else None
        )
        self._config_dir = os.path.expanduser(str(config_dir)) if config_dir else None
        # PI_CONFIG_DIR's value, relative to $HOME; set by _configuration_refusal.
        self._config_rel: Optional[str] = None
        self._binary = os.path.expanduser(binary) if binary else None
        self._session_dir = Path(session_dir) if session_dir is not None else None
        self._resume = resume or None
        self._on_session_file = on_session_file
        self.session_file: Optional[str] = None

        self._status = AGENT_CONNECTING
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # The throwaway PI_CODING_AGENT_DIR an omp_base_url run writes its
        # models.yml into, removed by close(); never where a conversation is.
        self._temp_agent_dir: Optional[Path] = None
        # The last turn of a resumed session's history (launch passes
        # bus.turn), so the next one continues its numbering.
        self._turn = int(turn)
        # Whether a turn is running, as far as this session knows: set when a
        # message is sent, cleared by omp's terminal agent_end. Decides only
        # whether a new message ends the running turn as "steered" in the
        # pane; omp itself decides what the message does (see the module
        # docstring on why every message is sent the steer way).
        self._running = False
        # end_turn_after_tool's tool call ids, read on the reader thread once
        # that call's result is delivered (_on_tool_execution_end), and the
        # stop reason the turn it ends reports.
        self._end_turn_calls: set = set()
        self._stop_reason: Optional[str] = None
        # The session's cumulative cost and tokens at the last turn end (or
        # at start), which the next turn's figures are the difference from.
        self._usage: Optional[dict] = None
        self._closing = False
        self._viewers_seen = 0
        # tool_call_id -> tool_name for whichever host-tool calls are
        # currently in flight, written and read only from `RpcClient`'s own
        # reader thread (`tool_execution_start`/`_end` and
        # `extension_ui_request` are all delivered on that one thread, in
        # wire order), so no lock is needed: see this module's docstring on
        # the confirm adapter's correlation.
        self._pending_tool_names: dict = {}

        # One small pool for every one-shot blocking `RpcClient` call
        # (start/stop/prompt/abort/get_state); see this module's docstring
        # on why no second "drain" pool is needed the way Codex's is.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omp-session")

    # -- AgentSession surface -------------------------------------------------

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
        """`omp` runs with every built-in tool disabled (``--no-tools``): the
        model's only capabilities are the product's own host tools, which never
        expose a shell or an unrestricted filesystem writer. There is
        therefore no sandbox to request in the bwrap/landlock/Seatbelt sense
        ``SdkSession``/``CodexSession`` report -- the "session with no shell
        to contain reports requested false" case ``base.py``'s own
        ``SandboxStatus`` docstring names.
        """
        return SandboxStatus(requested=False, active=False, missing=())

    def on_viewer_presence(self, count: int) -> None:
        """Keep the broker's view of viewer count in step with the
        registry's. Identical in intent to ``SdkSession.on_viewer_presence``
        and ``CodexSession.on_viewer_presence``; see either docstring."""
        if self._broker is None:
            return
        while self._viewers_seen < count:
            self._broker.viewer_connected()
            self._viewers_seen += 1
        while self._viewers_seen > count:
            self._broker.viewer_disconnected()
            self._viewers_seen -= 1

    async def submit_turn(self, blocks: list, viewer: Optional[str] = None) -> None:
        """Send one human message to `omp` and return once it is accepted.

        Always as a steer-capable prompt (this module's docstring says why):
        with no turn running it starts one, with one running it redirects
        it, and the running turn ends in the pane as ``steered`` while its
        further output carries this message's turn number. A message `omp`
        refuses is reported (``_refused``) and the session stays ready; only
        a dead `omp` process marks it unavailable.

        Unlike Codex's per-turn notification scope, `omp`'s event listeners
        (registered once in ``start()``) keep delivering
        ``message_update``/``tool_execution_*``/``agent_end`` frames for the
        rest of this session's life regardless of how many turns are sent,
        so there is nothing to hand off to a background drain here -- the
        same "the pump runs whether or not a browser is attached" invariant
        ``SdkSession``'s single persistent pump keeps.
        """
        if self._client is None or self._status == AGENT_UNAVAILABLE:
            self._emit(
                AgentError(
                    stderr="",
                    remediation="the agent is not running, so this turn was not sent; "
                    "check the startup output for why and use Retry",
                    viewer=viewer,
                )
            )
            return
        steering = self._running
        if steering:
            self._emit(TurnEnd(turn=self._turn, stop_reason="steered", cost_usd=0.0))
        # Before sending, not after: omp may start streaming the new turn
        # before it answers the prompt, and those events must carry this
        # turn's number.
        self._turn += 1
        self._running = True
        turn = self._turn
        try:
            loop = asyncio.get_running_loop()
            expanded = await loop.run_in_executor(
                None, turn_images.expand_turn_blocks, blocks, self.cwd
            )
            message, images = _to_omp_prompt(expanded)
            await self._run_blocking(
                self._client.prompt, message, images=images, streaming_behavior="steer"
            )
        except Exception as exc:
            if _process_gone(exc):
                self._running = False
                self._fail(exc, viewer=viewer)
            else:
                self._refused(str(exc) or type(exc).__name__, turn, ends_turn=not steering)

    def _refused(self, reason: str, turn: int, *, ends_turn: bool) -> None:
        """`omp` refused a message (at once, or later: ``_on_protocol_error``).
        Reported in the chat; the session stays ready. ``ends_turn`` closes
        the turn the message would have started, so the pane does not wait on
        it; a refused steer leaves the running turn running."""
        self._emit(
            AgentError(
                stderr=reason,
                remediation="omp did not accept that message, so the agent never saw it; "
                "the session is still ready, so send it again",
            )
        )
        if ends_turn:
            self._running = False
            self._emit(TurnEnd(turn=turn, stop_reason="rejected", cost_usd=0.0))

    def end_turn_after_tool(self) -> None:
        """A tool result asked to end the turn (``tools.ok``'s ``end_turn``).
        Called from the tool's handler, before omp has its result; omp's
        ``tool_execution_end`` for that same tool call (``_on_tool_execution_end``)
        is when the result is in the conversation, and the turn is aborted
        then. Keyed on the call's id, so a sibling tool running in parallel
        cannot trigger it, and forgotten when the turn settles, so a request
        whose end never came cannot end a later turn."""
        tool_call_id = _TOOL_CALL_ID.get()
        if tool_call_id is not None:
            self._end_turn_calls.add(tool_call_id)

    async def set_tool_table(self, tool_table: dict) -> bool:
        """Register ``tool_table`` (``ToolServer.host_tool_table()``'s shape)
        as this session's host tools in place of the old set, through omp's
        ``set_host_tools``, which applies before the next model call. False
        when `omp` is not running to take them."""
        self._tool_table = dict(tool_table)
        if self._client is None:
            return False
        await self._run_blocking(self._client.set_custom_tools, self._build_host_tools())
        return True

    async def decide_permission(self, request_id: str, decision: str, message: str = "") -> None:
        """Route a human's decision to the broker. Identical to
        ``SdkSession.decide_permission``/``CodexSession.decide_permission``;
        see either docstring for why ``UnknownRequest`` propagates
        deliberately."""
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
        """Deny every pending permission request before sending `omp`'s own
        ``abort``, not after -- the same ordering ``CodexSession.interrupt``
        uses and for the identical reason: a pending write-class approval or
        a pending ``confirm`` both block `omp_rpc`'s single reader thread
        synchronously (see this module's docstring), and that same thread is
        the only one that can ever deliver ``abort``'s own response. Waiting
        on ``abort`` first would hang until the pending decision resolves on
        its own.
        """
        if self._client is None:
            return
        if self._broker is not None:
            for request in list(self._broker.pending_requests()):
                try:
                    await self._broker.decide(request.request_id, "deny", "turn interrupted")
                except UnknownRequest:
                    # Already decided or timed out between the snapshot
                    # above and this call; nothing left to unblock.
                    pass
        try:
            await self._run_blocking(self._client.abort)
        except Exception as exc:
            # Best-effort, like the other two backends: the turn is either
            # already finished or the child is gone, and both surface
            # through the event stream's own failure handling.
            sys.stderr.write("warning: interrupt failed: %r\n" % (exc,))

    async def set_model(self, model: str) -> None:
        """Switch `omp`'s live session model via the RPC `set_model` command.

        ``omp://rpc.md`` documents the wire shape as
        ``{type: "set_model", provider, modelId}``; the installed
        ``omp_rpc`` source (``RpcClient.set_model(self, provider: str,
        model_id: str) -> ModelInfo``, confirmed by reading
        ``omp_rpc/client.py`` directly) is the method this calls through
        ``_run_blocking``, exactly like every other one-shot ``RpcClient``
        call in this file.

        With ``omp_base_url`` configured, ``_provider_id()`` is the same
        custom-provider id this session's own ``models.yml`` registers at
        ``start()``, so a live switch stays scoped to the provider `omp`
        already knows this session by, and ``model`` is the bare model id
        within it. Without ``omp_base_url`` (this session is using
        `omp`'s own already-configured providers directly -- see
        ``start()``), there is no such registered provider to assume, so
        ``model`` is split on the first ``"/"`` into a ``"provider/model"``
        pair, the same form `omp`'s own ``--model`` flag documents (e.g.
        ``"titan/qwen3.8-27b"``). Deliberately no local check of whether the
        split looks sensible (non-empty provider, non-empty model id,
        anything else): ``RpcClient.set_model`` does no client-side
        validation itself either (confirmed by reading ``omp_rpc/client.py``
        -- it just sends the request), so `omp`'s own response is the real
        validation here, exactly like every other RPC call this file makes.
        A malformed reference surfaces as whatever error `omp` itself
        returns, caught by ``ws.py``'s generic per-frame exception handler
        like any other ``set_model`` failure -- there is no case this method
        could reject locally that `omp` would not also reject, so guessing
        at that in advance would only risk being wrong about what `omp`
        actually accepts.
        """
        if model == self._model:
            return
        if self._base_url:
            provider, model_id = _provider_id(), model
        else:
            provider, _, model_id = model.partition("/")
        await self._run_blocking(self._client.set_model, provider, model_id)
        self._model = model
        self._emit(AgentModelChanged(model=model))

    # -- lifecycle --------------------------------------------------------------

    async def start(self) -> None:
        """Write this run's throwaway agent dir (if ``omp_base_url`` is
        set), launch `omp`, register every listener, switch to the
        conversation to resume (if any) and record the one `omp` is in;
        never raise. See ``SdkSession.start``'s docstring for why: the HTTP
        server starts independently and must keep serving the viewer whatever
        the agent does.
        """
        self._loop = asyncio.get_running_loop()
        refusal = self._configuration_refusal()
        if refusal is not None:
            self._fail(ValueError(refusal))
            return
        try:
            if self._base_url:
                # An arbitrary/self-hosted OpenAI-compatible endpoint `omp`
                # does not already know about: synthesize this session's own
                # throwaway custom provider (models.yml) pointing at it.
                model_id = self._model or _DEFAULT_MODEL_ID
                self._temp_agent_dir, api_key_env = _write_agent_dir(
                    self._base_url, self._api_key, model_id, self.session_id
                )
                env = {"PI_CODING_AGENT_DIR": str(self._temp_agent_dir)}
                env.update(api_key_env)
                model_arg = "%s/%s" % (_provider_id(), model_id)
            else:
                # No omp_base_url: `omp`'s own provider registry and
                # credentials, from the profile ``agent_dir`` names or, with
                # none, the human's own (no PI_CODING_AGENT_DIR override).
                # `self._model` (e.g. "titan/qwen3.8-27b", or None to take
                # omp's own default) is passed straight through to
                # ``--model``, the same fuzzy provider/model reference a
                # human would type at the CLI. `RpcClient.start()` merges
                # `env` onto `os.environ` rather than replacing it
                # (confirmed by reading ``omp_rpc/client.py``).
                env = {}
                if self._profile_dir is not None:
                    env["PI_CODING_AGENT_DIR"] = self._profile_dir
                model_arg = self._model
            if self._config_rel is not None:
                env["PI_CONFIG_DIR"] = self._config_rel
            client_kwargs = {}
            instructions = "\n\n".join(
                p for p in (self._instructions, _renamed_tools_note(self._tool_table)) if p
            )
            if instructions:
                client_kwargs["append_system_prompt"] = instructions
            if self._session_dir is not None:
                self._session_dir.mkdir(parents=True, exist_ok=True)
                client_kwargs["session_dir"] = str(self._session_dir)
            else:
                client_kwargs["no_session"] = True
            self._client = self._client_factory(
                **client_kwargs,
                executable=self._binary or "omp",
                model=model_arg,
                cwd=self.cwd,
                env=env,
                # Every built-in tool disabled: the model's only capabilities
                # are the host tools registered below. Extension discovery
                # disabled too: without it, a project-local .omp/.pi
                # extension in `cwd` would load as trusted code and could
                # register its own native tools, entirely outside this
                # file's own host_tool_call broker gate. See this module's
                # docstring on the permission design this makes possible.
                tools=(),
                custom_tools=self._build_host_tools(),
                extra_args=("--auto-approve", "--no-extensions"),
            )
            self._register_listeners()
            await self._run_blocking(self._client.start)
        except Exception as exc:
            # start() may already have spawned the subprocess and its
            # reader/stderr threads by the time a later step raises;
            # dropping the only reference without stopping it first would
            # leak both for the life of this still-serving process. Same
            # try/except/warn pattern SdkSession/CodexSession use in start().
            if self._client is not None:
                try:
                    await self._run_blocking(self._client.stop)
                except Exception as close_exc:
                    sys.stderr.write(
                        "warning: agent client did not close cleanly: %r\n" % (close_exc,)
                    )
                self._client = None
            self._discard_agent_dir()
            self._fail(exc)
            return
        if self._resume:
            await self._switch_to(self._resume)
        try:
            state = await self._run_blocking(self._client.get_state)
            self._remember_sdk_session(state.session_id)
            self._remember_session_file(state.session_file)
        except Exception as exc:
            # A session that cannot read this still works; what it costs is
            # the hello frame's session id and a later -c resuming it.
            sys.stderr.write("warning: could not read the omp session state: %r\n" % (exc,))
        self._usage = await self._read_usage()
        self._set_status(AGENT_READY)

    def _configuration_refusal(self) -> Optional[str]:
        """Why this session's configuration cannot start, or None."""
        if self._api_key and not self._base_url:
            return (
                "omp_api_key is set without omp_base_url; omp_api_key only "
                "applies to an arbitrary omp_base_url endpoint, since a provider "
                "omp already knows about carries its own credentials"
            )
        if self._base_url and self._profile_dir is not None:
            return (
                "omp_base_url and an omp agent directory are both set, and both "
                "would own omp's PI_CODING_AGENT_DIR; name the endpoint in the agent "
                "directory's own models.yml instead"
            )
        if self._binary is not None and not os.path.isabs(self._binary):
            return (
                "the omp binary must be an absolute path, not %r, which would be "
                "looked for relative to the served directory" % self._binary
            )
        if self._config_dir is not None:
            home = os.path.realpath(os.path.expanduser("~"))
            target = os.path.realpath(os.path.join(home, self._config_dir))
            rel = os.path.relpath(target, home)
            if rel == "." or rel == ".." or rel.startswith(".." + os.sep):
                return (
                    "the omp config directory must be inside %s (omp joins "
                    "PI_CONFIG_DIR to $HOME), not %r" % (home, self._config_dir)
                )
            self._config_rel = rel
        return None

    async def _switch_to(self, session_file: str) -> None:
        """Resume the conversation in ``session_file``; a conversation that
        cannot be resumed is reported (``SessionReset``) and this one goes on
        as a new conversation. A file that does not exist is nothing to
        resume rather than a failure: `omp` names the file at start but
        writes it with the first message, so a session that never had one
        leaves no file behind."""
        if not await self._run_blocking(os.path.isfile, session_file):
            return
        reason = None
        try:
            result = await self._run_blocking(self._client.switch_session, session_file)
            if getattr(result, "cancelled", False):
                reason = "omp cancelled the switch to %s" % session_file
        except Exception as exc:
            reason = "omp could not open %s (%s)" % (session_file, exc)
        if reason is not None:
            self._emit(
                SessionReset(
                    reason="asked to resume the previous omp conversation, but %s; "
                    "this is a new conversation" % reason
                )
            )

    async def close(self) -> None:
        self._closing = True
        if self._broker is not None:
            # Before the client goes, while there is still a socket to carry
            # the denial event and while the RPC it belongs to can still get
            # a result: see SdkSession.close's identical ordering.
            self._broker.shutdown()
        if self._client is not None:
            try:
                await self._run_blocking(self._client.stop)
            except Exception as exc:
                sys.stderr.write("warning: agent client did not close cleanly: %r\n" % (exc,))
        self._client = None
        self._executor.shutdown(wait=True)
        self._discard_agent_dir()
        self._set_status(AGENT_UNAVAILABLE)

    def _discard_agent_dir(self) -> None:
        if self._temp_agent_dir is not None:
            shutil.rmtree(self._temp_agent_dir, ignore_errors=True)
            self._temp_agent_dir = None

    # -- host tools -------------------------------------------------------------

    def _build_host_tools(self) -> tuple:
        """``omp_rpc.HostTool`` instances for every ``tool_table()`` entry,
        for ``RpcClient(custom_tools=...)``, which registers them via
        ``set_host_tools`` itself once ``start()`` succeeds. A tool named like
        one of `omp`'s own is registered under ``_omp_name`` (see there); the
        broker is still asked under the product's name for it.
        """
        return tuple(
            host_tool(
                name=_omp_name(name),
                description=spec.description,
                parameters=spec.schema,
                execute=self._make_execute(name, spec),
            )
            for name, spec in self._tool_table.items()
        )

    def _make_execute(self, name: str, spec):
        """The synchronous ``execute`` callback one host tool runs on its own
        per-call daemon thread (``omp_rpc``'s ``_handle_host_tool_call``
        spawns one per call, never the shared reader thread -- see this
        module's docstring).

        Write-class tools call ``broker.ask`` first and raise on a denial;
        raising, rather than returning an ``is_error`` result, is what makes
        ``omp_rpc`` set ``isError: true`` on the wire (its own
        ``_handle_host_tool_call`` only does that for an exception, not for
        a returned dict that happens to carry its own error marker) -- the
        same "a refusal must not read as a successful call" invariant
        ``tools/__init__.py``'s ``fail()`` documents for the Claude backend.
        ``tool_table()``'s own handler already never raises (it is `_wrap`'s
        job to turn every failure into ``{"content": [...], "is_error":
        True}``), so this is the one place that boundary gets translated
        into the wire's own error signal for this backend.
        """
        write = spec.write
        handler = spec.handler

        def execute(params, context):
            tool_call_id = getattr(context, "tool_call_id", None)

            async def run():
                # Seen by end_turn_after_tool if this tool's handler asks to
                # end the turn: the task running this coroutine carries it.
                _TOOL_CALL_ID.set(tool_call_id)
                if write and self._broker is not None:
                    decision = await self._broker.ask(name, dict(params), None)
                    if not decision.allow:
                        raise RuntimeError(decision.message)
                return await handler(dict(params))

            result = asyncio.run_coroutine_threadsafe(run(), self._loop).result()
            if result.get("is_error"):
                raise RuntimeError(_content_to_text(result.get("content")))
            return {"content": result.get("content") or [], "details": {}}

        return execute

    # -- event wiring -------------------------------------------------------------

    def _register_listeners(self) -> None:
        client = self._client
        client.on_message_update(self._on_message_update)
        client.on_tool_execution_start(self._on_tool_execution_start)
        client.on_tool_execution_end(self._on_tool_execution_end)
        client.on_agent_end(self._on_agent_end)
        client.on_ui_request(self._on_ui_request)
        client.on_protocol_error(self._on_protocol_error)

    def _on_message_update(self, event) -> None:
        """Runs on `omp_rpc`'s reader thread; every emit crosses back onto
        the session's own loop via ``call_soon_threadsafe``, the same
        marshalling ``CodexSession._drain_turn`` uses for its own
        off-loop-thread notifications."""
        assistant_event = event.assistant_message_event or {}
        kind = assistant_event.get("type")
        if kind == "text_delta":
            text = assistant_event.get("delta") or ""
            if text:
                turn = self._turn
                self._loop.call_soon_threadsafe(self._emit, TextDelta(turn=turn, text=text))
        elif kind == "error":
            error = assistant_event.get("error")
            if error is not None:
                message = _content_to_text(error)
            else:
                message = "the model reported an error"
            self._loop.call_soon_threadsafe(
                self._emit,
                AgentError(
                    stderr=message,
                    remediation="the model reported an error during this turn; "
                    "the session otherwise remains ready",
                ),
            )

    def _on_tool_execution_start(self, event) -> None:
        # The product's name for a tool _omp_name renamed, so the page and the
        # broker see the same names on every backend.
        name = _product_name(event.tool_name, self._tool_table)
        self._pending_tool_names[event.tool_call_id] = name
        args = event.args if isinstance(event.args, dict) else {}
        turn = self._turn
        self._loop.call_soon_threadsafe(
            self._emit,
            ToolUse(turn=turn, tool_use_id=event.tool_call_id, name=name, input=args),
        )

    def _on_tool_execution_end(self, event) -> None:
        self._pending_tool_names.pop(event.tool_call_id, None)
        result = event.result
        content = result.get("content") if isinstance(result, dict) else result
        text = _content_to_text(content)
        self._loop.call_soon_threadsafe(
            self._emit,
            ToolResult(tool_use_id=event.tool_call_id, is_error=bool(event.is_error), text=text),
        )
        if event.tool_call_id in self._end_turn_calls:
            # The result that asked to end the turn is in the conversation
            # now (a sibling tool's end, running in parallel, is not this
            # one), so aborting keeps it; omp's RPC has no way for a host
            # tool to end the turn itself, so this is the abort a human's
            # interrupt sends.
            self._end_turn_calls.discard(event.tool_call_id)
            self._loop.call_soon_threadsafe(self._end_turn_now)

    def _end_turn_now(self) -> None:
        self._stop_reason = "ended_by_tool"
        asyncio.ensure_future(self.interrupt())

    def _on_agent_end(self, event) -> None:
        if event.is_terminal is False:
            # Maintenance/async delivery scheduled more work; not the turn's
            # true final settle (omp://rpc.md's Event Stream Schema).
            return
        # Read here, on the reader thread, in wire order: the turn this run
        # ended in, whatever the loop has done by the time the settle runs.
        turn = self._turn
        self._loop.call_soon_threadsafe(self._turn_settled, turn)

    def _turn_settled(self, turn: int) -> None:
        """On the loop, for omp's terminal ``agent_end`` of a run that ended
        in ``turn``. Its ``TurnEnd`` follows once its cost is read.

        A settle for a turn no longer current is dropped: a message sent
        between omp ending the run and this callback already ended that turn
        as ``steered`` and started a run of its own, whose own ``agent_end``
        settles it."""
        if turn != self._turn:
            return
        self._running = False
        self._end_turn_calls.clear()
        stop_reason, self._stop_reason = self._stop_reason or "end", None
        asyncio.ensure_future(self._emit_turn_end(turn, stop_reason))

    async def _emit_turn_end(self, turn: int, stop_reason: str) -> None:
        cost, tokens = 0.0, None
        usage = await self._read_usage()
        if usage is not None and self._usage is not None:
            cost = max(0.0, usage["cost"] - self._usage["cost"])
            tokens = {key: usage[key] - self._usage[key] for key in _TOKEN_KEYS}
        if usage is not None:
            self._usage = usage
        self._emit(TurnEnd(turn=turn, stop_reason=stop_reason, cost_usd=cost, tokens=tokens))

    async def _read_usage(self) -> Optional[dict]:
        """The session's cumulative cost and token counts so far
        (``get_session_stats``), or None when omp cannot say."""
        try:
            stats = await self._run_blocking(self._client.get_session_stats)
        except Exception as exc:
            sys.stderr.write("warning: could not read the omp session's usage: %r\n" % (exc,))
            return None
        usage = {key: int(getattr(stats.tokens, key)) for key in _TOKEN_KEYS}
        usage["cost"] = float(stats.cost)
        return usage

    def _on_protocol_error(self, error) -> None:
        """Runs on `omp_rpc`'s reader thread for an error response nothing
        was waiting on. For ``prompt`` that is `omp` refusing a message it
        had already acknowledged (``omp://rpc.md``: async prompt scheduling
        can fail after the immediate success response)."""
        if getattr(error, "command", None) != "prompt":
            return
        reason = getattr(error, "remote_error", None) or str(error)
        turn = self._turn
        self._loop.call_soon_threadsafe(self._prompt_refused_late, reason, turn)

    def _prompt_refused_late(self, reason: str, turn: int) -> None:
        """Reported whatever the turn; the turn is closed only if it is still
        the current one (a later message has otherwise already moved on)."""
        self._refused(reason, turn, ends_turn=self._running and turn == self._turn)

    def _on_ui_request(self, request) -> None:
        """Runs on `omp_rpc`'s reader thread. See this module's docstring on
        why ``confirm`` is a defensive second layer here, not the primary
        write-class gate, and why blocking this thread while asking the
        broker is correct and intentional (mirrors
        ``CodexSession._approval_handler``'s identical choice for Codex's
        own single reader thread).

        A ``confirm`` frame carries no tool-call id on the wire, so it can
        only be attributed to "the tool currently in flight" when there is
        exactly one -- `omp`'s default parallel-tool-execution mode runs
        sibling tool calls from the same turn concurrently (see this
        module's docstring), so ``self._pending_tool_names`` can genuinely
        hold more than one entry. Zero or more than one pending tool call
        both fail closed (declined) rather than guess which one a confirm
        belongs to.
        """
        if request.method != "confirm":
            if request.requires_response():
                try:
                    self._client.cancel_ui_request(request.id)
                except Exception as exc:
                    sys.stderr.write(
                        "warning: could not answer a %s UI request: %r\n" % (request.method, exc)
                    )
            return
        pending = self._pending_tool_names
        broker = self._broker
        if broker is None or len(pending) != 1:
            self._send_ui_confirmation(request.id, False)
            return
        tool_name = next(iter(pending.values()))
        try:
            future = asyncio.run_coroutine_threadsafe(broker.ask(tool_name, {}, None), self._loop)
            decision = future.result()
        except Exception as exc:
            sys.stderr.write(
                "warning: could not reach the permission broker for a confirm "
                "request: %r\n" % (exc,)
            )
            self._send_ui_confirmation(request.id, False)
            return
        self._send_ui_confirmation(request.id, decision.allow)

    def _send_ui_confirmation(self, request_id: str, confirmed: bool) -> None:
        try:
            self._client.send_ui_confirmation(request_id, confirmed)
        except Exception as exc:
            sys.stderr.write("warning: could not answer a confirm UI request: %r\n" % (exc,))

    # -- helpers -------------------------------------------------------------

    async def _run_blocking(self, func, *args, **kwargs):
        """Run one blocking ``RpcClient`` call on ``self._executor``. The one
        helper every one-shot call in this file uses, mirroring
        ``CodexSession._run_blocking``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

    def _remember_sdk_session(self, session_id: Optional[str]) -> None:
        if session_id and session_id != self.sdk_session_id:
            self.sdk_session_id = session_id
            if self._on_sdk_session_id is not None:
                try:
                    self._on_sdk_session_id(session_id)
                except Exception as exc:
                    sys.stderr.write("warning: could not record the omp session id: %r\n" % (exc,))

    def _remember_session_file(self, session_file: Optional[str]) -> None:
        """Record `omp`'s conversation file, what a later ``-c`` resumes."""
        if session_file and session_file != self.session_file:
            self.session_file = session_file
            if self._on_session_file is not None:
                try:
                    self._on_session_file(session_file)
                except Exception as exc:
                    sys.stderr.write(
                        "warning: could not record the omp conversation file: %r\n" % (exc,)
                    )

    def _fail(self, exc: BaseException, viewer: Optional[str] = None) -> None:
        """Report a failure as an event and mark the session unavailable.
        Never raises. Identical in intent to ``SdkSession._fail``/
        ``CodexSession._fail``."""
        self._set_status(AGENT_UNAVAILABLE)
        self._emit(
            AgentError(
                stderr="%s: %s" % (type(exc).__name__, exc),
                remediation=_remediation_for(exc),
                viewer=viewer,
            )
        )

    def _emit(self, event) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            sys.stderr.write("warning: could not deliver %s: %r\n" % (type(event).__name__, exc))


#: The token counts a turn's ``TurnEnd`` reports, as ``omp_rpc``'s
#: ``TokenUsage`` names them.
_TOKEN_KEYS = ("input", "output", "cache_read", "cache_write")

#: `omp`'s own tool names (its built-in and hidden tools, 18.2, plus
#: ``search``, which it reads as ``grep``). `omp` keys behaviour on some of
#: these by name alone, whoever registered the tool: a successful result from
#: any tool named ``checkpoint`` puts the session in a context checkpoint that
#: re-samples the model until it calls ``rewind`` (Annealage Loom's own
#: ``checkpoint`` tool looped a turn for minutes this way, and the state is
#: restored from the session file on resume). A host tool with one of these
#: names, in any case (`omp` folds case when it reads tool names), is
#: therefore registered under ``_omp_name``.
_OMP_TOOL_NAMES = frozenset(
    (
        "read security_scan bash edit ast_grep ast_edit ask debug eval github glob grep "
        "find lsp checkpoint rewind context_notes new_context task hub todo web_search "
        "write memory_edit retain recall reflect learn manage_skill think yield goal "
        "delete move browser fetch search"
    ).split()
)


def _omp_name(name: str) -> str:
    """The name `omp` sees for host tool ``name``: unchanged, unless `omp`
    has a tool of that name, in which case the product's server name is put
    in front (``checkpoint`` becomes ``loom__checkpoint``), the same form a
    remote server's tools take."""
    if name.lower() in _OMP_TOOL_NAMES:
        return host_tool_name(product.current().mcp_server_name, name)
    return name


def _product_name(omp_name: str, tool_table: dict) -> str:
    """The inverse of ``_omp_name`` over ``tool_table``'s tools: the
    product's name for a tool `omp` reports as ``omp_name``."""
    for name in tool_table:
        if _omp_name(name) == omp_name:
            return name
    return omp_name


def _renamed_tools_note(tool_table: dict) -> Optional[str]:
    """A line for the system prompt naming each tool ``_omp_name`` renamed,
    so guidance written against the product's names still finds them."""
    renamed = [(name, _omp_name(name)) for name in tool_table if _omp_name(name) != name]
    if not renamed:
        return None
    return "On this agent backend these tools of %s are named differently: %s." % (
        product.current().title,
        ", ".join("`%s` is `%s`" % pair for pair in renamed),
    )


#: The id of the host tool call whose handler is running in this task
#: (``_make_execute``), for ``OmpSession.end_turn_after_tool``.
_TOOL_CALL_ID: contextvars.ContextVar = contextvars.ContextVar("omp_tool_call_id", default=None)


def _process_gone(exc: BaseException) -> bool:
    """Whether ``exc`` means the `omp` process itself is gone, which is the
    one failure of a sent message that makes the session unavailable; a
    command `omp` answered with an error, or one it did not answer in time,
    leaves a running process that can take the next message. Matched by
    class name, like ``_remediation_for``."""
    return type(exc).__name__ == "RpcProcessExitError" or isinstance(exc, BrokenPipeError)


# ---------------------------------------------------------------------------
# Custom-provider config generation (Q4, roadmap.md - DECIDED 2026-09-19).
# ---------------------------------------------------------------------------


def _api_key_env_name(session_id: object) -> str:
    """A collision-resistant environment-variable name for one session's
    ``omp_api_key``, derived from ``session_id`` so two concurrent
    ``OmpSession`` runs never share a name. `omp`'s own ``apiKey``
    resolution (`omp://providers.md`) tries an environment variable by
    this exact name first, before falling back to treating the string as a
    literal -- see this module's docstring on why writing the name here,
    not the secret, is the deliberate fix rather than a workaround.
    """
    digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:32]
    return "%s_OMP_API_KEY_%s" % (product.current().name.upper(), digest)


def _build_custom_provider(base_url: str, api_key_env_name: Optional[str]) -> dict:
    """The ``omp://providers.md`` custom-provider shape for one arbitrary
    OpenAI-compatible endpoint. ``api_key_env_name`` is the name of an
    environment variable set on the launched subprocess (see
    ``_api_key_env_name``/``_write_agent_dir``), never the literal secret:
    `omp` resolves ``apiKey`` as an env-var name first, so this lets that
    resolution path do the substitution rather than writing a secret to
    disk that could also collide with an existing env var name or be
    misread as a leading-``!`` shell command. Absent means a genuinely
    local, unauthenticated endpoint (Ollama, llama.cpp): ``auth: none`` is
    `omp`'s own keyless marker, never an empty-string ``Authorization``
    header.

    ``discovery: {type: "proxy"}`` is `omp://providers.md`'s
    "Discovery-enabled provider" shape: it makes `omp` fetch the endpoint's
    own model list at runtime (registry-assembly step 3, after the
    ``models.yml`` static entries in step 2), so every model
    ``base_url`` actually reports becomes selectable, not only the single
    ``model_id`` this session happened to start with. Without it,
    ``RpcClient.set_model`` -- which only accepts a provider/model pair
    `omp` already knows about -- rejects any live switch to a model other
    than the one ``_write_agent_dir`` registered at startup with "Model not
    found", which is exactly the failure the webui's live model picker
    exists to avoid. The static ``models`` entry ``_write_agent_dir`` still
    writes is kept alongside discovery, not replaced by it: it guarantees
    the startup model is registered even against an endpoint whose
    discovery probe fails or is unsupported (a bare llama.cpp server with
    no ``/v1/models`` route, for instance), while discovery is what makes
    every *other* model the endpoint reports live-switchable too.
    """
    provider = {"baseUrl": base_url, "api": "openai-completions", "discovery": {"type": "proxy"}}
    if api_key_env_name:
        provider["apiKey"] = api_key_env_name
    else:
        provider["auth"] = "none"
    return provider


def _write_agent_dir(
    base_url: str, api_key: Optional[str], model_id: str, session_id: object
) -> tuple:
    """A throwaway ``PI_CODING_AGENT_DIR``-scoped directory holding this
    run's custom-provider ``models.yml``, isolated from the human's real
    `omp` install, plus the ``{env_var_name: secret}`` mapping (empty when
    ``api_key`` is absent) the caller must add to the launched subprocess's
    environment. Caller (``start()``) removes the directory in
    ``close()``/its own failure path; see this module's docstring for why
    JSON content in a ``.yml`` file is intentional, not a mistake.

    ``mkdtemp`` and the ``models.yml`` write are wrapped in their own
    try/except: if the write fails after the directory already exists, the
    directory is removed here before re-raising, so a caller that has not
    yet recorded the path anywhere still cannot leak it.
    """
    agent_dir = Path(tempfile.mkdtemp(prefix="%s-omp-" % product.current().name))
    try:
        api_key_env_name = _api_key_env_name(session_id) if api_key else None
        provider = _build_custom_provider(base_url, api_key_env_name)
        provider["models"] = [{"id": model_id, "name": model_id}]
        models_doc = {"providers": {_provider_id(): provider}}
        (agent_dir / "models.yml").write_text(json.dumps(models_doc, indent=2))
    except Exception:
        shutil.rmtree(agent_dir, ignore_errors=True)
        raise
    extra_env = {api_key_env_name: api_key} if api_key_env_name else {}
    return agent_dir, extra_env


# ---------------------------------------------------------------------------
# Content-block translation: the agent layer's Anthropic-shaped turn blocks (what
# turn_images.expand_turn_blocks produces, the same shape SdkSession sends
# untouched) to the RPC ``prompt`` command's ``message``/``images`` shape.
# ---------------------------------------------------------------------------


def _to_omp_prompt(blocks: list):
    """``blocks`` (already expanded by ``turn_images.expand_turn_blocks``,
    Anthropic-shaped: ``{"type": "text", ...}`` and ``{"type": "image",
    "source": {"type": "base64", ...}}``) as an RPC ``prompt`` message plus
    an ``omp_rpc.ImageContent`` list.

    Unlike Codex's schema (`session/codex.py`'s ``_to_codex_input_items``),
    which has no inline-base64 image variant and needs a ``data:`` URI, the
    RPC wire's ``ImageContent`` (``{"type": "image", "data": ..., "mimeType":
    ...}``) already carries base64 data directly, so an expanded
    attachment's bytes pass through with no re-encoding.
    """
    text_parts = []
    images = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            if text:
                text_parts.append(text)
        elif block_type == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64" and source.get("data"):
                images.append(
                    {
                        "type": "image",
                        "data": source["data"],
                        "mimeType": source.get("media_type") or "application/octet-stream",
                    }
                )
    message = "\n\n".join(text_parts) if text_parts else "(this message arrived empty)"
    return message, images


def _content_to_text(content: Any) -> str:
    """Flatten a tool result's (or an assistant error's) content to text for
    the chat pane. Mirrors ``session/sdk.py``'s ``_content_to_text``: a
    string passes through, a list of blocks renders its text blocks and
    names anything else rather than dropping it silently.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = content.get("content")
        if inner is not None:
            return _content_to_text(inner)
        return json.dumps(content, default=str)
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                parts.append("[%s]" % (block.get("type") or "non-text content"))
        else:
            parts.append("[non-text content]")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Remediation text, matched by exception class name for the same reason
# session/sdk.py's _remediation_for and session/codex.py's _remediation_for
# are: a renamed or added omp_rpc error degrades to the generic message
# instead of an unhandled traceback at startup.
# ---------------------------------------------------------------------------


def _remediation_for(exc: BaseException) -> str:
    if isinstance(exc, ValueError):
        # This file's own configuration checks (e.g. a missing
        # omp_base_url) already write an actionable message; passing it
        # through avoids restating it more vaguely.
        return str(exc)
    name = type(exc).__name__
    if name == "FileNotFoundError":
        return (
            "the omp CLI could not be found, on PATH or at the omp binary path "
            "this product was given; install it (see https://omp.sh/) before "
            "using backend=omp"
        )
    if name == "RpcTimeoutError":
        return (
            "omp did not become ready in time; check that omp_base_url is "
            "reachable and that the omp CLI is not stuck waiting on input"
        )
    if name == "RpcProcessExitError":
        return "the omp process exited before it was ready; check its stderr above"
    return "the agent is unavailable; the captured output above is what it reported"
