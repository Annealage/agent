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
that entire agent directory (auth store, sessions, cache, and the model
config alongside it), so this session gives every launch its own throwaway
temp directory via that env var, writes a ``models.yml`` there, and tears
the directory down in ``close()``. This keeps the human's own `omp` install
(auth, other providers, saved sessions) completely untouched by a
product-launched local-backend run. The file is written as ``json.dumps(...)``
rather than through a YAML library: JSON is valid YAML, `omp`'s config
loader accepts a `.yml` path with JSON content without complaint, and this
avoids adding a YAML dependency for a one-off machine-generated file no
human ever hand-edits.

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
import functools
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from omp_rpc import RpcClient, host_tool

from .. import product
from . import turn_images
from .base import (
    AGENT_CONNECTING,
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentModelChanged,
    AgentStatus,
    SandboxStatus,
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
    """

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

        self._status = AGENT_CONNECTING
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._agent_dir: Optional[Path] = None
        self._turn = 0
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
        """Send one prompt to `omp` and return once it is accepted.

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
        self._turn += 1
        try:
            loop = asyncio.get_running_loop()
            expanded = await loop.run_in_executor(
                None, turn_images.expand_turn_blocks, blocks, self.cwd
            )
            message, images = _to_omp_prompt(expanded)
            await self._run_blocking(self._client.prompt, message, images=images)
        except Exception as exc:
            self._fail(exc, viewer=viewer)

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
        set), launch `omp`, and register every listener; never raise. See
        ``SdkSession.start``'s docstring for why: the HTTP server starts
        independently and must keep serving the viewer whatever the agent
        does.
        """
        self._loop = asyncio.get_running_loop()
        if self._api_key and not self._base_url:
            self._fail(
                ValueError(
                    "omp_api_key is set without omp_base_url; omp_api_key only "
                    "applies to an arbitrary omp_base_url endpoint, since a provider "
                    "omp already knows about carries its own credentials"
                )
            )
            return
        try:
            if self._base_url:
                # An arbitrary/self-hosted OpenAI-compatible endpoint `omp`
                # does not already know about: synthesize this session's own
                # throwaway custom provider (models.yml) pointing at it, the
                # same as before this method grew the branch below.
                model_id = self._model or _DEFAULT_MODEL_ID
                self._agent_dir, api_key_env = _write_agent_dir(
                    self._base_url, self._api_key, model_id, self.session_id
                )
                env = {"PI_CODING_AGENT_DIR": str(self._agent_dir)}
                env.update(api_key_env)
                model_arg = "%s/%s" % (_provider_id(), model_id)
            else:
                # No omp_base_url: use `omp` exactly as already configured
                # on this host -- no PI_CODING_AGENT_DIR override, so its own
                # provider registry and credential resolution are untouched.
                # `self._model` (e.g. "titan/qwen3.8-27b", or None to take
                # omp's own default) is passed straight through to
                # ``--model``, the same fuzzy provider/model reference a
                # human would type at the CLI. `RpcClient.start()` merges
                # `env` onto `os.environ` rather than replacing it
                # (confirmed by reading ``omp_rpc/client.py``), so the empty
                # dict here changes nothing about the child's environment.
                env = {}
                model_arg = self._model
            extra_client_kwargs = (
                {"append_system_prompt": self._instructions} if self._instructions else {}
            )
            self._client = self._client_factory(
                **extra_client_kwargs,
                executable="omp",
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
                no_session=True,
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
        try:
            state = await self._run_blocking(self._client.get_state)
            self._remember_sdk_session(state.session_id)
        except Exception as exc:
            # Cosmetic only (the hello frame's own session id field): a
            # session that cannot fetch this still works, it just reports
            # None until something else updates it.
            sys.stderr.write("warning: could not read the omp session id: %r\n" % (exc,))
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
                await self._run_blocking(self._client.stop)
            except Exception as exc:
                sys.stderr.write("warning: agent client did not close cleanly: %r\n" % (exc,))
        self._client = None
        self._executor.shutdown(wait=True)
        self._discard_agent_dir()
        self._set_status(AGENT_UNAVAILABLE)

    def _discard_agent_dir(self) -> None:
        if self._agent_dir is not None:
            shutil.rmtree(self._agent_dir, ignore_errors=True)
            self._agent_dir = None

    # -- host tools -------------------------------------------------------------

    def _build_host_tools(self) -> tuple:
        """``omp_rpc.HostTool`` instances for every ``tool_table()`` entry,
        for ``RpcClient(custom_tools=...)``, which registers them via
        ``set_host_tools`` itself once ``start()`` succeeds.
        """
        return tuple(
            host_tool(
                name=name,
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

        def execute(params, _context):
            async def run():
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
        self._pending_tool_names[event.tool_call_id] = event.tool_name
        args = event.args if isinstance(event.args, dict) else {}
        turn = self._turn
        self._loop.call_soon_threadsafe(
            self._emit,
            ToolUse(turn=turn, tool_use_id=event.tool_call_id, name=event.tool_name, input=args),
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

    def _on_agent_end(self, event) -> None:
        if event.is_terminal is False:
            # Maintenance/async delivery scheduled more work; not the turn's
            # true final settle (omp://rpc.md's Event Stream Schema).
            return
        turn = self._turn
        self._loop.call_soon_threadsafe(self._emit_turn_end, turn)

    def _emit_turn_end(self, turn: int) -> None:
        # No per-turn cost figure is available from an arbitrary
        # OpenAI-compatible endpoint the way a metered subscription reports
        # one; 0.0 is an honest "not applicable", the same reasoning
        # CodexSession's own TurnEnd uses for subscription billing.
        self._emit(TurnEnd(turn=turn, stop_reason="end", cost_usd=0.0))

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
            "the omp CLI could not be found on PATH; install it (see "
            "https://omp.sh/) before using backend=omp"
        )
    if name == "RpcTimeoutError":
        return (
            "omp did not become ready in time; check that omp_base_url is "
            "reachable and that the omp CLI is not stuck waiting on input"
        )
    if name == "RpcProcessExitError":
        return "the omp process exited before it was ready; check its stderr above"
    return "the agent is unavailable; the captured output above is what it reported"
