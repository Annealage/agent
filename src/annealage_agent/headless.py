"""One prompt, one turn, no human: ``run_prompt``.

A product that wants a sub-agent for a batch step (review a sheet, summarise
a file) does not want a chat pane, a browser, a project directory of session
state or an approval card. ``run_prompt`` gives it the rest of the agent layer
without those: it starts a session on the Claude or omp backend, sends one
user message, waits for the turn to end, and returns what the model said, what
the turn cost and which tools it called.

    result = await run_prompt(
        "Check the sheet against the datasheet.",
        tools=ToolSet(tools=[...], grading=Grading(read=(...), view=(), write=(...)),
                      grant=("submit_result",)),
        backend="omp", model="provider/model", cwd=scratch_dir, timeout=300,
    )

What is private to the run. It builds its own ``ViewerBus`` (over a registry
nobody connects to) and its own ``ToolServer`` from the ``ToolSet``; it touches
no viewer registry, event log, ``sessions`` state or ``permissions.toml``. The
session's state is a temporary directory, removed when the run ends. ``cwd``
is the caller's scratch directory and is used only as the backend's working
directory; the caller should make it empty, since the model is told nothing
about it and, on the Claude backend, cannot read it either (see below).

How permission works with nobody to ask. ``session.permissions.HeadlessBroker``
stands in for the broker: it never opens a request, so nothing waits on a card.
A call it is asked about is allowed only if its name is in the ``ToolSet``'s
``grant``, and denied at once, with a message that says why, otherwise. Read
and view tools never reach it on either backend (Claude's ``allowed_tools``;
omp's host-tool gate only asks about write-grade tools), so grading decides:
read and view are pre-allowed, a write-grade tool is refused unless granted.
A grant covers this run only and is never remembered.

Claude's built-ins. The session is built with an empty built-in tool list
(``--tools ""``), so the model has no Bash, Write, Read or the rest, only the
``ToolSet``'s tools; ``allowed_tools`` is the pre-allowed set plus the grants;
no settings file is loaded (``setting_sources`` empty), so nothing under
``cwd`` or in the user's own configuration adds a rule, a hook or a tool; the
shell sandbox is off since there is no shell. ``HeadlessBroker`` is the backstop
should anything still ask. omp already runs with every built-in disabled
(``--no-tools --no-extensions``), so its only tools are the ``ToolSet``'s.
Codex is refused: it reaches tools through the app's ``/mcp`` endpoint, which
a headless run has no listener for.

Failure. A model or backend failure (the agent would not start, its package is
not installed, the model errored, the process died, the turn timed out) comes
back as ``RunResult.error``, with whatever text arrived first; ``run_prompt``
raises only for a bad argument (``ValueError``) or because the caller
cancelled it (``CancelledError``, after the session is closed).

When a run ends early. A timeout or a cancel ends the run: the broker refuses
any further call, each tool handler starts refusing ("this run has ended") and
those running are cancelled and waited for (up to ``_DRAIN_TIMEOUT``, so their
cleanup is done before ``run_prompt`` returns), the backend process is killed
(omp's ``kill``, so a start still waiting on the process cannot hold the close
up), and the close itself is bounded by ``_CLOSE_TIMEOUT``. A handler that
is blocked inside a synchronous call, or past its last ``await``, cannot be
stopped and finishes on its own thread; keep per-run handlers free of that.

What a run leaves behind. Nothing of its own: the session state is under the
temporary directory. On Claude the CLI is also started with
``--no-session-persistence``, so the conversation (prompts, replies, tool
arguments and results) is not written under ``~/.claude/projects``.

Tools that ask the human themselves (``tools.asks_the_human``) are refused by
``ToolSet``: with no human they could never be answered, and the tool server
would otherwise file them under read.

Imported on demand (``annealage_agent.run_prompt`` and the other two names
resolve lazily), because it imports the Claude SDK through ``tools``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import sys
import tempfile
from typing import Any, List, Optional, Tuple

from . import product
from .session.base import (
    AGENT_READY,
    AGENT_UNAVAILABLE,
    AgentError,
    AgentStatus,
    TextDelta,
    ToolResult,
    ToolUse,
    TurnEnd,
)
from .session.permissions import HeadlessBroker
from .tools import _ASKS_THE_HUMAN, Grading, ToolServer, _verify, fail, namespaced
from .viewers import ViewerBus, ViewerRegistry

#: The backends ``run_prompt`` can drive.
BACKENDS = ("omp", "claude")

#: What a pause-gated tool answers while the view is paused. A headless run has
#: no human to pause it, so it is never said; ``ToolServer`` requires it.
_PAUSED_MESSAGE = "this run has no human and cannot be paused"

#: Seconds each step of closing a session (``kill``, then ``close``) may take
#: before ``run_prompt`` gives up waiting for it and returns anyway.
_CLOSE_TIMEOUT = 10.0

#: Seconds ``run_prompt`` waits for the tool handlers it cancelled to finish
#: their cleanup before it returns.
_DRAIN_TIMEOUT = 5.0


@dataclasses.dataclass(frozen=True)
class ToolSet:
    """The tools one headless run may use.

    ``tools`` are ``@tool`` definitions (what a product's ``build_tools``
    passes to ``ToolServer``), ``grading`` the ``tools.Grading`` sorting every
    one of them, and ``grant`` the write-grade names pre-granted for this run.
    Built per call: the handlers close over the caller's per-run data.

    ``ValueError`` for a ``tools``/``grading`` pair ``ToolServer`` would refuse
    (a tool not graded, a graded name with no tool, a name in two grades, a
    name declared twice), or for a ``grant`` entry that is not a write-grade
    tool in the set: a read or view tool needs no grant, and one that is not
    there grants nothing. Needs an installed product (``product.install``), as
    ``ToolServer`` does.
    """

    tools: Tuple[Any, ...]
    grading: Grading
    grant: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "tools", tuple(self.tools))
        grant = (self.grant,) if isinstance(self.grant, str) else self.grant
        object.__setattr__(self, "grant", tuple(grant))
        if not isinstance(self.grading, Grading):
            raise ValueError("grading must be a tools.Grading, not %r" % (self.grading,))
        # Outside the try: no installed product is not a bad tool set.
        product.current()
        try:
            _verify(self.tools, self.grading)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None
        asking = sorted(t.name for t in self.tools if getattr(t.handler, _ASKS_THE_HUMAN, False))
        if asking:
            raise ValueError(
                "tool(s) %s ask the human themselves (tools.asks_the_human), and a headless "
                "run has no human to ask; leave them out of the set" % ", ".join(asking)
            )
        for name in self.grant:
            if name in self.grading.write:
                continue
            if name in self.grading.read or name in self.grading.view:
                raise ValueError(
                    "grant names %r, which is a read or view tool and runs without a grant; "
                    "only write-grade tools are granted" % name
                )
            raise ValueError(
                "grant names %r, which is not a tool in this set; only write-grade tools "
                "in the set are granted" % name
            )


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What one headless run produced.

    ``text`` is what the model said after its last tool result (the whole
    reply when it called no tool; the narration before and between tool calls
    is dropped), and what had arrived so far when the run failed or timed out.
    A turn that ends on a tool call (``ended_by_tool``) has no text.
    ``stop_reason`` is the backend's (``end_turn``, ``max_tokens``, ...) or
    ``ended_by_tool``, ``rejected``, ``timeout`` or ``error``. ``cost_usd`` and
    ``tokens`` are the turn's, as the backend reports them (0.0 and ``None``
    where it does not). ``error`` is ``None`` for a clean run, else why not
    (``"timeout"`` for a timeout).

    ``calls`` is every tool the model called, in order, as ``(name, ok)``:
    the ``ToolSet``'s own name on both backends, and ``ok`` false for a call
    that failed, was refused, or never returned. Neither arguments nor results
    are kept, since they may be confidential.
    """

    text: str
    stop_reason: str
    cost_usd: float
    tokens: Optional[dict]
    error: Optional[str]
    calls: Tuple[Tuple[str, bool], ...] = ()


class _Run:
    """The events of one turn, collected. ``on_event`` is the session's sink and
    only ever runs on the event loop."""

    def __init__(self, server_prefix: str):
        self._prefix = server_prefix
        self.done = asyncio.Event()
        self.text: List[str] = []
        self.errors: List[str] = []
        self.stop_reason: Optional[str] = None
        self.cost_usd = 0.0
        self.tokens: Optional[dict] = None
        # [name, ok] per call in the order made, and where each call id is.
        self._calls: List[list] = []
        self._by_id: dict = {}
        # Set when the run is over, however it ended; a tool handler that starts
        # after that refuses (``guard``), and ``end`` cancels those running.
        self.over = False
        self._inflight: set = set()
        # The session, for what only it knows about how its turn ended.
        self.session: Any = None

    @property
    def finished(self) -> bool:
        return self.done.is_set()

    def guard(self, tool_def):
        """``tool_def`` with its handler made to refuse once the run is over,
        and to be cancelled (at its next ``await``) by ``end``. A late tool call
        from the backend (omp runs each on its own thread and can outlive its
        timeout) then cannot touch the caller's per-run state after
        ``run_prompt`` has returned. A handler already past its last ``await``
        cannot be stopped, and one blocking inside a synchronous call runs on."""
        inner = tool_def.handler

        async def handler(args):
            if self.over:
                return fail("this run has ended, so %s did not run" % tool_def.name)
            task = asyncio.current_task()
            self._inflight.add(task)
            try:
                return await inner(args)
            finally:
                self._inflight.discard(task)

        return dataclasses.replace(tool_def, handler=handler)

    def end(self) -> None:
        """The run is over: refuse new tool calls and cancel the running ones.
        Idempotent; ``drain`` waits for the cancellations to finish."""
        self.over = True
        for task in list(self._inflight):
            task.cancel()

    async def drain(self) -> None:
        """Wait, up to ``_DRAIN_TIMEOUT``, for the handlers ``end`` cancelled to
        finish their cleanup, so none runs on after ``run_prompt`` returns.
        One that will not stop in that time is abandoned."""
        pending = [t for t in self._inflight if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=_DRAIN_TIMEOUT)

    def error(self) -> Optional[str]:
        return "; ".join(self.errors) or None

    def calls(self) -> Tuple[Tuple[str, bool], ...]:
        return tuple((name, ok) for name, ok in self._calls)

    def on_event(self, event) -> None:
        if self.finished:
            return
        if isinstance(event, TextDelta):
            self.text.append(event.text)
        elif isinstance(event, ToolUse):
            name = event.name
            if name.startswith(self._prefix):
                name = name[len(self._prefix) :]
            self._by_id[event.tool_use_id] = len(self._calls)
            self._calls.append([name, False])
        elif isinstance(event, ToolResult):
            # What was said so far was before the tool; the answer follows it.
            self.text.clear()
            index = self._by_id.get(event.tool_use_id)
            if index is not None:
                self._calls[index][1] = not event.is_error
        elif isinstance(event, AgentError):
            message = (event.stderr or event.remediation or "").strip()
            if message and message not in self.errors:
                self.errors.append(message)
        elif isinstance(event, AgentStatus):
            if event.status == AGENT_UNAVAILABLE:
                # After this call returns: the session emits its AgentError
                # right behind the status change.
                asyncio.get_running_loop().call_soon(self._stopped)
        elif isinstance(event, TurnEnd):
            if event.stop_reason == "steered":
                # omp: another message redirected the turn; the real end follows.
                return
            self.stop_reason = event.stop_reason
            self.cost_usd = event.cost_usd
            self.tokens = event.tokens
            # A Claude result that said it failed (an API error, say) is still a
            # TurnEnd on the wire; the session keeps why, read here.
            failure = getattr(self.session, "last_result_error", None)
            if failure:
                self.errors.append(failure)
                self.stop_reason = "error"
            self.done.set()

    def _stopped(self, why: str = "the agent stopped before answering") -> None:
        """The session went unavailable: no turn end is coming."""
        if self.finished:
            return
        self.stop_reason = "error"
        if not self.errors:
            self.errors.append(why)
        self.done.set()

    def result(self, *, stop_reason: Optional[str] = None, error: Optional[str] = None):
        return RunResult(
            text="".join(self.text),
            stop_reason=stop_reason or self.stop_reason or "error",
            cost_usd=self.cost_usd,
            tokens=self.tokens,
            error=error if error is not None else self.error(),
            calls=self.calls(),
        )


def _check_arguments(prompt, tools, backend, model, cwd, timeout):
    if backend == "codex":
        raise ValueError(
            "run_prompt does not support the codex backend: it reaches its tools "
            "through the app's /mcp endpoint, which a headless run has no listener "
            "for; use backend='omp' or 'claude'"
        )
    if backend not in BACKENDS:
        raise ValueError("backend must be one of %s, not %r" % (", ".join(BACKENDS), backend))
    if not isinstance(tools, ToolSet):
        raise ValueError("tools must be a ToolSet, not %r" % (tools,))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string: a headless run has no default")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        raise ValueError("timeout must be a number of seconds greater than 0, not %r" % (timeout,))
    # A str or path only: os.path.isdir takes an int for a file descriptor.
    try:
        is_directory = isinstance(cwd, (str, os.PathLike)) and os.path.isdir(cwd)
    except ValueError:
        is_directory = False
    if not is_directory:
        raise ValueError("cwd must be an existing directory, not %r" % (cwd,))


def _build_session(
    backend,
    on_event,
    *,
    broker,
    server,
    allowed_tools,
    model,
    cwd,
    state_dir,
    instructions,
    omp_agent_dir,
    omp_binary,
    omp_config_dir,
):
    """The session ``backend`` names, built and not started. ``launch.build_session``
    is the app's: it is made of the served directory (the working directory and
    the state directory are one there), resume, settings and the product's own
    context, none of which a headless run has, so the two classes are built
    directly with only what a run needs."""
    session_id = "headless"
    if backend == "omp":
        from .session.omp import OmpSession

        return OmpSession(
            on_event,
            cwd=cwd,
            session_id=session_id,
            broker=broker,
            model=model,
            tool_table=server.host_tool_table(),
            instructions=instructions,
            agent_dir=omp_agent_dir,
            config_dir=omp_config_dir,
            binary=omp_binary,
            # The conversation file goes under the run's own state, and goes
            # with it.
            session_dir=os.path.join(state_dir, "omp"),
        )

    from .session.sdk import SdkSession

    return SdkSession(
        on_event,
        cwd=cwd,
        session_id=session_id,
        broker=broker,
        model=model,
        mcp_servers=server.mcp_servers,
        allowed_tools=allowed_tools,
        instructions=instructions,
        # No shell to contain, no built-in tools, no settings files, and no
        # transcript left under ~/.claude/projects: the model has the ToolSet
        # and nothing else, and the run leaves nothing behind (module docstring).
        sandbox=False,
        builtin_tools=(),
        setting_sources=(),
        persist_session=False,
    )


async def _drive(session, run: _Run, prompt: str) -> None:
    """Start the session and send the one turn. Never raises for a backend
    failure: that is left in ``run``, which the turn's end (or the failure)
    completes."""
    try:
        await session.start()
        if session.agent_status() != AGENT_READY:
            run._stopped("the agent did not start")
            return
        await session.submit_turn([{"type": "text", "text": prompt}])
    except Exception as exc:
        run.errors.append("%s: %s" % (type(exc).__name__, exc))
        run._stopped()


async def _cancelled(task: "asyncio.Future") -> None:
    """Cancel ``task`` and wait for it to be gone."""
    if task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _close(session, *, kill: bool) -> None:
    """Close ``session``, never for longer than ``_CLOSE_TIMEOUT``. ``kill``
    first ends a backend process that may be stuck behind a blocked call (omp
    mid-start), so the close has nothing to wait for."""
    try:
        if kill and hasattr(session, "kill"):
            await asyncio.wait_for(session.kill(), _CLOSE_TIMEOUT)
        await asyncio.wait_for(session.close(), _CLOSE_TIMEOUT)
    except asyncio.TimeoutError:
        sys.stderr.write("warning: headless session did not close within %gs\n" % _CLOSE_TIMEOUT)
    except Exception as exc:
        sys.stderr.write("warning: headless session did not close cleanly: %r\n" % (exc,))


async def run_prompt(
    prompt,
    *,
    tools,
    backend,
    model,
    cwd,
    timeout,
    system=None,
    omp_agent_dir=None,
    omp_binary=None,
    omp_config_dir=None,
) -> RunResult:
    """Run ``prompt`` as one turn of a fresh agent session with ``tools``, and
    return its ``RunResult``. See the module docstring.

    ``backend`` is ``"omp"`` or ``"claude"``; ``model`` is required (omp's
    ``provider/model``, or a Claude model name). ``cwd`` is an existing
    directory, the caller's scratch space, used only as the backend's working
    directory. ``timeout`` is seconds for the whole run, start included: when it
    passes the session is closed and the result is ``error="timeout"``.
    ``system`` is appended to the backend's system prompt. ``omp_agent_dir``,
    ``omp_binary`` and ``omp_config_dir`` are ``launch.build_session``'s omp
    profile, executable and config root, ``None`` for omp as installed.

    Raises ``ValueError`` for a bad argument, ``CancelledError`` if cancelled
    (the session closed first); never for a model or backend failure.
    """
    _check_arguments(prompt, tools, backend, model, cwd, timeout)

    with tempfile.TemporaryDirectory(prefix="annealage-headless-") as state_dir:
        bus = ViewerBus(ViewerRegistry(), url="")
        run = _Run(namespaced(product.current().mcp_server_name, ""))
        server = ToolServer(
            [run.guard(t) for t in tools.tools],
            grading=tools.grading,
            bus=bus,
            paused_message=_PAUSED_MESSAGE,
        )
        # Claude asks under ``mcp__<server>__<tool>``; omp's host-tool gate under the bare name.
        claude_grants = tuple(namespaced(server.name, g) for g in tools.grant)
        broker = HeadlessBroker(run.on_event, granted=tuple(tools.grant) + claude_grants)
        # The channel a tool that reads its broker uses (app.py's, launch.py's).
        bus.tools = server
        bus.broker = broker
        try:
            try:
                session = _build_session(
                    backend,
                    run.on_event,
                    broker=broker,
                    server=server,
                    allowed_tools=server.pre_allowed + claude_grants,
                    model=model,
                    cwd=cwd,
                    state_dir=state_dir,
                    instructions=system or None,
                    omp_agent_dir=omp_agent_dir,
                    omp_binary=omp_binary,
                    omp_config_dir=omp_config_dir,
                )
            except Exception as exc:
                # The backend's package missing (omp_rpc is installed apart), say.
                return run.result(stop_reason="error", error="%s: %s" % (type(exc).__name__, exc))
            run.session = session
            bus.end_turn_handler = getattr(session, "end_turn_after_tool", None)

            task = asyncio.ensure_future(_drive(session, run, prompt))
            waiter = asyncio.ensure_future(run.done.wait())
            try:
                await asyncio.wait({waiter}, timeout=timeout)
                if not run.finished:
                    return run.result(stop_reason="timeout", error="timeout")
                return run.result()
            finally:
                # Also when cancelled: nothing outlives the call. Late tool calls
                # are refused first, then the backend is stopped.
                unfinished = not run.finished
                run.end()
                broker.shutdown()
                await _cancelled(waiter)
                await _cancelled(task)
                await _close(session, kill=unfinished)
                await run.drain()
        finally:
            run.end()
