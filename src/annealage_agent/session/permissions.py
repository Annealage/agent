"""``can_use_tool`` permission broker: one ``asyncio.Future`` per outstanding
request, project-scoped allow-always grants, and a provider-neutral
``Decision`` that each backend's own session adapter maps onto whatever its
SDK expects. ``session/sdk.py``'s ``_to_claude_result`` is the Claude SDK's
adapter: it maps a remembered grant onto ``PermissionUpdate`` so that SDK's
own rule matcher can take over for the rest of a running session; other
backends have their own, different remembered-grant mechanisms and write
their own adapter instead (see ``planning/roadmap.md``).

A bug here means a tool ran that the human never approved, so every path
through ``ask`` returns a ``Decision`` and nothing here ever raises out of
``ask`` itself (fact 14 in ``docs/agent-chat-plan.md`` section 2a): a
returned dict, or an uncaught exception, both surface to the model as an
infrastructure error rather than a decision, which is worse than a denial
with a clear reason.

Calling contract, mirroring ``viewers.py``'s own documented pattern rather
than inventing a second one:

- ``session/sdk.py`` binds ``options.can_use_tool`` to its own thin adapter,
  which calls ``ask`` and converts the ``Decision`` it returns into what the
  Claude SDK requires; the SDK calls it once per tool invocation that its own
  rules evaluate to "ask" (fact 2: ``allowed_tools`` and ``permissions.allow``
  rules never reach it at all).
- Whatever dispatches an inbound ``permission`` frame (``ws.py`` via
  ``AgentSession.decide_permission``) calls ``decide`` with the frame's
  ``request_id``, ``decision`` and ``message``.
- Whatever tracks connection count (the same wiring that calls
  ``ViewerRegistry.add``/``remove``) calls ``viewer_connected`` and
  ``viewer_disconnected`` on the same transitions, so ``ask`` knows whether
  anyone exists to answer before it ever creates a request, and a request
  already outstanding is denied shortly after its last possible answerer
  leaves rather than at the end of a timeout that teaches nothing new.
  Shortly, not immediately: a page reload is a disconnect followed by a
  connect, so see ``DEFAULT_NO_VIEWER_GRACE``.
- ``pending_requests`` reports what is still outstanding. Nothing in the
  reconnection path needs it: a viewer that reconnects replays the
  ``permission_request`` and ``permission_resolved`` pair from the
  ``EventLog``, and a request with no resolution in that replay is exactly
  the card it should show. It is an inspection method, and what the tests
  assert "nothing is left pending" against.
- The shutdown sequence (plan section 3.4) calls ``shutdown`` once.

Every request this broker opens is announced twice: a ``PermissionRequest``
when it starts waiting, and exactly one ``PermissionResolved`` when it stops,
whichever of the six paths ended it. The pair is what lets a browser that did
not send the deciding frame retire the card, and what stops replay
reconstructing an answered request as a live one on every reload. ``decide``
raising ``UnknownRequest`` is the other half of the same problem: a second tab
answering an already-answered card is owed that reply rather than silence.

Nothing here reads or writes a socket, holds a ``ViewerRegistry``, or
touches ``<state dir>/config.toml`` or the user's ``settings.toml``; those are
reached through ``on_event`` and the four methods above, all of it
call-and-return with no reference back into this module's caller.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

from .. import product
from .base import AgentEvent, PermissionRequest, PermissionResolved, UnknownRequest

try:
    import tomllib  # Python 3.11+, stdlib, read-only.
except ImportError:  # Python 3.10: no tomllib, and this project takes no TOML dependency.
    tomllib = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class Decision:
    """The provider-neutral result of one permission decision: whatever
    ``ask``/``decide`` produce, and every backend's own session adapter
    (``session/sdk.py``'s ``_to_claude_result``, for the Claude SDK) maps
    onto its own SDK's result type from here, so this module never has to
    know what that type is.

    ``remember_tool`` is set only on an ``allow_always`` grant that was not
    downgraded by ``NEVER_REMEMBERED``, naming the tool a backend's own
    remembered-grant mechanism (the Claude SDK's session-scoped
    ``PermissionUpdate``, Codex's ``acceptForSession``, or omp's broker-side
    interception) should also cover for the rest of this running session.
    """

    allow: bool
    remember_tool: Optional[str] = None
    message: str = ""


# A human reviewing a running agent may be away from the keyboard for a
# while without that meaning "no"; a session nobody is watching should
# still not hold one tool call open forever. Five minutes is long enough
# for the first and short enough for the second. Always injectable, per
# the M5 brief: tests pass a small value, never sleep past a real one.
DEFAULT_TIMEOUT = 300.0

# How long the last viewer leaving is treated as possibly temporary before the
# requests nobody is left to answer are denied.
#
# Reloading the page is a disconnect followed by a connect, and with one browser
# open, which is the ordinary case, denying the instant the count reached zero
# meant a reload cancelled whatever was being asked. The human came back to a
# pane with no card and the model to a refusal it had done nothing to earn.
#
# A few seconds separates a reload from someone closing the laptop: long enough
# that a page coming back finds its request still waiting, short enough that a
# genuinely abandoned session is not held open on the chance somebody returns.
# The five-minute timeout is what covers the case of a browser that never
# returns at all.
DEFAULT_NO_VIEWER_GRACE = 8.0

# Never persisted and never covered by an allow-always grant (plan section
# 5): a broad standing grant for the one tool with unrestricted shell
# access would turn one careless click into a permanent bypass. Enforced
# here, not only in the pane, because a compromised or buggy client could
# send ``allow_always`` for this tool regardless of what the UI offers.
NEVER_REMEMBERED = frozenset({"Bash"})

DEFAULT_DENY_MESSAGE = "the human reviewing this session denied the request"

# The three templates below name the product (``%(product)s``, its lowercase
# name), because "reopen the viewer" alone does not say which one to a human
# running more than one of these tools.
_DENY_TIMEOUT_TEMPLATE = (
    "no decision was received within %(timeout)g seconds and the request has expired; "
    "if this tool call is still needed, ask the human to reopen the %(product)s "
    "viewer and try again"
)

_DENY_ALL_GONE_TEMPLATE = (
    "every browser viewer disconnected while this request was awaiting a "
    "decision; ask the human to reopen the %(product)s viewer and try again"
)

_DENY_SHUTDOWN_TEMPLATE = (
    "the %(product)s session is shutting down and cannot ask a human to decide "
    "this; the request was not approved"
)


def _deny_message(template, **values):
    return template % dict(values, product=product.current().name)


_DENY_EMIT_FAILED_TEMPLATE = (
    "could not deliver this permission request to the browser due to an "
    "internal error; the request was not approved"
)


OUTCOME_TIMEOUT = "timeout"
OUTCOME_NO_VIEWER = "no_viewer"
OUTCOME_SHUTDOWN = "shutdown"


class PermissionBroker:
    """One broker per running agent session. See the module docstring for
    the full calling contract."""

    def __init__(
        self,
        on_event: Callable[[AgentEvent], None],
        *,
        permissions_path: Optional[Union[str, Path]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        viewer_url: Optional[str] = None,
        no_viewer_grace: float = DEFAULT_NO_VIEWER_GRACE,
    ):
        self._on_event = on_event
        self._permissions_path = Path(permissions_path) if permissions_path is not None else None
        self._timeout = timeout
        self._viewer_url = viewer_url
        self._no_viewer_grace = no_viewer_grace
        self._viewer_count = 0
        # The pending call_later that will deny everything if no viewer comes
        # back, or None when at least one viewer is present or the grace period
        # has already elapsed.
        self._no_viewer_timer = None
        self._shutdown = False
        self._next_id = 1
        # One future per outstanding request, keyed by the id ``ask``
        # allocates; cleaned up in ``ask``'s own ``finally`` regardless of
        # how the request ends, mirroring viewers.py's ``call``.
        self._pending: Dict[str, asyncio.Future] = {}
        # The PermissionRequest event for each entry above, kept alongside
        # it so decide() can look up which tool a request_id was for
        # without ask() having to pass it back out of band, and so
        # pending_requests() has something to hand a reconnecting viewer.
        self._open: Dict[str, PermissionRequest] = {}
        # How each outstanding request ended, written by whichever path
        # resolved it and read by ``ask``'s ``finally``, which is the one place
        # every resolution passes through and therefore the one place a
        # ``PermissionResolved`` can be emitted exactly once per request
        # without each of the six resolution paths remembering to.
        self._outcomes: Dict[str, str] = {}
        # Serialises the disk write inside _remember: two allow_always
        # decisions racing on the executor could otherwise complete out of
        # order and leave the file holding a smaller set than memory,
        # transiently losing a grant on disk (never in this process's own
        # memory, and never a false grant, only ever an under-write that a
        # later decision or restart-time re-prompt self-heals). See
        # _remember's docstring.
        self._write_lock = asyncio.Lock()
        self._granted_tools: FrozenSet[str] = _load_grants(self._permissions_path)

    # -- can_use_tool itself ----------------------------------------------

    async def ask(self, tool_name: str, input_data: dict, context: Any) -> Decision:
        """The ``can_use_tool`` callback: every backend's session adapter
        calls this directly and converts the ``Decision`` it returns into
        whatever its own SDK expects (``session/sdk.py``'s
        ``_to_claude_result``, for the Claude SDK). That conversion must
        itself never raise or return anything but a value of the target
        SDK's own result type, or it defeats the one guarantee this module
        exists to give (fact 14: a returned dict or a raised exception both
        read to the model as the permission system erroring, not as a
        decision) - this method's own "never raises, always returns a
        decision" contract is only as good as the adapter on the other end
        of it.

        ``context`` is an opaque, backend-defined value (a Claude SDK
        ``ToolPermissionContext`` for ``session/sdk.py``) this broker does
        not act on; it is accepted only to match each backend's own
        permission-callback signature, and every field beyond ``tool_name``
        and ``input_data`` reaches the pane, if at all, through the
        ``permission_request`` event this method emits, which does not
        currently carry them (plan section 3.3's wire example has none).

        Every branch below returns a ``Decision``, and the two checks
        before a request is ever created (already granted, no viewer to
        ask) run with no ``await`` between them and this method's entry, so
        nothing else on this event loop can change either answer out from
        under this call before it commits to one.
        """
        if self._shutdown:
            return Decision(allow=False, message=_deny_message(_DENY_SHUTDOWN_TEMPLATE))
        if tool_name in self._granted_tools:
            return Decision(allow=True, remember_tool=tool_name)
        if self._viewer_count == 0:
            return Decision(allow=False, message=self._no_viewer_message())

        request_id = "pr_%d" % self._next_id
        self._next_id += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        event = PermissionRequest(request_id=request_id, tool=tool_name, input=input_data)
        self._pending[request_id] = future
        self._open[request_id] = event
        try:
            try:
                self._on_event(event)
            except Exception as exc:
                # on_event is the wiring layer's own broadcast path
                # (EventLog.append plus ViewerRegistry.broadcast); a bug
                # there must not turn into this coroutine raising, which
                # is exactly the infrastructure-error failure fact 14
                # warns against. The request never reached a browser, so
                # there is nothing to wait for; deny at once instead of
                # awaiting a future nothing will ever resolve.
                sys.stderr.write(
                    "error: failed to emit permission_request %r for %r: %r\n"
                    % (request_id, tool_name, exc)
                )
                return Decision(allow=False, message=_DENY_EMIT_FAILED_TEMPLATE)
            try:
                return await asyncio.wait_for(future, timeout=self._timeout)
            except asyncio.TimeoutError:
                # asyncio.wait_for has already cancelled `future` (it
                # wraps any bare Future the same way it would a Task);
                # this method's own Deny is returned directly rather than
                # inspecting that future's now-cancelled state, so a
                # decide() that was mid-flight when the deadline hit and
                # loses the race against it is told "already decided" by
                # its own done() check, not by a value read off a future
                # this method has already stopped trusting.
                self._outcomes[request_id] = OUTCOME_TIMEOUT
                return Decision(
                    allow=False,
                    message=_deny_message(_DENY_TIMEOUT_TEMPLATE, timeout=self._timeout),
                )
        finally:
            self._pending.pop(request_id, None)
            self._open.pop(request_id, None)
            outcome = self._outcomes.pop(request_id, OUTCOME_SHUTDOWN)
            # After the pops, so a viewer that reconnects on the strength of
            # this event and asks for what is still outstanding cannot be told
            # about the request this event just closed.
            try:
                self._on_event(PermissionResolved(request_id=request_id, outcome=outcome))
            except Exception as exc:
                # The tool call has already been answered by this point, so a
                # failure to announce that costs a card left on screen until
                # the next reload, not a wrong decision.
                sys.stderr.write(
                    "warning: failed to emit permission_resolved %r (%s): %r\n"
                    % (request_id, outcome, exc)
                )

    # -- resolving a request ------------------------------------------------

    async def decide(self, request_id: str, decision: str, message: str = "") -> None:
        """Resolve one outstanding request. Called for an inbound
        ``permission`` frame (plan section 3.3), dispatched here via
        ``AgentSession.decide_permission``.

        Raises ``UnknownRequest`` if ``request_id`` names nothing pending;
        see that exception's docstring for the four ways that happens.

        The lookup-and-check above, and the ``future.set_result`` that
        answers the request, happen with no ``await`` between them: two
        ``decide`` calls racing the same ``request_id`` (two tabs both
        holding the same card, or a human's click racing the timeout
        inside ``ask``) can therefore only ever observe the future
        strictly before or strictly after this method resolves it, never
        mid-resolution, so the loser's ``future.done()`` check always sees
        a definite answer. Without that guarantee, a second ``decide``
        that read ``done() is False`` before the first one's own
        ``set_result`` would call ``set_result`` again itself once it got
        the chance to run, and ``asyncio.Future.set_result`` on an
        already-done future raises ``InvalidStateError`` rather than
        losing gracefully, which would surface as an unhandled error deep
        inside whatever task happened to run second, not as the ordinary
        "already decided" reply the loser is owed. This is why the disk
        write for ``allow_always`` happens *after* this section, in
        ``_remember``, rather than before ``future.set_result``:
        persistence is not on the critical path of answering the tool
        call this decision unblocks, and putting it before the resolve
        would reintroduce exactly the ``await``-shaped gap this method is
        written to avoid.
        """
        future = self._pending.get(request_id)
        if future is None or future.done():
            raise UnknownRequest(
                "permission request %r was already decided or does not exist" % (request_id,)
            )
        event = self._open.get(request_id)
        tool_name = event.tool if event is not None else ""
        result, grant = _build_result(tool_name, decision, message)
        self._outcomes[request_id] = decision
        future.set_result(result)
        if grant is not None:
            await self._remember(grant)

    async def _remember(self, tool_name: str) -> None:
        """Add ``tool_name`` to the in-memory grant set ``ask`` consults
        first, and persist it to ``<state dir>/permissions.toml`` if this
        broker was given a path.

        The in-memory set is kept independently of the SDK's own
        in-session rule matching (the ``updated_permissions`` this
        broker's ``ask`` and this method's caller both return): if a
        future SDK version's live rule matching ever failed to cover a
        call this grant should have, ``ask``'s first check still catches
        it, since it never has to trust a second system's memory of a
        decision this process already made.

        The disk write is serialised by ``self._write_lock`` and always
        writes the *current* ``self._granted_tools`` at the moment it
        acquires the lock, not a value captured earlier: two grants
        arriving close together therefore never race each other's write
        to completion out of order, which could otherwise leave the file
        holding only the first grant even though memory holds both.
        """
        if tool_name in self._granted_tools:
            return
        self._granted_tools = self._granted_tools | {tool_name}
        if self._permissions_path is None:
            return
        async with self._write_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, _write_grants, self._permissions_path, self._granted_tools
            )

    # -- viewer presence ------------------------------------------------------

    def viewer_connected(self) -> None:
        """Record one browser connection becoming available to answer a
        request. Call once per successful WebSocket upgrade, on the same
        transition that calls ``ViewerRegistry.add``, so ``ask`` stops
        denying immediately once at least one viewer exists to ask.

        Cancels any pending no-viewer denial, which is what makes a reload
        keep an outstanding request rather than cancel it.
        """
        self._viewer_count += 1
        self._cancel_no_viewer_timer()

    def viewer_disconnected(self) -> None:
        """Record one browser connection going away, on the same
        transition that calls ``ViewerRegistry.remove``. Never lets the
        count go negative, so a caller that reports one disconnection
        twice cannot make a later, real connection look like it was never
        registered.

        When this drops the count to zero, every request still awaiting a
        decision is denied after ``no_viewer_grace`` seconds: waiting out the
        rest of ``ask``'s five-minute timeout to learn that nobody is left to
        answer teaches the model nothing a prompt denial would not, the same
        reasoning ``viewers.py``'s own ``ViewerGone`` applies to a stalled tool
        call. This is what keeps a request from outliving every viewer that
        could have answered it, rather than merely from outliving one
        particular viewer.

        The grace period is what separates "the human reloaded the page" from
        "the human left". Denying on the transition itself made a reload cancel
        whatever was being asked, since a reload is a disconnect followed a
        moment later by a connect.

        Scheduled on the running loop when there is one. Without a loop there is
        nothing to schedule and nothing to wait for, so the denial happens
        immediately, which is the behaviour a synchronous caller can observe and
        the safe direction besides.
        """
        self._viewer_count = max(0, self._viewer_count - 1)
        if self._viewer_count > 0:
            return
        self._cancel_no_viewer_timer()
        if self._no_viewer_grace <= 0:
            self._deny_all_pending(_deny_message(_DENY_ALL_GONE_TEMPLATE), OUTCOME_NO_VIEWER)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._deny_all_pending(_deny_message(_DENY_ALL_GONE_TEMPLATE), OUTCOME_NO_VIEWER)
            return
        self._no_viewer_timer = loop.call_later(self._no_viewer_grace, self._deny_for_no_viewer)

    def _deny_for_no_viewer(self) -> None:
        """The grace period elapsed with nobody back. Guarded on the count
        because a viewer that reconnected cancels the timer, and a timer already
        firing cannot be cancelled."""
        self._no_viewer_timer = None
        if self._viewer_count == 0:
            self._deny_all_pending(_deny_message(_DENY_ALL_GONE_TEMPLATE), OUTCOME_NO_VIEWER)

    def _cancel_no_viewer_timer(self) -> None:
        if self._no_viewer_timer is not None:
            self._no_viewer_timer.cancel()
            self._no_viewer_timer = None

    def _no_viewer_message(self) -> str:
        if self._viewer_url:
            return (
                "no browser viewer is connected to decide this permission request; "
                "ask the human to open %s, then try again" % self._viewer_url
            )
        return (
            "no browser viewer is connected to decide this permission request; "
            "ask the human to open the %s viewer, then try again" % product.current().name
        )

    # -- replay and shutdown -----------------------------------------------

    def pending_requests(self) -> List[PermissionRequest]:
        """Every ``permission_request`` still awaiting a decision, for a
        freshly connected or reconnected viewer.

        These are the same event objects originally passed to
        ``on_event``, not a re-fetch from the ``EventLog``: a request
        outstanding long enough to fall off the log's 500-event ring is
        exactly the case a plain seq replay cannot recover, and a request
        already answered is absent here the instant ``decide``, a
        timeout, ``viewer_disconnected`` reaching zero, or ``shutdown``
        resolves it, so a stale card is never handed to a fresh
        connection. The caller decides how to deliver these (send
        directly to the one new connection, or re-broadcast through
        ``on_event`` so every tab's log gains a fresh entry); this method
        only reports what is still open.
        """
        return list(self._open.values())

    def shutdown(self) -> None:
        """Deny every outstanding request and make every future ``ask``
        call deny immediately, for process shutdown (plan section 3.4).
        Idempotent: a second call finds nothing left pending and a flag
        already set.

        Cancels any pending no-viewer denial first: the process is going, so
        there is nothing left to wait for a browser to come back to, and a timer
        outliving the loop it was scheduled on is worth not leaving behind.
        """
        self._shutdown = True
        self._cancel_no_viewer_timer()
        self._deny_all_pending(_deny_message(_DENY_SHUTDOWN_TEMPLATE), OUTCOME_SHUTDOWN)

    def _deny_all_pending(self, message: str, outcome: str) -> None:
        for request_id, future in list(self._pending.items()):
            if not future.done():
                self._outcomes[request_id] = outcome
                future.set_result(Decision(allow=False, message=message))


# ---------------------------------------------------------------------------
# Decision application, pure and synchronous (see decide()'s docstring for
# why this must never await).
# ---------------------------------------------------------------------------


def _build_result(tool_name: str, decision: str, message: str) -> Tuple[Decision, Optional[str]]:
    """The ``Decision`` for one decision, and the tool name to remember
    afterward (``None`` if nothing should be persisted)."""
    if decision == "allow":
        return Decision(allow=True), None
    if decision == "deny":
        return Decision(allow=False, message=message or DEFAULT_DENY_MESSAGE), None
    if decision == "allow_always":
        if tool_name in NEVER_REMEMBERED:
            # The pane should never offer this button for this tool, but
            # this broker does not trust that it never will: downgraded to
            # a one-time allow rather than denied outright, because the
            # human already said "allow this call" and refusing the call
            # itself over a labelling mismatch would be a surprise in the
            # wrong direction. What is refused is only the memory of it.
            sys.stderr.write(
                "warning: allow_always for %r downgraded to a one-time allow; "
                "%r is never remembered (plan section 5)\n" % (tool_name, tool_name)
            )
            return Decision(allow=True), None
        return Decision(allow=True, remember_tool=tool_name), tool_name
    raise ValueError("unknown permission decision: %r" % (decision,))


# ---------------------------------------------------------------------------
# <state dir>/permissions.toml: load at construction, write on every new grant.
# ---------------------------------------------------------------------------

_GRANTS_KEY = "allow_always_tools"

_FALLBACK_ARRAY_RE = re.compile(_GRANTS_KEY + r"\s*=\s*\[(.*?)\]", re.DOTALL)
_FALLBACK_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _load_grants(path: Optional[Path]) -> FrozenSet[str]:
    """The allow-always grants recorded in ``path``, or an empty set if
    there is no path, no file yet, or the file cannot be read or parsed.

    Every failure here falls back to "no grants", never to raising:
    losing a remembered grant only costs one re-prompt for a tool the
    human already trusted, which is the safe direction (fact 17's own
    reasoning: fail toward more asking, not toward less). ``Bash`` is
    filtered out even if present, so a hand-edited file cannot reintroduce
    the one grant this module refuses to ever create itself.
    """
    if path is None:
        return frozenset()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return frozenset()
    except OSError as exc:
        sys.stderr.write(
            "warning: could not read %s: %r; starting with no remembered grants\n" % (path, exc)
        )
        return frozenset()
    try:
        tools = _parse_grants(text)
    except ValueError as exc:
        sys.stderr.write(
            "warning: could not parse %s: %r; starting with no remembered grants\n" % (path, exc)
        )
        return frozenset()
    return frozenset(t for t in tools if t not in NEVER_REMEMBERED)


def _parse_grants(text: str) -> List[str]:
    """Extract the ``allow_always_tools`` array from ``text``, the shape
    ``_write_grants`` produces.

    Uses the stdlib ``tomllib`` when available (Python 3.11+). On 3.10,
    where ``tomllib`` does not exist and this project takes no TOML
    dependency, falls back to a parser for exactly this module's own
    shape: one ``allow_always_tools = [...]`` assignment, one quoted
    string per element. A file in that shape, however reordered,
    recommented or reformatted, still parses under the fallback; a file
    using TOML features this shape has no need for (inline tables,
    multiline strings, a differently named key) does not, and is treated
    by the caller as unreadable rather than partially trusted.
    """
    if tomllib is not None:
        data = tomllib.loads(text)
        tools = data.get(_GRANTS_KEY, [])
    else:
        tools = _parse_grants_fallback(text)
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise ValueError("%s must be an array of strings" % _GRANTS_KEY)
    return tools


def _parse_grants_fallback(text: str) -> List[str]:
    match = _FALLBACK_ARRAY_RE.search(text)
    if match is None:
        return []
    return [
        s.replace('\\"', '"').replace("\\\\", "\\")
        for s in _FALLBACK_STRING_RE.findall(match.group(1))
    ]


def _write_grants(path: Path, tools: FrozenSet[str]) -> None:
    """Write ``tools`` to ``path`` as valid, hand-editable TOML.

    Runs on the executor thread ``_remember`` schedules it on, not the
    event loop, matching the project's rule that blocking file IO never
    runs inline on the loop (plan section 3.1); this file is a handful of
    lines, but the rule is simplest to keep by never making an exception
    for "this one is small".

    Written to a sibling temp file and renamed into place, which on POSIX
    is atomic with respect to any concurrent open of ``path`` by name: a
    reader (this module's own ``_load_grants``, at the next process
    start) either sees the old complete file or the new complete one,
    never a half-written one from a write in progress.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Allow-always permission grants for this project (%s)." % product.current().distribution,
        "# Managed by %s; hand-editing is safe, this is plain TOML."
        % product.current().distribution,
        "# Bash is never recorded here (plan section 5); an entry here would",
        "# be ignored on load regardless.",
        "%s = [" % _GRANTS_KEY,
    ]
    for tool in sorted(tools):
        lines.append("    %s," % _toml_string(tool))
    lines.append("]")
    lines.append("")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return '"%s"' % escaped
