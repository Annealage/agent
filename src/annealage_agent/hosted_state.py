"""Hosted durable records: what a human wrote, handed to an injected sink.

In hosted mode the worker's own files are a deletable cache. What a human
authored (a comment, a status change, a message to the agent, a permission
decision) is also given, as it happens, to ``state_sink(record)``, a plain
callable the hosted side supplies (it appends to the project's record log).
This module knows the record *shape* only; the agent package imports nothing
from the hosted library.

A record is a dict::

    {"record_id": "comment:3", "kind": "comment", "principal_id": "usr_<uuid>",
     "display": "Ada", "payload": {...}, "rev": "<40 hex>", "ts": "...Z"}

``rev`` is present only when the record refers to a revision. ``record_id`` is
stable across retries (derived from the thing recorded, never random), so the
sink can be at-least-once and de-duplicate on it. ``principal_id`` is the
delegated ``usr_<uuid>`` (or ``usr_public``), never a name; ``display`` is a
label snapshot for showing beside the record and is never read for
authorisation. A record whose human has no valid principal (a standalone login,
a malformed id) is not emitted: the hosted log refuses non-legacy records
without one, and a login is not a principal.

The sink is called from an executor thread, so it must be thread-safe. A sink
that raises loses nothing locally and fails no request: the failure is logged,
the record is kept (bounded, in memory) and offered again, in order, before the
next record; after ``MAX_ATTEMPTS`` it is dropped with a warning. That retry
is best effort (a restart forgets it), which is why the sink must be
idempotent by ``record_id`` and the producer's file remains the source to
reconcile from. With no sink every method here is a no-op.

The session's own lifecycle (launch, cancel, end, error) goes to the same sink
as ``session_event`` records (``SessionEvents``), and each provider request's
token counts go to a second, telemetry-only ``usage_sink`` (``usage_event``,
``UsageRecorder``) under the same delivery policy. The usage sink is not tied
to hosted mode: ``create_app`` takes one in hosted mode, and
``headless.run_prompt`` takes one for any omp run (a product's sub-sessions),
so a caller can cross-check the inference proxy's meter. It reports tokens
only, never a cost, and the proxy remains the authoritative meter.
"""

import asyncio
import collections
import hashlib
import re
import sys
import threading
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable, Deque, List, Optional

#: The record kinds the hosted log accepts that this package produces.
KINDS = (
    "comment",
    "comment_status",
    "user_turn",
    "permission_decision",
    "session_event",
    "review",
    "callout",
    "note",
)
PUBLIC_PRINCIPAL_ID = "usr_public"
DISPLAY_LIMIT = 120
#: How long an async caller waits for the sink before carrying on without it.
SINK_TIMEOUT = 10.0
MAX_ATTEMPTS = 5
MAX_PENDING = 1000

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_PRINCIPAL_RE = re.compile("usr_" + _UUID)
_RECORD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_REV_RE = re.compile(r"[0-9a-f]{40}")
# Control characters (C0, DEL, C1), line and paragraph separators, bidi
# overrides and isolates, lone surrogates: what a display must not hold.
_BAD_DISPLAY = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029\u202a-\u202e\u2066-\u2069\ud800-\udfff]")
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime(_TS_FORMAT)


def record_id(*parts) -> str:
    """A stable record id from ``parts`` joined with ``:``. Text outside the
    id charset (or over 128 characters) is replaced by a digest of the same
    parts, so the id is still deterministic and always valid."""
    text = ":".join(str(part) for part in parts)
    if _RECORD_ID_RE.fullmatch(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:32]
    return "%s:%s" % (parts[0], digest)


def clean_display(raw) -> Optional[str]:
    """``raw`` as an attribution label (at most ``DISPLAY_LIMIT`` characters, no
    control characters, nothing around it), or ``None`` when it is not one."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()[:DISPLAY_LIMIT].strip()
    if not value or _BAD_DISPLAY.search(value):
        return None
    if any(unicodedata.category(char).startswith("C") for char in value):
        return None
    return value


def attribution(human) -> Optional[tuple]:
    """``(principal_id, display)`` for ``human`` (an ``identity.Human``), or
    ``None`` when they carry no valid principal. The display falls back to the
    principal id itself, which is clean by construction."""
    principal = getattr(human, "principal_id", None)
    if not isinstance(principal, str):
        return None
    if principal != PUBLIC_PRINCIPAL_ID and not _PRINCIPAL_RE.fullmatch(principal):
        return None
    return principal, clean_display(getattr(human, "name", None)) or principal


def claim_rev(human) -> Optional[str]:
    """The revision the human's delegation is bound to, when it is a full
    commit hash."""
    claims = getattr(human, "hosted_claims", None)
    rev = getattr(claims, "rev", None)
    return rev if isinstance(rev, str) and _REV_RE.fullmatch(rev) else None


def _warn(message: str) -> None:
    sys.stderr.write("warning: %s\n" % message)


class Sink:
    """An injected ``sink(record_or_event) -> None`` with the failure policy in
    the module docstring. ``None`` sink: ``enabled`` is false and ``emit`` does
    nothing."""

    def __init__(self, sink: Optional[Callable[[dict], None]], *, label: str = "state"):
        if sink is not None and not callable(sink):
            raise TypeError("a %s sink is a callable taking one dict" % label)
        self._sink = sink
        self._label = label
        self._lock = threading.Lock()
        self._pending: Deque[List[Any]] = collections.deque()
        self._tasks: set = set()

    @property
    def enabled(self) -> bool:
        return self._sink is not None

    def emit(self, item: dict, key: Optional[str] = None) -> None:
        """Give ``item`` to the sink after any earlier one that failed. Never
        raises. ``key`` (its id) keeps one item from being queued twice."""
        if self._sink is None:
            return
        with self._lock:
            if key is None or all(entry[2] != key for entry in self._pending):
                if len(self._pending) >= MAX_PENDING:
                    dropped = self._pending.popleft()
                    _warn("%s sink: dropped %s, too many undelivered" % (self._label, dropped[2]))
                self._pending.append([item, 0, key])
            kept: Deque[List[Any]] = collections.deque()
            for entry in self._pending:
                try:
                    self._sink(entry[0])
                except Exception as exc:
                    entry[1] += 1
                    _warn(
                        "%s sink failed for %s (attempt %d): %r"
                        % (self._label, entry[2], entry[1], exc)
                    )
                    if entry[1] < MAX_ATTEMPTS:
                        kept.append(entry)
                    else:
                        _warn("%s sink: gave up on %s" % (self._label, entry[2]))
            self._pending = kept

    async def emit_async(self, item: dict, key: Optional[str] = None) -> None:
        """``emit`` on an executor thread, so a slow sink does not hold the
        event loop; the caller carries on after ``SINK_TIMEOUT`` (the delivery
        goes on in its thread)."""
        if self._sink is None:
            return
        loop = asyncio.get_running_loop()
        delivery = loop.run_in_executor(None, self.emit, item, key)
        try:
            await asyncio.wait_for(asyncio.shield(delivery), SINK_TIMEOUT)
        except asyncio.TimeoutError:
            _warn("%s sink still running after %ss; carrying on" % (self._label, SINK_TIMEOUT))

    def emit_soon(self, item: dict, key: Optional[str] = None) -> None:
        """``emit_async`` as a task, for synchronous code on the event loop
        (or ``emit`` inline when there is no running loop)."""
        if self._sink is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.emit(item, key)
            return
        task = loop.create_task(self.emit_async(item, key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait (bounded by ``SINK_TIMEOUT``) for the deliveries ``emit_soon``
        started. Never raises."""
        tasks = list(self._tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=SINK_TIMEOUT)
        if pending:
            _warn("%s sink still delivering after %ss; carrying on" % (self._label, SINK_TIMEOUT))


class StateRecorder:
    """Builds and emits the human-authored records, over an injected sink."""

    def __init__(self, sink: Optional[Callable[[dict], None]]):
        self._sink = Sink(sink, label="state")

    @property
    def enabled(self) -> bool:
        return self._sink.enabled

    def build(
        self,
        kind: str,
        rid: str,
        human,
        payload: dict,
        *,
        rev: Optional[str] = None,
        when: Optional[datetime] = None,
    ) -> Optional[dict]:
        """The record, or ``None`` (warned) when ``human`` has no principal."""
        if kind not in KINDS:
            raise ValueError("unknown record kind %r" % (kind,))
        who = attribution(human)
        if who is None:
            _warn("not recording %s: the request carries no valid principal id" % rid)
            return None
        record = {
            "record_id": rid,
            "kind": kind,
            "principal_id": who[0],
            "display": who[1],
            "payload": payload,
        }
        if rev is not None:
            record["rev"] = rev
        record["ts"] = timestamp(when or now())
        return record

    def _send(self, record: Optional[dict]) -> None:
        if record is not None:
            self._sink.emit(record, record["record_id"])

    async def _send_async(self, record: Optional[dict]) -> None:
        if record is not None:
            await self._sink.emit_async(record, record["record_id"])

    # -- comments (called from the review routes' executor threads) ----------

    def comment(self, human, comment) -> None:
        """A human's new comment (``review.model.Comment``). Its record id is
        ``comment:<store id>``, which a later status record names."""
        if not self.enabled or comment.id is None:
            return
        payload = {
            "id": comment.id,
            "anchor": dict(comment.anchor),
            "text": comment.text,
            "author": comment.author,
        }
        if comment.ref is not None:
            payload["ref"] = comment.ref
        if comment.extra:
            payload["extra"] = dict(comment.extra)
        self._send(
            self.build(
                "comment", record_id("comment", comment.id), human, payload, rev=claim_rev(human)
            )
        )

    def comment_status(self, human, comment, status: str) -> None:
        """The human set ``comment``'s status to ``status`` (``open`` or
        ``resolved``). Its id carries the microsecond it was made at: distinct
        changes get distinct ids, and the id is fixed when the record is built,
        so a retry of this one is the same record."""
        if not self.enabled or comment.id is None:
            return
        when = now()
        rid = record_id("comment_status", comment.id, int(when.timestamp() * 1_000_000))
        payload = {"comment_id": record_id("comment", comment.id), "status": status}
        self._send(self.build("comment_status", rid, human, payload, when=when))

    # -- chat (called from the WebSocket handler) -----------------------------

    async def user_turn(self, human, session_id, turn: int, blocks, client_id=None) -> None:
        if not self.enabled:
            return
        payload = {"session_id": str(session_id), "turn": turn, "blocks": blocks}
        if client_id is not None:
            payload["client_id"] = client_id
        await self._send_async(
            self.build("user_turn", record_id("user_turn", session_id, turn), human, payload)
        )

    async def permission_decision(
        self, human, request_id: str, decision: str, message: str = "", tool: Optional[str] = None
    ) -> None:
        if not self.enabled:
            return
        payload = {"request_id": request_id, "decision": decision}
        if tool:
            payload["tool"] = tool
        if message:
            payload["message"] = message
        await self._send_async(
            self.build(
                "permission_decision", record_id("permission_decision", request_id), human, payload
            )
        )


class SessionEvents:
    """The ``session_event`` records of one hosted app's agent session.

    A launch or a cancel is made by a human, who is named. How a session ends
    by itself (idle close, shutdown) or fails has no human, so it is
    attributed to the principal that last launched, cancelled or sent a turn
    to it (``actor``): the delegation the session runs under, never a name
    invented here. With no actor known nothing is recorded (warned).
    Record ids are ``session_event:<session id>:<event>:<microsecond>``.
    """

    def __init__(self, recorder: StateRecorder, session_id):
        self._recorder = recorder
        self._session_id = session_id
        self.actor = None

    def note(self, human) -> None:
        """``human`` is the principal now driving the session."""
        if attribution(human) is not None:
            self.actor = human

    def _record(self, event: str, detail: Optional[dict], human) -> Optional[dict]:
        if not self._recorder.enabled:
            return None
        when = now()
        rid = record_id("session_event", self._session_id, event, int(when.timestamp() * 1_000_000))
        payload = {"session_id": str(self._session_id), "event": event}
        payload.update(detail or {})
        return self._recorder.build(
            "session_event", rid, human if human is not None else self.actor, payload, when=when
        )

    def emit(self, event: str, detail: Optional[dict] = None, human=None) -> None:
        """Record ``event`` without waiting for the sink (from synchronous code
        on the event loop)."""
        record = self._record(event, detail, human)
        if record is not None:
            self._recorder._sink.emit_soon(record, record["record_id"])

    async def emit_async(self, event: str, detail: Optional[dict] = None, human=None) -> None:
        """Record ``event``, waiting (bounded) for the sink."""
        await self._recorder._send_async(self._record(event, detail, human))


def usage_event(
    *,
    session_id,
    turn_index: int,
    request_n: int,
    backend: str,
    model: Optional[str],
    provider: Optional[str],
    provider_request_id: Optional[str],
    tokens: dict,
    when: Optional[datetime] = None,
) -> dict:
    """One provider request's usage, as ``usage_sink`` receives it::

        {"event_id": "<session>:<turn>:<request n>", "session_id": ..,
         "turn_index": .., "backend": "omp", "model": .., "provider": ..,
         "provider_request_id": null or the provider's own id,
         "tokens_input": .., "tokens_output": .., "tokens_cache_read": ..,
         "tokens_cache_write": .., "tokens_reasoning": .., "ts": ".."}

    Token fields are this request's counts, never running totals, and ``None``
    where the backend did not say (never 0). There is no cost field: the agent
    does not price anything. ``event_id`` is always the position of the
    request in the conversation (a provider's own id is not trusted to be
    unique: some OpenAI-compatible servers repeat one), so a redelivery of this
    event is recognised, and an event without a provider id is still
    identified; ``provider_request_id`` is carried beside it for the meter to
    cross-check against.
    """
    return {
        "event_id": "%s:%s:%s" % (session_id, turn_index, request_n),
        "session_id": str(session_id),
        "turn_index": turn_index,
        "backend": backend,
        "model": model,
        "provider": provider,
        "provider_request_id": provider_request_id,
        "tokens_input": tokens.get("input"),
        "tokens_output": tokens.get("output"),
        "tokens_cache_read": tokens.get("cache_read"),
        "tokens_cache_write": tokens.get("cache_write"),
        "tokens_reasoning": tokens.get("reasoning"),
        "ts": timestamp(when or now()),
    }


class UsageRecorder:
    """Hands per-request usage events (``usage_event``) to the injected
    ``usage_sink``, with the same failure policy as the state sink. Telemetry
    and attribution only: the inference proxy is the authoritative meter. Used
    by ``create_app`` (hosted mode) and by ``headless.run_prompt``."""

    def __init__(self, sink: Optional[Callable[[dict], None]]):
        self._sink = Sink(sink, label="usage")

    @property
    def enabled(self) -> bool:
        return self._sink.enabled

    def report(self, event: dict) -> None:
        """Deliver ``event`` from the event loop without waiting for the sink."""
        self._sink.emit_soon(event, event["event_id"])

    async def drain(self) -> None:
        """Wait (bounded by ``SINK_TIMEOUT``) for the deliveries ``report``
        started, for a caller that is about to leave the event loop."""
        await self._sink.drain()


__all__ = [
    "DISPLAY_LIMIT",
    "KINDS",
    "PUBLIC_PRINCIPAL_ID",
    "SessionEvents",
    "Sink",
    "StateRecorder",
    "UsageRecorder",
    "attribution",
    "claim_rev",
    "clean_display",
    "record_id",
    "usage_event",
]
