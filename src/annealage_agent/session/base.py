"""AgentSession protocol and the AgentEvent dataclasses a session emits.

``AgentSession`` is the seam between the transport (``http/ws.py``) and
whatever is actually driving a conversation. Neither ``protocol.py`` nor
``viewers.py`` imports a concrete session; they, and ``ws.py``, see only
this Protocol, which is what lets the WebSocket layer be built and tested
in M4 before an agent exists. ``session/fake.py`` is the one concrete
implementation this milestone ships; ``session/sdk.py``, wrapping the real
``claude-agent-sdk`` client, is M5's.

A session owns its turn and keeps producing events regardless of whether a
browser is attached (plan section 3.4): it does not read from or write to
a WebSocket itself, and does not hold an ``EventLog`` or a
``ViewerRegistry``. It only calls the ``on_event`` callback given to its
constructor for every event it produces; whatever wires a session together
with an event log and a viewer registry (``http/ws.py`` or ``app.py``, not
this module) decides what that callback does. Keeping the session ignorant
of both is what lets ``session/fake.py`` cover every WebSocket and viewer
test with no event log and no registry in play at all, and what will let
``session/sdk.py`` be swapped in later without touching either.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Optional, Protocol, Tuple, runtime_checkable

# session.agent values for the hello frame (plan section 3.3): the SDK
# client is not yet constructed, is constructed and answering, or failed
# to start. A product's own routes (Mesh's pins, /submit, /callouts) never depend
# on this value; only the chat pane (M5) does, per plan section 3.4's
# "agent health never gates viewer health".
AGENT_CONNECTING = "connecting"
AGENT_READY = "ready"
AGENT_UNAVAILABLE = "unavailable"


def turn_not_sent(status: str) -> str:
    """The remediation a backend reports for a turn it will not take because
    its status is ``status``, not ready.

    Every backend refuses a turn unless it is ready, whatever else it could
    do with one: it numbers the turns it is given, and ``http/ws.py`` counts
    one on the ``ViewerBus`` only for a ready session, so taking one in any
    other state would put its reply under a number the page pairs with the
    wrong message."""
    if status == AGENT_CONNECTING:
        return (
            "the agent is still starting, so this turn was not sent; send it again once it is ready"
        )
    return (
        "the agent is not running, so this turn was not sent; "
        "check the startup output for why and use Retry"
    )


@dataclasses.dataclass(frozen=True)
class SandboxStatus:
    """Whether bash is actually contained, as opposed to requested.

    ``requested`` is what this process asked for, known before the child says
    anything.

    ``active`` is a prediction until the child contradicts it, and the wording
    of the banner reflects that rather than overclaiming. It cannot be anything
    stronger: the CLI announces a sandbox it could **not** engage, and says
    nothing at all about one it could, so the only positive evidence available
    is the absence of a denial, which is not evidence until a bash command has
    actually run. What this process can check cheaply and correctly is the
    negative case, and it does.

    ``missing`` starts from a PATH check for the dependencies the CLI's own
    message names, and is replaced by the child's list the moment the child
    reports one. The child's word wins because it is the only party that knows
    which sandbox implementation it tried; the PATH check exists so that the
    startup banner is right on a machine that is plainly missing a dependency,
    instead of staying silent until the first command runs.
    """

    requested: bool
    active: bool
    missing: Tuple[str, ...] = ()


class UnknownRequest(Exception):
    """Raised when a ``permission`` frame names no request awaiting a decision:
    never asked, or already resolved by another connection's decision, a
    timeout, the last viewer leaving, or shutdown.

    Lives here rather than beside the broker that raises it because
    ``http/ws.py`` has to catch it to answer the connection the losing frame
    arrived on, and ``ws.py`` must keep working with no agent SDK installed;
    importing the broker's module to name its exception would give the
    WebSocket layer an SDK dependency in viewer-only mode.

    Two tabs answering one card is the ordinary way this happens, so it is not
    an error in the sense of something being broken: the loser is owed the
    reply that their click did not decide anything, which is the one outcome a
    silently swallowed exception makes indistinguishable from success.
    """


class AgentEvent:
    """Base for every server-originated event kind.

    ``kind`` is the wire discriminator inside the ``event`` object of a
    server ``event`` frame; ``protocol.build_event`` wraps whatever
    ``to_wire()`` returns. It is a plain class attribute rather than a
    dataclass field, via the ``ClassVar`` annotation subclasses repeat, so
    each subclass fixes its own ``kind`` without every instance carrying
    a redundant copy of it.

    ``viewer`` is the forward seam plan section 3.3 asks for: the
    originating tab id when a later, collaborative mode traces an event
    back to the browser tab that caused it, and unset (``None``) when the
    server originated the event on its own.

    ``by``, on the three events a human causes directly (``UserTurn``,
    ``PermissionResolved`` for a decision, ``PauseChanged``), is that human's
    tailnet login (``identity.Human.login``), and unset for the browser
    token's holder, whose identity is unknown. Every allowed login is the whole
    human (no per-user permissions), so it records who, never what they may
    do. Both are optional and dropped from the wire when unset, so a log
    written before either existed replays unchanged and a page that knows
    neither parses every event.

    Every subclass is a frozen dataclass, immutable once built: an event
    already appended to an ``EventLog`` and already sitting in another
    connection's outbound queue must not change out from under either.
    """

    kind: ClassVar[str] = ""

    def to_wire(self) -> dict:
        """The JSON-able object this event appears as inside an ``event`` frame.

        Every dataclass field on the concrete subclass, plus ``kind``,
        with ``None`` values dropped: none of the wire examples in plan
        section 3.3 show a field written out as null, and an *absent*
        ``viewer`` is exactly how a client tells a server-originated event
        apart from one attributed to a tab.
        """
        data = {"kind": self.kind}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                data[field.name] = value
        return data


@dataclasses.dataclass(frozen=True)
class TextDelta(AgentEvent):
    kind: ClassVar[str] = "text_delta"
    turn: int
    text: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ToolUse(AgentEvent):
    kind: ClassVar[str] = "tool_use"
    turn: int
    tool_use_id: str
    name: str
    input: dict
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ToolResult(AgentEvent):
    kind: ClassVar[str] = "tool_result"
    tool_use_id: str
    is_error: bool
    text: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PermissionRequest(AgentEvent):
    """A tool call waiting on the human's decision (``PermissionBroker.ask``).

    ``rememberable`` is ``False`` for a request the broker will never
    remember an "always allow" for (its ``never_remembered`` names, or a call
    the human started), so the pane offers no such button; absent otherwise.

    ``action``, on a call the human started from the page rather than one
    the agent made (an upload action, ``uploads.py``), is that action's
    label, and ``by`` the login of the human who started it (unset for the
    browser token's holder). The card says so, since the human is approving
    what leaves the page on their own behalf.
    """

    kind: ClassVar[str] = "permission_request"
    request_id: str
    tool: str
    input: dict
    suggestions: list = dataclasses.field(default_factory=list)
    rememberable: Optional[bool] = None
    viewer: Optional[str] = None
    action: Optional[str] = None
    by: Optional[str] = None


#: How an upload action ended (``UploadActionEnded.outcome``): the call ran
#: and answered, the call ran and failed (or never reached the server), or
#: the human (or the broker, on a timeout) did not approve it.
ACTION_DONE = "done"
ACTION_FAILED = "failed"
ACTION_DENIED = "denied"


@dataclasses.dataclass(frozen=True)
class UploadActionEnded(AgentEvent):
    """An action the human started on an uploaded file ended
    (``uploads.py``): which (``label``), on what (``file``, ``bytes``, and
    ``upload``, the upload's id, which the page's chip for it is keyed by),
    the call it made (``tool``, as the card named it), ``outcome`` (one of the
    ``ACTION_`` values) and ``text``, what the call answered or why it did not
    run. ``by`` is the login of the human who started it. ``id`` is the
    action's own, so a page adds it to the conversation once."""

    kind: ClassVar[str] = "upload_action"
    id: str
    label: str
    file: str
    bytes: int
    tool: str
    outcome: str
    text: str
    by: Optional[str] = None
    viewer: Optional[str] = None
    upload: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PermissionResolved(AgentEvent):
    """A permission request is no longer awaiting a decision.

    Emitted exactly once per ``PermissionRequest``, whatever ended it, which is
    what makes a permission card's lifetime a pair of events rather than an
    event and an assumption. Two things depend on that.

    A browser that did not send the deciding frame has no other way to learn
    the card is answered, so without this a request answered on a phone stays
    clickable on a laptop, and the second click is refused for reasons that
    look like a bug.

    Replay reconstructs a reconnecting viewer's pane from the event log, so a
    request with no resolution in the log comes back as a live card on every
    reload, however long ago it was answered.

    ``outcome`` says how it ended, not merely that it did: the human's own
    ``allow``, ``allow_always`` or ``deny``, or one of the resolutions nobody
    clicked, ``timeout``, ``no_viewer`` and ``shutdown``. A pane that submitted
    a decision and sees a different outcome can then say so, rather than
    silently showing the human's deny as if it had taken effect. ``by`` is the
    login of the human who decided, when one did and was signed in by it.
    """

    kind: ClassVar[str] = "permission_resolved"
    request_id: str
    outcome: str
    viewer: Optional[str] = None
    by: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class TurnEnd(AgentEvent):
    """A turn ended. ``cost_usd`` is what this turn cost, 0.0 where the
    backend reports none; ``tokens``, where the backend reports them (omp,
    Claude), is this turn's ``{"input", "output", "cache_read",
    "cache_write"}`` token counts. Both of those backends report running
    totals for the conversation, so these are a total less the one before.
    ``stop_reason`` is the backend's own, or one of the session's:
    ``steered`` (omp: the human sent another message while this turn was
    running, and the agent carries on under the next turn number),
    ``rejected`` (the backend refused the message that started it) or
    ``ended_by_tool`` (a tool result carried ``end_turn``, ``tools.ok``)."""

    kind: ClassVar[str] = "turn_end"
    turn: int
    stop_reason: str
    cost_usd: float
    viewer: Optional[str] = None
    tokens: Optional[dict] = None


#: The token counts a ``Usage`` reports.
USAGE_TOKEN_KEYS = ("input", "output", "cache_read", "cache_write")


@dataclasses.dataclass(frozen=True)
class Usage(AgentEvent):
    """What the conversation has used so far, as the backend reports it:
    emitted after every ``TurnEnd`` and, by a backend that knows it before
    any turn (omp, on a resumed conversation), once it starts.

    Every figure is the whole conversation's, a resumed one included where
    the backend carries it across (omp reads it from the conversation file,
    and the Claude CLI restores it with the conversation), and none is ever
    a sum this layer made: a page summing ``turn_end`` events would be
    wrong as soon as its replay was cut short. ``cost_usd`` is in US
    dollars; ``tokens`` has the four ``USAGE_TOKEN_KEYS``, ``input`` being
    the input tokens not read from the cache; ``context`` is how full the
    model's context window is, ``{"used_tokens", "window_tokens"}``, or
    ``None``. Anything the backend does not say is ``None``, never 0, and
    unlike other events every field is on the wire, null when unknown.
    """

    kind: ClassVar[str] = "usage"
    cost_usd: Optional[float] = None
    tokens: Optional[dict] = None
    context: Optional[dict] = None
    viewer: Optional[str] = None

    def to_wire(self) -> dict:
        data = {"kind": self.kind, **self.snapshot()}
        if self.viewer is not None:
            data["viewer"] = self.viewer
        return data

    def snapshot(self) -> dict:
        """The figures alone, as the ``hello`` frame's ``session.usage``
        carries them."""
        tokens = self.tokens or {}
        return {
            "cost_usd": self.cost_usd,
            "tokens": {key: tokens.get(key) for key in USAGE_TOKEN_KEYS},
            "context": dict(self.context) if self.context is not None else None,
        }


def context_figures(used_tokens, window_tokens) -> Optional[dict]:
    """``Usage.context`` from a backend's two numbers: ``None`` unless it
    gave both, and a window of at least one token, since a fill without
    either is not a figure anyone can read."""
    if not isinstance(used_tokens, int) or not isinstance(window_tokens, int):
        return None
    if window_tokens <= 0:
        return None
    return {"used_tokens": used_tokens, "window_tokens": window_tokens}


@dataclasses.dataclass(frozen=True)
class PauseChanged(AgentEvent):
    """The human's pause switch moved, so the product tools that change the view or
    the project now refuse (or have stopped refusing).

    Broadcast rather than answered to the tab that flipped it, because the
    switch is one property of the running server and every attached view has a
    control showing it: a phone still offering to pause something the laptop
    already paused invites a click that changes nothing.

    The current value also travels in the ``hello`` frame, since a tab that
    connects later has no event to learn it from. ``by`` is the login of the
    human who moved it, when they were signed in by one.
    """

    kind: ClassVar[str] = "pause_changed"
    paused: bool = False
    viewer: Optional[str] = None
    by: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ViewerPrimary(AgentEvent):
    """Announces which connection now receives ``call`` frames.

    ``primary`` is the newly primary connection's tab id, or ``None`` when
    the last viewer disconnected and no connection holds the role. Every
    connection is broadcast this event on a change, including the
    connection that just became primary, so a chat pane can show whether
    the tab it is running in is the one driving tool calls.
    """

    kind: ClassVar[str] = "viewer_primary"
    primary: Optional[str] = None
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class AgentStatus(AgentEvent):
    """The agent's status changed to ``status``, one of the three AGENT_ values.

    The ``hello`` frame carries the status as it stood when that connection was
    accepted, which is a snapshot and nothing more. A browser reaches the page
    and opens its socket faster than the CLI child starts, so the ordinary case
    is a viewer told "connecting" a moment before the agent becomes ready;
    without this event that first answer is also the last one, and a working
    agent reads as permanently starting up.
    """

    kind: ClassVar[str] = "agent_status"
    status: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class AgentModelChanged(AgentEvent):
    """The active model changed mid-conversation, via ``session.set_model``.

    Not to be confused with a product's own events about the files it serves
    (Mesh's ``models_changed``, about its STL files), which share the word
    "model" by coincidence. This one is ``Agent``-prefixed, matching
    ``AgentStatus``/``AgentError``: it is about the conversation driver
    itself, never the served project.

    Emitted once the switch actually took effect (after the live
    control-plane call each driver's ``set_model`` makes), so a picker
    showing the CLI-configured starting model (``hello``'s ``session.model``)
    reflects a later human choice without a page reload.
    """

    kind: ClassVar[str] = "agent_model_changed"
    model: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SessionReset(AgentEvent):
    """Emitted when a requested resume (``-c``/``-r``) fails and the
    session falls back to starting fresh instead (plan section 3.4)."""

    kind: ClassVar[str] = "session_reset"
    reason: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class AgentError(AgentEvent):
    """Emitted when the CLI is missing, unauthenticated, or its child dies
    (plan section 3.4): the product's page and its own routes keep working,
    and this is how the chat pane learns to show a Retry affordance
    instead of hanging."""

    kind: ClassVar[str] = "agent_error"
    stderr: str
    remediation: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ReviewChanged(AgentEvent):
    """The product's review changed: a comment or a callout was added,
    resolved or deleted, by the model, the human or anything else writing the
    review's files (``review/watcher.py`` publishes it).

    Carries no payload: the page refetches the review for itself
    (``static/review.js``, or a product's own route), which keeps the review
    in the page to one writer; putting the changed content here would give it
    a second. Generic rather than a product's own event, so every product's
    page reacts to the same kind and none hand-rolls its own.
    """

    kind: ClassVar[str] = "review_changed"
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Attention(AgentEvent):
    """Something needs the human: the agent has stopped and is waiting on
    them (Annealage Loom's ``checkpoint`` tool), published through
    ``ViewerBus.attention``. The page shows a browser notification and
    flashes its title until it is focused, for a live event only: a replayed
    one is history, not a call for attention now. Recorded in the event log
    like every event, so the session's history says when the agent asked."""

    kind: ClassVar[str] = "attention"
    title: str
    body: str
    viewer: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class UserTurn(AgentEvent):
    """What the human sent as turn ``turn``: the turn frame's own ``blocks``,
    exactly as the page sent them, and the ``client_id`` the sending tab gave
    the frame.

    ``http/ws.py`` logs it when, and only when, the turn goes to a ready
    session (the moment ``ViewerBus.begin_turn`` counts it), and before the
    session sees the turn, so it precedes every event the turn produces. It
    is what makes the human's side of the conversation part of the history:
    every tab, a reload and a restart pair a turn with what the human said by
    this event, and ``client_id`` is how the tab that sent it retires its own
    optimistic copy. ``turn`` is ``bus.turn`` after ``begin_turn``, which is
    the number every backend gives the reply (a steer's included).

    ``blocks`` are the human's, never the product's notes ``begin_turn`` puts
    in front of them, and an image is its ``image_path`` block, a path under
    the served directory: the pixels stay in the file, out of the log. ``by``
    is the sender's login, when they were signed in by one.
    """

    kind: ClassVar[str] = "user_turn"
    turn: int
    blocks: list
    client_id: Optional[str] = None
    viewer: Optional[str] = None
    by: Optional[str] = None


#: Every event kind the agent layer emits itself. A product's own event
#: classes (``Product.events``) are registered beside these by
#: ``product.install`` and may not reuse one of their kinds: the chat pane and
#: the transcript export both dispatch on ``kind`` alone, so a product event
#: whose kind was ``turn_end`` would be read as the end of a turn.
GENERIC_EVENTS = (
    TextDelta,
    ToolUse,
    ToolResult,
    PermissionRequest,
    PermissionResolved,
    TurnEnd,
    Usage,
    PauseChanged,
    ViewerPrimary,
    AgentStatus,
    AgentModelChanged,
    SessionReset,
    AgentError,
    ReviewChanged,
    Attention,
    UserTurn,
    UploadActionEnded,
)

#: The installed product's event classes; see ``register_product_events``.
PRODUCT_EVENTS: Tuple[type, ...] = ()


def check_product_events(events) -> None:
    """Raise ``ValueError`` if ``events`` cannot be registered: a class that
    is not an ``AgentEvent`` with a kind of its own, or a kind already taken by
    a generic event or by another of ``events``. Changes nothing."""
    taken = {event.kind for event in GENERIC_EVENTS}
    for event in events:
        if not (isinstance(event, type) and issubclass(event, AgentEvent)) or not event.kind:
            raise ValueError("product event %r is not an AgentEvent with a kind" % (event,))
        if event.kind in taken:
            raise ValueError("product event kind %r is already taken" % event.kind)
        taken.add(event.kind)


def register_product_events(events) -> None:
    """Make ``events`` the product's event classes, replacing any registered
    before. Called by ``product.install`` (after ``check_product_events``) and
    ``product.reset`` only."""
    global PRODUCT_EVENTS
    PRODUCT_EVENTS = tuple(events)


@dataclasses.dataclass(frozen=True)
class BackendLog:
    """One of the agent backend's own logs, as ``AgentSession.backend_logs``
    lists it: what the backend itself wrote about this session, which is where
    the real reason for a failure is when the remediation text is only a guess
    at it.

    ``kind`` is ``"file"``, with ``path`` naming a file this process found for
    this session (the Claude CLI's transcript, a Codex rollout, omp's process
    log and conversation file), or ``"text"``, with ``text`` holding what this
    process kept in memory (the backend's stderr), which exists even when the
    backend never got as far as writing a file. ``format`` is ``"jsonl"`` for a
    file of JSON objects one per line, which the page can filter by their
    ``level``, and ``"text"`` otherwise.

    A session only ever lists a path it decided itself, from where its backend
    is known to keep its files, and only when that is a regular file inside
    that place (``session/logfiles.py``): ``GET /agent/logs`` serves these to
    the browser, so a path the backend or the agent could have planted must
    never reach this list.
    """

    name: str
    kind: str
    path: Optional[str] = None
    text: Optional[str] = None
    format: str = "text"

    def to_wire(self) -> dict:
        """The listing's JSON shape: everything but the text, which is served
        one entry at a time, capped (``http/routes_logs.py``)."""
        return {"name": self.name, "kind": self.kind, "path": self.path, "format": self.format}


@runtime_checkable
class AgentSession(Protocol):
    """What ``http/ws.py`` needs from whatever is driving a conversation.

    The five frame-handling coroutines are what ``ws.py`` dispatches to. The
    four members after them are the lifecycle ``app.py`` (``create_app`` and
    ``serve``) and the product's CLI (Mesh's ``cli.py``) drive,
    and they are declared here rather than left to duck typing because both
    callers reach for them unconditionally: a session missing ``start`` fails
    with ``AttributeError`` before a single request is served, which is a
    contract worth stating once here instead of guarding at each call site.

    ``session_id``, ``sdk_session_id`` and ``cwd`` are read once per
    connection to build the ``hello`` frame's ``session`` object.
    ``sdk_session_id`` is ``None`` until the real SDK client has one to
    report (M5); it is never fabricated. The five coroutine methods are
    where ``ws.py`` dispatches an inbound ``turn``, ``permission``,
    ``interrupt`` or ``set_model`` frame; there is deliberately no method for
    an inbound product frame (Mesh's ``state``), because a browser's report
    of its own view is viewer state, not agent state, and belongs wherever a tool
    call reads "the current view" from, not here.

    A session never touches a WebSocket, an ``EventLog`` or a
    ``ViewerRegistry`` directly; it only calls ``on_event`` from its
    constructor for every event it produces. Whatever builds a session
    (``http/ws.py`` or ``app.py``) is responsible for making that callback
    append to an ``EventLog`` and broadcast through a ``ViewerRegistry``.

    ``backend_logs()`` lists the backend's own logs for this session
    (``BackendLog``) for the page's Agent log section, ``GET /agent/logs`` and
    the diagnostics block. Every session implements it; one whose backend
    writes nothing of its own returns an empty list.

    Three optional members, outside the Protocol so a session without them
    (a test's fake, an older product's) still is one, and read with
    ``getattr`` where they are used:

    - ``steers``: true when a turn sent while one is running redirects it
      (omp) rather than waiting behind it; the ``hello`` frame reports it so
      the page labels its Send button.
    - ``end_turn_after_tool()``: a tool result asked to end the turn
      (``tools.ok(..., end_turn=True)``, via ``ViewerBus.request_end_turn``).
      Called from inside the tool's handler, before its result reaches the
      backend; the session stops the turn once that result is delivered.
    - ``set_tool_table(table)``: replace the tools the model sees mid-session
      (``ToolServer.host_tool_table()``'s shape), returning True if the
      backend took them; used when a remote MCP server that was unreachable at
      startup is reached later (``ToolServer.reconnect``). Only omp can.
    """

    session_id: str
    sdk_session_id: Optional[str]
    cwd: str

    def agent_status(self) -> str:
        """One of AGENT_CONNECTING, AGENT_READY, AGENT_UNAVAILABLE."""
        ...

    async def submit_turn(self, blocks: list, viewer: Optional[str] = None) -> None:
        """Handle an inbound ``turn`` frame's ``blocks``.

        ``viewer`` is the tab id of the connection the frame arrived on,
        for a session that wants to stamp the events a turn produces
        (it is not required to)."""
        ...

    async def decide_permission(
        self, request_id: str, decision: str, message: str = "", by: Optional[str] = None
    ) -> None:
        """Handle an inbound ``permission`` frame. ``by`` is the deciding
        human's login (``None`` when it is not known), which the
        ``PermissionResolved`` it produces records."""
        ...

    async def interrupt(self) -> None:
        """Handle an inbound ``interrupt`` frame."""
        ...

    async def set_model(self, model: str) -> None:
        """Handle an inbound ``set_model`` frame: switch the live conversation
        to ``model`` and emit ``AgentModelChanged`` once it takes effect.

        All three backends support a genuine live switch natively (no
        reconnect or restart); a session that cannot honour a given value
        raises rather than silently no-opping, which ``ws.py`` turns into a
        ``refused`` frame naming the failure.
        """
        ...

    async def start(self) -> None:
        """Bring the session up. Called once, by the app's ``agent_start``
        once its socket is listening (``app.serve`` and a front door call
        it), or as the app resumes after closing for idle.

        Must not raise: the HTTP server starts first and independently and has
        to keep serving the viewer whatever the agent does, so a failure here
        belongs in an ``AgentError`` event with ``agent_status()`` left at
        unavailable.
        """
        ...

    async def close(self) -> None:
        """Shut the session down. Called once: by the app's ``agent_stop``,
        while there is still a socket to carry any last event, or by its idle
        sweep, with no page connected. A closed session is not started again;
        an app that resumes builds a new one."""
        ...

    def on_viewer_presence(self, count: int) -> None:
        """Note that ``count`` browser connections now exist.

        Called on the same transitions as ``ViewerRegistry.add``/``remove``.
        Reaching zero is what lets a session deny a request nobody is left to
        answer, rather than holding a tool call open for the whole timeout.
        """
        ...

    def sandbox_status(self):
        """What containment is actually in effect, for the startup banner.

        Returns an object with ``requested``, ``active`` and ``missing``. A
        session with no shell to contain reports ``requested`` false.
        """
        ...

    def backend_logs(self) -> list:
        """The backend's own logs for this session, as ``BackendLog`` entries,
        in the order the page lists them.

        Called off the event loop (it looks at files), and again on every
        request for a log, so the list is always the session's current view:
        a file the backend has only now created appears, one that has gone is
        no longer listed. Must not raise; a log that cannot be found is left
        out rather than reported as an error.
        """
        return []
