/**
 * Browser side of the /ws transport: reads the per-run token out of the URL
 * once, connects, exchanges hello and replay, dispatches server events, and
 * drives the topbar connection indicator. Reconnection uses capped
 * exponential backoff and never stops on its own except for the two cases
 * that retrying cannot fix within this page load: a protocol-version
 * mismatch, and a confirmed token refusal.
 *
 * This module is the only place the token is held; it lives in the
 * TOKEN module variable, never in localStorage or sessionStorage, because a
 * token that outlives the run it belongs to is a liability and a fresh one
 * is generated every server start anyway.
 *
 * This module never writes the product's state and knows nothing about what
 * the product shows. What the product needs from the socket it hands in to
 * `initWs` (the product's main module is where the two meet):
 *
 * - `onEvent`, `{kind: fn(event)}`, handles the event kinds the chat pane has
 *   no use for: the product's own (the ones it registered server-side, e.g.
 *   Mesh's `models_changed`) and `review_changed`, which review.js's
 *   `onEvent` handles for a product that uses it. A kind listed there goes to
 *   its handler and nowhere else; the server already refuses a product kind
 *   that collides with a generic one. Only a live event is handed on: the
 *   replay a connection opens with (the session's whole history, for a new
 *   page) is what already happened, which `onLive` covers.
 * - `onLive` runs on every successful handshake, the first and every
 *   reconnect, after the connection is marked live: the product stops its
 *   fallback poll there and refetches whatever an event it missed while the
 *   socket was down would have told it, since a replayed product event is not
 *   handed to its handler (above).
 * - `onFallback` runs when the socket stops being the live channel for a
 *   while (a downtime longer than one backoff interval, a protocol mismatch,
 *   a confirmed refusal): the product starts whatever poll keeps its view
 *   current without the socket. ws.js owns the decision of when; the product
 *   owns the poll's mechanics, so the push and the poll never both run.
 * - `connTitles` overrides the indicator's tooltips by state, for a product
 *   whose live channel means something more specific than "updates".
 *
 * `onHello` and `onAgentEvent` are the chat pane's two inbound hooks:
 * `onHello` receives the hello frame's `session` object once per connection
 * (including every reconnect, since agent status can change between them),
 * and `onAgentEvent` receives every event neither this module nor a product
 * handler takes, replayed ones included, with `{replayed}` saying which.
 * Neither is called from here except at those two points;
 * chat.js, not this module, decides what an event means. The returned `send`
 * is this module's only outbound capability, so a turn, permission, pause or
 * interrupt frame still goes out over the one socket this closure owns, with
 * no second connection or second reconnect policy.
 *
 * `dispatchCall` is the product's method table, and it is what makes a `call`
 * frame do something: this module owns the correlation (answer the id, exactly
 * once, whatever happened) and knows nothing about what any method means.
 * `onPaused` receives the pause flag from the hello frame and from each live
 * `pause_changed` event, so its caller has one path to reconcile rather than
 * two; a replayed one is not passed on, since the hello already said what the
 * flag is now.
 */

import { store } from "./store.js";
import { showError, toast } from "./ui.js";
import { appUrl } from "./url.js";

// protocol.py's PROTOCOL_VERSION. Stamped on every frame `send` writes, so
// no caller carries a version of its own that could fall behind this one.
const PROTOCOL_VERSION = 2;

const BASE_BACKOFF_MS = 500; // also the "one backoff interval" the fallback poll waits out
const MAX_BACKOFF_MS = 15000;

// How long a connection attempt is given to prove it is alive, from the
// moment connect() constructs the socket: past this, with no message
// (open's own hello included) having reset the clock, the socket is
// treated as dead whether it never reached OPEN at all or opened and then
// went quiet with no close event of its own (a laptop that slept, a NAT
// mapping dropped silently).
//
// This must stay at least twice the server's PING_INTERVAL (http/ws.py,
// currently 5 seconds), because those pings are the only traffic an
// otherwise idle connection carries and they are what distinguishes idle
// from dead. Set below that and a healthy connection with nothing to say
// gets closed and reopened forever; set far above it and a phone that lost
// its network waits too long before falling back to polling. Three missed
// pings is the compromise.
const LIVENESS_TIMEOUT_MS = 15000;

const REFUSED_MESSAGE =
  "This page's link has gone stale (the server was restarted, or this " +
  "tab was opened without the printed link, since a plain reload does not " +
  "keep the token). Reopen the URL printed in the terminal to get a " +
  "working one.";
const MISMATCH_MESSAGE =
  "This page is running an older build than the server now speaks. " +
  "Reload the page to pick up the current one.";

const CONN_LABEL = {
  connecting: "Connecting…",
  live: "Live",
  polling: "Polling",
  refused: "Reopen URL",
};
// The generic tooltips; a product names what "live" and "polling" mean for
// it through `initWs`'s `connTitles`.
const CONN_TITLE = {
  connecting: "Connecting to the live update channel.",
  live: "Live: updates arrive without a reload.",
  polling: "Live updates are unavailable right now; falling back to polling.",
  refused: REFUSED_MESSAGE,
};

/**
 * Reads and consumes the per-run token from location.hash: "#t=<token>"
 * directly, or "#n=<nonce>", a single-use login nonce traded once through
 * POST /login for the token. The nonce form is what the server puts in the
 * URL it launches a browser with, because that URL sits on a command line
 * any local process can read; the printed "#t=" link is the reusable one.
 * A fragment is never sent to a server, which is why the token travels
 * here rather than in the path or a query parameter for the initial page
 * load; it is promoted to a query parameter only for the one request that
 * needs it, the /ws handshake, because a browser cannot set headers on
 * that handshake and this is the only channel left that is not the
 * subprotocol field.
 *
 * The fragment is stripped from the visible URL unconditionally, whether
 * or not a token was found in it, and before the nonce is traded: a token
 * left visible in the address bar survives a reload, a bookmark and a
 * screen share, all of which are wider exposure than the "never sent to a
 * server" property the fragment was chosen for in the first place. A nonce
 * the server refuses (spent or expired) leaves no token, which ends in the
 * same "refused" state as a stale link.
 */
async function extractToken() {
  const hash = location.hash;
  const params = hash.startsWith("#") ? new URLSearchParams(hash.slice(1)) : null;
  if (hash) {
    history.replaceState(null, "", location.pathname + location.search);
  }
  if (!params) return "";
  const token = params.get("t");
  if (token) return token;
  const nonce = params.get("n");
  if (!nonce) return "";
  try {
    const res = await fetch(appUrl("login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nonce }),
    });
    if (!res.ok) return "";
    const data = await res.json();
    return typeof data.token === "string" ? data.token : "";
  } catch (err) {
    return "";
  }
}

// Top-level await: every module that imports this one (and so authToken)
// evaluates only once the token is known, whichever form it arrived in.
const TOKEN = await extractToken();

/**
 * The per-run token, for the modules that authenticate a plain HTTP request
 * with it: an upload, the settings window, a transcript export, and any
 * token-gated route of the product's own. Reading it here rather than a
 * second `extractToken()` call is what keeps this module the token's only
 * holder (see the header comment): the fragment is already stripped from
 * `location.hash` by the time any other module's top-level code runs, since
 * this module evaluates first.
 */
export function authToken() {
  return TOKEN;
}

function makeTabId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  // Every browser this viewer targets has crypto.randomUUID; this only
  // covers an embedded WebView that stripped it. Uniqueness here only
  // needs to hold across the handful of tabs one person opens, not
  // cryptographic strength, since the tab id carries no authority: the
  // token is what proves who is allowed to connect at all.
  return "t" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

const TAB_ID = makeTabId();

function wsPath() {
  return appUrl("ws") + "?t=" + encodeURIComponent(TOKEN);
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}${wsPath()}`;
}

/**
 * Connects, and keeps connecting, for the life of the page. See this
 * module's header for what each hook is for. `indicator` is the element the
 * connection state is shown on (default: `#connIndicator`); a page without
 * one passes null.
 */
export function initWs({
  onEvent = {},
  onLive = () => {},
  onFallback = () => {},
  onHello = () => {},
  onAgentEvent = () => {},
  onPaused = () => {},
  onRefused = () => {},
  dispatchCall = null,
  connTitles = {},
  indicator = document.getElementById("connIndicator"),
} = {}) {
  let ws = null;
  let opened = false; // true once this attempt's WebSocket has reached readyState OPEN
  let lastSeq = 0;
  // The session lastSeq is a position in: the hello's `session.id` when the
  // events counted in it arrived. A seq means nothing in another session's
  // log, so the server answers a hello whose session differs from the start.
  let seqSession = null;
  let liveAfter = 0;
  let attempt = 0;
  let stopped = false; // permanently done trying: protocol mismatch or a confirmed 403
  let fallbackFired = false; // whether this downtime episode has already switched to polling
  let fallbackTimer = null;
  let reconnectTimer = null;
  let livenessTimer = null;

  // A Map rather than the object itself, so an event kind can never resolve
  // to something the object inherited ("constructor", "toString").
  const productHandlers = new Map(Object.entries(onEvent));
  const titles = { ...CONN_TITLE, ...connTitles };

  function applyConnIndicator(state) {
    if (!indicator) return;
    indicator.textContent = CONN_LABEL[state.connection] || state.connection;
    indicator.title = titles[state.connection] || "";
    indicator.dataset.state = state.connection;
  }
  applyConnIndicator(store.getState());
  store.subscribe("connection", applyConnIndicator);

  function backoffDelay(n) {
    const raw = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * Math.pow(2, n));
    // +/-25% jitter, so a server restart does not get every tab's next
    // retry landing on it in the same instant.
    return raw * (0.75 + Math.random() * 0.5);
  }

  // Returns whether the frame was actually written to the socket, rather
  // than nothing: a caller with no correlation id to match a reply against
  // (chat.js's set_model, tracked locally via pendingSetModel) needs to
  // know when this no-ops, because a false return means no frame went out
  // at all, so neither a confirming event nor a `refused` frame is ever
  // coming for it, and the caller must reconcile its own optimistic state
  // immediately instead of waiting on an answer that will never arrive.
  function send(frame) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ ...frame, v: PROTOCOL_VERSION }));
      return true;
    }
    return false;
  }

  function clearLiveness() {
    clearTimeout(livenessTimer);
    livenessTimer = null;
  }

  function resetLiveness() {
    clearTimeout(livenessTimer);
    livenessTimer = setTimeout(handleLivenessExpiry, LIVENESS_TIMEOUT_MS);
  }

  function handleLivenessExpiry() {
    livenessTimer = null;
    if (stopped || !ws) return;
    // Nothing reset the deadline: this attempt's socket never delivered a
    // single message since connect(), whether because it never reached
    // OPEN, opened but never got a hello back, or went quiet sometime
    // after that with no close event of its own. ws.close() is the one
    // action that produces a decision in every one of those cases: fired
    // while still CONNECTING it closes without ever having opened, which
    // handleClose routes to probeRefusal; fired on an OPEN socket it
    // routes to scheduleRetry instead.
    try {
      ws.close();
    } catch (err) {
      // already closing or closed; handleClose was already called or is imminent
    }
  }

  function connect() {
    if (stopped) return;
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
    opened = false;
    try {
      ws = new WebSocket(wsUrl());
    } catch (err) {
      // Malformed URL or a browser that refuses to construct a WebSocket
      // (e.g. mixed content); nothing to learn from this beyond "try again
      // later", the same as any other pre-handshake failure.
      probeRefusal();
      return;
    }
    resetLiveness();
    ws.addEventListener("open", handleOpen);
    ws.addEventListener("message", handleMessage);
    ws.addEventListener("close", handleClose);
    // No separate handling in "error": the WebSocket spec fires close right
    // after error for every failure mode, and close is where readyState and
    // the close code are both available, so all decisions are made there.
    ws.addEventListener("error", () => {});
  }

  function handleOpen() {
    opened = true;
    const hello = {
      v: PROTOCOL_VERSION,
      type: "hello",
      token: TOKEN,
      last_seq: lastSeq,
      viewer: { tab_id: TAB_ID, w: window.innerWidth, h: window.innerHeight },
    };
    if (seqSession !== null) hello.session_id = seqSession;
    send(hello);
  }

  function handleMessage(event) {
    // Any inbound frame, parseable or not, proves the transport is still
    // alive; this is the one reset shared by the "never got a hello"
    // and "went idle after hello" halves of the liveness deadline.
    resetLiveness();
    let frame;
    try {
      frame = JSON.parse(event.data);
    } catch (err) {
      return; // not a frame this client understands; nothing was sent that depends on a reply
    }
    if (frame.v !== PROTOCOL_VERSION) {
      // The server is expected to close with 4400 rather than send a frame
      // with the wrong v to a client that announced the right one; this is
      // a defensive second check in case a future server build ever sends
      // one frame before closing.
      handleProtocolMismatch();
      return;
    }
    if (frame.type === "hello") {
      handleHello(frame);
    } else if (frame.type === "event") {
      handleEvent(frame);
    } else if (frame.type === "call") {
      handleCall(frame);
    } else if (frame.type === "refused") {
      // Always sent in answer to something this page sent, and always
      // carrying a reason. Dropping it would leave the human watching for an
      // effect that is never coming, with the server's explanation of why
      // discarded one layer below the UI. A refused turn names itself with
      // the turn frame's own `client_id`, which chat.js uses to stop holding
      // that message; any other refusal carries no correlation id, so
      // onRefused cannot know which outstanding request it answers, and
      // chat.js uses it only to reset optimistic UI that has no other way to
      // learn a change was rejected.
      toast(frame.reason || "the server refused that request", false);
      onRefused(frame.reason || "", frame.client_id || null);
    }
    // "ping": nothing to do beyond the liveness reset above, which every
    // inbound frame already did. An unrecognised type from a same-version
    // server is simply a frame this build of the page has no use for, not an
    // error.
  }

  /**
   * Run one server-initiated `call` and answer it, exactly once.
   *
   * Every path builds a reply, and that is the point: the server holds a
   * pending future per call id with a model waiting on it, so a call this page
   * cannot serve has to say so rather than be left to time out. `dispatchCall`
   * is optional for the same reason, rather than assumed present.
   *
   * Not awaited by the caller: `handleMessage` must return to the socket's
   * message loop immediately, or a capture that takes 200 ms would stall every
   * event queued behind it, including the `turn_end` ending the very turn this
   * call belongs to. Nothing in here may therefore reject, since there is no
   * caller left to catch it.
   */
  async function handleCall(frame) {
    let reply;
    try {
      if (!dispatchCall) throw new Error("this viewer does not serve calls");
      const result = await dispatchCall(frame.method, frame.params);
      // `result` is a required key on this frame, and `undefined` would drop
      // out of JSON.stringify entirely and be refused by the server's
      // whitelist validation, so a method that returns nothing sends {}.
      reply = { v: PROTOCOL_VERSION, type: "result", id: frame.id,
                result: result === undefined ? {} : result };
    } catch (err) {
      reply = { v: PROTOCOL_VERSION, type: "error", id: frame.id,
                error: { code: (err && err.code) || "viewer_failed",
                         message: String((err && err.message) || err) } };
    }
    try {
      send(reply);
    } catch (err) {
      // The socket went between the guard inside `send` and the write itself.
      // Nothing to do and nothing lost: the server fails every call pending on
      // a closed connection the moment it notices, without waiting out the
      // timeout.
    }
  }

  function handleHello(frame) {
    if (frame.protocol !== PROTOCOL_VERSION) {
      handleProtocolMismatch();
      return;
    }
    // lastSeq is not moved forward here: it advances only as events arrive,
    // so a connection that drops partway through the replay below resumes
    // from the last event it actually received rather than skipping the
    // rest. It goes back to the start when the server answers from the
    // start: a different session than the one it counted (the server ignored
    // the position), or a log that has not reached it (a different run of
    // the log), so the replay that follows is not taken for duplicates.
    const session = frame.session && frame.session.id ? frame.session.id : null;
    if (session !== seqSession || frame.seq < lastSeq) lastSeq = 0;
    seqSession = session;
    // Everything up to here is replay of what already happened; only a later
    // seq is live (chat.js raises no notification for a replayed attention).
    liveAfter = frame.seq;
    attempt = 0;
    fallbackFired = false;
    clearTimeout(fallbackTimer);
    fallbackTimer = null;
    store.setConnection("live");
    // The product's resync, on every (re)connect: it stops its fallback poll
    // and refetches what it shows, because a replayed product event is not
    // handed to its handler (handleEvent), so a change announced while this
    // page was away would otherwise leave it stale with no further event to
    // prompt a refetch.
    onLive();
    // The pause flag comes with the greeting for the same reason: this tab may
    // have connected long after it was set, and `pause_changed` only reaches a
    // client that was attached when it happened.
    onPaused(!!(frame.session && frame.session.paused));
    onHello(frame.session);
  }

  function handleEvent(frame) {
    // Each event once, in order: the server can send an event twice around
    // the moment a connection is registered (its replay, then a broadcast
    // scheduled before the replay finished), and never sends seqs out of
    // order otherwise.
    if (frame.seq <= lastSeq) return;
    lastSeq = frame.seq;
    const event = frame.event;
    const kind = event && event.kind;
    // History rather than news: the hello already carried the current pause
    // flag, and onLive already refetched what a product event would, so
    // replaying either would only put back a state that has since changed
    // (a pause set before a restart, which the new process does not hold),
    // or refetch once per event in the history.
    const replayed = frame.seq <= liveAfter;
    if (kind === "pause_changed") {
      if (!replayed) onPaused(!!event.paused);
    } else if (productHandlers.has(kind)) {
      if (!replayed) productHandlers.get(kind)(event);
    } else {
      onAgentEvent(event, { replayed });
    }
  }

  function handleProtocolMismatch() {
    stopped = true;
    clearLiveness();
    clearTimeout(fallbackTimer);
    clearTimeout(reconnectTimer);
    if (ws) {
      try {
        ws.close();
      } catch (err) {
        // already closing or closed; nothing left to do
      }
    }
    showError(MISMATCH_MESSAGE);
    store.setConnection("polling");
    onFallback();
  }

  function handleClose(event) {
    const wasOpened = opened;
    opened = false;
    ws = null;
    clearLiveness();
    if (stopped) return;
    if (wasOpened && event.code === 4400) {
      handleProtocolMismatch();
      return;
    }
    if (wasOpened) {
      // A drop after a successful handshake: the token, Origin and Host
      // checks already passed once for this page, so there is nothing to
      // diagnose, only a network blip, a laptop sleep or a server restart
      // to wait out.
      //
      // The state has to stop saying "live" here, before scheduleRetry arms
      // the fallback timer, because that timer's own guard skips the switch
      // to polling when it finds the connection live. Leaving it live across
      // a drop makes that guard read a state this close never cleared, and
      // the fallback never fires at all: the page then shows a live socket
      // and runs neither the push nor the poll, which is the exact outcome
      // the fallback exists to prevent.
      store.setConnection("connecting");
      scheduleRetry();
      return;
    }
    // Never reached "open". Per the WebSocket platform contract this looks
    // identical whether the cause was a wrong token, a rejected Origin or
    // Host, or the server simply being unreachable: every browser reports
    // a pre-handshake failure as the same reasonless abnormal closure, with
    // no status code and no body exposed to script. probeRefusal tells
    // a stale token apart from a transient outage by asking the same
    // question over plain HTTP instead, where the status code is visible.
    probeRefusal();
  }

  async function probeRefusal() {
    if (stopped) return;
    try {
      const res = await fetch(wsPath(), { cache: "no-store" });
      if (res.status === 403) {
        // Confirmed: this exact request, over a channel that does expose
        // its status, was refused before any upgrade was attempted. Per
        // the auth design, 403 on /ws is emitted by the token, Origin and
        // Host checks, which is the only way a request already reaching
        // this same origin fails here; further retries with the same
        // fixed token would repeat forever, so the socket path is
        // abandoned for this page load and the fallback poll takes over
        // for good.
        //
        // Only 403 is terminal, and every other status falls through to a
        // retry on purpose. A healthy server answers this same probe with
        // 400, because the probe is a plain GET carrying no upgrade headers
        // and the route reaches the handshake and rejects it there: that
        // status means the token was accepted and the socket failure was
        // transient, which is exactly the case worth retrying.
        stopped = true;
        clearTimeout(fallbackTimer);
        fallbackTimer = null;
        store.setConnection("refused");
        showError(REFUSED_MESSAGE);
        onFallback();
        return;
      }
    } catch (err) {
      // The probe failed too (offline, DNS, TLS): no more information than
      // the WebSocket attempt already gave. Falls through to the ordinary
      // retry path below.
    }
    scheduleRetry();
  }

  function scheduleRetry() {
    if (stopped) return;
    if (!fallbackFired && fallbackTimer === null) {
      // Only armed once per downtime episode: repeated failures before it
      // fires must not stack a second timer that fires again later and
      // re-runs the switch-to-polling step redundantly.
      fallbackTimer = setTimeout(() => {
        fallbackTimer = null;
        if (stopped || fallbackFired || store.getState().connection === "live") return;
        fallbackFired = true;
        store.setConnection("polling");
        onFallback();
      }, BASE_BACKOFF_MS);
    }
    reconnectTimer = setTimeout(connect, backoffDelay(attempt));
    attempt++;
  }

  connect();

  // `send` is the only capability this module hands back: chat.js's turn,
  // permission and interrupt frames all go out through it, so there is
  // still exactly one WebSocket and one place (this closure) that owns its
  // lifecycle, reconnects and replay bookkeeping.
  return { send };
}
