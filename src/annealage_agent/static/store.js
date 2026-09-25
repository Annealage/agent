/**
 * The single writer for the page's application state: the agent layer's own
 * keys (the chat pane, the connection, the pause flag, the layout) and every
 * key a product adds through `defineSlice`.
 *
 * Every other module reads state through `store.getState()` or a subscriber
 * callback and changes it only by calling a mutator: one of those exported on
 * `store` below, or one a product builds on the `update` function its
 * `defineSlice` call returned. No line outside this file ever assigns to
 * `state`, and each top-level key has exactly one owner: this file for the
 * keys declared here, the slice that declared it for a product's. `update`
 * refuses to touch a key its slice does not own, so two slices cannot both
 * write one key and a value with two writers cannot come back by accident.
 *
 * There is one state object and one notification pass for all of it, not a
 * store per slice, because modules read across the boundary (a product's
 * pause control is disabled by the chat slice's `agentStatus`) and because
 * the re-entrancy rule below only holds if every mutator, whoever owns it,
 * goes through the same queue.
 *
 * State is plain, serialisable data: numbers, strings, booleans, plain arrays
 * and plain objects only. No DOM node and no rendering-library object is ever
 * a state value; a module that draws a piece of state keeps its own side
 * table of whatever it drew and reconciles it against the store from a
 * subscriber. That is what lets an event arriving over the socket and a
 * click change state the same way, through a mutator, with no second writer
 * to reconcile against.
 *
 * State shape (this file's keys; a product documents its own slice):
 *   activeTab     string | null
 *                 which narrow-viewport tab is showing; the id of one of the
 *                 tabs the page handed layout.js's `initLayout`, which sets
 *                 the first one. Null only before that.
 *   panelOpen     boolean
 *                 whether the side panel is shown at wide viewports.
 *   paused        boolean
 *                 whether the human has paused the agent's control of this
 *                 page. Written only from what the server reports (the
 *                 hello frame's `session.paused` and every `pause_changed`
 *                 event), never optimistically from the click that asked for
 *                 the change: the flag is enforced in the server, where the
 *                 tools it gates run, so a local value could show "paused"
 *                 while they were still running.
 *   connection    'connecting' | 'live' | 'polling' | 'refused'
 *                 ws.js's view of the /ws socket, read by the topbar
 *                 indicator. 'live' means updates arrive by push; 'polling'
 *                 means ws.js has handed over to the product's fallback poll
 *                 (its `onFallback` hook), whether because the socket has not
 *                 yet reconnected or because it never will again this page
 *                 load (a protocol-version mismatch); 'refused' means a
 *                 pre-handshake 403 was confirmed, most likely a stale token
 *                 from a server restart, and the fallback runs under this
 *                 state too.
 *   chat          {turns, pendingUser, pending, agentStatus, model, banner,
 *                  attachments}
 *                 the whole chat pane's state.
 *                 `turns` is one record per SDK turn number, `{turn, user,
 *                 text, tools, stopReason, costUsd, complete}`, built up
 *                 as text_delta/tool_use/tool_result/turn_end events
 *                 arrive; `tools` is `[{tool_use_id, name, input, result}]`
 *                 with `result` null until a tool_result names that
 *                 tool_use_id. `user` pairs the record with the composer
 *                 blocks that started it, taken off `pendingUser` the
 *                 first time any event for that turn number is seen,
 *                 because the outbound `turn` frame carries no id the
 *                 server echoes back to correlate the two. `pendingUser`
 *                 is that queue: blocks the composer has sent whose turn
 *                 number has not appeared yet.
 *                 `pending` is outstanding permission requests, one entry
 *                 per live `request_id`, `{request_id, tool, input,
 *                 suggestions}`; a replayed duplicate of a still-unanswered
 *                 request is not added twice.
 *                 `agentStatus` mirrors the hello frame's `session.agent`
 *                 ('connecting' | 'ready' | 'unavailable'). `banner` is the most
 *                 recent `session_reset` or `agent_error` event as
 *                 `{kind, text}`, or null once dismissed. A `session_reset`
 *                 also clears `turns` (`resetChatTurns`), because the new
 *                 session's turn numbers start over from 1.
 *                 `model` mirrors the hello frame's `session.model` (the
 *                 CLI-configured starting model, possibly null) until an
 *                 `agent_model_changed` event corrects it to whatever a live
 *                 `set_model` frame actually took effect as: the LLM
 *                 backend's active model, read and written only by the chat
 *                 pane.
 *                 `attachments` is one entry per image attached to the
 *                 message being composed, in the order they were attached,
 *                 whatever state each is in: `[{id, kind, state, path, url,
 *                 bytes, mediaType, message}, ...]` where `state` is
 *                 'uploading' | 'done' | 'error'. The slot is appended before
 *                 the upload starts and filled in when it lands, so the order
 *                 is the human's, not the order the network happened to
 *                 answer in, and both the chips and the `image_path` blocks a
 *                 turn carries follow it. `id` is the identity, assigned here
 *                 and monotonic: `path` cannot be, since an entry has no path
 *                 until its upload has already succeeded. `path` is the
 *                 "images/<name>" string a turn frame carries and `url` is
 *                 "/asset/<name>", what the chip's thumbnail and the
 *                 sent-message thumbnail both fetch; both are null until
 *                 `state` is 'done', as `message` is until it is 'error'.
 *                 An upload in flight is therefore visible as a state on the
 *                 entry itself rather than as a separate count, which is what
 *                 keeps the cap, the strip and Send reading one record of one
 *                 fact. `uploads.js` is the only module that writes any of
 *                 it, since it is the only module that performs an upload.
 *
 *   toolCardsCollapsed
 *                 whether a tool card is created closed. A saved preference
 *                 read once at page load from `GET /settings`, held here
 *                 rather than in `chat.js` because `settings.js` writes it
 *                 and `chat.js` reads it, and a value two modules touch has
 *                 exactly one writer by living in this file.
 */

let state = Object.freeze({
  activeTab: null,
  panelOpen: false,
  paused: false,
  connection: "connecting",
  toolCardsCollapsed: true,
  chat: Object.freeze({
    turns: Object.freeze([]),
    pendingUser: Object.freeze([]),
    pending: Object.freeze([]),
    agentStatus: "connecting",
    model: null,
    banner: null,
    attachments: Object.freeze([]),
  }),
});

// How many images one message may carry. Enforced in exactly one place,
// `reserveChatAttachment` below, and against uploads in flight as well as
// finished ones, since both hold a slot: every caller (the composer's own
// paste, drop and picker paths, and whatever a product attaches through
// uploads.js, such as Mesh's sketch overlay) is refused the same way, before
// its request is sent, once a message already has four.
export const MAX_CHAT_ATTACHMENTS = 4;

// Identity for an attachment slot, assigned in attach order and never reused
// within a page's lifetime. Not the server's path: a slot exists, and is
// already drawn as a chip, before any path is known.
let nextAttachmentId = 1;

const listeners = new Map(); // key (or '*') -> Set<fn(state)>

// True for the duration of a notify pass (the forEach in `apply`). Guards
// against a mutator being called synchronously from inside a subscriber,
// which is a real shape: a subscriber may re-validate its own state against
// the change it was told about and correct it from inside that same
// notification (Mesh's measure.js drops a dropdown selection whose pin has
// gone). Running that call inline would notify a second time from inside the
// first pass's own listener loop, so a listener later in the pass would
// observe a state change its earlier siblings never saw. Queueing defers it,
// so every listener sees one complete state change at a time, in the order
// the mutators were actually called, whichever slice's mutators they are.
let notifying = false;
const pending = [];

function commit(mutate, changedKeys) {
  if (notifying) {
    pending.push([mutate, changedKeys]);
    return;
  }
  apply(mutate, changedKeys);
  // Drained here, by the outermost commit, as a flat loop. A queued mutator
  // that queues another one appends to this same array rather than nesting a
  // commit inside a commit, so a long chain of re-entrant changes iterates
  // instead of growing the stack.
  while (pending.length) {
    const next = pending.shift();
    apply(next[0], next[1]);
  }
}

function apply(mutate, changedKeys) {
  mutate();
  state = Object.freeze(state);
  notifying = true;
  try {
    const keys = new Set([...changedKeys, "*"]);
    keys.forEach((k) => {
      const set = listeners.get(k);
      if (set) set.forEach((fn) => fn(state));
    });
  } finally {
    notifying = false;
  }
}

function getState() {
  return state;
}

function subscribe(key, fn) {
  if (!listeners.has(key)) listeners.set(key, new Set());
  listeners.get(key).add(fn);
  return () => listeners.get(key).delete(fn);
}

// Every key declared so far, this file's and every slice's, so a second
// declaration of one is refused rather than silently giving it two owners.
const declared = new Set(Object.keys(state));

/**
 * Adds a product's slice of state and returns the one function that may
 * change it: `update(keys, fn)`.
 *
 * `initial` is a plain object of the slice's top-level keys and their
 * starting values; a key this file or an earlier slice already declared is
 * refused. Called once, from the product's own store module, at module
 * evaluation, so every key exists before any subscriber can read it.
 *
 * `update(keys, fn)` runs `fn(state)` through the same commit as every other
 * mutator, and notifies the subscribers of `keys`. `fn` returns the new
 * values of the keys it changes (a shallow patch of top-level keys), or
 * nothing to leave state as it is; subscribers of `keys` are notified either
 * way, as they are for this file's own mutators. Every key in `keys` and in
 * the patch must belong to this slice, and every patched key must be named
 * in `keys`, since a key changed without being named would leave its own
 * subscribers unaware of the change.
 */
export function defineSlice(initial) {
  const owned = new Set(Object.keys(initial));
  owned.forEach((key) => {
    if (declared.has(key)) throw new Error("store key " + JSON.stringify(key) + " is already declared");
  });
  owned.forEach((key) => declared.add(key));
  state = Object.freeze({ ...state, ...initial });

  return function update(keys, fn) {
    keys.forEach((key) => {
      if (!owned.has(key)) throw new Error("store key " + JSON.stringify(key) + " is not this slice's");
    });
    commit(() => {
      const patch = fn(state);
      if (!patch) return;
      Object.keys(patch).forEach((key) => {
        if (!keys.includes(key)) {
          throw new Error("store key " + JSON.stringify(key) + " changed without being notified");
        }
      });
      state = { ...state, ...patch };
    }, keys);
  };
}

function setToolCardsCollapsed(collapsed) {
  commit(() => {
    state = { ...state, toolCardsCollapsed: !!collapsed };
  }, ["toolCardsCollapsed"]);
}

function setActiveTab(tabId) {
  commit(() => {
    state = { ...state, activeTab: tabId };
  }, ["activeTab"]);
}

function setPanelOpen(v) {
  commit(() => {
    state = { ...state, panelOpen: !!v };
  }, ["panelOpen"]);
}

function setConnection(v) {
  commit(() => {
    state = { ...state, connection: v };
  }, ["connection"]);
}

function setPaused(v) {
  commit(() => {
    state = { ...state, paused: !!v };
  }, ["paused"]);
}

// Finds the turn record for `turn` in `chat.turns`, or builds one, pairing it
// with the oldest still-unmatched entry in `chat.pendingUser` (the outbound
// `turn` frame carries no id the server echoes back, so the first event for
// a turn number is what associates it with the composer blocks that started
// it). Callers must fold the returned `turns` and `pendingUser` back into a
// new chat object themselves; this only computes the two arrays, it does not
// touch `state`.
function ensureTurn(chat, turn) {
  const idx = chat.turns.findIndex((t) => t.turn === turn);
  if (idx !== -1) {
    return { turns: chat.turns, pendingUser: chat.pendingUser, idx };
  }
  const user = chat.pendingUser.length ? chat.pendingUser[0] : null;
  const pendingUser = chat.pendingUser.length
    ? Object.freeze(chat.pendingUser.slice(1))
    : chat.pendingUser;
  const record = Object.freeze({
    turn,
    user,
    text: "",
    tools: Object.freeze([]),
    stopReason: null,
    costUsd: null,
    complete: false,
  });
  const turns = Object.freeze([...chat.turns, record]);
  return { turns, pendingUser, idx: turns.length - 1 };
}

function appendChatTextDelta(turn, text) {
  commit(() => {
    const chat = state.chat;
    const { turns, pendingUser, idx } = ensureTurn(chat, turn);
    const nextTurns = Object.freeze(
      turns.map((t, i) => (i === idx ? Object.freeze({ ...t, text: t.text + text }) : t)),
    );
    state = { ...state, chat: Object.freeze({ ...chat, turns: nextTurns, pendingUser }) };
  }, ["chat"]);
}

function addChatToolUse(turn, toolUseId, name, input) {
  commit(() => {
    const chat = state.chat;
    const { turns, pendingUser, idx } = ensureTurn(chat, turn);
    const tool = Object.freeze({ tool_use_id: toolUseId, name, input, result: null });
    const nextTurns = Object.freeze(
      turns.map((t, i) =>
        i === idx ? Object.freeze({ ...t, tools: Object.freeze([...t.tools, tool]) }) : t,
      ),
    );
    state = { ...state, chat: Object.freeze({ ...chat, turns: nextTurns, pendingUser }) };
  }, ["chat"]);
}

// Attaches a tool's result to the tool_use record it belongs to, wherever in
// `turns` that record is; a tool_result's frame carries only the
// tool_use_id, not the turn number, so every turn is searched rather than
// assuming the result lands in the same turn its tool_use did.
function setChatToolResult(toolUseId, isError, text) {
  commit(() => {
    const chat = state.chat;
    const turns = Object.freeze(
      chat.turns.map((t) => {
        if (!t.tools.some((tool) => tool.tool_use_id === toolUseId)) return t;
        const tools = Object.freeze(
          t.tools.map((tool) =>
            tool.tool_use_id === toolUseId
              ? Object.freeze({ ...tool, result: Object.freeze({ isError: !!isError, text }) })
              : tool,
          ),
        );
        return Object.freeze({ ...t, tools });
      }),
    );
    state = { ...state, chat: Object.freeze({ ...chat, turns }) };
  }, ["chat"]);
}

function endChatTurn(turn, stopReason, costUsd) {
  commit(() => {
    const chat = state.chat;
    const { turns, pendingUser, idx } = ensureTurn(chat, turn);
    const nextTurns = Object.freeze(
      turns.map((t, i) =>
        i === idx ? Object.freeze({ ...t, stopReason, costUsd, complete: true }) : t,
      ),
    );
    state = { ...state, chat: Object.freeze({ ...chat, turns: nextTurns, pendingUser }) };
  }, ["chat"]);
}

function queueChatUserTurn(blocks) {
  commit(() => {
    const chat = state.chat;
    state = {
      ...state,
      chat: Object.freeze({ ...chat, pendingUser: Object.freeze([...chat.pendingUser, blocks]) }),
    };
  }, ["chat"]);
}

// Adds a permission_request to the pending list unless its request_id is
// already there. Replay re-emits every still-unanswered request on
// reconnect, so this call is not proof of a first sighting.
function addChatPermissionRequest(requestId, tool, input, suggestions) {
  commit(() => {
    const chat = state.chat;
    if (chat.pending.some((p) => p.request_id === requestId)) return;
    const entry = Object.freeze({
      request_id: requestId,
      tool,
      input,
      suggestions: suggestions || null,
    });
    state = {
      ...state,
      chat: Object.freeze({ ...chat, pending: Object.freeze([...chat.pending, entry]) }),
    };
  }, ["chat"]);
}

// Records that this view has sent a decision for `requestId` and is waiting to
// hear that it landed. The card stays on screen, showing that it is in flight,
// because it is the server's `permission_resolved` that says a request is over,
// and only that event distinguishes a decision that took effect from one
// another view had already answered.
function markChatPermissionSubmitted(requestId, decision) {
  commit(() => {
    const chat = state.chat;
    let changed = false;
    const pending = Object.freeze(
      chat.pending.map((p) => {
        if (p.request_id !== requestId || p.submitted) return p;
        changed = true;
        return Object.freeze({ ...p, submitted: decision });
      }),
    );
    if (!changed) return;
    state = { ...state, chat: Object.freeze({ ...chat, pending }) };
  }, ["chat"]);
}

function removeChatPermissionRequest(requestId) {
  commit(() => {
    const chat = state.chat;
    const pending = Object.freeze(chat.pending.filter((p) => p.request_id !== requestId));
    state = { ...state, chat: Object.freeze({ ...chat, pending }) };
  }, ["chat"]);
}


function setChatAgentStatus(status) {
  commit(() => {
    state = { ...state, chat: Object.freeze({ ...state.chat, agentStatus: status }) };
  }, ["chat"]);
}

function setChatModel(model) {
  commit(() => {
    state = { ...state, chat: Object.freeze({ ...state.chat, model: model || null }) };
  }, ["chat"]);
}

function setChatBanner(kind, text) {
  commit(() => {
    state = { ...state, chat: Object.freeze({ ...state.chat, banner: Object.freeze({ kind, text }) }) };
  }, ["chat"]);
}

function clearChatBanner() {
  commit(() => {
    state = { ...state, chat: Object.freeze({ ...state.chat, banner: null }) };
  }, ["chat"]);
}

// Empties `turns` without touching `pendingUser`. Called when a
// session_reset event reports the SDK started a fresh session: that
// session's own turn numbering starts from 1 again, so leaving the old
// turns in place risks a later ensureTurn call finding a stale record
// under the same turn number and folding new content into it. A message
// already queued in `pendingUser` still belongs to the next turn the new
// session produces, so it is left alone.
function resetChatTurns() {
  commit(() => {
    state = { ...state, chat: Object.freeze({ ...state.chat, turns: Object.freeze([]) }) };
  }, ["chat"]);
}

// Appends an entry for an upload about to start, in the 'uploading' state,
// and returns its id, or returns null and changes nothing when the message
// already carries MAX_CHAT_ATTACHMENTS. Reserving the slot before the request
// rather than on its response is what keeps the strip and the sent blocks in
// the order the human attached things, and what lets the cap refuse a file
// before a byte leaves the page or a file is written into the project.
function reserveChatAttachment(kind) {
  let id = null;
  commit(() => {
    const chat = state.chat;
    if (chat.attachments.length >= MAX_CHAT_ATTACHMENTS) return;
    id = nextAttachmentId++;
    const entry = Object.freeze({
      id,
      kind,
      state: "uploading",
      path: null,
      url: null,
      bytes: null,
      mediaType: null,
      message: null,
    });
    state = {
      ...state,
      chat: Object.freeze({ ...chat, attachments: Object.freeze([...chat.attachments, entry]) }),
    };
  }, ["chat"]);
  return id;
}

// Fills a reserved entry in place, so it keeps the position it was attached
// at. An id no longer present (the human removed the chip, or sent the
// message, while the upload was still in flight) matches nothing and changes
// nothing, rather than reappearing: the bytes are on the server either way,
// but a turn references only what is still in this list when it is sent.
function updateChatAttachment(id, fields) {
  commit(() => {
    const chat = state.chat;
    const attachments = Object.freeze(chat.attachments.map(
      (a) => (a.id === id ? Object.freeze({ ...a, ...fields }) : a)));
    state = { ...state, chat: Object.freeze({ ...chat, attachments }) };
  }, ["chat"]);
}

function completeChatAttachment(id, { path, url, bytes, mediaType }) {
  updateChatAttachment(id, { state: "done", path, url, bytes, mediaType });
}

function failChatAttachment(id, message) {
  updateChatAttachment(id, { state: "error", message });
}

function dropChatAttachment(id) {
  commit(() => {
    const chat = state.chat;
    const attachments = Object.freeze(chat.attachments.filter((a) => a.id !== id));
    state = { ...state, chat: Object.freeze({ ...chat, attachments }) };
  }, ["chat"]);
}

function clearChatAttachments() {
  commit(() => {
    state = { ...state, chat: Object.freeze({ ...state.chat, attachments: Object.freeze([]) }) };
  }, ["chat"]);
}

export const store = Object.freeze({
  getState,
  subscribe,
  setToolCardsCollapsed,
  setActiveTab,
  setPanelOpen,
  setConnection,
  setPaused,
  appendChatTextDelta,
  addChatToolUse,
  setChatToolResult,
  endChatTurn,
  queueChatUserTurn,
  addChatPermissionRequest,
  markChatPermissionSubmitted,
  removeChatPermissionRequest,
  setChatAgentStatus,
  setChatModel,
  setChatBanner,
  clearChatBanner,
  resetChatTurns,
  reserveChatAttachment,
  completeChatAttachment,
  failChatAttachment,
  dropChatAttachment,
  clearChatAttachments,
});
