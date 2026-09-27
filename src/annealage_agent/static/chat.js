/**
 * The chat pane: composer, streamed transcript, tool cards, permission
 * cards and interrupt. The store stays the single writer; this module only
 * reads `store.getState().chat` and calls the `chat*` mutators store.js
 * exports, and only builds DOM.
 *
 * It mounts into whatever page hands it a root: every element it drives is
 * found by id under `root` (default: the whole document), through an id map
 * whose defaults (`DEFAULT_IDS` below) are the ids a page gets by copying
 * the pane's markup unchanged, so a product renames only what it has to.
 * `agentTitles` lets the product say what keeps working without an agent,
 * which only it knows.
 *
 * Model text is never inserted as HTML. Every value that came from the
 * agent or from a tool, a tool name, a tool's JSON input, a tool's result
 * text, a permission request's fields, a banner's text, is written with
 * `.textContent`, which cannot be interpreted as markup no matter what it
 * contains. The one exception is the assistant's own free-form turn text,
 * which is allowed a small, fixed set of inline styles; `renderModelText`
 * below is the only place this module ever assigns `.innerHTML`, and it
 * only ever does so after `escapeHtml` has run over the raw string first,
 * so every character that could open a tag has already become an entity
 * before any of the markdown-style regexes see it. Each regex wraps an
 * already-escaped substring in a fixed literal tag; none of them re-reads
 * or re-matches text a previous one inserted, so there is no path back
 * from "escaped text" to "text a later step interprets as markup".
 *
 * Reconciliation uses a Map keyed by a stable id
 * (turn number, tool_use_id, permission request_id) so a re-render updates
 * an existing element in place rather than replacing it, which is what
 * keeps a `<details>` tool card's open/closed state and the composer's
 * focus and caret position untouched by an unrelated event arriving.
 *
 * Attachment chips follow the same reconciliation pattern, keyed by the
 * store's own attachment id rather than by `path`, because a chip exists (in
 * its "uploading" state) before an upload has produced a path at all. Every
 * chip is drawn from `state.chat.attachments`, whatever state the entry is
 * in, so this module keeps no second record of what is attached and cannot
 * disagree with the one the cap and Send both read. `uploadImage` is the one
 * function behind all three ways an image enters the composer: the file
 * picker, a paste onto the textarea, and a drop onto the chat pane. A sent
 * turn's own thumbnails are a second, simpler case: built once from
 * `record.user`'s `image_path` blocks and never reconciled again, since a
 * turn's composer blocks do not change after the turn is sent.
 */

import { initAttention, notifyAttention } from "./attention.js";
import { store } from "./store.js";
import { uploadImage } from "./uploads.js";
import { toast } from "./ui.js";
import { authToken } from "./ws.js";

// The three image types the upload route accepts (files.py's
// `_IMAGE_NAME_RE`/`sniff_image`) and the same byte cap it enforces
// (`files.MAX_IMAGE_BYTES`). Checking both here means a wrong-typed file or
// an oversized drop is refused before a single byte leaves the page, rather
// than after the whole body has streamed to the server only to be refused
// there. A file this accepts may still be too large to send to the model
// inline, which the server decides and reports per attachment; it is stored
// and named either way, so that is not a reason to refuse it here.
const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024;

function localAttachmentRefusal(file) {
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
    return file.name + ": only PNG, JPEG or WEBP images can be attached";
  }
  if (file.size > MAX_ATTACHMENT_BYTES) {
    return file.name + " is larger than the 8 MB upload limit";
  }
  return null;
}

// What a card shows while its decision is in flight, keyed by the decision
// this view sent.
const DECISION_SENT = Object.freeze({
  allow: "Allow sent…",
  allow_always: "Always allow sent…",
  deny: "Deny sent…",
});

// How a resolution reads in a toast. The three that nobody clicked are worth
// naming precisely: a card that vanishes because it expired looks, without
// this, exactly like one somebody answered.
const OUTCOME_TEXT = Object.freeze({
  allow: "was allowed",
  // Not "for this session": `_remember` writes the grant to the project's
  // permissions.toml (in the product's state directory) and `_load_grants`
  // reads it back when the broker is constructed, so it holds for every later
  // run in this directory too. The grant is also per-tool rather than
  // per-argument, so a card showing one file's contents grants that tool for
  // any path. This string is the only description the human necessarily
  // reads at the moment they decide, so it says what the grant actually does.
  allow_always: "will be allowed from now on, for that tool, in this project",
  deny: "was denied",
  timeout: "expired before it was answered",
  no_viewer: "was cancelled when every view disconnected",
  shutdown: "was cancelled by shutdown",
});

// The outcomes that mean a person clicked something, as opposed to the request
// running out of time or of viewers.
const HUMAN_DECISIONS = new Set(["allow", "allow_always", "deny"]);

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Applies the non-fence inline styles (inline code, bold, italic, newline)
// to a string that has already been through escapeHtml. Order matters:
// inline code runs first so a literal `*` or `**` inside a code span is
// never read as emphasis, and bold runs before italic so `**x**` is not
// left as an unmatched pair of single asterisks once the double-asterisk
// match is gone.
function renderInline(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}

/**
 * Escapes `text`, then applies a fixed, small set of transforms: fenced
 * code blocks become `<pre>`, and within the rest, inline code, bold,
 * italic and newlines get their usual HTML equivalents. This is a
 * hand-written renderer over already-escaped text, not a markdown parser
 * and not `innerHTML` of the model's own output; see this module's header
 * comment for why that distinction is the whole point of it.
 */
function renderModelText(text) {
  const escaped = escapeHtml(text);
  const fenceRe = /```[^\n`]*\n([\s\S]*?)```/g;
  const segments = [];
  let lastIndex = 0;
  let m;
  while ((m = fenceRe.exec(escaped))) {
    segments.push({ code: false, text: escaped.slice(lastIndex, m.index) });
    segments.push({ code: true, text: m[1] });
    lastIndex = m.index + m[0].length;
  }
  segments.push({ code: false, text: escaped.slice(lastIndex) });
  return segments
    .map((seg) => (seg.code ? '<pre class="code">' + seg.text + "</pre>" : renderInline(seg.text)))
    .join("");
}

// Ordinary Claude Code tools (Bash, Edit, Write, ...) keep their plain
// name; an in-process MCP tool arrives as `mcp__<server>__<tool>` and is
// shown as `<server>: <tool>` instead of the wire form.
function shortToolName(name) {
  const m = /^mcp__([^_]+)__(.+)$/.exec(name);
  return m ? m[1] + ": " + m[2] : name;
}

function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

// A tool's arguments, laid out to be read rather than parsed. JSON alone is the
// wrong shape for the arguments that matter most here: a Bash command is a
// script, and stringifying it turns every line break into a literal \n, so the
// one field the human most needs to inspect before approving it arrives as a
// single unreadable line. String values are therefore printed as themselves,
// with their newlines intact; anything else falls back to JSON.
//
// The result is always assigned with textContent, never innerHTML: these values
// come from the model.
function formatToolInput(input) {
  if (input === null || input === undefined) return "";
  if (typeof input !== "object") return String(input);
  const keys = Object.keys(input);
  if (!keys.length) return "";
  // The overwhelmingly common shape, and the one where a key label adds
  // nothing: one string argument, such as a command or a path.
  if (keys.length === 1 && typeof input[keys[0]] === "string") {
    return input[keys[0]];
  }
  return keys
    .map((key) => {
      const value = input[key];
      const shown = typeof value === "string" ? value : safeJson(value);
      return shown.includes("\n") ? key + ":\n" + shown : key + ": " + shown;
    })
    .join("\n");
}

// Joins the text blocks of a turn's composer blocks for the bubble's text;
// the same blocks' `image_path` entries are rendered separately, as
// thumbnails, by renderUserAttachments below.
function blocksToText(blocks) {
  if (!blocks) return "";
  return blocks
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n\n");
}

// An `image_path` block's `path` is "images/<name>" (protocol.py's block
// shape, one directory component); the image itself is served one
// component further in, at "/asset/<name>".
function assetUrlForImagePath(path) {
  const name = path.startsWith("images/") ? path.slice("images/".length) : path;
  return "/asset/" + name;
}

// The name this page gives one message it sends: the turn frame's
// `client_id`, echoed back in the `user_turn` event that logs the message or
// the `refused` frame that turns it away. It only has to be unique among the
// tabs of one conversation, and protocol.py takes up to 64 letters, digits,
// '-' and '_'. randomUUID is missing on a plain-http tailnet address, which
// is not a secure context, hence the fallback.
function makeClientId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return "m" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

// Builds a sent turn's own thumbnail row from its blocks, once: `container`
// starts empty and this only ever appends, since `record.user` is set once,
// from the turn's `user_turn` event, and never changes afterwards. This is
// why a sent turn's images come from that event, the blocks exactly as the
// page sent them, rather than from the agent's echo of the turn: the SDK's
// message parser drops an image block out of that echo with no error, so the
// echo cannot be relied on to show what was actually sent.
// Every element is built with createElement and given its `src` directly;
// this pane never assigns HTML from a value that did not originate here.
function renderUserAttachments(container, blocks) {
  if (container.childElementCount || !blocks) return;
  blocks
    .filter((b) => b.type === "image_path")
    .forEach((b) => {
      const img = document.createElement("img");
      img.src = assetUrlForImagePath(b.path);
      img.alt = "";
      container.appendChild(img);
    });
}

const AGENT_LABEL = { connecting: "Connecting…", ready: "Ready", unavailable: "Unavailable" };
// The generic tooltips; `initChat`'s `agentTitles` replaces any of them.
const AGENT_TITLE = {
  connecting: "The agent process is starting.",
  ready: "The agent is ready to receive a message.",
  unavailable:
    "The agent is not available right now. The rest of the page keeps working regardless.",
};

// The ids the pane's elements have in the markup a page copies, by role.
// `pane` is the whole pane (the file drop target); `exportButton` is the one
// optional element, for a page that offers no transcript export.
export const DEFAULT_IDS = Object.freeze({
  pane: "chat",
  log: "chatLog",
  pending: "chatPending",
  agentStatus: "agentStatus",
  banner: "chatBanner",
  bannerText: "chatBannerText",
  bannerClose: "chatBannerClose",
  input: "chatInput",
  send: "chatSend",
  modelInput: "chatModelInput",
  interrupt: "chatInterrupt",
  attachButton: "chatAttachBtn",
  fileInput: "chatFileInput",
  attachStrip: "chatAttachStrip",
  exportButton: "chatExport",
});

const OPTIONAL_ELEMENTS = new Set(["exportButton"]);

// The element with id `id` under `root`, `root` itself included: a page may
// hand over the pane element as the root and keep its default id on it.
function findById(root, id) {
  if (root.nodeType === Node.ELEMENT_NODE && root.id === id) return root;
  return root.querySelector("#" + CSS.escape(id));
}

// The block under the banner's message that holds the backend's own text (an
// agent_error's stderr): collapsible, and capped in height by agent.css. Built
// here rather than asked of the page's markup, so every product's banner has
// it; reused if this banner already has one.
function bannerDetail(bannerEl) {
  const existing = bannerEl.querySelector(":scope > details.chatbannerdetail");
  if (existing) return existing;
  const details = document.createElement("details");
  details.className = "chatbannerdetail";
  details.hidden = true;
  const summary = document.createElement("summary");
  summary.textContent = "What the agent backend said";
  details.append(summary, document.createElement("pre"));
  return bannerEl.appendChild(details);
}

/**
 * Mounts the pane. `send` is ws.js's frame sender; `root` is where the pane's
 * elements are looked up (default: the document); `ids` overrides any of
 * `DEFAULT_IDS`; `agentTitles` overrides any of the agent status tooltips.
 * A required element that is missing is an error here, at mount, naming the
 * id, rather than a null dereference on the first event.
 */
export function initChat({ send, root = document, ids = {}, agentTitles = {} }) {
  const idMap = { ...DEFAULT_IDS, ...ids };
  const els = {};
  for (const [role, id] of Object.entries(idMap)) {
    els[role] = findById(root, id);
    if (!els[role] && !OPTIONAL_ELEMENTS.has(role)) {
      throw new Error("chat pane: no element #" + id + " (" + role + ") under the given root");
    }
  }
  const titles = { ...AGENT_TITLE, ...agentTitles };
  const chatLogEl = els.log;
  const chatPendingEl = els.pending;
  const agentStatusEl = els.agentStatus;
  const bannerEl = els.banner;
  const bannerTextEl = els.bannerText;
  const bannerCloseBtn = els.bannerClose;
  const bannerDetailEl = bannerDetail(bannerEl);
  const chatInputEl = els.input;
  const chatSendBtn = els.send;
  const chatModelInputEl = els.modelInput;
  const chatInterruptBtn = els.interrupt;
  const chatAttachBtn = els.attachButton;
  const chatFileInput = els.fileInput;
  const chatAttachStripEl = els.attachStrip;
  const chatPaneEl = els.pane;
  const chatExportBtn = els.exportButton;

  // The session this pane belongs to, learned from the hello frame. Null until
  // then, and null for a viewer-only run, which has no conversation to write.
  let sessionId = null;
  // Whether a message sent while a turn runs redirects it (the hello's
  // `steers`, the omp backend), which is what the Send button then says.
  let steers = false;
  initAttention(root);

  // turn number -> {row, textEl, toolsEl, metaEl, userEl, tools: Map<tool_use_id, {card, resultEl}>}
  const turnEls = new Map();
  // request_id -> element
  const pendingEls = new Map();
  // "att:<id>" for each entry in state.chat.attachments, whatever state it is
  // in. Keyed by the store's id rather than by `path`, since a chip is drawn
  // for an upload still in flight and that has no path yet.
  const attachEls = new Map();

  function buildAttachChip() {
    const chip = document.createElement("div");
    chip.className = "attachchip";
    const spinner = document.createElement("div");
    spinner.className = "attachspinner";
    const thumb = document.createElement("img");
    thumb.className = "attachthumb";
    thumb.hidden = true;
    thumb.alt = "";
    const errText = document.createElement("span");
    errText.className = "attacherr";
    errText.hidden = true;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "attachremove";
    removeBtn.textContent = "×";
    removeBtn.setAttribute("aria-label", "Remove attachment");
    chip.appendChild(spinner);
    chip.appendChild(thumb);
    chip.appendChild(errText);
    chip.appendChild(removeBtn);
    return { chip, spinner, thumb, errText, removeBtn };
  }

  // `att` is one entry of state.chat.attachments; the chip element is reused
  // across its states, keyed by `attachEls` above.
  function updateAttachChip(rec, att) {
    const uploading = att.state === "uploading";
    const errored = att.state === "error";
    const done = att.state === "done";
    rec.chip.classList.toggle("error", errored);
    rec.spinner.hidden = !uploading;
    rec.thumb.hidden = !done;
    if (done) rec.thumb.src = att.url;
    rec.errText.hidden = !errored;
    if (errored) rec.errText.textContent = att.message;
    // Not removable while still in flight: dropping the slot now would leave
    // the upload with nowhere to land and no failure yet to dismiss.
    rec.removeBtn.hidden = uploading;
    // Triggers the store's own "chat" subscriber (render(), below), which
    // redraws this strip with the entry gone.
    rec.removeBtn.onclick = () => store.dropChatAttachment(att.id);
  }

  function renderAttachStrip(chat) {
    const keys = chat.attachments.map((a) => "att:" + a.id);
    chatAttachStripEl.hidden = keys.length === 0;
    chat.attachments.forEach((att) => {
      const key = "att:" + att.id;
      let rec = attachEls.get(key);
      if (!rec) {
        rec = buildAttachChip();
        attachEls.set(key, rec);
        chatAttachStripEl.appendChild(rec.chip);
      }
      updateAttachChip(rec, att);
    });
    for (const [key, rec] of attachEls) {
      if (!keys.includes(key)) {
        rec.chip.remove();
        attachEls.delete(key);
      }
    }
  }

  // The one function all three ways of attaching an image call: the file
  // picker, a paste onto the textarea, and a drop onto this pane. Each file
  // gets its own chip immediately, in the "uploading" state, so a slow
  // upload never leaves the human wondering whether the drop or paste was
  // even noticed.
  function handleFile(file) {
    const refusal = localAttachmentRefusal(file);
    if (refusal) {
      toast(refusal, false);
      return;
    }
    // Not awaited: several files dropped at once each get their chip and their
    // request straight away, and each lands in the slot it reserved here.
    uploadImage(file, "upload");
  }

  chatAttachBtn.addEventListener("click", () => chatFileInput.click());
  chatFileInput.addEventListener("change", () => {
    Array.from(chatFileInput.files || []).forEach(handleFile);
    // Cleared so picking the same file again still fires "change".
    chatFileInput.value = "";
  });

  chatInputEl.addEventListener("paste", (e) => {
    const files = Array.from((e.clipboardData && e.clipboardData.files) || []).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (!files.length) return; // an ordinary text paste: leave it to the browser
    e.preventDefault();
    files.forEach(handleFile);
  });

  // dragover must be prevented for `drop` to fire at all (the platform
  // default is to refuse the pane as a drop target); drop must be prevented
  // separately, or a browser that reaches this handler navigates the whole
  // tab to the dropped file once the handler returns.
  chatPaneEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    chatPaneEl.classList.add("dragover");
  });
  chatPaneEl.addEventListener("dragleave", (e) => {
    if (!e.relatedTarget || !chatPaneEl.contains(e.relatedTarget)) {
      chatPaneEl.classList.remove("dragover");
    }
  });
  chatPaneEl.addEventListener("drop", (e) => {
    e.preventDefault();
    chatPaneEl.classList.remove("dragover");
    Array.from((e.dataTransfer && e.dataTransfer.files) || []).forEach(handleFile);
  });

  let stickToBottom = true;
  chatLogEl.addEventListener("scroll", () => {
    stickToBottom =
      chatLogEl.scrollTop + chatLogEl.clientHeight >= chatLogEl.scrollHeight - 24;
  });

  function buildToolCard(tool) {
    const card = document.createElement("details");
    card.className = "toolcard";
    card.dataset.toolId = tool.tool_use_id;
    // A saved preference, read from the store rather than decided here, so the
    // settings window and this builder cannot disagree about the default.
    card.open = store.getState().toolCardsCollapsed === false;
    const summary = document.createElement("summary");
    const inputEl = document.createElement("pre");
    inputEl.className = "toolinput";
    const resultEl = document.createElement("pre");
    resultEl.className = "toolresult";
    resultEl.hidden = true;
    card.appendChild(summary);
    card.appendChild(inputEl);
    card.appendChild(resultEl);
    return { card, summary, inputEl, resultEl };
  }

  function updateToolCard(rec, tool) {
    rec.summary.textContent = shortToolName(tool.name);
    rec.inputEl.textContent = formatToolInput(tool.input);
    if (tool.result) {
      rec.resultEl.hidden = false;
      rec.resultEl.textContent = tool.result.text;
      rec.resultEl.classList.toggle("error", !!tool.result.isError);
    } else {
      rec.resultEl.hidden = true;
    }
  }

  function buildTurnRow(turn) {
    const row = document.createElement("div");
    row.className = "turn";
    row.dataset.turn = String(turn);

    const userEl = document.createElement("div");
    userEl.className = "msg user";
    userEl.hidden = true;
    const userText = document.createElement("div");
    userText.className = "text";
    const userAttachmentsEl = document.createElement("div");
    userAttachmentsEl.className = "attachthumbs";
    userEl.appendChild(userText);
    userEl.appendChild(userAttachmentsEl);

    const assistantEl = document.createElement("div");
    assistantEl.className = "msg assistant";
    const textEl = document.createElement("div");
    textEl.className = "text";
    const toolsEl = document.createElement("div");
    toolsEl.className = "tools";
    const metaEl = document.createElement("div");
    metaEl.className = "turnmeta";
    assistantEl.appendChild(textEl);
    assistantEl.appendChild(toolsEl);
    assistantEl.appendChild(metaEl);

    row.appendChild(userEl);
    row.appendChild(assistantEl);

    return { row, userEl, userText, userAttachmentsEl, textEl, toolsEl, metaEl, tools: new Map() };
  }

  function updateTurnRow(rec, t) {
    if (t.user) {
      rec.userEl.hidden = false;
      rec.userText.textContent = blocksToText(t.user);
      renderUserAttachments(rec.userAttachmentsEl, t.user);
    }
    rec.textEl.innerHTML = renderModelText(t.text);

    const seenTools = new Set();
    t.tools.forEach((tool) => {
      seenTools.add(tool.tool_use_id);
      let toolRec = rec.tools.get(tool.tool_use_id);
      if (!toolRec) {
        toolRec = buildToolCard(tool);
        rec.tools.set(tool.tool_use_id, toolRec);
        rec.toolsEl.appendChild(toolRec.card);
      }
      updateToolCard(toolRec, tool);
    });
    for (const [id, toolRec] of rec.tools) {
      if (!seenTools.has(id)) {
        toolRec.card.remove();
        rec.tools.delete(id);
      }
    }

    if (t.complete) {
      const tokens = t.tokens
        ? " · " + t.tokens.input + " in / " + t.tokens.output + " out"
        : "";
      rec.metaEl.textContent = "Stop: " + t.stopReason + " · $" + t.costUsd.toFixed(4) + tokens;
    } else {
      rec.metaEl.textContent = "";
    }
  }

  function renderTurns(chat) {
    const seen = new Set();
    chat.turns.forEach((t) => {
      seen.add(t.turn);
      let rec = turnEls.get(t.turn);
      if (!rec) {
        rec = buildTurnRow(t.turn);
        turnEls.set(t.turn, rec);
        chatLogEl.appendChild(rec.row);
      }
      updateTurnRow(rec, t);
    });
    for (const [turn, rec] of turnEls) {
      if (!seen.has(turn)) {
        rec.row.remove();
        turnEls.delete(turn);
      }
    }
    if (stickToBottom) chatLogEl.scrollTop = chatLogEl.scrollHeight;
  }

  function buildPermissionCard(req) {
    const card = document.createElement("div");
    card.className = "permcard";
    card.dataset.requestId = req.request_id;

    const toolEl = document.createElement("div");
    toolEl.className = "ptool";
    const inputEl = document.createElement("pre");
    inputEl.className = "toolinput";

    const actions = document.createElement("div");
    actions.className = "pactions";
    const allowBtn = document.createElement("button");
    allowBtn.type = "button";
    allowBtn.textContent = "Allow";
    const allowAlwaysBtn = document.createElement("button");
    allowAlwaysBtn.type = "button";
    // "Always" rather than "for this session", matching what the grant does.
    allowAlwaysBtn.textContent = "Always allow (this project)";
    const reasonEl = document.createElement("textarea");
    reasonEl.className = "preason";
    reasonEl.placeholder = "Reason (sent to the agent if you deny)";
    reasonEl.rows = 1;
    const denyBtn = document.createElement("button");
    denyBtn.type = "button";
    denyBtn.className = "danger";
    denyBtn.textContent = "Deny";

    // The card is not removed here. A decision is a request to the server, not
    // the answer: another view may already have answered this card, or it may
    // have expired, in which case this click decides nothing. Removing it now
    // would render those cases identically to a decision that took effect,
    // which for a Deny the human believes they made is the worst of the three.
    // `permission_resolved` retires the card, and carries the outcome that
    // actually applied.
    function decide(decision) {
      const message = decision === "deny" ? reasonEl.value : "";
      send({ type: "permission", request_id: req.request_id, decision, message });
      store.markChatPermissionSubmitted(req.request_id, decision);
    }
    allowBtn.addEventListener("click", () => decide("allow"));
    allowAlwaysBtn.addEventListener("click", () => decide("allow_always"));
    denyBtn.addEventListener("click", () => decide("deny"));

    // A request the server never remembers (resolving one of the human's own
    // review comments, say) is a question about this one call, so it gets
    // no "always" button at all; the server downgrades one anyway.
    const buttons = req.rememberable ? [allowBtn, allowAlwaysBtn, denyBtn] : [allowBtn, denyBtn];
    actions.appendChild(allowBtn);
    if (req.rememberable) actions.appendChild(allowAlwaysBtn);
    actions.appendChild(reasonEl);
    actions.appendChild(denyBtn);

    const statusEl = document.createElement("div");
    statusEl.className = "pstatus";

    card.appendChild(toolEl);
    card.appendChild(inputEl);
    card.appendChild(actions);
    card.appendChild(statusEl);

    return { card, toolEl, inputEl, statusEl, buttons, reasonEl };
  }

  // Tells the human when a card ended in a way they would otherwise have to
  // guess at: a decision this view sent that did not apply, and a resolution
  // nobody chose. A card another view answered while this one was only
  // watching just disappears, which is what a second device answering is
  // supposed to look like.
  function reportResolution(event) {
    const req = store.getState().chat.pending.find(
      (p) => p.request_id === event.request_id,
    );
    if (!req) return;
    const what = OUTCOME_TEXT[event.outcome] || ("ended as " + event.outcome);
    const tool = shortToolName(req.tool);
    if (req.submitted && req.submitted !== event.outcome) {
      toast(tool + ": your decision did not apply, the request " + what, false);
    } else if (!req.submitted && !HUMAN_DECISIONS.has(event.outcome)) {
      toast(tool + ": the request " + what, false);
    }
  }

  function renderPending(chat) {
    const seen = new Set();
    chat.pending.forEach((req) => {
      seen.add(req.request_id);
      let rec = pendingEls.get(req.request_id);
      if (!rec) {
        rec = buildPermissionCard(req);
        pendingEls.set(req.request_id, rec);
        chatPendingEl.appendChild(rec.card);
      }
      rec.toolEl.textContent = shortToolName(req.tool);
      rec.inputEl.textContent = formatToolInput(req.input);
      // A card whose decision is in flight stays visible and stops accepting
      // clicks, so a second click cannot send a second decision for one
      // request and the human can see which answer is on its way.
      const submitted = req.submitted || "";
      rec.buttons.forEach((b) => { b.disabled = !!submitted; });
      rec.reasonEl.disabled = !!submitted;
      rec.card.classList.toggle("submitted", !!submitted);
      rec.statusEl.textContent = submitted ? DECISION_SENT[submitted] : "";
    });
    for (const [id, rec] of pendingEls) {
      if (!seen.has(id)) {
        rec.card.remove();
        pendingEls.delete(id);
      }
    }
  }

  function renderAgentStatus(chat) {
    agentStatusEl.textContent = AGENT_LABEL[chat.agentStatus] || chat.agentStatus;
    agentStatusEl.title = titles[chat.agentStatus] || "";
    agentStatusEl.dataset.state = chat.agentStatus;
  }

  // Set to the requested model right before sending `set_model`, and
  // cleared by whichever of two frames answers it first: a matching
  // `agent_model_changed` (the switch took effect) or the next `refused`
  // frame seen afterwards that names no turn (protocol.py's build_refused
  // correlates a refused turn by its client_id and nothing else, so "the
  // next uncorrelated one" is the only signal this pane has that its own
  // request -- not some unrelated permission refusal -- was the one
  // rejected). `handleRefused` below reads this to decide whether a given
  // refusal is its business at all.
  let pendingSetModel = null;

  // A refusal that needs to revert the field, but arrived while the field
  // had focus: renderModel intentionally skips a focused field (see
  // below), so the revert cannot be applied immediately without stomping
  // on whatever the human is mid-typing. Recorded here instead of dropped,
  // so the next blur (or a further edit, which supersedes it) applies it
  // -- otherwise a rejected value can stay displayed forever, since only
  // another edit's `change` event would ever call renderModel again.
  let queuedModelRevert = false;

  // The model picker is a free-text field, not a dropdown: enumerating what
  // a given backend/endpoint actually offers is out of this ticket's scope
  // (protocol.py's build_hello and this pane only carry the *current*
  // value). `change` fires on blur once the value differs from what it was
  // on focus, which is what lets a human type a full model id without a
  // frame going out on every keystroke.
  //
  // This field tracks `agent_model_changed`, the LLM backend's active model;
  // a product's own event about its files (Mesh's `models_changed`) is taken
  // by the product's handler in ws.js and never reaches this pane.
  function applyModelInput() {
    // Any fresh edit supersedes a revert queued by an earlier refusal: the
    // human has already moved past that rejected value, so there is
    // nothing left for the queued revert to correct.
    queuedModelRevert = false;
    const value = chatModelInputEl.value.trim();
    const current = store.getState().chat.model || "";
    if (!value || value === current) {
      // Empty or unchanged: not a switch request, so the field is put back
      // to the last known value rather than left showing something that was
      // never sent and never took effect.
      chatModelInputEl.value = current;
      return;
    }
    pendingSetModel = value;
    if (!send({ type: "set_model", model: value })) {
      // send() no-ops silently when the socket is not OPEN; no refusal
      // (or confirming agent_model_changed) will ever arrive for a
      // request that was never transmitted, so pendingSetModel cannot be
      // left set waiting for an answer that is not coming. Revert through
      // the same focus-aware path a genuine refusal uses -- from this
      // pane's point of view a request that never went out failed exactly
      // as thoroughly as one the server rejected.
      pendingSetModel = null;
      revertPendingModel();
    }
  }

  chatModelInputEl.addEventListener("change", applyModelInput);
  chatModelInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      chatModelInputEl.blur(); // fires "change" above if the value differs
    }
  });
  // `change` alone cannot apply a queued revert: it only fires when the
  // field's value differs from its value at focus time, which is exactly
  // false in the stuck case (the human refocused, made no further edit,
  // then blurred) that queuedModelRevert exists to fix. `blur` fires
  // regardless of whether the value changed.
  chatModelInputEl.addEventListener("blur", () => {
    if (queuedModelRevert) {
      queuedModelRevert = false;
      renderModel(store.getState().chat);
    }
  });

  // The one writer of the model field's value and disabled state; mirrors
  // renderSendButton's reasoning for gating on `agentStatus`. Skipped while
  // the field has focus, so a store update racing a human mid-edit (a
  // reconnect's hello, another tab's own AgentModelChanged) cannot overwrite
  // what they are typing.
  function renderModel(chat) {
    chatModelInputEl.disabled = chat.agentStatus === "unavailable";
    if (document.activeElement !== chatModelInputEl) {
      chatModelInputEl.value = chat.model || "";
    }
  }

  // The one writer of the Send button's state, because there are two
  // independent reasons to refuse a send and a function per reason would leave
  // whichever ran last deciding for both.
  //
  // Only a ready agent takes a message: the server refuses one sent while the
  // agent is starting or gone, so the button says so rather than offering a
  // send that comes straight back.
  //
  // An upload still in flight is a refusal rather than a partial send: posting
  // the turn without it would silently omit the image the human is waiting on
  // and then carry it on whatever they typed next, which reads as the picture
  // having been ignored.
  function renderSendButton(chat) {
    const ready = chat.agentStatus === "ready";
    const uploading = chat.attachments.some((a) => a.state === "uploading");
    const busy = chat.pendingUser.length > 0 || chat.turns.some((t) => !t.complete);
    const steering = steers && busy && ready;
    chatSendBtn.disabled = !ready || uploading;
    chatSendBtn.textContent = steering ? "Steer" : "Send";
    chatSendBtn.title = uploading
      ? "Waiting for an attachment to finish uploading"
      : (!ready ? titles[chat.agentStatus] || ""
        : (steering ? "The agent is working: this message redirects it now" : ""));
  }

  function renderBanner(chat) {
    if (chat.banner) {
      bannerEl.hidden = false;
      bannerEl.dataset.kind = chat.banner.kind;
      bannerTextEl.textContent = chat.banner.text;
      // Opened whenever it shows something new, so the backend's words are
      // seen, and left as the human set it while the same text stays.
      const detail = chat.banner.detail || "";
      const pre = bannerDetailEl.querySelector("pre");
      if (pre.textContent !== detail) {
        pre.textContent = detail;
        bannerDetailEl.open = true;
      }
      bannerDetailEl.hidden = !detail;
    } else {
      bannerEl.hidden = true;
    }
  }

  function renderInterrupt(chat) {
    const busy = chat.pendingUser.length > 0 || chat.turns.some((t) => !t.complete);
    chatInterruptBtn.disabled = !busy;
  }

  function render() {
    const chat = store.getState().chat;
    renderTurns(chat);
    renderPending(chat);
    renderAgentStatus(chat);
    renderModel(chat);
    renderBanner(chat);
    renderInterrupt(chat);
    renderAttachStrip(chat);
    renderSendButton(chat);
  }

  store.subscribe("chat", render);
  render();

  function doSend() {
    const text = chatInputEl.value.trim();
    const chat = store.getState().chat;
    const attachments = chat.attachments;
    // Held while the agent is not ready or anything is still uploading: the
    // button is already disabled for both, and this covers the Enter key,
    // which is not the button. The message stays in the composer.
    if (chat.agentStatus !== "ready") return;
    if (attachments.some((a) => a.state === "uploading")) return;
    const ready = attachments.filter((a) => a.state === "done");
    if (!text && !ready.length) return; // an attachment with no text is allowed; neither is not
    const blocks = ready.map((a) => ({ type: "image_path", path: a.path }));
    // Only when there is something in it. An empty text block is refused by
    // the API outright, failing the whole turn including the image, and an
    // attachment with no typed message is a case this composer allows.
    if (text) blocks.push({ type: "text", text });
    const clientId = makeClientId();
    if (!send({ type: "turn", blocks, client_id: clientId })) {
      // Nothing went out, so nothing will answer it: the message stays in
      // the composer to be sent again once the socket is back.
      toast("Not connected, so the message was not sent", false);
      return;
    }
    // Held as pending, not drawn: the server's `user_turn` for it, which
    // every tab receives, is what shows the message.
    store.queueChatUserTurn(clientId, blocks);
    chatInputEl.value = "";
    store.clearChatAttachments();
  }

  chatSendBtn.addEventListener("click", doSend);
  chatInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSend();
    }
  });

  chatInterruptBtn.addEventListener("click", () => {
    send({ type: "interrupt" });
  });

  bannerCloseBtn.addEventListener("click", () => store.clearChatBanner());

  function renderExportButton() {
    if (!chatExportBtn) return;
    const exportable = !!sessionId && sessionId !== "viewer-only";
    chatExportBtn.disabled = !exportable || exporting;
    chatExportBtn.title = exportable
      ? "Write this conversation into review/ as a file you can commit or share"
      : "There is no conversation to export in this run";
  }

  let exporting = false;

  async function exportTranscript() {
    if (exporting || !sessionId) return;
    exporting = true;
    renderExportButton();
    try {
      const res = await fetch(
        `/session/${encodeURIComponent(sessionId)}/export?t=${encodeURIComponent(authToken())}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ format: "markdown", include: "full" }),
        });
      let payload = null;
      try {
        payload = await res.json();
      } catch (err) {
        payload = null;
      }
      if (!res.ok || !payload || payload.ok !== true) {
        store.setChatBanner("error",
          "The transcript could not be written: " +
          ((payload && payload.error) || "the server refused the request"));
        return;
      }
      // The path is reported rather than a bare "done", because the file is the
      // point and it is written into the human's own project.
      store.setChatBanner("info", "Transcript written to " + payload.path);
    } finally {
      exporting = false;
      renderExportButton();
    }
  }

  if (chatExportBtn) {
    chatExportBtn.addEventListener("click", () => { exportTranscript(); });
    renderExportButton();
  }

  // Messages this pane sent over an earlier connection that neither a
  // user_turn nor a refusal had answered when a new connection opened. The
  // new connection's replay says what became of each: one the server logged
  // arrives as its user_turn, which retires it; one that never reached the
  // server (the socket dropped with the frame in flight) does not, and goes
  // back into the composer once the replay is over, rather than vanishing.
  // The replay is over at the first live event, or, when none comes, once
  // the replay has been quiet for SETTLE_MS.
  const SETTLE_MS = 1500;
  let unconfirmed = new Set();
  let settleTimer = null;

  function settleUnconfirmed() {
    clearTimeout(settleTimer);
    settleTimer = null;
    const lost = [];
    for (const clientId of unconfirmed) {
      const blocks = store.dropChatUserTurn(clientId);
      if (blocks) lost.push(...blocks);
    }
    unconfirmed = new Set();
    if (!lost.length) return;
    const complete = restoreToComposer(lost);
    toast(restoredText("A message sent as the connection dropped never arrived", complete), false);
  }

  function handleHello(session) {
    if (session) {
      // A fresh connection's hello is authoritative, the same way it
      // already is for agent status and model below: any set_model this
      // pane sent on a prior connection is moot by now, whether the
      // request itself raced the disconnect and was never transmitted, or
      // it went out but its response (agent_model_changed or a refused
      // frame) was lost when the socket dropped before it arrived.
      // Clearing both here, rather than leaving them for a refusal that
      // may never come, keeps a stale pending flag from leaving the field
      // stuck on a rejected value and from this connection's first
      // unrelated refusal being blamed on that earlier request.
      pendingSetModel = null;
      queuedModelRevert = false;
      store.setChatAgentStatus(session.agent);
      store.setChatModel(session.model);
      // Why the agent is down, for a page opened after it went down: the
      // agent_error event that said so is history in this connection's
      // replay, which raises no banner.
      if (session.agent_error) {
        store.setChatBanner(
          "error",
          session.agent_error.remediation || "The agent reported an error.",
          session.agent_error.stderr,
        );
      }
    }
    // What this pane sent over an earlier connection is settled by the replay
    // that follows (see `unconfirmed`).
    unconfirmed = new Set(store.getState().chat.pendingUser.map((p) => p.clientId));
    clearTimeout(settleTimer);
    settleTimer = unconfirmed.size ? setTimeout(settleUnconfirmed, SETTLE_MS) : null;
    // A different session from the one this pane was showing (the server
    // restarted into a new conversation): the turns on screen belong to
    // another conversation, and this one's history arrives in the replay.
    const nextSessionId = session && session.id ? session.id : null;
    if (sessionId !== null && nextSessionId !== sessionId) store.resetChatTurns();
    // Held for the Export button, which needs the id of the session it is
    // writing out. Viewer-only runs report "viewer-only" here and have no
    // conversation to export, which the button reflects by staying disabled.
    sessionId = nextSessionId;
    steers = !!(session && session.steers);
    renderSendButton(store.getState().chat);
    renderExportButton();
  }

  // `meta.replayed` marks the history a connection opens with. It builds the
  // conversation like any live event, but state the hello already reported
  // afresh (agent status, model) is not taken from it, and nothing in it
  // calls for the human now: a banner, a toast or a notification raised
  // from history would report an earlier process's trouble as current.
  function handleEvent(event, meta = {}) {
    if (!event) return;
    const replayed = !!meta.replayed;
    switch (event.kind) {
      case "user_turn":
        store.setChatUserTurn(event.turn, event.blocks, event.client_id || null);
        // The human answering is what the agent asked for, so its question
        // leaves the banner (it stays in the conversation, with the tool
        // call that raised it). Replayed too: a tab reconnecting after the
        // human answered elsewhere catches up here. Nothing replayed raises
        // an attention banner, so this can only clear a live one. Any other
        // banner is left as it is.
        {
          const banner = store.getState().chat.banner;
          if (banner && banner.kind === "attention") store.clearChatBanner();
        }
        break;
      case "text_delta":
        store.appendChatTextDelta(event.turn, event.text);
        break;
      case "tool_use":
        store.addChatToolUse(event.turn, event.tool_use_id, event.name, event.input);
        break;
      case "tool_result":
        store.setChatToolResult(event.tool_use_id, !!event.is_error, event.text);
        break;
      case "turn_end":
        store.endChatTurn(event.turn, event.stop_reason, event.cost_usd, event.tokens || null);
        break;
      case "attention":
        if (!replayed) {
          notifyAttention(event.title, event.body);
          store.setChatBanner("attention", event.title + ": " + event.body);
        }
        break;
      case "permission_request":
        store.addChatPermissionRequest(
          event.request_id,
          event.tool,
          event.input,
          event.suggestions,
          event.rememberable,
        );
        break;
      case "permission_resolved":
        if (!replayed) reportResolution(event);
        store.removeChatPermissionRequest(event.request_id);
        break;
      case "agent_status":
        // The hello frame's `session.agent` is the status at the moment this
        // connection was accepted; a live event is what keeps it current.
        if (!replayed) store.setChatAgentStatus(event.status);
        break;
      case "agent_model_changed":
        // The LLM backend's active model, confirmed to have taken effect
        // by whichever driver's `set_model` emitted it. Clears
        // pendingSetModel only when this is the change this pane itself
        // asked for -- a different tab's switch, or a value nobody here
        // requested, is still worth displaying but must not be mistaken for
        // an answer to a request that, as far as this pane knows, is still
        // outstanding. A replayed one is older than the hello's model.
        if (replayed) break;
        if (pendingSetModel !== null && event.model === pendingSetModel) {
          pendingSetModel = null;
        }
        store.setChatModel(event.model);
        break;
      case "session_reset":
        store.resetChatTurns();
        if (!replayed) store.setChatBanner("reset", event.reason);
        break;
      case "agent_error":
        // What to do, and under it the backend's own words, which are often
        // the only real reason given.
        if (!replayed) {
          store.setChatBanner(
            "error", event.remediation || "The agent reported an error.", event.stderr);
        }
        break;
      default:
        // "viewer_primary" and any future kind: no rendering in this pane yet.
        break;
    }
    if (unconfirmed.size) {
      if (replayed) {
        clearTimeout(settleTimer);
        settleTimer = setTimeout(settleUnconfirmed, SETTLE_MS);
      } else {
        settleUnconfirmed();
      }
    }
  }

  // Reverts the model field after a set_model request is known to have
  // failed -- either an explicit `refused` frame or a send that never
  // reached the socket at all (applyModelInput above). Mirrors renderModel's
  // own focus-awareness: the field cannot be overwritten while mid-edit
  // without stomping on what the human is typing, so the revert is queued
  // instead and applied by the blur listener above.
  function revertPendingModel() {
    if (document.activeElement === chatModelInputEl) {
      // The field is mid-edit; renderModel would no-op against it (see its
      // own comment), silently dropping the revert and leaving a rejected
      // value stuck until some later edit happens to fire `change`. Queue
      // it instead so the blur listener above applies it the moment focus
      // leaves, regardless of whether the human edits anything further.
      queuedModelRevert = true;
      return;
    }
    renderModel(store.getState().chat);
  }

  // A message that will not appear (refused, or lost in a dropped
  // connection) goes back into the composer rather than being lost: it was
  // cleared on Send. Its text goes in front of anything typed since, a blank
  // line between, and its images come back as finished attachments, since
  // their files are already on the server. Returns false when some images
  // did not fit beside those already attached (MAX_CHAT_ATTACHMENTS).
  function restoreToComposer(blocks) {
    const text = blocksToText(blocks);
    const typed = chatInputEl.value.trim();
    if (text) chatInputEl.value = typed ? text + "\n\n" + typed : text;
    let complete = true;
    blocks
      .filter((b) => b.type === "image_path")
      .forEach((b) => {
        const id = store.reserveChatAttachment("upload");
        if (id === null) {
          complete = false;
          return;
        }
        store.completeChatAttachment(id, {
          path: b.path,
          url: assetUrlForImagePath(b.path),
          bytes: null,
          mediaType: null,
        });
      });
    return complete;
  }

  // The toast that says a message went back into the composer, true to what
  // restoreToComposer managed.
  function restoredText(what, complete) {
    return complete
      ? what + "; it is back in the box"
      : what + "; it is back in the box, but not every image fitted beside the ones attached";
  }

  // A refused turn names itself (`clientId`, the turn frame's own): it is
  // not coming back as a user_turn, so it stops being pending and is offered
  // again. Any other refusal carries no correlation id, so this cannot tell
  // which outstanding request it answers. What it can do is track its own
  // pending `set_model` call locally (`pendingSetModel`, set by
  // applyModelInput above and cleared by a matching `agent_model_changed`)
  // and treat only the next such refusal seen while that is still set as
  // "probably mine". An unrelated refusal (a permission frame, with no
  // set_model outstanding) is left alone entirely, rather than reverting a
  // model change that has not actually failed.
  function handleRefused(reason, clientId = null) {
    if (clientId) {
      const blocks = store.dropChatUserTurn(clientId);
      if (blocks) {
        const complete = restoreToComposer(blocks);
        toast(restoredText(reason || "The message was not sent", complete), false);
      }
      return;
    }
    if (pendingSetModel === null) return;
    pendingSetModel = null;
    revertPendingModel();
  }

  return { handleHello, handleEvent, handleRefused };
}
