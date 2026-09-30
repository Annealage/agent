"""The chat pane (``static/chat.js``) run under Node, against a minimal fake
DOM: just enough elements for ``initChat`` to mount and for the composer, the
banner and the toast to be read back.

What these pin is what the human sees at the edges of a connection: why the
agent is down, for a page opened after it went down (the hello, since a
replayed ``agent_error`` raises no banner); and a message that will not
appear, refused or lost in a dropped connection, going back into the
composer in front of whatever was typed since, with a toast saying so. And the
state a stylesheet reads off the pane element (``data-working``,
``data-turn-started``, ``data-pending``) through a turn, its permission
requests, a reconnect, a steer and a mid-turn connect; and a finished turn's
meta line. The same harness shape as ``test_review_js.py``.
"""

import json
import shutil
import subprocess

import pytest

from annealage_agent import product

STATIC = __import__("pathlib").Path(product.__file__).resolve().parent / "static"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")

HARNESS = r"""
import { pathToFileURL } from "node:url";

globalThis.window = globalThis;
globalThis.location = { hash: "#t=tok", pathname: "/", search: "", protocol: "http:", host: "x" };
globalThis.history = { replaceState() {} };
globalThis.Node = { ELEMENT_NODE: 1 };
globalThis.CSS = { escape: (s) => s };
globalThis.addEventListener = () => {};

class El {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.dataset = {}; this.style = {};
    this.hidden = false; this.textContent = ""; this.value = ""; this.listeners = {};
    this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
  }
  get nodeType() { return 1; }
  get childElementCount() { return this.children.length; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  appendChild(child) { this.children.push(child); return child; }
  append(...kids) { this.children.push(...kids); }
  remove() {}
  before(...nodes) { (this.inserted ||= []).push(...nodes); }
  after(...nodes) { (this.inserted ||= []).push(...nodes); }
  replaceChildren(...kids) { this.children = [...kids]; }
  setAttribute() {}
  getAttribute() { return null; }
  focus() {}
  blur() {}
  querySelector(sel) {
    const find = (el) => {
      for (const c of el.children) {
        if (c.tagName === sel) return c;
        const hit = find(c);
        if (hit) return hit;
      }
      return null;
    };
    return find(this);
  }
  querySelectorAll() { return []; }
  set innerHTML(v) { this._html = v; }
  get innerHTML() { return this._html || ""; }
}
const byId = new Map();
const el = (id) => { if (!byId.has(id)) byId.set(id, new El("div")); return byId.get(id); };
globalThis.document = {
  nodeType: 9, title: "t", activeElement: null, hasFocus: () => true,
  createElement: (tag) => new El(tag),
  getElementById: el,
  querySelector: (sel) => (sel.startsWith("#") ? el(sel.slice(1)) : null),
  addEventListener() {},
};

const dir = process.argv[2];
const { initChat } = await import(pathToFileURL(dir + "/chat.js").href);
const { store } = await import(pathToFileURL(dir + "/store.js").href);

const sent = [];
let online = true;
const chat = initChat({ send: (frame) => { if (!online) return false; sent.push(frame); return true; } });
const input = el("chatInput");
const sendMessage = (text) => { input.value = text; el("chatSend").listeners.click[0](); };
const out = {};

// A page opened while the agent is down: the hello says why.
chat.handleHello({ id: "s", agent: "unavailable",
                   agent_error: { remediation: "log omp in", stderr: "401 no credentials" } });
out.bannerFromHello = store.getState().chat.banner;
out.sendDisabledWhileDown = el("chatSend").disabled;

// A replayed agent_error is history; a live one is news.
store.clearChatBanner();
chat.handleEvent({ kind: "agent_error", remediation: "old trouble", stderr: "" }, { replayed: true });
out.bannerFromReplay = store.getState().chat.banner;
chat.handleEvent({ kind: "agent_error", remediation: "new trouble", stderr: "" }, { replayed: false });
out.bannerFromLive = store.getState().chat.banner && store.getState().chat.banner.text;
// Why it was down stays while it starts again, and goes once it is up.
chat.handleEvent({ kind: "agent_status", status: "connecting" }, { replayed: false });
out.errorWhileStarting = store.getState().chat.banner && store.getState().chat.banner.text;
chat.handleEvent({ kind: "agent_status", status: "ready" }, { replayed: false });
out.errorOnceReady = store.getState().chat.banner;
store.clearChatBanner();

// The agent's question (attention) leaves the banner once the human answers;
// an error banner outlives the next message.
chat.handleEvent({ kind: "attention", title: "demo: circuit checkpoint", body: "OK?" }, { replayed: false });
out.attentionBanner = store.getState().chat.banner && store.getState().chat.banner.text;
chat.handleEvent({ kind: "user_turn", turn: 7, blocks: [{ type: "text", text: "yes" }] }, { replayed: false });
out.afterAnswer = store.getState().chat.banner;
chat.handleEvent({ kind: "agent_error", remediation: "still broken", stderr: "" }, { replayed: false });
chat.handleEvent({ kind: "user_turn", turn: 8, blocks: [{ type: "text", text: "and?" }] }, { replayed: false });
out.errorAfterTurn = store.getState().chat.banner && store.getState().chat.banner.text;
store.clearChatBanner();
// A tab that reconnects after the human answered from another one.
chat.handleEvent({ kind: "attention", title: "demo: schematic checkpoint", body: "Pin comments" }, { replayed: false });
chat.handleEvent({ kind: "user_turn", turn: 9, blocks: [{ type: "text", text: "done" }] }, { replayed: true });
out.afterReplayedAnswer = store.getState().chat.banner;

// Ready: a message is sent, and refused after the human started another.
chat.handleHello({ id: "s", agent: "ready" });
sendMessage("first");
out.clearedOnSend = input.value;
input.value = "draft";
chat.handleRefused("the agent is connecting, so this message was not sent", sent[0].client_id);
out.afterRefusal = input.value;
out.refusalToast = el("toast").textContent;

// A message lost as the connection dropped: the next hello's replay has no
// user_turn for it, so once the first live event arrives it comes back.
input.value = "";
sendMessage("lost one");
chat.handleHello({ id: "s", agent: "ready" });
chat.handleEvent({ kind: "text_delta", turn: 1, text: "old" }, { replayed: true });
out.beforeReplayEnds = input.value;
chat.handleEvent({ kind: "viewer_primary" }, { replayed: false });
out.afterReplayEnds = input.value;
out.lostToast = el("toast").textContent;
out.pending = store.getState().chat.pendingUser.length;

// Usage, before the agent status: from the hello, then from live events only.
const usageChip = el("agentStatus").inserted.find((n) => n.className === "chatusage");
chat.handleHello({ id: "s", agent: "ready", usage: {
  cost_usd: 0.4312,
  tokens: { input: 1200, output: 300, cache_read: 45000, cache_write: 900 },
  context: { used_tokens: 24500, window_tokens: 200000 },
} });
out.usageFromHello = { text: usageChip.textContent, title: usageChip.title, hidden: usageChip.hidden };
chat.handleEvent({ kind: "usage", cost_usd: 0.01, tokens: {}, context: null }, { replayed: true });
out.usageAfterReplay = usageChip.textContent;
chat.handleEvent({ kind: "usage", cost_usd: null,
                   tokens: { input: 5, output: null, cache_read: null, cache_write: null },
                   context: { used_tokens: 150000, window_tokens: 200000 } }, { replayed: false });
out.usageLive = { text: usageChip.textContent, title: usageChip.title };
chat.handleHello({ id: "s", agent: "ready", usage: null });
out.usageHiddenWhenUnknown = usageChip.hidden;
chat.handleHello({ id: "s", agent: "ready", usage: { cost_usd: 1, tokens: {}, context: null } });
chat.handleEvent({ kind: "session_reset", reason: "new" }, { replayed: false });
out.usageHiddenAfterReset = usageChip.hidden;

// A document's chip says how its action ended, and offers the action no more.
const docId = store.reserveChatDocument("sheet.pdf");
store.updateChatDocument(docId, { state: "done", upload: "u-1", bytes: 10,
  actions: [{ name: "datum", label: "Submit to Datum", accepts: "application/pdf" }] });
chat.handleEvent({ kind: "upload_action", id: "ua_1", upload: "u-1", label: "Submit to Datum",
  file: "sheet.pdf", bytes: 10, tool: "mcp__datum__submit", outcome: "done", text: "job 1" },
  { replayed: false });
const endedDoc = store.getState().chat.documents.find((d) => d.id === docId);
out.documentAfterAction = { state: endedDoc.state, message: endedDoc.message };

// Permission cards across connections.
const pendingOf = () => store.getState().chat.pending.map((p) => [p.request_id, p.submitted || null]);
const allowOn = (index) => el("chatPending").children[index].children[3].children[0].listeners.click[0]();
chat.handleEvent({ kind: "permission_request", request_id: "pr_a_1", tool: "t", input: {} }, { replayed: false });
online = false;
allowOn(0);
out.cardWhileOffline = { pending: pendingOf(), toast: el("toast").textContent };
online = true;
allowOn(0);
out.cardSent = pendingOf();
chat.handleHello({ id: "s", agent: "ready" });
out.cardAfterReconnect = pendingOf();
chat.handleHello({ id: "another", agent: "ready" });
out.cardsAfterNewSession = pendingOf();

// The pane's state attributes, through a turn and its permission requests.
let clock = 1000;
Date.now = () => clock;
const paneData = () => ({ ...el("chat").dataset });
const live = { replayed: false };
const replay = { replayed: true };
out.paneIdle = paneData();
chat.handleEvent({ kind: "user_turn", turn: 1, blocks: [{ type: "text", text: "go" }] }, live);
out.paneStarted = paneData();
clock = 2000;
chat.handleEvent({ kind: "text_delta", turn: 1, text: "working" }, live);
out.paneLater = paneData();
chat.handleEvent({ kind: "permission_request", request_id: "pr_b_1", tool: "t", input: {} }, live);
chat.handleEvent({ kind: "permission_request", request_id: "pr_b_2", tool: "t", input: {} }, live);
out.paneTwoOpen = paneData();
chat.handleEvent({ kind: "permission_resolved", request_id: "pr_b_1", outcome: "allow" }, live);
out.paneOneOpen = paneData();
chat.handleEvent({ kind: "permission_resolved", request_id: "pr_b_2", outcome: "deny" }, live);
// A reconnect mid-turn: the replay repeats the turn, which keeps its start.
clock = 3000;
chat.handleHello({ id: "another", agent: "ready" });
chat.handleEvent({ kind: "user_turn", turn: 1, blocks: [{ type: "text", text: "go" }] }, replay);
out.paneAfterReconnect = paneData();
chat.handleEvent({ kind: "turn_end", turn: 1, stop_reason: "end_turn", cost_usd: 0 }, live);
out.paneEnded = paneData();
// A turn the server ended as interrupted (an interrupt, or the agent gone).
clock = 4000;
chat.handleEvent({ kind: "user_turn", turn: 2, blocks: [{ type: "text", text: "again" }] }, live);
out.paneSecondStarted = paneData();
chat.handleEvent({ kind: "turn_end", turn: 2, stop_reason: "interrupted", cost_usd: 0 }, live);
out.paneInterrupted = paneData();
// An omp steer: the new turn is logged, then the running one ends "steered".
clock = 10000;
chat.handleEvent({ kind: "user_turn", turn: 3, blocks: [{ type: "text", text: "do x" }] }, live);
clock = 70000;
chat.handleEvent({ kind: "user_turn", turn: 4, blocks: [{ type: "text", text: "y too" }] }, live);
chat.handleEvent({ kind: "turn_end", turn: 3, stop_reason: "steered", cost_usd: 0 }, live);
out.paneSteered = paneData();
chat.handleEvent({ kind: "turn_end", turn: 4, stop_reason: "end_turn", cost_usd: 0.0412,
                   tokens: { input: 6120, output: 814 } }, live);
out.paneSteerEnded = paneData();
// The finished turns' meta lines: one span per part that is known.
const metaEls = [];
const collectMeta = (node) => {
  for (const c of node.children || []) {
    if (typeof c !== "object") continue;
    if (c.className === "turnmeta") metaEls.push(c);
    collectMeta(c);
  }
};
collectMeta(el("chatLog"));
const metaParts = (m) => m.children.map((c) => (typeof c === "string" ? c : [c.className, c.textContent]));
out.metaSteered = metaParts(metaEls[metaEls.length - 2]);
out.metaFull = metaParts(metaEls[metaEls.length - 1]);
// A page that connects mid-turn: the replay is the first it hears of it.
clock = 90000;
chat.handleHello({ id: "third", agent: "ready" });
chat.handleEvent({ kind: "user_turn", turn: 1, blocks: [{ type: "text", text: "hi" }] }, replay);
chat.handleEvent({ kind: "text_delta", turn: 1, text: "on it" }, replay);
chat.handleEvent({ kind: "permission_request", request_id: "pr_c_1", tool: "t", input: {} }, replay);
out.paneMidTurnConnect = paneData();
chat.handleEvent({ kind: "permission_resolved", request_id: "pr_c_1", outcome: "allow" }, replay);
chat.handleEvent({ kind: "session_reset", reason: "new" }, live);
out.paneAfterReset = paneData();

// The model field: the listed models as its suggestions, emptied on focus so
// all of them are offered, and the current model back if nothing is typed.
chat.handleHello({ id: "m", agent: "ready", model: "opus", models: ["opus", "haiku"] });
const modelInput = el("chatModelInput");
const fire = (type) => { for (const fn of modelInput.listeners[type] || []) fn({ key: "" }); };
const focusModel = () => { document.activeElement = modelInput; fire("focus"); };
const blurModel = () => { document.activeElement = null; fire("blur"); fire("change"); };
out.modelOptions = modelInput.inserted.find((n) => n.id === "chatModelList").children.map((o) => o.value);
focusModel();
out.modelOnFocus = [modelInput.value, modelInput.placeholder];
blurModel();
out.modelOnBlurUntyped = modelInput.value;
sent.length = 0;
focusModel();
modelInput.value = "hai";
// The window losing and regaining focus refocuses the field mid-edit.
fire("blur"); fire("focus");
out.modelAfterRefocus = modelInput.value;
modelInput.value = "haiku";
blurModel();
out.modelSent = sent.filter((f) => f.type === "set_model").map((f) => f.model);

console.log(JSON.stringify(out));
process.exit(0);
"""


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    harness = tmp_path_factory.mktemp("chat-js") / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(harness), str(STATIC)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_page_opened_while_the_agent_is_down_shows_why(observed):
    banner = observed["bannerFromHello"]
    assert (banner["kind"], banner["text"]) == ("error", "log omp in")
    assert "401 no credentials" in json.dumps(banner)
    assert observed["sendDisabledWhileDown"] is True


def test_only_a_live_agent_error_raises_the_banner(observed):
    assert observed["bannerFromReplay"] is None
    assert observed["bannerFromLive"] == "new trouble"
    assert observed["errorWhileStarting"] == "new trouble"
    assert observed["errorOnceReady"] is None


def test_the_agents_question_leaves_the_banner_when_the_human_answers(observed):
    assert observed["attentionBanner"] == "demo: circuit checkpoint: OK?"
    assert observed["afterAnswer"] is None
    assert observed["errorAfterTurn"] == "still broken"
    assert observed["afterReplayedAnswer"] is None


def test_a_refused_message_goes_back_in_front_of_what_was_typed_since(observed):
    assert observed["clearedOnSend"] == ""
    assert observed["afterRefusal"] == "first\n\ndraft"
    assert observed["refusalToast"] == (
        "the agent is connecting, so this message was not sent; it is back in the box"
    )


def test_a_message_lost_in_a_drop_comes_back_when_the_replay_ends(observed):
    assert observed["beforeReplayEnds"] == ""
    assert observed["afterReplayEnds"] == "lost one"
    assert "never arrived; it is back in the box" in observed["lostToast"]
    assert observed["pending"] == 0


def test_the_header_shows_the_conversation_s_usage_from_the_hello_and_live_events(observed):
    assert observed["usageFromHello"] == {
        "text": "12% context · $0.431",
        "title": "Context window: 24,500 of 200,000 tokens (12%)\n"
        "Cost so far: $0.4312\n"
        "Tokens: 1,200 in, 300 out, 45,000 read from the cache, 900 written to it",
        "hidden": False,
    }
    # A replayed usage event is older than the hello; each is the whole
    # conversation's, so the live one replaces what was shown.
    assert observed["usageAfterReplay"] == "12% context · $0.431"
    assert observed["usageLive"] == {
        "text": "75% context",
        "title": "Context window: 150,000 of 200,000 tokens (75%)\n"
        "Cost so far: unknown\n"
        "Tokens: 5 in, unknown out, unknown read from the cache, unknown written to it",
    }
    assert observed["usageHiddenWhenUnknown"] is True
    assert observed["usageHiddenAfterReset"] is True, (
        "a new conversation's usage is not the old one's"
    )


def test_a_document_s_chip_says_how_its_action_ended(observed):
    assert observed["documentAfterAction"] == {
        "state": "ended",
        "message": "Submit to Datum: sent",
    }


def test_a_card_whose_decision_could_not_be_sent_stays_answerable(observed):
    assert observed["cardWhileOffline"]["pending"] == [["pr_a_1", None]]
    assert "Not sent" in observed["cardWhileOffline"]["toast"]
    assert observed["cardSent"] == [["pr_a_1", "allow"]]


def test_a_new_connection_reconciles_the_cards_shown(observed):
    # The same conversation: the card may be answered again (the replay
    # retires it if the decision did land).
    assert observed["cardAfterReconnect"] == [["pr_a_1", None]]
    # Another conversation: its cards are not this one's.
    assert observed["cardsAfterNewSession"] == []


def test_the_pane_root_says_whether_a_turn_runs_since_when_and_what_waits(observed):
    idle = {"pending": "0"}
    running = {"working": "", "turnStarted": "1000", "pending": "0"}
    assert observed["paneIdle"] == idle
    # The start is when the live user_turn arrived, and stays put.
    assert observed["paneStarted"] == running
    assert observed["paneLater"] == running
    assert observed["paneTwoOpen"] == {**running, "pending": "2"}
    assert observed["paneOneOpen"] == {**running, "pending": "1"}
    assert observed["paneAfterReconnect"] == running
    assert observed["paneEnded"] == idle
    assert observed["paneSecondStarted"] == {**running, "turnStarted": "4000"}
    assert observed["paneInterrupted"] == idle
    # A steer carries on the work the steered turn started.
    assert observed["paneSteered"] == {**running, "turnStarted": "10000"}
    assert observed["paneSteerEnded"] == idle
    # Connected mid-turn: the start is when this page first saw the turn.
    assert observed["paneMidTurnConnect"] == {
        "working": "",
        "turnStarted": "90000",
        "pending": "1",
    }
    assert observed["paneAfterReset"] == idle


def test_a_finished_turn_s_meta_is_a_span_per_known_part(observed):
    assert observed["metaFull"] == [
        ["turnstop", "end_turn"],
        ", ",
        ["turncost", "$0.0412"],
        ", ",
        ["turntokens", "6,120 in, 814 out"],
    ]
    # No tokens reported: no tokens part.
    assert observed["metaSteered"] == [["turnstop", "steered"], ", ", ["turncost", "$0.0000"]]


def test_the_model_field_offers_the_listed_models_from_an_empty_field(observed):
    assert observed["modelOptions"] == ["opus", "haiku"]
    # Emptied on focus, so the datalist offers every model; the current one
    # stays visible as the placeholder and comes back if nothing is typed.
    assert observed["modelOnFocus"] == ["", "opus"]
    assert observed["modelOnBlurUntyped"] == "opus"


def test_a_half_typed_model_survives_a_refocus_and_is_sent(observed):
    assert observed["modelAfterRefocus"] == "hai"
    assert observed["modelSent"] == ["haiku"]
