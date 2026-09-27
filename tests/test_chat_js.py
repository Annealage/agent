"""The chat pane (``static/chat.js``) run under Node, against a minimal fake
DOM: just enough elements for ``initChat`` to mount and for the composer, the
banner and the toast to be read back.

What these pin is what the human sees at the edges of a connection: why the
agent is down, for a page opened after it went down (the hello, since a
replayed ``agent_error`` raises no banner); and a message that will not
appear, refused or lost in a dropped connection, going back into the
composer in front of whatever was typed since, with a toast saying so. The
same harness shape as ``test_review_js.py``.
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
const chat = initChat({ send: (frame) => { sent.push(frame); return true; } });
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
