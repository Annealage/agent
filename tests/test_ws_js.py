"""The page's socket client (``static/ws.js``), run under Node with a fake
``WebSocket`` and the few browser globals it touches stubbed.

A connection opens with the history (every event up to the hello's ``seq``)
and then carries on live. What these pin: each event is handled once and in
order even when the server sends one twice around registration; the replayed
history does not re-apply state the hello already reported afresh (the pause
flag) or re-run a product's refetch per historical event; and a position in
one session's log is never offered to another session. The same harness shape
as ``test_review_js.py``: a scripted run printing one JSON object.
"""

import json
import shutil
import subprocess

import pytest

from annealage_agent import product

WS_JS = __import__("pathlib").Path(product.__file__).resolve().parent / "static" / "ws.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")

HARNESS = r"""
import { pathToFileURL } from "node:url";

globalThis.window = globalThis;
globalThis.location = { hash: "#t=tok", pathname: "/", search: "", protocol: "http:", host: "x" };
globalThis.history = { replaceState() {} };
globalThis.innerWidth = 800;
globalThis.innerHeight = 600;
globalThis.document = { getElementById: () => ({ textContent: "", style: {}, dataset: {} }) };
globalThis.fetch = async () => ({ status: 400 });

const sockets = [];
class FakeWebSocket {
  static OPEN = 1;
  constructor(url) { this.readyState = 0; this.sent = []; this.on = {}; sockets.push(this); }
  addEventListener(type, fn) { (this.on[type] ||= []).push(fn); }
  send(data) { this.sent.push(JSON.parse(data)); }
  close() { this.readyState = 3; (this.on.close || []).forEach((f) => f({ code: 1006 })); }
  open() { this.readyState = 1; (this.on.open || []).forEach((f) => f({})); }
  deliver(frame) { (this.on.message || []).forEach((f) => f({ data: JSON.stringify({ v: 2, ...frame }) })); }
}
globalThis.WebSocket = FakeWebSocket;

const { initWs } = await import(pathToFileURL(process.argv[2]).href);

const paused = [];
const agent = [];
const refetches = [];
initWs({
  onEvent: { models_changed: () => refetches.push("models") },
  onPaused: (v) => paused.push(v),
  onAgentEvent: (event, meta) => agent.push([event.kind, event.text || null, meta.replayed]),
  indicator: null,
});
const out = {};
const hello = (seq, id, pausedNow) => ({
  type: "hello", seq, protocol: 2, session: { id, paused: pausedNow },
});
const event = (seq, ev) => ({ type: "event", seq, event: ev });
const delta = (seq, text) => event(seq, { kind: "text_delta", turn: 1, text });

// First connection: no position yet. The server was restarted with the pause
// switch off; its history still holds a pause_changed(true) and a product
// event from before.
let ws = sockets[0];
ws.open();
out.firstHello = ws.sent[0];
ws.deliver(hello(4, "s1", false));
ws.deliver(event(1, { kind: "pause_changed", paused: true }));
ws.deliver(event(2, { kind: "models_changed" }));
ws.deliver(delta(3, "Hel"));
ws.deliver(delta(4, "lo"));
// Live from here; seq 4 once more, as a broadcast scheduled before the
// replay finished can deliver it, then a live pause and a product event.
ws.deliver(delta(4, "lo"));
ws.deliver(delta(5, "!"));
ws.deliver(event(6, { kind: "pause_changed", paused: true }));
ws.deliver(event(7, { kind: "models_changed" }));
out.paused = paused.slice();
out.refetches = refetches.slice();
out.agent = agent.slice();

// Reconnect to the same session: the position and its session go with it.
ws.close();
await new Promise((r) => setTimeout(r, 1200));
ws = sockets[1];
ws.open();
out.sameSessionHello = { last_seq: ws.sent[0].last_seq, session_id: ws.sent[0].session_id };
ws.deliver(hello(7, "s1", false));

// The server comes back on another session with a longer log: it answers
// from the start, and the page takes that replay rather than dropping it as
// already seen.
ws.close();
await new Promise((r) => setTimeout(r, 2500));
ws = sockets[2];
ws.open();
out.otherSessionHello = { last_seq: ws.sent[0].last_seq, session_id: ws.sent[0].session_id };
agent.length = 0;
ws.deliver(hello(40, "s2", false));
ws.deliver(delta(1, "other"));
out.otherSessionReplay = agent.slice();

console.log(JSON.stringify(out));
process.exit(0);
"""


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    harness = tmp_path_factory.mktemp("ws-js") / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(harness), str(WS_JS)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_first_hello_offers_no_position(observed):
    assert observed["firstHello"]["last_seq"] == 0
    assert "session_id" not in observed["firstHello"]


def test_a_replayed_pause_does_not_override_the_hello(observed):
    """After a restart the server holds the switch off; a pause from the
    earlier process's history must not show it on."""
    assert observed["paused"] == [False, True]


def test_a_product_handler_runs_for_live_events_only(observed):
    assert observed["refetches"] == ["models"]


def test_each_event_is_handled_once_and_in_order(observed):
    assert observed["agent"] == [
        ["text_delta", "Hel", True],
        ["text_delta", "lo", True],
        ["text_delta", "!", False],
    ]


def test_a_reconnect_offers_its_position_with_its_session(observed):
    assert observed["sameSessionHello"] == {"last_seq": 7, "session_id": "s1"}


def test_another_session_s_replay_is_taken_from_the_start(observed):
    assert observed["otherSessionHello"] == {"last_seq": 7, "session_id": "s1"}
    assert observed["otherSessionReplay"] == [["text_delta", "other", True]]
