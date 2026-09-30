"""The pause control (``static/pause.js``), run under Node.

It shows what the server says (``aria-pressed`` and ``.on``) and never
replaces markup the page put in the button when the page gives it a
``[data-label]`` child: a theme's icon survives every change. The same
harness shape as ``test_chat_store_js.py``.
"""

import json
import shutil
import subprocess

import pytest

from annealage_agent import product

PAUSE_JS = __import__("pathlib").Path(product.__file__).resolve().parent / "static" / "pause.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")

HARNESS = r"""
import { pathToFileURL } from "node:url";

// Just enough of a button for pause.js: children, classes, attributes.
function el(tag, attrs = {}) {
  const node = {
    tag, attrs: { ...attrs }, children: [], classes: new Set(), listeners: {},
    disabled: false,
    get textContent() {
      return this.text !== undefined ? this.text : this.children.map((c) => c.textContent).join("");
    },
    set textContent(v) { this.text = String(v); this.children = []; },
    classList: null,
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k] ?? null; },
    addEventListener(kind, fn) { this.listeners[kind] = fn; },
    querySelector(sel) {
      const key = sel.replace(/^\[|\]$/g, "");
      for (const c of this.children) {
        if (key in c.attrs) return c;
        const found = c.querySelector(sel);
        if (found) return found;
      }
      return null;
    },
    append(...kids) { this.text = undefined; this.children.push(...kids); return this; },
  };
  node.classList = {
    toggle: (c, on) => (on ? node.classes.add(c) : node.classes.delete(c)),
    contains: (c) => node.classes.has(c),
  };
  return node;
}

const { store } = await import(pathToFileURL(process.argv[2].replace(/pause\.js$/, "store.js")).href);
const { initPause } = await import(pathToFileURL(process.argv[2]).href);

const icon = el("svg", { class: "i" });
const label = el("span", { "data-label": "" });
label.textContent = "Pause";
const themed = el("button").append(icon, label);
const plain = el("button");
const sent = [];
initPause({ send: (f) => sent.push(f), button: themed });
initPause({ send: () => {}, button: plain });

const look = (b) => ({
  pressed: b.getAttribute("aria-pressed"), on: b.classList.contains("on"), text: b.textContent,
});
const out = { before: look(themed), plainBefore: look(plain) };
themed.listeners.click();
out.sent = sent;
out.beforeServerSays = look(themed);
store.setPaused(true);
out.paused = look(themed);
out.iconKept = themed.children[0] === icon && themed.children.length === 2;
out.plainPaused = look(plain);
store.setPaused(false);
out.resumed = look(themed);
console.log(JSON.stringify(out));
"""


def test_the_pause_button_shows_the_server_s_flag_and_keeps_the_page_s_markup(tmp_path):
    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS)
    run = subprocess.run(
        ["node", str(harness), str(PAUSE_JS)], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)

    assert out["before"] == {"pressed": "false", "on": False, "text": "Pause"}
    # A click asks; the button changes only once the server says so.
    assert out["sent"] == [{"type": "pause", "paused": True}]
    assert out["beforeServerSays"]["pressed"] == "false"
    assert out["paused"] == {"pressed": "true", "on": True, "text": "Paused"}
    # The page's icon is still there: only the label's text was written.
    assert out["iconKept"] is True
    assert out["resumed"] == {"pressed": "false", "on": False, "text": "Pause"}
    # A button with no label child gets its whole text, as before.
    assert out["plainBefore"]["text"] == "❙❙ Pause"
    assert out["plainPaused"] == {"pressed": "true", "on": True, "text": "❙❙ Paused"}
