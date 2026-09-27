"""Every URL the page's modules request, from a page mounted under a front
door's prefix (``/p/demo/``), run under Node like ``test_chat_js.py``.

``static/url.js``'s ``appUrl`` resolves a route against the page's own
directory, so the same modules reach their own app wherever it is mounted and
still name ``/settings`` at the root. What this pins is that every request
actually goes through it: the login a ``#n=`` link trades, the WebSocket and
the probe that tells a refused socket from an outage, an upload, the review;
and that a page at the root is unchanged, which is what Annealage Mesh's
browser suite relies on (a thumbnail's ``src`` still starts ``/asset/``).
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

const dir = process.argv[2];
globalThis.window = globalThis;
globalThis.location = {
  hash: "#n=nonce-1", pathname: "/p/demo/", search: "", protocol: "http:", host: "x:8765",
};
globalThis.history = { replaceState() {} };
globalThis.document = { getElementById: () => ({ textContent: "", style: {}, dataset: {} }) };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};

const requests = [];
globalThis.fetch = async (url, opts = {}) => {
  requests.push(url);
  if (url.endsWith("/login")) return { ok: true, status: 200, json: async () => ({ token: "tok" }) };
  if (url.includes("/upload?")) {
    return { ok: true, status: 200, json: async () => ({
      ok: true, path: "images/upload-1.png", url: "/p/demo/asset/upload-1.png", bytes: 3,
      media_type: "image/png" }) };
  }
  return { ok: true, status: 400, json: async () => ({ ok: true, comments: [], capabilities: {} }) };
};
const sockets = [];
globalThis.WebSocket = class {
  static OPEN = 1;
  constructor(url) { this.url = url; this.on = {}; sockets.push(this); }
  addEventListener(type, fn) { (this.on[type] ||= []).push(fn); }
  send() {}
  close() { (this.on.close || []).forEach((f) => f({ code: 1006 })); }
};
const flush = () => new Promise((r) => setTimeout(r, 0));

const { appUrl } = await import(pathToFileURL(dir + "/url.js").href);
const out = {};
out.resolved = {};
for (const page of ["/p/demo/", "/p/demo/index.html", "/"]) {
  location.pathname = page;
  out.resolved[page] = [appUrl("settings"), appUrl("agent/logs/" + "a%20b")];
}
location.pathname = "/p/demo/";

// ws.js trades the #n= nonce as it is imported.
const { initWs } = await import(pathToFileURL(dir + "/ws.js").href);
out.login = requests.slice();
initWs({ indicator: null });
out.socket = sockets[0].url;
// Closed before it opened: the probe asks the same URL over plain HTTP.
requests.length = 0;
sockets[0].close();
await flush();
out.probe = requests.slice();

const { uploadImage } = await import(pathToFileURL(dir + "/uploads.js").href);
requests.length = 0;
await uploadImage(new Blob(["abc"]), "upload");
out.upload = requests.slice();

const { initReview } = await import(pathToFileURL(dir + "/review.js").href);
requests.length = 0;
initReview({});
await flush();
out.review = requests.slice();

// The same page at the root: nothing moves.
location.pathname = "/";
requests.length = 0;
await uploadImage(new Blob(["abc"]), "upload");
out.rootUpload = requests.slice();

console.log(JSON.stringify(out));
process.exit(0);
"""


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    harness = tmp_path_factory.mktemp("url-js") / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(harness), str(STATIC)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_route_resolves_under_the_page_s_own_directory(observed):
    assert observed["resolved"] == {
        "/p/demo/": ["/p/demo/settings", "/p/demo/agent/logs/a%20b"],
        "/p/demo/index.html": ["/p/demo/settings", "/p/demo/agent/logs/a%20b"],
        "/": ["/settings", "/agent/logs/a%20b"],
    }


def test_a_mounted_page_logs_in_and_connects_to_its_own_app(observed):
    assert observed["login"] == ["/p/demo/login"]
    assert observed["socket"] == "ws://x:8765/p/demo/ws?t=tok"
    assert observed["probe"] == ["/p/demo/ws?t=tok"]


def test_a_mounted_page_uploads_to_and_reads_the_review_of_its_own_app(observed):
    assert observed["upload"] == ["/p/demo/upload?t=tok&kind=upload"]
    assert observed["review"] == ["/p/demo/review?t=tok"]


def test_a_page_at_the_root_requests_what_it_always_did(observed):
    assert observed["rootUpload"] == ["/upload?t=tok&kind=upload"]
