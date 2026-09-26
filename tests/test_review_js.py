"""The page's review client (``static/review.js``), run under Node with the
browser globals it touches stubbed.

The package has no browser suite of its own (a product's page is what loads
these modules), and no product loads this one yet, so its behaviour is pinned
here: one writer (``onChange``, only from a refetch), a reply that arrives
after a later refetch has started is dropped, an unchanged list is not handed
on twice, an unreadable review is reported rather than shown as empty, the
fallback poll runs only while the socket is not live, and ``add`` and
``setStatus`` post with the browser token.
"""

import json
import shutil
import subprocess

import pytest

from annealage_agent import product

REVIEW_JS = __import__("pathlib").Path(product.__file__).resolve().parent / "static" / "review.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")

# Stubs for exactly what ws.js and review.js touch at import and call time,
# then a scripted run whose observations are printed as one JSON object.
HARNESS = r"""
import { pathToFileURL } from "node:url";

globalThis.window = globalThis;
globalThis.location = { hash: "#t=tok%2B1", pathname: "/", search: "", protocol: "http:", host: "x" };
globalThis.history = { replaceState() {} };

const requests = [];
const pending = [];
globalThis.fetch = (url, opts = {}) => {
  requests.push({ url, method: opts.method || "GET", body: opts.body || null });
  return new Promise((resolve) => pending.push(resolve));
};
function reply(status, body) {
  const resolve = pending.shift();
  resolve({ ok: status >= 200 && status < 300, status, json: async () => body });
}
const flush = () => new Promise((r) => setTimeout(r, 0));

const intervals = new Map();
let nextTimer = 1;
globalThis.setInterval = (fn, ms) => { intervals.set(nextTimer, ms); return nextTimer++; };
globalThis.clearInterval = (id) => { intervals.delete(id); };

const { initReview } = await import(pathToFileURL(process.argv[2]).href);

const changes = [];
const errors = [];
const review = initReview({
  onChange: (comments, capabilities) => changes.push({ comments, capabilities }),
  onError: (message) => errors.push(message),
  pollMs: 700,
});
const A = [{ id: 1, anchor: { card: "front" }, text: "a", author: "human" }];
const B = [{ id: 2, anchor: { card: "back" }, text: "b", author: "model" }];
const C = [{ id: 3, anchor: { card: "back" }, text: "c", author: "model" }];
const caps = { can_resolve: true };
const out = {};

// First paint: one fetch at init, with the browser token.
out.firstUrl = requests[0].url;
reply(200, { ok: true, comments: A, capabilities: caps });
await flush();
out.afterFirst = changes.length;

// The event refetches; an unchanged list is not handed on again.
review.onEvent.review_changed({ kind: "review_changed" });
reply(200, { ok: true, comments: A, capabilities: caps });
await flush();
out.afterUnchanged = changes.length;

// Two refetches in flight; the later one's reply lands first, the earlier
// one's afterwards and is dropped.
review.refetch();
review.refetch();
const early = pending.shift();
reply(200, { ok: true, comments: B, capabilities: caps });
await flush();
early({ ok: true, status: 200, json: async () => ({ ok: true, comments: C, capabilities: caps }) });
await flush();
out.afterRace = changes.map((c) => c.comments[0].id);
out.current = review.comments().map((c) => c.id);

// An unreadable review is reported; the same list afterwards is handed on
// again, since the product was told something was wrong.
review.refetch();
reply(409, { ok: false, error: "toy.review.json does not parse; fix it by hand" });
await flush();
review.refetch();
reply(200, { ok: true, comments: B, capabilities: caps });
await flush();
out.errors = errors;
out.afterRecovery = changes.map((c) => c.comments[0].id);

// The fallback poll: started once however often it is asked for, with an
// immediate fetch; stopped, with a refetch, when the socket is live again.
const before = requests.length;
review.onFallback();
review.onFallback();
out.pollIntervals = [...intervals.values()];
out.fallbackFetches = requests.length - before;
reply(200, { ok: true, comments: B, capabilities: caps });
await flush();
review.onLive();
out.intervalsAfterLive = intervals.size;
reply(200, { ok: true, comments: B, capabilities: caps });
await flush();

// add posts the human's comment and refetches at once.
const adding = review.add({ card: "front", x: 1, y: 2 }, "thin");
const post = requests[requests.length - 1];
out.post = { url: post.url, method: post.method, body: JSON.parse(post.body) };
reply(200, { ok: true, comment: { id: 4 } });
out.added = await adding;
out.refetchedAfterAdd = requests[requests.length - 1].method;
reply(200, { ok: true, comments: B, capabilities: caps });
const refused = review.add({ card: "side" }, "x");
reply(400, { ok: false, error: "no card 'side'" });
out.refused = await refused;

// setStatus posts the new status to the comment's own URL and refetches.
const setting = review.setStatus(4, "open");
const statusPost = requests[requests.length - 1];
out.statusPost = {
  url: statusPost.url, method: statusPost.method, body: JSON.parse(statusPost.body),
};
reply(200, { ok: true, comment: { id: 4, status: "open" } });
out.statusSet = await setting;
out.refetchedAfterStatus = requests[requests.length - 1].method;
reply(200, { ok: true, comments: B, capabilities: caps });

console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    harness = tmp_path_factory.mktemp("review-js") / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(harness), str(REVIEW_JS)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_first_fetch_carries_the_browser_token(observed):
    assert observed["firstUrl"] == "/review?t=tok%2B1"
    assert observed["afterFirst"] == 1


def test_an_unchanged_list_is_not_handed_on_twice(observed):
    assert observed["afterUnchanged"] == 1


def test_a_reply_overtaken_by_a_later_refetch_is_dropped(observed):
    assert observed["afterRace"] == [1, 2]
    assert observed["current"] == [2]


def test_an_unreadable_review_is_reported_and_recovery_is_handed_on(observed):
    assert observed["errors"] == ["toy.review.json does not parse; fix it by hand"]
    assert observed["afterRecovery"] == [1, 2, 2]


def test_the_fallback_poll_runs_once_and_stops_when_live(observed):
    assert observed["pollIntervals"] == [700]
    assert observed["fallbackFetches"] == 1
    assert observed["intervalsAfterLive"] == 0


def test_add_posts_the_comment_and_refetches(observed):
    assert observed["post"] == {
        "url": "/review?t=tok%2B1",
        "method": "POST",
        "body": {"anchor": {"card": "front", "x": 1, "y": 2}, "text": "thin"},
    }
    assert observed["added"] == {"ok": True, "comment": {"id": 4}}
    assert observed["refetchedAfterAdd"] == "GET"
    assert observed["refused"] == {"ok": False, "error": "no card 'side'"}


def test_set_status_posts_to_the_comment_and_refetches(observed):
    assert observed["statusPost"] == {
        "url": "/review/4?t=tok%2B1",
        "method": "POST",
        "body": {"status": "open"},
    }
    assert observed["statusSet"] == {"ok": True, "comment": {"id": 4, "status": "open"}}
    assert observed["refetchedAfterStatus"] == "GET"
