/**
 * The page's client for the product's review: the human's comments on the
 * view and the model's callouts, as `GET /review` reports them, refetched on
 * every `review_changed` event. It renders nothing. Every product draws
 * anchors its own way (a marker on a part, a box on a sheet), so this module
 * hands the product the list and leaves the drawing to it.
 *
 * Wiring, in the product's main module:
 *
 *   const review = initReview({ onChange: (comments, capabilities) => draw(comments) });
 *   initWs({
 *     onEvent: { ...review.onEvent, ...theProductsOwnHandlers },
 *     onLive: () => { review.onLive(); ... },
 *     onFallback: () => { review.onFallback(); ... },
 *   });
 *
 * `onChange(comments, capabilities)` is the one way the list reaches the
 * product, and it is only ever called from `refetch`, so the review in the
 * page has one writer however many things ask for a refetch: the event, a
 * reconnect, the fallback poll, or the product itself after `add`. Each
 * comment is the server's product-neutral shape: `id`, `anchor` (in the
 * product's anchor space, `capabilities`' sibling `anchor_space` names it),
 * `ref` when known, `text`, `author` ("human" or "model"), and `status` and
 * `resolution` in a review that keeps status. `onError(message)` is called
 * instead when the server has the review but cannot read it (a hand-broken
 * file), so the product can say so rather than show an empty review over the
 * human's comments.
 *
 * `onLive` and `onFallback` follow ws.js's contract: on every (re)connect the
 * poll stops and the list is refetched once, since an event pushed while the
 * socket was down may be older than the replay; while the socket is not the
 * live channel, the list is polled instead.
 *
 * `add(anchor, text)` posts one of the human's comments, for a product whose
 * review takes them one at a time (`capabilities.human_adds_via_api`), and
 * resolves to `{ok, comment}` or `{ok: false, error}`.
 */

import { authToken } from "./ws.js";

// How often the list is fetched while the socket is not live: the same period
// a product's own fallback poll uses.
const POLL_MS = 1500;

function reviewUrl() {
  return "/review?t=" + encodeURIComponent(authToken());
}

export function initReview({ onChange = () => {}, onError = () => {}, pollMs = POLL_MS } = {}) {
  let comments = [];
  let capabilities = null;
  // The last list handed to onChange, serialised, so a refetch that finds
  // nothing new does not make the product redraw.
  let signature = null;
  // Bumped on every refetch; a reply that arrives after a later refetch has
  // started is stale and dropped, whatever order the two replies land in.
  let generation = 0;
  let pollTimer = null;

  async function refetch() {
    const mine = ++generation;
    try {
      const res = await fetch(reviewUrl(), { cache: "no-store" });
      const body = await res.json();
      if (mine !== generation) return;
      if (!res.ok || !body || body.ok !== true || !Array.isArray(body.comments)) {
        // Forgotten, so the next good reply is handed on even if it matches
        // the list from before the failure: the product has been told
        // something is wrong and needs telling that it is not any more.
        signature = null;
        onError(
          body && typeof body.error === "string"
            ? body.error
            : "the review could not be read (" + res.status + ")",
        );
        return;
      }
      const next = JSON.stringify([body.comments, body.capabilities]);
      if (next === signature) return;
      signature = next;
      comments = body.comments;
      capabilities = body.capabilities || null;
      onChange(comments, capabilities);
    } catch (err) {
      // Only the fetch and the JSON parse are expected to fail transiently (a
      // dropped connection; a refusal, whose body is not JSON, which ws.js
      // reports as a stale link). An error thrown by the product's onChange
      // lands here too and is a real defect, so it is not swallowed.
      if (!(err instanceof TypeError || err instanceof SyntaxError)) throw err;
    }
  }

  function startPoll() {
    if (pollTimer !== null) return;
    refetch();
    pollTimer = setInterval(refetch, pollMs);
  }

  function stopPoll() {
    if (pollTimer === null) return;
    clearInterval(pollTimer);
    pollTimer = null;
  }

  async function add(anchor, text) {
    let res;
    let body = null;
    try {
      res = await fetch(reviewUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anchor, text }),
      });
      body = await res.json();
    } catch (err) {
      if (!(err instanceof TypeError || err instanceof SyntaxError)) throw err;
    }
    if (!res || !res.ok || !body || body.ok !== true) {
      const status = res ? res.status : "no answer";
      return {
        ok: false,
        error: (body && body.error) || "the comment was not saved (" + status + ")",
      };
    }
    // The server announces the change to every page with review_changed;
    // this page refetches at once rather than wait for its own copy.
    refetch();
    return { ok: true, comment: body.comment };
  }

  // Unconditional, so the first paint of the review never waits for the
  // socket's hello.
  refetch();

  return {
    onEvent: { review_changed: () => refetch() },
    onLive: () => {
      stopPoll();
      refetch();
    },
    onFallback: startPoll,
    refetch,
    add,
    comments: () => comments,
    capabilities: () => capabilities,
  };
}
