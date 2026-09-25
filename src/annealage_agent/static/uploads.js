/**
 * The one module that POSTs `/upload`, and the only writer of
 * `state.chat.attachments`, so an attachment always reaches the store through
 * this one path regardless of which UI started it: the composer's file picker,
 * paste and drop handlers in `chat.js`, or anything the product attaches
 * itself (Mesh's sketch overlay composite). None of those callers writes
 * `state.chat.attachments` itself; they call `uploadImage` and read what it
 * returns.
 *
 * The slot is reserved before the request goes out, which is what makes the
 * attachment cap a refusal rather than an apology: a file offered when the
 * message is already full is turned away before a byte leaves the page, so
 * nothing is written into the human's project that no message will reference.
 * It is also what keeps the strip in the order things were attached, since the
 * position is taken at attach time rather than whenever the network answers.
 *
 * An upload that fails leaves its slot behind, in the 'error' state, for the
 * human to see and dismiss. An upload that succeeds leaves a file in the
 * served project whether or not the message that prompted it is ever sent:
 * nothing here deletes it, because `images/` is git-tracked evidence of what a
 * part looked like at some moment and silently removing a file the server
 * already wrote would be the surprising half of that bargain.
 */

import { store, MAX_CHAT_ATTACHMENTS } from "./store.js";
import { authToken } from "./ws.js";
import { toast } from "./ui.js";

const UPLOAD_FAILED_MESSAGE = "The upload failed. Try again.";
const ATTACHMENT_LIMIT_MESSAGE =
  "This message already carries " + MAX_CHAT_ATTACHMENTS + " attachments.";

/**
 * Uploads `blob`'s bytes as `kind` and records the result in
 * `state.chat.attachments`. `kind` is "upload" for the composer's own
 * attachments, or a kind the product declared (`Product.upload_kinds` on the
 * server, which refuses any other); it names the written file and nothing
 * else.
 *
 * Returns the completed entry on success, or `null` on any failure (the cap
 * already being full, a network failure, or a non-2xx response), having
 * already shown a `toast` for the caller. A Blob body, not a stream, is what
 * lets the browser set `Content-Length` itself: the server refuses a request
 * with no declared length before it reads a single byte.
 */
export async function uploadImage(blob, kind) {
  const id = store.reserveChatAttachment(kind);
  if (id === null) {
    toast(ATTACHMENT_LIMIT_MESSAGE, false);
    return null;
  }
  const url = "/upload?t=" + encodeURIComponent(authToken()) + "&kind=" + encodeURIComponent(kind);
  let res;
  try {
    res = await fetch(url, { method: "POST", body: blob });
  } catch (err) {
    store.failChatAttachment(id, "Upload failed");
    toast(UPLOAD_FAILED_MESSAGE, false);
    return null;
  }
  let data = null;
  try {
    data = await res.json();
  } catch (err) {
    // The token-refusal response is `text/plain` by design (it is the same
    // response `/ws` gives an unauthenticated caller), so a missing or stale
    // token lands here with no JSON to parse; every other failure this route
    // defines answers JSON, which is what "not `data`" below is there to
    // catch the one case that does not.
  }
  if (!res.ok || !data || !data.ok) {
    const message = (data && data.error) || UPLOAD_FAILED_MESSAGE;
    store.failChatAttachment(id, "Upload failed");
    toast(message, false);
    return null;
  }
  store.completeChatAttachment(id, {
    path: data.path,
    url: data.url,
    bytes: data.bytes,
    mediaType: data.media_type,
  });
  return store.getState().chat.attachments.find((a) => a.id === id) || null;
}
