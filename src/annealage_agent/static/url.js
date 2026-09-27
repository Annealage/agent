/**
 * Where every route this page calls lives: resolved against the page's own
 * URL, so the same modules work whether the app is served at the root
 * (`/settings`) or mounted under a front door's prefix (`/p/demo/settings`,
 * for a page at `/p/demo/`).
 *
 * Every request the agent layer's modules make goes through `appUrl`, the
 * WebSocket included, and a product's page names its own routes the same
 * way. A route is written relative, with no leading slash (`appUrl("ws")`),
 * and the result is path-absolute, ready for `fetch`, `new WebSocket` (with
 * the scheme and host in front) or an `img` src. The page's own directory is
 * the base, not a `<base>` element: the server's policy forbids one
 * (`base-uri 'none'`), and a page reached without its trailing slash is
 * redirected to it by the front door.
 */

/** `route` (relative, e.g. "agent/logs") under the page's own directory. */
export function appUrl(route) {
  return location.pathname.replace(/[^/]*$/, "") + route;
}
