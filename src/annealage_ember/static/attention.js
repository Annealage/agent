/**
 * The agent asking for the human (an `attention` event, `ViewerBus.attention`
 * server-side): a browser notification while the page is not focused, and
 * the document title flashing until it is.
 *
 * Notification permission is asked on the human's first click anywhere on the
 * page, never on load: browsers refuse (or penalise) a prompt nobody asked
 * for, and a click is the gesture they require.
 */

const FLASH_MS = 1000;

let flashTimer = null;
let baseTitle = null;

function stopFlash() {
  if (flashTimer === null) return;
  clearInterval(flashTimer);
  flashTimer = null;
  document.title = baseTitle;
}

function startFlash(title) {
  if (flashTimer !== null) return;
  baseTitle = document.title;
  let on = false;
  flashTimer = setInterval(() => {
    on = !on;
    document.title = on ? "\u25cf " + title : baseTitle;
  }, FLASH_MS);
}

export function initAttention(root = document) {
  if (typeof Notification !== "undefined" && Notification.permission === "default") {
    root.addEventListener("click", () => {
      if (Notification.permission === "default") Notification.requestPermission();
    }, { once: true });
  }
  window.addEventListener("focus", stopFlash);
}

export function notifyAttention(title, body) {
  if (document.hasFocus()) return;
  startFlash(title);
  if (typeof Notification !== "undefined" && Notification.permission === "granted") {
    const note = new Notification(title, { body, tag: "agent-attention" });
    note.onclick = () => {
      window.focus();
      note.close();
    };
  }
}
