/**
 * The pause control: the human's switch for the agent's control of this
 * page.
 *
 * What it pauses is the product's view-grade tools (the server's pause gate,
 * `tools.py`), the ones that act on what the human is looking at; the
 * agent's own file edits and commands are not affected, and it can still read
 * the view. The button's wording is the page's own, since only the product
 * knows what its view-grade tools do.
 *
 * It is a pure reader of `store.paused`: clicking sends a frame and nothing
 * else, and the label only changes when the server says the flag moved. A
 * control that latched locally would show "paused" while the tools it claims
 * to gate were still running.
 */

import { store } from "./store.js";

/**
 * @param send    ws.js's frame sender, for the outbound pause frame
 * @param button  the control (default: the page's `#pauseBtn`). Its state is
 *                `.on` and `aria-pressed`. With a `[data-label]` child, only that
 *                child's text is written ("Pause" / "Paused"), so an icon or other
 *                markup the page put in the button stays; without one, the
 *                button's whole text is.
 * Returns `{setPausedFromServer}`, which the page hands to ws.js's `onPaused`.
 */
export function initPause({ send, button = document.getElementById("pauseBtn") }) {
  const label = button.querySelector("[data-label]");
  function renderPause(state) {
    button.classList.toggle("on", state.paused);
    if (label) label.textContent = state.paused ? "Paused" : "Pause";
    else button.textContent = state.paused ? "❙❙ Paused" : "❙❙ Pause";
    button.setAttribute("aria-pressed", state.paused ? "true" : "false");
    // Disabled when there is no agent, which is both viewer-only mode and a
    // session that failed to start. In viewer-only mode the server refuses a
    // pause frame outright, since there are no tools to pause, and an enabled
    // control whose every click produces a refusal toast is worse than one that
    // plainly cannot be pressed. Keyed on the same status the composer's Send
    // button reads, so the pane and the topbar agree about what is available.
    button.disabled = state.chat.agentStatus === "unavailable";
  }
  renderPause(store.getState());
  store.subscribe("paused", renderPause);
  store.subscribe("chat", renderPause);

  button.addEventListener("click", () => {
    // The current state is inverted and sent; the button's own appearance
    // does not change until the server broadcasts that the flag moved. That
    // is deliberate: the flag lives in the server because the tools it gates
    // run there, so a local latch would be a claim this page cannot make.
    send({ type: "pause", paused: !store.getState().paused });
  });

  return { setPausedFromServer: (paused) => store.setPaused(paused) };
}
