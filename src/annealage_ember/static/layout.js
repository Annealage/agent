/**
 * Panel toggle at wide viewports, tab bar at narrow ones.
 *
 * At 901px and up there is no tab bar: the Panel button toggles the body's
 * `panel-open` class, which the page's own stylesheet uses to show or hide
 * its side panel, and every other pane stays as the page lays it out. At
 * 900px and below the Panel button is hidden by CSS and a tab bar governs
 * which one of the panes is displayed; arbitrating a drawer transform and a
 * tab bar over the same element would be the complexity the narrow layout
 * exists to avoid, so at that width exactly one of them is ever shown.
 *
 * The page names its panes: `initLayout`'s `tabs` is the one list both the
 * tab bar and the show/hide logic below read, so a pane (the chat pane
 * included) needs only an entry there and a target element to exist, not a
 * change to the logic that renders or switches tabs.
 */

import { store } from "./store.js";

const NARROW_QUERY = "(max-width: 900px)";

/** Whether the tab bar is the one governing pane visibility right now. */
export function isNarrow() {
  return matchMedia(NARROW_QUERY).matches;
}

/**
 * Brings tab `id`'s pane forward. A no-op at wide viewports, where the tab
 * bar has no say over which panes are visible. This is the one entry point
 * a module outside this file should use to change which tab is showing: it
 * goes through the same `store.setActiveTab` the tab bar's own buttons use,
 * so the tab bar's `.on` state and the pane's `display` stay in sync with
 * whatever caused the switch, rather than a caller reaching past this
 * module into the DOM or the store directly.
 */
export function activateTab(id) {
  store.setActiveTab(id);
}

/**
 * `tabs` is `[{id, label, target}]`, `target` a selector for the pane the
 * tab shows, in tab-bar order; the first is the tab a narrow viewport starts
 * on. `tabbar` and `panelButton` default to the page's `#tabbar` and
 * `#panelBtn`.
 */
export function initLayout({
  tabs,
  tabbar = document.getElementById("tabbar"),
  panelButton = document.getElementById("panelBtn"),
}) {
  const mq = matchMedia(NARROW_QUERY);

  tabbar.innerHTML = "";
  tabs.forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.tab = t.id;
    btn.textContent = t.label;
    btn.addEventListener("click", () => store.setActiveTab(t.id));
    tabbar.appendChild(btn);
  });

  function applyTabButtons(state) {
    tabbar.querySelectorAll("button").forEach((btn) => {
      btn.classList.toggle("on", btn.dataset.tab === state.activeTab);
    });
  }

  function applyPanes() {
    const state = store.getState();
    document.body.classList.toggle("panel-open", state.panelOpen);
    panelButton.classList.toggle("on", state.panelOpen);
    if (mq.matches) {
      tabs.forEach((t) => {
        const el = document.querySelector(t.target);
        if (el) el.style.display = t.id === state.activeTab ? "" : "none";
      });
    } else {
      // Wide layout: clear any inline override the narrow branch left
      // behind so the panel-open CSS rule (keyed off the class just above)
      // governs the side panel, and every other pane is shown.
      tabs.forEach((t) => {
        const el = document.querySelector(t.target);
        if (el) el.style.display = "";
      });
    }
  }

  store.subscribe("activeTab", (state) => {
    applyTabButtons(state);
    applyPanes();
  });
  store.subscribe("panelOpen", applyPanes);
  // Crossing the breakpoint (e.g. rotating a tablet) does not change either
  // stored value, only which CSS rule and which of the two branches above
  // currently applies; re-running applyPanes is what makes that visible.
  mq.addEventListener("change", applyPanes);

  panelButton.addEventListener("click", () => store.setPanelOpen(!store.getState().panelOpen));

  // Wide viewports start with the panel open; narrow viewports start on the
  // first tab. These are independent initial choices: the Panel button is
  // hidden by CSS at narrow widths, so panelOpen has no visible effect
  // there, and the tab bar is hidden at wide widths, so activeTab has no
  // visible effect there either.
  store.setPanelOpen(!mq.matches);
  store.setActiveTab(tabs[0].id);
  applyTabButtons(store.getState());
  applyPanes();
}
