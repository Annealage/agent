/**
 * The settings window, and the boot-time read that precedes it.
 *
 * Two jobs, and the first happens whether or not anyone opens the window.
 * `load` (on what `initSettings` returns) runs at page load and applies the
 * settings whose effect is "load" to this page: whether tool cards start
 * closed, here, and whatever the product's own load-effect keys mean, through
 * the `onLoad` hook the product passes (Mesh: which axis is up). Those go
 * through store setters rather than being applied to the DOM here, because
 * other modules read them and a value with two writers is the defect
 * `store.js` exists to prevent.
 *
 * The second job is the window itself. Every field shows where its value came
 * from, because a three-layer scheme is otherwise mystifying: "port 9000, from
 * user settings" tells you which file to edit, while a bare "9000" does not.
 * Fields whose effect is "restart" say so instead of pretending to apply, and
 * there is deliberately no live rebind: moving a listening socket while
 * connections are open is complexity with a security-relevant failure mode,
 * and restarting is cheap.
 *
 * The server answers `GET /settings` with what is in effect plus a `pending` map
 * of anything saved that differs, and "in effect" is per key: a restart-effect
 * key reports what the running process resolved at startup, while a load-effect
 * key reports what is on disk, because each page applies those as it loads.
 * Every control here edits the *saved* value, and the note beside it carries
 * what the running server is still using, so pressing Save with no edits can
 * never revert a change made earlier.
 *
 * Nothing here names a setting beyond the one generic load-effect key. The
 * payload says which keys the window shows and under which heading
 * (`sections`, from each key's own declaration on the server), which of them
 * are a closed set of choices (`choices`, `nullable`), and which product it is
 * describing (`product`), so a product's key appears and is editable here
 * because the product declared it, not because this file knows its name.
 */

import { store } from "./store.js";
import { authToken } from "./ws.js";
import { toast } from "./ui.js";

// Which layer a value came from, in words a person can act on. The keys are
// the layer names `settings.py` reports; the project file lives in the
// product's own state directory, which the payload names.
function originText(from, product) {
  switch (from) {
    case "flag":
      return "from a command-line flag, this run only";
    case "project":
      return `from this project's ${product.state_dirname}/config.toml`;
    case "user":
      return "from your user settings";
    case "default":
      return "built-in default";
    default:
      return from;
  }
}

// The options a key with a closed set of values is offered as, so the window
// shows a select rather than a free-text box that can only be got wrong. An
// empty value means "not set", offered first for a key where that is legal.
function choicesOf(entry) {
  const choices = entry.choices || [];
  if (!choices.length) return null;
  return entry.nullable ? ["", ...choices] : choices;
}

/**
 * Fetches `/settings` and returns the payload, or null if it could not be
 * read. A failure is not fatal to the page: the page works without ever
 * knowing what the saved preferences are, so this reports and carries on.
 */
async function fetchSettings() {
  const token = authToken();
  if (!token) return null;
  try {
    const res = await fetch("/settings?t=" + encodeURIComponent(token));
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    return null;
  }
}

/**
 * Applies the load-effect settings to this page, through the store: the
 * product's first, through its `onLoad(settings)` hook, then the chat pane's
 * own tool-card preference.
 *
 * Called at boot and after every save. Anything else in the payload describes
 * the run rather than the page, and is only shown when the window is opened.
 * Only a value that differs is written, here and (by the same rule) in the
 * product's hook: this read arrives well after the first frame, and
 * announcing a value the page already has can undo something done since,
 * a whole HTTP round trip later.
 */
function applyToPage(payload, onLoad) {
  const settings = payload && payload.settings;
  if (!settings) return;
  onLoad(settings);
  if (settings.tool_cards_collapsed) {
    const collapsed = settings.tool_cards_collapsed.value !== false;
    if (collapsed !== store.getState().toolCardsCollapsed) {
      store.setToolCardsCollapsed(collapsed);
    }
  }
}

function fieldRow(name, entry, pending, product) {
  const row = document.createElement("div");
  row.className = "setrow";
  row.dataset.key = name;

  const label = document.createElement("label");
  label.className = "setlabel";
  label.textContent = name;
  label.htmlFor = "set-" + name;
  row.appendChild(label);

  // The control edits the value that is *saved*, which after a write is the
  // pending one rather than the one this run is using. Showing the running
  // value instead would mean that opening the window and pressing Save with no
  // edits silently reverted the last change.
  const editing = pending ? pending.value : entry.value;

  let input;
  const choices = choicesOf(entry);
  if (choices) {
    input = document.createElement("select");
    for (const choice of choices) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice === "" ? "(not set)" : choice;
      input.appendChild(option);
    }
    input.value = editing === null ? "" : String(editing);
  } else if (entry.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = editing === true;
  } else {
    input = document.createElement("input");
    input.type = entry.type === "int" ? "number" : "text";
    input.value = editing === null ? "" : String(editing);
  }
  input.id = "set-" + name;
  input.className = "setinput";
  input.dataset.key = name;
  input.dataset.valueType = entry.type;
  if (!entry.editable) input.disabled = true;
  row.appendChild(input);

  const note = document.createElement("div");
  note.className = "setnote";
  const origin = originText(entry.from, product);
  const parts = [origin];
  if (entry.effect === "restart") parts.push("takes effect next run");
  if (pending) {
    // The field above holds the saved value, so what the note has to supply is
    // the other half: what the running server is still using.
    parts.push("saved; this run is still using " + JSON.stringify(entry.value));
  }
  note.textContent = parts.join(" · ");
  row.appendChild(note);

  const help = document.createElement("div");
  help.className = "sethelp";
  help.textContent = entry.description || "";
  row.appendChild(help);

  return row;
}

function diagnosticsBlock(facts, product) {
  const lines = [];
  const cli = facts.claude_cli || {};
  lines.push([product.title + " version", product.version]);
  lines.push(["Python", (facts.python || {}).version]);
  lines.push(["claude CLI", cli.source === "missing"
    ? "not found: agent mode cannot run"
    : (cli.version || "version not reported") + " (" +
      (cli.source === "bundled" ? "bundled with the SDK" : "found on PATH") + ")"]);
  lines.push(["git", facts.git ? facts.git.version || facts.git.path : "not installed"]);
  const sandbox = facts.sandbox || {};
  if (!(sandbox.dependencies || []).length) {
    lines.push(["Sandbox", "provided by this platform"]);
  } else if ((sandbox.missing || []).length) {
    lines.push(["Sandbox", "missing " + sandbox.missing.join(", ")]);
  } else {
    lines.push(["Sandbox", sandbox.dependencies.join(", ") + " present"]);
  }
  lines.push(["Project", facts.project_root]);
  lines.push(["Session", facts.session_id || "none (viewer only)"]);
  lines.push(["Bound to", facts.bind ? facts.bind + ":" + facts.port : "unknown"]);
  const reachable = facts.reachable || [];
  if (reachable.length) {
    lines.push(["Addresses", reachable
      .map((entry) => entry.address + " (" + entry.interface + ")").join(", ")]);
  }
  const files = facts.settings_files || {};
  lines.push(["User settings", files.user + (files.user_present ? "" : " (not written yet)")]);
  lines.push(["Project config",
    files.project + (files.project_present ? "" : " (not written yet)")]);

  const wrap = document.createElement("div");
  wrap.className = "diagblock";
  for (const [name, value] of lines) {
    const row = document.createElement("div");
    row.className = "diagrow";
    const key = document.createElement("span");
    key.className = "diagkey";
    key.textContent = name;
    const val = document.createElement("span");
    val.className = "diagval";
    val.textContent = value === undefined || value === null ? "unknown" : String(value);
    row.appendChild(key);
    row.appendChild(val);
    wrap.appendChild(row);
  }
  return wrap;
}

/**
 * Reads every input back and returns only the values that differ from what the
 * payload reported, so a save writes what was actually changed rather than
 * rewriting every key with what it already held.
 */
function collectChanges(root, payload) {
  const changes = {};
  for (const input of root.querySelectorAll(".setinput")) {
    const name = input.dataset.key;
    const entry = payload.settings[name];
    if (!entry || input.disabled) continue;
    let value;
    if (input.type === "checkbox") {
      value = input.checked;
    } else if (input.value === "") {
      value = null;
    } else if (entry.type === "int") {
      const parsed = Number(input.value);
      if (!Number.isInteger(parsed)) {
        throw new Error(name + " must be a whole number");
      }
      value = parsed;
    } else {
      value = input.value;
    }
    const saved = payload.pending && payload.pending[name]
      ? payload.pending[name].value
      : entry.value;
    if (value !== saved) changes[name] = value;
  }
  return changes;
}

async function saveChanges(changes) {
  const token = authToken();
  const res = await fetch("/settings?t=" + encodeURIComponent(token), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ changes }),
  });
  let payload = null;
  try {
    payload = await res.json();
  } catch (err) {
    payload = null;
  }
  if (!res.ok || !payload || payload.ok !== true) {
    throw new Error((payload && payload.error) || "the server refused the change");
  }
  return payload;
}

/**
 * Wires the topbar gear to a modal built fresh each time it opens, and
 * returns `{open, close, settings, load}`.
 *
 * Rebuilt per open rather than kept and updated, because it is opened rarely
 * and every field in it is a rendering of a payload that may have changed
 * since last time; a cached panel would be one more thing that can disagree
 * with the server.
 *
 * `openButton` and `container` default to the page's `#settingsBtn` and
 * `#settingsModal`; a page with neither still gets `load`. `onLoad(settings)`
 * is the product's hook for its own load-effect keys, given the payload's
 * `settings` map at boot and after every save. `load()` does the boot-time
 * read; the page calls it once, when it no longer minds the round trip.
 */
export function initSettings({
  openButton = document.getElementById("settingsBtn"),
  container = document.getElementById("settingsModal"),
  onLoad = () => {},
} = {}) {
  let current = null;

  async function load() {
    current = await fetchSettings();
    applyToPage(current, onLoad);
    return current;
  }

  if (!openButton || !container) {
    return { open: () => {}, close: () => {}, settings: () => current, load };
  }

  let onKeydown = null;

  function close() {
    container.hidden = true;
    container.textContent = "";
    if (onKeydown) {
      document.removeEventListener("keydown", onKeydown);
      onKeydown = null;
    }
  }

  async function open() {
    const payload = await fetchSettings();
    if (!payload || !payload.settings) {
      toast("Could not read the settings from the server.");
      return;
    }
    current = payload;
    container.textContent = "";

    const panel = document.createElement("div");
    panel.className = "setpanel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Settings");

    const header = document.createElement("header");
    const title = document.createElement("h2");
    title.textContent = "Settings";
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "setclose";
    closeBtn.setAttribute("aria-label", "Close settings");
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", close);
    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    if (payload.saved_error) {
      const warn = document.createElement("div");
      warn.className = "setwarn";
      warn.textContent = "A settings file could not be read: " + payload.saved_error;
      panel.appendChild(warn);
    }

    const body = document.createElement("div");
    body.className = "setbody";
    const product = payload.product || {};
    for (const section of payload.sections || []) {
      const block = document.createElement("section");
      const heading = document.createElement("h3");
      heading.textContent = section.title;
      block.appendChild(heading);
      for (const name of section.keys) {
        const entry = payload.settings[name];
        if (!entry) continue;
        block.appendChild(
          fieldRow(name, entry, payload.pending && payload.pending[name], product));
      }
      body.appendChild(block);
    }

    const diagSection = document.createElement("section");
    const diagHeading = document.createElement("h3");
    diagHeading.textContent = "Diagnostics";
    diagSection.appendChild(diagHeading);
    diagSection.appendChild(diagnosticsBlock(payload.diagnostics || {}, product));
    body.appendChild(diagSection);
    panel.appendChild(body);

    const footer = document.createElement("footer");
    const status = document.createElement("span");
    status.className = "setstatus";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "primary";
    save.textContent = "Save";
    save.addEventListener("click", async () => {
      let changes;
      try {
        changes = collectChanges(panel, payload);
      } catch (err) {
        status.textContent = err.message;
        return;
      }
      if (!Object.keys(changes).length) {
        status.textContent = "Nothing changed.";
        return;
      }
      save.disabled = true;
      status.textContent = "Saving…";
      try {
        const saved = await saveChanges(changes);
        current = saved;
        applyToPage(saved, onLoad);
        close();
        const names = Object.keys(changes);
        const restart = names.filter(
          (name) => saved.settings[name] && saved.settings[name].effect === "restart");
        toast(restart.length
          ? "Saved. " + restart.join(", ") + " takes effect next run."
          : "Saved.");
      } catch (err) {
        status.textContent = err.message;
        save.disabled = false;
      }
    });
    footer.appendChild(status);
    footer.appendChild(save);
    panel.appendChild(footer);

    container.appendChild(panel);
    container.hidden = false;

    onKeydown = (event) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKeydown);
  }

  openButton.addEventListener("click", () => { open(); });
  container.addEventListener("click", (event) => {
    // Only the backdrop closes; a click inside the panel must not.
    if (event.target === container) close();
  });

  return { open, close, settings: () => current, load };
}
