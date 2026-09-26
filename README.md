# Annealage Agent

An embedded coding agent for tools that already have a browser view. Your product serves its own page (a 3D model, a schematic sheet, whatever it shows). This package puts a chat pane beside it, runs a coding agent behind that pane in the project directory, and gives the agent your product's tools under one permission model.

I split it out of [Annealage Mesh](https://github.com/Annealage/mesh) when Annealage Loom's schematic review needed the same thing. Both are built on it now: Mesh supplies a three.js viewer and its CAD tools, Loom a sheet viewer and its build and review tools, and everything agent-side is this package. It's a library for building a product like those, not something you run on its own.

## What a product gets

- **Three agent backends** behind one session interface: Claude through the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) (a base dependency), Codex through `openai-codex` (the `codex` extra), and [Oh My Pi](https://github.com/can1357/oh-my-pi) through `omp-rpc`, which works with every provider omp is configured for, local models included. What's installed is detected, and a scripted fake session is there for tests.
- **Permission cards.** Every write-class action reaches the human as a card in the chat pane with the full input on it: Allow, Always allow (recorded for this project), or Deny with a reason that goes to the model verbatim. A request with nobody connected to answer it is denied rather than left hanging.
- **Graded tools.** You declare each of your tools as read (changes nothing), view (changes only what's on screen) or write (leaves something behind). Read and view run without asking, write gets a card, and a pause switch in the page refuses view and write while the human wants the view to hold still. An ungraded tool is refused at startup.
- **Remote MCP servers** beside your own tools: name a server on the network and grade its tools, and every backend reaches them through the same grades, pause switch and cards.
- **Workspace trust.** A served directory's `.claude/` settings and hooks, `.mcp.json`, or a git config or hook that names something executable can run commands before any prompt is sent. The package provides the gate that keeps agent mode off until the human has accepted exactly those contents, and your command line runs it (the example below does).
- **Secret and write-protected paths.** Tool calls reaching for credentials (`~/.ssh`, `~/.aws` and the like) are refused, and so are writes to files you name as only your own tools' to change.
- **A WebSocket event channel with replay.** `/ws` streams the conversation as numbered events, the human's own messages included, and a reloaded tab or a second device catches up on it. Recent events come from an in-memory ring of the last 500; in agent mode every event is also written to the session's `events.jsonl`, and anything older than the ring, or from before a restart that resumes the session (`-c` in Mesh and Loom), is replayed from there, with each streamed reply joined into one event per run. A plain rerun starts a new session.
- **The chat pane's front end**: plain ES modules and one stylesheet served at `/agent/static/`, no build step.
- **Settings** in four layers (flag, project config, user settings, built-in default), shown in a settings window with where each value came from, plus a diagnostics block.
- **What the backend itself says.** Every agent error goes to the server's stderr (a service's journal) as well as the page, where the chat banner shows the backend's own words under its advice. The settings window's Agent log section lists the backend's own logs for the session (omp's process log, the Claude CLI's transcript, a Codex rollout, each one's stderr) with their paths and shows the end of any of them.
- **A review model** for comments the human pins on your view and callouts the agent pins back, anchored in your product's own coordinates, with the tools the agent uses to read and answer them.
- **A login link and two tokens.** The browser and the agent get separate per-run secrets, and the browser the run opens gets a single-use login link rather than a reusable one.
- Session persistence and resume, transcript export, image uploads, a per-project lock, and bind modes for loopback, a chosen address or your tailnet.

## Install

Python 3.10 or later (the omp backend needs 3.11).

### From PyPI

It isn't published yet. Once it is:

    pip install annealage-agent
    pip install 'annealage-agent[codex]'   # the Codex backend as well

The Claude Agent SDK is a base dependency and its wheel bundles the Claude Code CLI, so an install pulls about 90 MB. The omp backend's client isn't on PyPI, so it isn't an extra either. Install it from the commit this package was verified against:

    pip install "omp-rpc @ git+https://github.com/can1357/oh-my-pi.git@71c5eec978b0e7ce9ff057eb4e311f67f4f03eb9#subdirectory=python/omp-rpc"

On Linux the Claude backend's shell sandbox needs `bubblewrap` and `socat` (`apt install bubblewrap socat`). macOS has its sandbox built in.

### As a git submodule

This is how to use it before a release, or to pin a commit you've tested:

    git submodule add https://github.com/Annealage/agent vendor/annealage-agent

Then point uv at the submodule in your `pyproject.toml`:

```toml
[project]
dependencies = ["annealage-agent"]

[tool.uv.sources]
annealage-agent = { path = "vendor/annealage-agent", editable = true }
```

`uv sync` builds it from the submodule, with the version taken from the submodule's git tags. Editable matters here: uv doesn't rebuild a non-editable path dependency when only its source files change, so moving the submodule to a new commit would otherwise leave the old code installed. The code follows the submodule, but the installed version stays at the old number, and a raised version floor isn't enforced until you run `uv sync --reinstall-package annealage-agent`. Anyone cloning your project needs `git clone --recurse-submodules` (or `git submodule update --init` afterwards).

A path source is uv-only and never reaches published metadata, so a product you publish to an index has to depend on a release of this package instead.

### As a uv git source

Without a submodule, uv can fetch it straight from the repository:

```toml
[tool.uv.sources]
annealage-agent = { git = "https://github.com/Annealage/agent", branch = "main" }
```

Pin a `tag = "v..."` or `rev = "<commit>"` instead of `branch` once you depend on a particular version.

## A minimal product

Two files: a page and a Python module. The page is your product's view with the chat pane beside it:

```html
<!-- page.html -->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hello Agent</title>
<link rel="stylesheet" href="/agent/static/agent.css">
<script type="importmap">
{"imports": {"agent/": "/agent/static/"}}
</script>
</head>
<body>
<main id="view">
  <h1>Hello</h1>
  <p>Your product's own view goes here.</p>
</main>
<aside id="chat">
  <header><h1>Chat</h1><span id="agentStatus" class="agentstatus"></span></header>
  <div id="chatBanner" class="chatbanner" hidden>
    <span id="chatBannerText"></span>
    <button id="chatBannerClose" type="button" aria-label="Dismiss">&times;</button>
  </div>
  <div id="chatLog"></div>
  <div id="chatPending"></div>
  <div id="chatComposer">
    <div class="chatmodelrow">
      <label for="chatModelInput">Model</label>
      <input id="chatModelInput" type="text" placeholder="backend default">
    </div>
    <div id="chatAttachStrip" class="attachstrip" hidden></div>
    <div id="chatComposerRow">
      <input id="chatFileInput" type="file" accept="image/png,image/jpeg,image/webp" multiple hidden>
      <button id="chatAttachBtn" type="button" class="attachbtn">Attach</button>
      <textarea id="chatInput" rows="2" placeholder="Message the agent"></textarea>
      <div class="chatbtns">
        <button id="chatInterrupt" type="button">Interrupt</button>
        <button id="chatSend" type="button">Send</button>
      </div>
    </div>
  </div>
</aside>
<div id="toast"></div>
<div id="err"></div>
<script type="module">
import { initChat } from "agent/chat.js";
import { initWs } from "agent/ws.js";

let ws;
const chat = initChat({ send: (frame) => ws && ws.send(frame) });
ws = initWs({
  onHello: chat.handleHello,
  onAgentEvent: chat.handleEvent,
  onRefused: chat.handleRefused,
});
</script>
</body>
</html>
```

The module describes the product, gives the agent two tools (one read, one write), and does what a product's command line has to do before it serves anything:

```python
# hello.py
import argparse
import asyncio
import sys
import webbrowser
from pathlib import Path

from annealage_agent import app as agent_app
from annealage_agent import launch, lock, net, product, sessions, settings
from annealage_agent.http import Response
from annealage_agent.http.routes_login import LoginNonces
from annealage_agent.session import workspace_trust

PAGE = Path(__file__).with_name("page.html")


def build_tools(bus, serve_dir, session_id):
    # Called in agent mode only, so a viewer-only run never imports the SDK.
    from claude_agent_sdk import tool

    from annealage_agent.tools import Grading, ToolServer, ok

    @tool("list_files", "List the files in the project directory.", {})
    async def list_files(args):
        return ok(sorted(p.name for p in Path(serve_dir).iterdir() if p.is_file()))

    @tool("write_note", "Write NOTE.txt in the project directory.", {"text": str})
    async def write_note(args):
        (Path(serve_dir) / "NOTE.txt").write_text(args["text"], encoding="utf-8")
        return ok({"written": "NOTE.txt"})

    return ToolServer(
        [list_files, write_note],
        grading=Grading(read=("list_files",), view=(), write=("write_note",)),
        bus=bus,
        paused_message="Paused by the human; list_files still works.",
    )


HELLO = product.Product(
    name="hello",
    title="Hello",
    display_name="Hello Agent",
    distribution="hello-agent",
    module="hello",
    version="0.1.0",
    state_dirname=".hello",
    config_dirname="hello-agent",
    mcp_server_name="hello",
    viewer_only_command="python hello.py DIR --backend none",
    build_tools=build_tools,
    cli_command="python hello.py DIR",
    doctor_command="",
    codex_install_hint="pip install 'annealage-agent[codex]'",
)


def register_routes(app, allowed_origins):
    @app.get("/")
    async def page(req):
        return Response(PAGE.read_bytes(), headers={"Content-Type": "text/html; charset=utf-8"})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hello")
    parser.add_argument("dir", type=Path)
    parser.add_argument("--backend", choices=("claude", "codex", "omp", "none"), default="claude")
    parser.add_argument("--trust-project-config", action="store_true")
    args = parser.parse_args(argv)

    product.install(HELLO)
    serve_dir = args.dir.resolve()
    resolved = settings.resolve(serve_dir)
    bind = net.resolve_bind(resolved["host"])
    port = resolved["port"]
    token, agent_token = net.generate_token(), net.generate_token()
    login = LoginNonces()

    session_id = digest = held = None
    if args.backend != "none":
        if args.backend == "claude":
            from annealage_agent.session import sdk

            if sdk.missing_sandbox_dependencies():
                sys.exit("the agent's shell runs sandboxed: install " + sdk.SANDBOX_PACKAGES)
        # A .claude/, .mcp.json or executable git config here can run commands
        # before the first prompt: accept exactly what is there, or refuse.
        digest = workspace_trust.config_digest(serve_dir)
        if digest != workspace_trust.EMPTY_DIGEST:
            trust = workspace_trust.TrustStore()
            if args.trust_project_config:
                trust.accept(serve_dir, digest)
            elif not trust.accepted(serve_dir, digest):
                sys.exit(
                    workspace_trust.refusal_message(serve_dir, workspace_trust.present(serve_dir))
                )
        held = lock.acquire(sessions.state_dir(serve_dir), port)
        session_id = sessions.create_session(serve_dir)

    def build_session(on_event, *, bus):
        if session_id is None:
            return None  # viewer-only
        return launch.build_session(
            args.backend,
            on_event,
            bus=bus,
            serve_dir=serve_dir,
            session_id=session_id,
            resumed=False,
            settings=resolved,
            mcp_host=bind.address,
            mcp_port=port,
            agent_token=agent_token,
            trusted_config_digest=digest,
        )

    app = agent_app.create_app(
        serve_dir,
        page_html=PAGE,
        port=port,
        host=bind.address,
        token=token,
        agent_token=agent_token,
        session_id=session_id,
        build_session=build_session,
        register_routes=register_routes,
        settings=resolved,
        login=login,
    )

    async def on_ready():
        print(net.format_banner(bind, port, token), flush=True)
        if resolved["open_browser"]:
            url = net.login_url(bind, port, login.issue())
            await asyncio.get_running_loop().run_in_executor(None, webbrowser.open, url)

    try:
        asyncio.run(agent_app.serve(app, bind.address, port, on_ready=on_ready))
    except KeyboardInterrupt:
        pass
    finally:
        if held is not None:
            held.release()


if __name__ == "__main__":
    main()
```

Run it against a directory:

    python hello.py ./project

It prints a banner with a reusable `#t=` link and opens your browser on a single-use `#n=` login link (both below, under the security model). Ask the agent what's in the folder and it calls `list_files` without asking. Ask it to leave a note and `write_note` puts a card in the pane first. The agent's sessions, the lock, standing grants and project settings live in `./project/.hello/`. `--backend none` serves the page with no agent.

A real product adds what this leaves out: `-c` to resume the last session (`sessions.resolve_continue`, then `resumed=True`), choosing a backend from settings and `backends.detect()` rather than a flag, and its own routes, events and settings keys. Annealage Mesh's `cli.py` and `app.py` are the fuller worked example.

## The Product contract

A product describes itself with one `annealage_agent.product.Product`, installed once per process with `product.install()`. Generic code reads it back with `product.current()`, and `product.py`'s docstring documents every field. In brief:

- **Identity**:
  - `name`, `title`, `display_name`, `distribution`, `module` and `version`, used in refusals, banners, the `Server` header and diagnostics.
  - `state_dirname`, the per-project directory holding sessions, the lock, `permissions.toml` and the project `config.toml`.
  - `config_dirname`, the per-user directory holding `settings.toml` and the workspace trust store.
  - `mcp_server_name`, which every tool is namespaced under (`mcp__<name>__<tool>`).
  - `viewer_only_command`, which every refusal that stops agent mode offers as the way out.
- **Tools**: `build_tools(bus, serve_dir, session_id)` returns a `tools.ToolServer` over the product's `@tool` definitions and their `Grading`. The grade decides the pre-allowed list every backend gets, what the pause switch refuses and what reaches the permission broker. `bus` is how a tool drives the page (`await bus.call(method, params)`, answered by the `dispatchCall(method, params)` function the page hands `initWs`), and it carries the product's review store as `bus.review_store` and the run's resolved settings as `bus.settings`.
- **Remote MCP servers**: `ToolServer(..., remote=(RemoteServer(name, url, grading, prime=None, headers=None),))`, with `RemoteServer` from `annealage_agent.remote`, adds a streamable-HTTP MCP server the product wants its agent to reach. The package connects when the tool server is built, makes the optional `prime` call (`(tool, arguments)`, for a server that lists more tools once one has been called), lists the tools and proxies each one the remote's `Grading` names, with the remote's own schema and description, graded, paused and failure-mapped like the product's own. Each remote is a server namespace of its own: Claude sees `mcp__<name>__<tool>`, Codex gets a bridge of its own at `/mcp/<name>`, and omp host tools named `<name>__<tool>`. A tool the remote lists but the grading doesn't name is left out, one the grading names but the remote doesn't list is skipped, and a remote that can't be reached within 10 seconds is left out of the session, each with a warning on stderr; none of them stops startup. The remote's `initialize` instructions are added to every backend's system prompt under a heading naming it. Each call opens its own connection, so the tool set is the one listed at startup. A remote named like the product's own server, or two with one name, is refused.
- **omp's own tool names**: omp keys some behaviour on a tool's name alone, whoever registered it. A successful result from any tool named `checkpoint` puts the session in a context checkpoint that re-samples the model until it calls `rewind` (which a product's tools don't include), so a turn never ends, and the state comes back from the session file on resume. A product tool that shares a name with one of omp's is therefore registered on omp as `<mcp_server_name>__<tool>` (Loom's `checkpoint` is `loom__checkpoint`), and omp's system prompt gets a line naming each renamed tool so guidance written against the product's names still finds it. Claude and Codex see the product's names unchanged.
- **Settings keys**: extra `settings.Key` rows, each with the settings-window section and choices it's shown with.
- **Events**: the product's own `AgentEvent` subclasses, broadcast and replayed like the generic ones.
- **Inbound frames**: `protocol.FrameSpec`s for what the page reports over `/ws` (Mesh: its camera and selection).
- **Upload kinds** beyond the generic `upload` (Mesh: `sketch`).
- **Write-protected files**: glob patterns under the served directory that only the product's own tools may write (Loom: `designs/src/*.review.json`).
- **Command hints**: `cli_command`, `doctor_command` and `codex_install_hint`, so refusals name commands the product really has.
- **Session context**: `session_context(bus, serve_dir)` returns a short text every backend adds to its system prompt (Loom: the design under review and the skills to use).

`install` refuses a product whose event kind, frame type or settings key collides with the package's own.

`app.create_app` takes the product's page (`page_html`, whose inline scripts the Content-Security-Policy hashes at startup), a `register_routes(app, allowed_origins)` for its own routes, the two tokens, the session factory and, optionally, a `review_store` and `external_agents=True` (below). `app.serve` binds, starts the session once the socket is listening, runs the product's background tasks and shuts it all down on Ctrl-C.

For a service that runs persistently (behind `tailscale serve`, say), `create_app` also takes:

- `token`: the browser token, given explicitly. `net.load_token(path)` keeps one in a 0600 file, creating it on first use, so a bookmarked link survives restarts; `net.generate_token()` is the per-run default.
- `host` and `port`: the bind. `extra_origins` and `extra_hosts`: the `Origin` (`https://box.tailnet.ts.net`) and `Host` (`box.tailnet.ts.net`) a proxy fronts the server under, accepted verbatim beside what the bind allows.
- `write_protected`: this app's write-protected patterns, relative to the served directory, in place of `Product.write_protected` (`None`, the default, keeps the product's).

`launch.build_session` takes `omp_agent_dir` (omp's agent directory, set as `PI_CODING_AGENT_DIR`: its own auth, settings and models), `omp_config_dir` (omp's config root, set as `PI_CONFIG_DIR` in place of `~/.omp` and its `agent/APPEND_SYSTEM.md`, `plugins/` and `.env`; omp joins it to `$HOME`, so it must lie under `$HOME`) and `omp_binary` (an absolute path to `omp`). The agent directory alone still leaves the user's `~/.omp` in effect; a service passes both. A project's own context files (`AGENTS.md` and the like in and above the served directory) are read either way. They are arguments for the product's CLI to pass, not settings keys, because settings are writable from the page. An omp conversation is kept under `<state dir>/omp`, the session's `meta.json` records omp's conversation file, and `-c`/`-r` resume it with `switch_session`, whether omp uses its own providers or `omp_base_url`. `OmpSession` itself takes these as `agent_dir`, `config_dir`, `binary`, `session_dir`, `resume` and `on_session_file`.

What the bus gives a product's tools beyond `call`:

- `bus.turn`: the conversation's current turn number: 0 before the first human turn of a new session (and in a viewer-only or external-agent run); a resumed session continues from the last turn in its history, so a new turn never reuses a number the page already shows. `bus.turn_at_open` is that starting value (0 for a new session), so `bus.turn - bus.turn_at_open` counts the turns this process has accepted.
- `bus.queue_note(text)`: text put in front of the next human turn, as a text block of its own marked as a note from the product rather than the human, on every backend (omp joins it to the message text; marker strings inside a note are defused, so a note cannot close itself early). Several notes go together; none is sent twice. A turn counts, and takes the notes, only when it goes to a ready session.
- `bus.on_turn_start(callback)`: `callback(turn)` runs synchronously as each human turn is accepted, with the new turn number, before that turn's notes are taken, so a note it queues goes with the same turn. A callback that raises is logged and does not stop the turn.
- `bus.attention(title, body)`: an `attention` event. The page shows a browser notification (permission is asked on the first click) and flashes its title until focused; a replayed one does neither.
- `tools.ok(..., end_turn=True)`: the result ends the agent's turn once the backend has it. omp aborts after its `tool_execution_end`, Claude interrupts once the tool result comes back through the stream, Codex interrupts once the MCP tool call completes. The turn ends as `ended_by_tool`. The text should still tell the model to stop and wait.

On omp, a message sent while a turn runs steers it (the page's Send button reads "Steer" then), a message omp refuses is reported and leaves the session ready, and each turn's cost and tokens come from `get_session_stats`. A remote MCP server that could not be reached at startup is tried again on the first turn and every minute; omp gets its tools at once (`set_host_tools`) and the model a note saying so. Claude's SDK servers and allow list, and Codex's bridges and tool list, are fixed when the session starts, so there the remote's tools arrive with the next start.

## The backend's own logs

Every product gets all of this from `create_app`, with nothing to wire:

- **The journal.** Every `AgentError` a session emits is also written to the server's stderr as `agent error: <remediation>: <stderr>`, the backend's text trimmed to its last 4000 characters with later lines indented. An error identical to the one written last isn't written again, so a burst of them is one line.
- **The banner.** The chat banner shows an `agent_error`'s remediation and, under it, the backend's own text in a collapsible block capped in height. `chat.js` adds that block to the page's `#chatBanner` itself, so the page's markup needs nothing new.
- **`AgentSession.backend_logs()`**, part of the session protocol (`session/base.py`) and implemented by every session, returns `BackendLog(name, kind, path=None, text=None, format="text")` entries: `kind` is `"file"` with a path, or `"text"` held in memory, and `format` is `"jsonl"` for JSON lines. `session/logfiles.py` finds them:
  - Claude: the CLI's transcript, `projects/<cwd slug>/<session id>.jsonl` under `$CLAUDE_CONFIG_DIR` (`~/.claude` by default), and the CLI's last 200 stderr lines.
  - Codex: the thread's rollout, `sessions/YYYY/MM/DD/rollout-<time>-<thread id>.jsonl` under `$CODEX_HOME` (`~/.codex`), and the app-server's last 400 stderr lines.
  - omp: its process log, `logs/omp.<date>.<pid>.log` under omp's config root (`~/.omp`, or the `omp_config_dir` a service passes), its conversation file, and the stderr `omp-rpc` keeps. When omp exits before it is ready, so its pid is never known, the log is guessed only once the start has failed: the newest one written between the start and the failure by an omp that is no longer running. Before that no process log is listed.

  A path is listed only when it's a regular file, not a symlink, inside the place that backend keeps it. Some of these names come from the backend itself, and omp's conversations sit inside the project, where the agent's shell can write.
- **`GET /agent/logs`** lists them (name, kind, path, format) and **`GET /agent/logs/<name>`** serves the last 256 KiB of the entry called `name`, starting at a whole line. The name is the key, not a position, because the list grows while the page shows it (omp's conversation file appears with the first message), and it's only ever looked up in the session's own list. Only the browser token opens them: they hold the conversation and whatever the provider said, so the agent token is refused, no request names a file, and no agent tool reads them.
- **The settings window's Agent log section** lists each log with its path and shows its end on request. A JSON-lines log (omp's) hides its debug-level lines until asked.
- **Diagnostics.** `diagnostics.collect` names them (without their text) under `backend_logs`: the running session's in `GET /settings`, and, for a `doctor` run with no server, the files the project's most recent session left, found from the ids it recorded. omp's process log isn't among those, since nothing records its pid.

## The front end

The page loads the pane's modules through one import map entry, `"agent/": "/agent/static/"`, and its stylesheet from `/agent/static/agent.css`. Nothing is bundled, and the page's own inline scripts are allowed by hash, so the product needs no build step either.

- `ws.js`: `initWs({onEvent, onLive, onFallback, onHello, onAgentEvent, onPaused, onRefused, dispatchCall, connTitles, indicator})` connects to `/ws`, reads the token out of the URL fragment (trading an `#n=` nonce at `POST /login`), replays what the tab missed and reconnects with backoff. `onEvent` takes handlers for the product's own event kinds, called for live events only (the history a connection opens with is not handed to them, so `onLive` should refetch what they would). `onRefused(reason, clientId)` gets each refusal, with the refused turn's `client_id` when it was a turn (pass chat.js's `handleRefused`). It returns `{send}`. `authToken()` gives the token to anything that calls a token-gated route.
- `chat.js`: `initChat({send, root, ids, agentTitles})` mounts the pane, finding its elements by id under `root`. The ids in the example are the defaults. `ids` maps a role (`log`, `input`, `banner`, `exportButton` and the rest, as in `DEFAULT_IDS`) to a different id. Every role's element has to exist except `exportButton` (default `#chatExport`, for transcript export).
- `settings.js`: `initSettings({openButton, container, onLoad})` is the settings window, on the page's `#settingsBtn` and `#settingsModal` by default.
- `pause.js`: `initPause({send, button})` is the pause control, on the page's `#pauseBtn` by default. It returns `{setPausedFromServer}`, which the page passes to `initWs` as `onPaused`.
- `layout.js`: `initLayout({tabs, tabbar, panelButton})` turns the panes into tabs at narrow widths. `tabs` is `[{id, label, target}]`, with `target` a selector for the pane, and the page needs `#tabbar` and `#panelBtn` unless it passes its own elements.
- `review.js`: `initReview({onChange, onError})` fetches and follows the review, returning the `onEvent`, `onLive` and `onFallback` hooks for `initWs` plus `add` and `setStatus`. It renders nothing, since every product draws anchors its own way.
- `store.js`: the page's one state store. `defineSlice(initial)` gives the product keys of its own.
- `ui.js` writes to the page's `#toast` and `#err` elements, which the page positions.

## The review

`annealage_agent.review` is the comment model both products share. A `Comment` has an id that's never reused, an anchor in the product's `AnchorSpace` (a sheet and a point in millimetres for Loom, a point on a 3D part for Mesh), what's at that point (`ref`), the text, the author (`human` or `model`), and a status where the store keeps one. The product supplies the anchor space (a tool-input schema, `validate`, `ref_at`) and a `ReviewStore` whose `Capabilities` say what it supports.

`JsonReviewStore(path, anchor_space)` is the package's own store, one JSON file replaced atomically and never overwritten when it doesn't parse. A product with an existing file format keeps it behind a store of its own. `review.tools.review_tools(store, bus=bus)` gives the agent `list_comments`, `add_callout` and, where the store supports them, `resolve_comment` and `delete_callout`, graded by the product like its own tools with one exception: resolving one of the human's comments always asks the human, with the comment on the card, whatever the product graded it. With a store passed to `create_app`, the page gets `GET /review` (plus `POST` routes where the store takes the human's comments or status changes), and a watcher publishes `review_changed` whenever the store's files change, however they changed.

## An agent in another process

An agent that isn't embedded (another Claude Code session, say) reaches the product's tools through the stdio MCP bridge:

    python -m annealage_agent.session.codex_mcp_stdio_bridge --host H --port P --server-name NAME --server-version V

with the run's agent token in its environment as `ANNEALAGE_AGENT_TOKEN`. Build the app with `create_app(external_agents=True)` and this works in viewer-only mode too: the run then has no conversation, but its write-grade calls still reach the page as cards.

## Security model

The agent holds a shell in a directory whose contents may have come from anywhere, and a browser tab can approve what it does. What sits between the two:

- **Two per-run secrets.** The browser token is the only credential `/ws`, the chat, upload, settings, review, agent log and login routes accept, and it's what authorises permission decisions. The agent token is the only one `/mcp` accepts. It reaches the stdio bridge through the environment, never a command line, and stays out of the agent's shell. Neither route family accepts the other's token, an app whose two tokens are equal is refused, and neither is written to the lock file, the event log, the transcript or anywhere else in the project directory.
- **A single-use login link.** A browser launched with a URL has that URL on its command line, which any local process can read through `ps`. So the link the run opens carries `#n=<nonce>` instead of the token, and the page trades it once at `POST /login`. A nonce is spent by its first use and expires after 60 seconds. The banner's `#t=<token>` link stays reusable, for a second tab or a phone, and the page scrubs either fragment from the address bar.
- **Origin and Host checks** on every route, computed from the actual bind, so a rebound DNS name gets nothing and a tailnet-bound viewer still works.
- **Workspace trust.** The product's command line runs the gate before it builds a session, as the example does, and nothing in `create_app` or `launch` runs it for you. `session.workspace_trust.config_digest` covers the directory's `.claude/settings.json`, `.claude/settings.local.json`, `.claude/hooks/`, `.mcp.json` and any `.git/config` or `.git/hooks/` that names something executable. A `TrustStore` in the user's config directory (never the project) records the contents the human accepted, and `refusal_message` says what to do when they haven't. Passing the accepted digest to `launch.build_session` as `trusted_config_digest` adds a tripwire on the Claude backend: while those files differ from what was accepted, every tool call is refused.
- **Secret paths.** The sandbox stops writes and network, not reads, so tool calls naming `~/.ssh`, `~/.aws`, `~/.config/gcloud`, `~/.kube`, `~/.gnupg`, `~/.netrc`, `~/.docker/config.json`, `~/.config/gh` or the agent's own `~/.claude/.credentials.json` are refused by a `PreToolUse` hook. For the file tools that check is exact, symlinks included. For shell commands it's text matching, so a path built from a variable or a glob gets through. It raises the floor against accidents and direct attempts, and isn't a wall against a determined agent. The same hook refuses writes to the product's write-protected files, with the same two halves.
- **The permission broker.** Every write-grade product tool reaches the human as a card on every backend, including calls arriving through `/mcp` and a remote MCP server's write-grade tools. On the Claude backend, file edits through its own tools and any command leaving the sandbox get cards too, while a contained shell command runs without asking (which is the point of containing it) and `git` runs outside the sandbox so it can work on the project's repository. Standing grants are per project and never cover the shell.
- **The pause switch** lives in the server: while the human holds it, view- and write-grade tools refuse whatever the page shows.
- **Response headers**: a `default-src 'none'` Content-Security-Policy with the page's inline scripts hashed at startup, `no-referrer`, `nosniff` and `no-store`.

### Known limits

- **Codex doesn't enforce secret paths or write-protected files.** Its own shell and patch tools run under its workspace sandbox and write inside the served directory without asking, and their requests carry no path to check. A product that relies on write-protected files must not offer the Codex backend (Loom refuses it). omp has no file or shell tools of its own, so only the product's tools apply there.
- **The tokens aren't a boundary against your own user.** They keep an honest agent to its tools. Anything running as you that can read the banner's `#t=` link, the browser's history (a `#t=` link once opened can persist there, and the sandboxed shell can read it) or the server's memory can act as the human: approve its own cards over `/ws` or change the review through the page's routes.
- **The state directory is writable by the sandboxed shell.** `permissions.toml` (the standing grants) and the project `config.toml` sit inside the served directory, where a contained command writes without a card, so an agent could add a grant for itself. Mesh's `.mesh/` and Loom's `.loom/` are the same.
- **Review store locking is per process.** `JsonReviewStore` serialises writers inside one process, so two processes serving the same review file can lose each other's changes.
- **A remote MCP server sees what its read-grade tools are called with.** Read and view run without a card, so whatever the agent passes a remote's pre-allowed tool leaves the machine unasked, and what comes back (results, descriptions, its `initialize` instructions) is text the model reads like any tool result. Declare only servers you trust with the project, and put any tool that sends something you'd want to see first in the write grade.

## Development

    uv sync --extra dev --extra codex
    uv pip install "omp-rpc @ git+https://github.com/can1357/oh-my-pi.git@71c5eec978b0e7ce9ff057eb4e311f67f4f03eb9#subdirectory=python/omp-rpc"
    uv run --extra dev --extra codex pytest -q

[CONTRIBUTING.md](CONTRIBUTING.md) covers why omp-rpc is installed separately (and what to do on Python 3.10, where it isn't available), the live `integration` tier and the pre-commit hooks. The suite runs as a small made-up product (`tests/toy_product.py`). The front end has no browser suite of its own, because only a product has a page: Annealage Mesh's end-to-end suite drives these modules in Chromium, so run it too after changing them. [RELEASING.md](RELEASING.md) covers publishing.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE), free to use for any noncommercial purpose. Commercial use needs a separate licence, covered in [COMMERCIAL.md](COMMERCIAL.md). A commercial licence for Annealage Mesh or Annealage Loom includes this package for use with that product, and it can also be licensed on its own. A commercial licence covers this package's own code only: the Claude Agent SDK (MIT, with its use governed by Anthropic's Commercial Terms of Service), the Claude Code CLI it bundles (© Anthropic PBC, all rights reserved), and the Codex and omp backends come under their owners' terms.

Contributions are welcome under the terms in [CONTRIBUTING.md](CONTRIBUTING.md). Every commit needs a Developer Certificate of Origin sign-off (`git commit -s`), which CI checks.
