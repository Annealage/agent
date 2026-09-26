# Annealage Agent

An embedded coding agent for tools that already have a browser view. Your product serves its own page (a 3D model, a schematic sheet, whatever it shows). This package puts a chat pane beside it, runs a coding agent behind that pane in the project directory, and gives the agent your product's tools under one permission model.

I split it out of [Annealage Mesh](https://github.com/Annealage/mesh) when Annealage Loom's schematic review needed the same thing. Both are built on it now: Mesh supplies a three.js viewer and its CAD tools, Loom a sheet viewer and its build and review tools, and everything agent-side is this package. It's a library for building a product like those, not something you run on its own.

## What a product gets

- **Three agent backends** behind one session interface: Claude through the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) (a base dependency), Codex through `openai-codex` (the `codex` extra), and [Oh My Pi](https://github.com/can1357/oh-my-pi) through `omp-rpc`, which works with every provider omp is configured for, local models included. What's installed is detected, and a scripted fake session is there for tests.
- **Permission cards.** Every write-class action reaches the human as a card in the chat pane with the full input on it: Allow, Always allow (recorded for this project), or Deny with a reason that goes to the model verbatim. A request with nobody connected to answer it is denied rather than left hanging.
- **Graded tools.** You declare each of your tools as read (changes nothing), view (changes only what's on screen) or write (leaves something behind). Read and view run without asking, write gets a card, and a pause switch in the page refuses view and write while the human wants the view to hold still. An ungraded tool is refused at startup.
- **Workspace trust.** A served directory's `.claude/` settings and hooks, `.mcp.json`, or a git config or hook that names something executable can run commands before any prompt is sent. The package provides the gate that keeps agent mode off until the human has accepted exactly those contents, and your command line runs it (the example below does).
- **Secret and write-protected paths.** Tool calls reaching for credentials (`~/.ssh`, `~/.aws` and the like) are refused, and so are writes to files you name as only your own tools' to change.
- **A WebSocket event channel with replay.** `/ws` streams the conversation as numbered events, and a reloaded tab or a second device catches up from an in-memory ring of the last 500. Agent mode also writes every event to the session's `events.jsonl`, so a restart that resumes the session (`-c` in Mesh and Loom) carries the numbering on, though the earlier history isn't replayed to the page. A plain rerun starts a new session.
- **The chat pane's front end**: plain ES modules and one stylesheet served at `/agent/static/`, no build step.
- **Settings** in four layers (flag, project config, user settings, built-in default), shown in a settings window with where each value came from, plus a diagnostics block.
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
- **Tools**: `build_tools(bus, serve_dir, session_id)` returns a `tools.ToolServer` over the product's `@tool` definitions and their `Grading`. The grade decides the pre-allowed list every backend gets, what the pause switch refuses and what reaches the permission broker. `bus` is how a tool drives the page (`await bus.call(method, params)`, answered by the `dispatchCall(method, params)` function the page hands `initWs`), and it carries the product's review store as `bus.review_store`.
- **Settings keys**: extra `settings.Key` rows, each with the settings-window section and choices it's shown with.
- **Events**: the product's own `AgentEvent` subclasses, broadcast and replayed like the generic ones.
- **Inbound frames**: `protocol.FrameSpec`s for what the page reports over `/ws` (Mesh: its camera and selection).
- **Upload kinds** beyond the generic `upload` (Mesh: `sketch`).
- **Write-protected files**: glob patterns under the served directory that only the product's own tools may write (Loom: `designs/src/*.review.json`).
- **Command hints**: `cli_command`, `doctor_command` and `codex_install_hint`, so refusals name commands the product really has.
- **Session context**: `session_context(bus, serve_dir)` returns a short text every backend adds to its system prompt (Loom: the design under review and the skills to use).

`install` refuses a product whose event kind, frame type or settings key collides with the package's own.

`app.create_app` takes the product's page (`page_html`, whose inline scripts the Content-Security-Policy hashes at startup), a `register_routes(app, allowed_origins)` for its own routes, the two tokens, the session factory and, optionally, a `review_store` and `external_agents=True` (below). `app.serve` binds, starts the session once the socket is listening, runs the product's background tasks and shuts it all down on Ctrl-C.

## The front end

The page loads the pane's modules through one import map entry, `"agent/": "/agent/static/"`, and its stylesheet from `/agent/static/agent.css`. Nothing is bundled, and the page's own inline scripts are allowed by hash, so the product needs no build step either.

- `ws.js`: `initWs({onEvent, onLive, onFallback, onHello, onAgentEvent, onPaused, onRefused, dispatchCall, connTitles, indicator})` connects to `/ws`, reads the token out of the URL fragment (trading an `#n=` nonce at `POST /login`), replays what the tab missed and reconnects with backoff. `onEvent` takes handlers for the product's own event kinds. It returns `{send}`. `authToken()` gives the token to anything that calls a token-gated route.
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

- **Two per-run secrets.** The browser token is the only credential `/ws`, the chat, upload, settings, review and login routes accept, and it's what authorises permission decisions. The agent token is the only one `/mcp` accepts. It reaches the stdio bridge through the environment, never a command line, and stays out of the agent's shell. Neither route family accepts the other's token, an app whose two tokens are equal is refused, and neither is written to the lock file, the event log, the transcript or anywhere else in the project directory.
- **A single-use login link.** A browser launched with a URL has that URL on its command line, which any local process can read through `ps`. So the link the run opens carries `#n=<nonce>` instead of the token, and the page trades it once at `POST /login`. A nonce is spent by its first use and expires after 60 seconds. The banner's `#t=<token>` link stays reusable, for a second tab or a phone, and the page scrubs either fragment from the address bar.
- **Origin and Host checks** on every route, computed from the actual bind, so a rebound DNS name gets nothing and a tailnet-bound viewer still works.
- **Workspace trust.** The product's command line runs the gate before it builds a session, as the example does, and nothing in `create_app` or `launch` runs it for you. `session.workspace_trust.config_digest` covers the directory's `.claude/settings.json`, `.claude/settings.local.json`, `.claude/hooks/`, `.mcp.json` and any `.git/config` or `.git/hooks/` that names something executable. A `TrustStore` in the user's config directory (never the project) records the contents the human accepted, and `refusal_message` says what to do when they haven't. Passing the accepted digest to `launch.build_session` as `trusted_config_digest` adds a tripwire on the Claude backend: while those files differ from what was accepted, every tool call is refused.
- **Secret paths.** The sandbox stops writes and network, not reads, so tool calls naming `~/.ssh`, `~/.aws`, `~/.config/gcloud`, `~/.kube`, `~/.gnupg`, `~/.netrc`, `~/.docker/config.json`, `~/.config/gh` or the agent's own `~/.claude/.credentials.json` are refused by a `PreToolUse` hook. For the file tools that check is exact, symlinks included. For shell commands it's text matching, so a path built from a variable or a glob gets through. It raises the floor against accidents and direct attempts, and isn't a wall against a determined agent. The same hook refuses writes to the product's write-protected files, with the same two halves.
- **The permission broker.** Every write-grade product tool reaches the human as a card on every backend, including calls arriving through `/mcp`. On the Claude backend, file edits through its own tools and any command leaving the sandbox get cards too, while a contained shell command runs without asking (which is the point of containing it) and `git` runs outside the sandbox so it can work on the project's repository. Standing grants are per project and never cover the shell.
- **The pause switch** lives in the server: while the human holds it, view- and write-grade tools refuse whatever the page shows.
- **Response headers**: a `default-src 'none'` Content-Security-Policy with the page's inline scripts hashed at startup, `no-referrer`, `nosniff` and `no-store`.

### Known limits

- **Codex doesn't enforce secret paths or write-protected files.** Its own shell and patch tools run under its workspace sandbox and write inside the served directory without asking, and their requests carry no path to check. A product that relies on write-protected files must not offer the Codex backend (Loom refuses it). omp has no file or shell tools of its own, so only the product's tools apply there.
- **The tokens aren't a boundary against your own user.** They keep an honest agent to its tools. Anything running as you that can read the banner's `#t=` link, the browser's history (a `#t=` link once opened can persist there, and the sandboxed shell can read it) or the server's memory can act as the human: approve its own cards over `/ws` or change the review through the page's routes.
- **The state directory is writable by the sandboxed shell.** `permissions.toml` (the standing grants) and the project `config.toml` sit inside the served directory, where a contained command writes without a card, so an agent could add a grant for itself. Mesh's `.mesh/` and Loom's `.loom/` are the same.
- **Review store locking is per process.** `JsonReviewStore` serialises writers inside one process, so two processes serving the same review file can lose each other's changes.

## Development

    uv sync --extra dev --extra codex
    uv pip install "omp-rpc @ git+https://github.com/can1357/oh-my-pi.git@71c5eec978b0e7ce9ff057eb4e311f67f4f03eb9#subdirectory=python/omp-rpc"
    uv run --extra dev --extra codex pytest -q

[CONTRIBUTING.md](CONTRIBUTING.md) covers why omp-rpc is installed separately (and what to do on Python 3.10, where it isn't available), the live `integration` tier and the pre-commit hooks. The suite runs as a small made-up product (`tests/toy_product.py`). The front end has no browser suite of its own, because only a product has a page: Annealage Mesh's end-to-end suite drives these modules in Chromium, so run it too after changing them. [RELEASING.md](RELEASING.md) covers publishing.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE), free to use for any noncommercial purpose. Commercial use needs a separate licence, covered in [COMMERCIAL.md](COMMERCIAL.md). A commercial licence for Annealage Mesh or Annealage Loom includes this package for use with that product, and it can also be licensed on its own. A commercial licence covers this package's own code only: the Claude Agent SDK (MIT, with its use governed by Anthropic's Commercial Terms of Service), the Claude Code CLI it bundles (© Anthropic PBC, all rights reserved), and the Codex and omp backends come under their owners' terms.

Contributions are welcome under the terms in [CONTRIBUTING.md](CONTRIBUTING.md). Every commit needs a Developer Certificate of Origin sign-off (`git commit -s`), which CI checks.
