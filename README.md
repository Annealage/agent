# Annealage Agent

The embedded coding agent the Annealage products share. A product serves its own view in a browser page (a 3D model, a schematic sheet, an LVGL canvas); this package puts a chat pane beside it, runs a coding agent behind that pane in the product's project directory, and gives the agent the product's own tools under one permission model.

It provides:

- **Sessions for three agent backends**: Claude (the Claude Agent SDK, the default dependency), Codex (`openai-codex`, the `codex` extra) and omp ([Oh My Pi](https://github.com/can1357/oh-my-pi)'s `omp-rpc`, installed from a pinned commit because it is not on PyPI), plus a scripted fake session for tests. Which backends are installed is detected; with more than one, the product asks once.
- **The permission broker**: every write-class action, from any backend, reaches the human as an approval card (allow, allow for the session, always allow for this project, deny with a reason that goes to the model verbatim).
- **A WebSocket event channel with replay**: `/ws` streams the conversation as numbered events from a ring backed by `events.jsonl`, so a reloaded tab, a second device and a restarted process all catch up the same way.
- **The chat pane's front end**: ES modules and a stylesheet served at `/agent/static/` (chat, socket client, settings window, uploads, layout, pause control, a store products extend with their own slice, and the page's review client).
- **A shared review model**: the human's comments pinned on the product's view and the model's callouts beside them, anchored in the product's own terms, with the review tools the agent uses and one approval rule no product can grade away (resolving a human's comment is always a permission card).
- **Settings** in three layers (flag, project config, user settings, default) with provenance, **session persistence and resume**, **transcript export**, **image uploads**, a **per-project lock**, **bind modes** (loopback, all interfaces, the tailnet) and **diagnostics** (`doctor` and the settings window read one collector).

It is internal infrastructure, not a sixth product: nothing here runs on its own. Annealage Mesh uses it today; Annealage Loom's review tool is next.

## Status

Unpublished. Products depend on it from a sibling checkout through a uv path source:

```toml
[project]
dependencies = ["annealage-agent>=0.0.0"]

[tool.uv.sources]
annealage-agent = { path = "../agent", editable = true }
```

The version floor matches this package's `hatch-vcs` fallback version (`0.0.0`), since it has no tag yet. A path source never reaches published metadata, so **no product that depends on this package can be released until it is published** (to PyPI or a private index; that is an open decision). Annealage Mesh's publish workflow refuses to build while the path source is present.

## Development

    uv sync --extra dev --extra codex
    uv pip install "omp-rpc @ git+https://github.com/can1357/oh-my-pi.git@71c5eec978b0e7ce9ff057eb4e311f67f4f03eb9#subdirectory=python/omp-rpc"
    uv run --extra dev --extra codex pytest -q

The suite runs as a small made-up product (`tests/toy_product.py`) whose every name differs from the agent layer's own, so an assertion that sees one of its names proves the value came from the product. Tests marked `integration` drive real backend CLIs against real accounts and are excluded by default (`-m integration` runs them). The front end has no browser suite here, because only a product has a page: Annealage Mesh's `tests/test_viewer_e2e.py` drives these modules in Chromium, so run a product's suite too after changing them.

Python 3.10+. Code style follows Annealage Mesh's (the code came from there): ruff as configured in `pyproject.toml`, `Optional[X]`, prose comments. See [CONTRIBUTING.md](CONTRIBUTING.md).

## The Product contract

A product describes itself with one `annealage_agent.product.Product`, installed once per process (`product.install(PRODUCT)`); generic code reads it back with `product.current()`. `product.py` documents every field. In brief:

- **Identity**: `name`, `title`, `display_name`, `distribution`, `module`, `version`, the per-project state directory (`state_dirname`, where sessions, `state.json`, the lock, `permissions.toml` and the project `config.toml` live), the per-user config directory (`config_dirname`, the user `settings.toml` and the workspace trust store), the MCP server name every tool is namespaced under (`mcp__<name>__<tool>`), and the command that runs the product with no agent.
- **Tools**: `build_tools(bus, serve_dir, session_id)` returns a `tools.ToolServer` over the product's `@tool` definitions, each graded **read** (changes nothing: pre-allowed), **view** (changes only what is on screen: pre-allowed, gated by the human's pause switch) or **write** (leaves something behind: reaches the broker as a card, and pause-gated). The grade drives the pre-allowed list for every backend; a tool left ungraded is refused at startup. A product that keeps a review finds its store on `bus.review_store` and adds the shared review tools beside its own.
- **Settings keys** added to the generic set, each with the settings-window section and choices it is shown with.
- **Events**: the product's own `AgentEvent` subclasses, broadcast and replayed like the generic ones. A change to the review is the generic `review_changed`.
- **Inbound frames**: `protocol.FrameSpec`s for what the product's page reports over `/ws` (for Mesh, its camera and selection).
- **Upload kinds** beyond the generic `upload` (Mesh: `sketch`).

A kind, frame type or key that collides with the agent layer's own is refused at install.

## The review

`annealage_agent.review` owns the comment model every product shares. A `Comment` has an id that is never reused, an anchor in the product's `AnchorSpace` (a sheet and a point in millimetres, a point on a 3D part), what is at that point (`ref`), its text, its author (`human` or `model`), and a status and resolution where the store keeps them. The product supplies the anchor space (its tool-input schema, `validate`, `ref_at`) and a `ReviewStore`, whose `Capabilities` say what it supports: resolving, deleting the model's own callouts, the page adding the human's comments one at a time, and a limit on the model's open callouts.

- **Storage** is pluggable. `JsonReviewStore(path, anchor_space)` is the review's own file (format in `review/native.py`: `{"version": 2, "next_id": N, "comments": [...]}`, each comment's anchor nested as `"anchor": {...}`), re-read on every call, replaced atomically under a per-file lock, never overwritten when it does not parse, and never reusing an id. A product whose files are already a published contract keeps them behind its own store (Annealage Mesh's `mesh-comments.json` and `mesh-callouts.json`).
- **Tools**: `review.tools.review_tools(store, bus=bus, grading=GRADING)` returns `list_comments`, `add_callout`, and, where the store supports them, `resolve_comment` and `delete_callout`; names, descriptions, schemas and results are configurable, so a product with a published tool surface keeps it. The product grades them like its own tools, with one exception: resolving a human's comment always reaches the human as a card, every time. The resolve tool's handler asks the broker itself, with the comment and the model's note on the card, only for an open comment of the human's (the model's own callouts resolve freely); the tool is therefore pre-allowed and never gated by a transport or the pause switch whatever grade the product gave it (`tools.asks_the_human`), so each such resolve is asked exactly once on every backend, and the broker never remembers the answer: the pane offers no "always allow" for it and a grant already in `permissions.toml` is ignored.
- **The page**: `create_app(review_store=store)` serves the review at `GET /review` (and `POST /review` for a store whose page adds comments one at a time), behind the browser token, and runs a watcher that publishes `review_changed` whenever the store's files change, through the store or around it (a hand edit, an external agent). `static/review.js` is the page's client: `initReview({onChange, onError})` returns the `onEvent`, `onLive` and `onFallback` hooks to hand `initWs`, and renders nothing, since every product draws anchors its own way.

## How a product plugs in

The product's application module builds the app with `annealage_agent.app.create_app`, handing it the product's page (whose inline scripts the Content-Security-Policy hashes) and a function that registers the product's own routes:

```python
from annealage_agent import app as agent_app

app = agent_app.create_app(
    serve_dir,
    page_html=PAGE_HTML,
    port=port,
    host=bind.address,
    token=browser_token,
    agent_token=agent_token,
    session_id=session_id,  # None for viewer-only
    build_session=build_session,  # factory(on_event, *, bus) -> session or None
    register_routes=register_product_routes,  # (app, allowed_origins)
    settings=resolved_settings,
    login=login_nonces,
    review_store=review_store,  # a review.ReviewStore, or None
)
await agent_app.serve(app, host, port, on_ready=on_ready, background=(watcher.run,))
```

The product's CLI resolves what a run needs from the agent layer's helpers: the bind (`net.resolve_bind`), two tokens (`net.generate_token`), the lock (`lock.acquire(sessions.state_dir(dir), port)`), the session id (`sessions`), the settings (`settings.resolve`), the workspace trust gate (`session.workspace_trust`) and a `http.routes_login.LoginNonces`. Its `build_session` factory hands them to `launch.build_session(backend, on_event, bus=bus, ...)`, which builds the one broker and the backend's session around the product's tool server. Annealage Mesh's `app.py` and `cli.py` are the worked example.

The page loads the chat pane through an import map entry, `"agent/": "/agent/static/"`, and its stylesheet from `/agent/static/agent.css`. The product's main module wires the two halves together: `initWs({onEvent, onLive, onFallback, ...})` takes handlers for the product's own event kinds, `initChat({send, root, ids, agentTitles})` mounts the pane into the product's DOM, `initSettings`, `initLayout({tabs})` and `initPause({send})` do the rest, and `defineSlice()` in `store.js` gives the product its own slice of the page store.

## Security model

The agent holds a shell in a directory whose contents may have come from anywhere, and a browser tab can approve what it does. What stands between the two:

- **Two per-run secrets.** The *browser token* is the only credential `/ws`, the chat, upload, settings, review and login routes and any product route that asks for one accept; it authorises permission decisions. The *agent token* is the only credential `/mcp` accepts, handed to the Codex stdio bridge through its environment (never on a command line), and excluded from the agent's shell environment. Neither route family accepts the other's token, and an app whose two tokens are equal is refused at construction. Neither token is written to the lock file, the event log, the transcript or anything else in the project directory the agent can read.
- **A single-use login nonce for the browser the run opens.** A browser launched with a URL has that URL on its command line, readable through `ps`. So the auto-opened URL carries `#n=<nonce>`, which the page trades once at `POST /login` for the browser token; a nonce is spent by its first use and expires after 60 seconds. The banner's reusable `#t=<token>` link stays for a second tab or another device, and the page scrubs either fragment from the address bar. Residual risk, recorded as an open decision: a `#t=` link once opened can persist in the browser's history database, which the agent's shell could read.
- **Origin and Host checks** on every route, computed from the actual bind, so a rebound DNS name gets nothing and a remote viewer on a chosen bind works.
- **Workspace trust.** A directory's `.claude/settings.json`, `.claude/settings.local.json`, `.claude/hooks/`, `.mcp.json`, and a `.git/config` or `.git/hooks/` that names something executable, can run commands before any prompt is sent. The product's CLI refuses agent mode until the human has accepted exactly those contents (recorded in the user's config directory, never the project), and an in-session tripwire denies every tool call once they change.
- **Secret paths.** The sandbox stops writes and network, not reads, so tool calls naming a short list of credential paths (`~/.ssh`, `~/.aws`, `~/.config/gcloud`, `~/.kube`, `~/.gnupg`, `~/.netrc`, `~/.docker/config.json`, `~/.config/gh`, the agent's own credentials) are refused by a `PreToolUse` hook: exactly, symlinks included, for the file tools; by text matching, and so only partially, for shell commands.
- **The permission broker.** Every write-grade product tool, every edit through the backend's own tools, and every command outside the sandbox reaches the human as a card, from every backend, including write-class calls that arrive through `/mcp`. A request with nobody connected to answer it is denied rather than left hanging; standing grants are per project and never cover unrestricted shell access.
- **The pause switch** lives in the server: while the human holds it, view- and write-grade tools refuse whatever the page shows.
- **Response headers**: a `default-src 'none'` Content-Security-Policy with the page's inline scripts hashed at startup, `no-referrer`, `nosniff`, `no-store`.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE), free to use for any noncommercial purpose. Commercial use needs a separate licence; see [COMMERCIAL.md](COMMERCIAL.md).
