"""Annealage Agent: the embedded coding agent the Annealage products share.

A product (Annealage Mesh for 3D parts, for example) serves its own view in a
browser page; this package puts a chat pane beside it, runs the agent behind
that pane on one of three backends, and gives the agent the product's tools
under one permission model. Nothing here knows what the product's page shows
or what its tools do (``tests/test_agent_boundary.py`` checks that no module
imports a product and none uses a product's vocabulary). What it needs from
the product it runs in comes through ``product.py``.

Layout:

    product.py     the Product contract: identity, tool builder, settings keys,
                   events, inbound frames and upload kinds; install()/current()
    app.py         generic app assembly (routes, CSP, security headers, event
                   publisher, tool server, session) and serve()
    launch.py      the backend switch: one broker, one session per run
    headless.py    run_prompt: one prompt as one turn of a private session
                   with no human, for a product's batch sub-agents
    tools.py       ToolSpec, READ/VIEW/WRITE Grading, ok/fail, the pause gate,
                   failure mapping, ToolServer (SDK server, pre-allowed list,
                   tool_table)
    remote.py      remote MCP servers a product declares: discovery, grading
                   and the per-call proxy behind their tools
    files.py       path safety, guarded fixed-file reads/appends, atomic
                   replace, image sniffing, images/ and review/ files
    settings.py    generic settings keys, layering, product key registration
    sessions.py    <state dir>/sessions and state.json
    lock.py        <state dir>/lock (pid and port, never a token)
    protocol.py    /ws frames, generic and product inbound frame specs
    viewers.py     ViewerRegistry and ViewerBus
    net.py         bind modes, tokens, Origin/Host allowlists, banner
    backends.py    which agent backends are installed, and choosing one
    diagnostics.py what doctor and GET /settings report
    session/       AgentSession, generic events, the Claude, Codex, omp and
                   fake sessions, permissions, workspace trust, secret paths,
                   turn images, the event log, the Codex stdio MCP bridge
    review/        the shared review model: Comment, AnchorSpace, ReviewStore
                   and its capabilities, the native JsonReviewStore, the
                   review watcher (review_changed) and the review tools with
                   their approval policy
    http/          shared route helpers, /ws, /login, chat (/upload, /asset,
                   export), /settings, /review, /mcp and /agent/static/ routes
    static/        the chat pane's front end (ES modules and agent.css) and
                   the page's review client (review.js), served at
                   /agent/static/

This code was developed inside Annealage Mesh and extracted from it (Mesh
commit 59036ba). Comments that cite "plan section N", a ``planning/`` file or
``docs/agent-chat-plan.md`` refer to that repository's design records.
"""

# ``run_prompt``, ``RunResult`` and ``ToolSet`` (``headless.py``) are the
# package's one public import path for a headless run. They resolve on first
# use, because ``headless`` imports the Claude SDK and a viewer-only run, or
# anything else that only imports a submodule, must not pay for that.
_HEADLESS = ("run_prompt", "RunResult", "ToolSet")


def __getattr__(name):
    if name in _HEADLESS:
        from . import headless

        return getattr(headless, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(_HEADLESS))
