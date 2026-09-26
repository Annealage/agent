"""The Product contract: how a product describes itself to the agent layer.

The agent layer (this package) knows how to run an embedded
coding agent beside a browser page: sessions for three backends, the
permission broker, the event log, the ``/ws`` protocol, the chat, settings
and ``/mcp`` routes. It does not know what the page shows, what the agent's
tools do, what the product is called, or where the product keeps its state.
A product supplies all of that through one ``Product`` object, and this
module is the only place that object's shape is written down.

**One process runs one product.** ``install`` makes a product the one this
process runs as, and ``current`` returns it to the generic code that needs
the product's identity: the state directory sessions, settings and the lock
live under, the user config directory, the names that appear in refusals,
banners and the transcript heading, the MCP server name and the Codex bridge
module. Identity is read this way, rather than passed as an argument, because
it is needed deep inside module-level helpers (``sessions.state_dir``,
``settings.user_settings_path``, a lock refusal's message) whose every caller
would otherwise have to carry it through for a value that never varies within
a run. What does vary with the construction of an app or a session (the tool
server, its grading, the tokens) is still passed explicitly at construction.

Installing a second, different product into a process that already has one
raises: the registrations below are process-wide, and two products sharing
them would each see the other's settings keys and frame types. ``reset`` is
for tests alone, which is the one place a process legitimately swaps its
product (a product's ``tests/conftest.py`` has the fixture that does it and
puts the suite's own product back afterwards).

What a product supplies, and what each field drives. The values quoted in
parentheses are Annealage Mesh's, as an example:

``name``
    Lowercase short name ("mesh"). Used in prose the model or the human reads
    ("restart mesh", "the mesh viewer"), and as the stem of internal
    identifiers the agent layer has to invent per product: the omp provider
    id, the omp API-key environment variable, the omp temp directory prefix,
    and the diagnostics key that carries the product's version
    (``<name>_version``, which the product's ``doctor`` output reads; the
    settings window takes the version from ``GET /settings``'s ``product``).
``title``
    The same name capitalised for the start of a sentence ("Mesh refuses this
    call") and the transcript heading ("# Mesh transcript").
``display_name``
    The product's full name ("Annealage Mesh"): the exposure banner, and the
    client title Codex's app-server is told.
``distribution``
    The command and package distribution name ("annealage-mesh"): install and
    ``doctor`` hints, lock messages, the settings and grants file headers,
    the ``Server`` response header, and the name the Codex stdio bridge's MCP
    server reports.
``module``
    The importable package name ("annealage_mesh"), which is the client name
    Codex's app-server is told.
``version``
    The product's version string, for the ``Server`` header, diagnostics and
    the MCP server version.
``state_dirname``
    The per-project state directory (".mesh"): sessions, ``state.json``,
    the lock, ``permissions.toml`` and the project settings file live in it.
``config_dirname``
    The per-user config directory name under the platform's config home
    ("annealage-mesh"): the user ``settings.toml`` and the workspace trust
    store live in it.
``mcp_server_name``
    The MCP server name every tool is namespaced under ("mesh", so a tool is
    ``mcp__mesh__<tool>`` to the model) and the key the Codex bridge is
    registered as.
``viewer_only_command``
    How to run the product with no agent ("annealage-mesh view"), which every
    refusal that stops agent mode from starting offers as the way out.
``build_tools``
    ``build_tools(bus, serve_dir, session_id) -> tools.ToolServer``: the
    product's tools, each graded read, view or write (``tools.py``). The
    grade drives the pre-allowed list every backend receives, the pause gate
    and which calls reach the permission broker. Called once per agent-mode
    app (``app.py``'s ``create_app``, the only reader), and only then,
    so a viewer-only run never imports an agent SDK. A product with none
    (``None``) can serve viewer-only; building an agent-mode app for it is
    refused at startup. A product that keeps a review hands its
    ``review.ReviewStore`` to ``create_app`` and finds it here on
    ``bus.review_store``, to build the shared review tools
    (``review/tools.py``) over it beside its own. The run's resolved
    settings are ``bus.settings``, so a product's own settings key can shape
    what it builds. Beside its own tools, the product may declare remote MCP
    servers (``ToolServer(..., remote=(remote.RemoteServer(name, url,
    grading),))``), which every backend then reaches as server namespaces of
    their own, graded by the product like its own tools (``remote.py``).
``settings_keys``
    Extra ``settings.Key`` rows the product adds to the generic key set (Mesh:
    ``up_axis``), registered by ``install``. A key's ``section`` names the
    settings window section it is shown in (a section of the same title as a
    generic one is shared with it; ``None`` leaves the key out of the window),
    and its ``choices`` become the window's select options, so the product's
    front end registers nothing for a key to be shown and edited. Applying a
    ``load``-effect value to the page is the product front end's own job
    (Mesh's ``main.js`` passes ``initSettings`` an ``onLoad`` hook for it).
``events``
    The product's own ``AgentEvent`` subclasses (Mesh: ``models_changed``),
    registered by ``install`` so a kind that collides with a generic one is
    refused before anything is published under it. A change to the review is
    the generic ``review_changed``, not a product event.
``inbound_frames``
    ``{type: protocol.FrameSpec}`` for browser-to-server frames the product's
    page sends beyond the generic protocol (Mesh: ``state``), registered by
    ``install``. The agent layer validates them like any other frame and
    counts one as interaction with that tab; it does nothing else with them.
``upload_kinds``
    Extra values ``POST /upload``'s ``kind`` parameter accepts beyond the
    generic ``upload`` (Mesh: ``sketch``, the sketch overlay's composite).
    The kind names the written file (``sketch-<stamp>-<hex>.png``), so each
    must be a short lowercase slug; ``install`` refuses anything else. The
    agent layer does nothing with a kind beyond naming the file by it.
``write_protected``
    Glob patterns naming files the agent's own file tools and shell must not
    write, relative to the served directory with ``/`` separators (Mesh: none;
    Annealage Loom: ``designs/src/*.review.json``). A file only the product's
    tools may write, because a change made around them would skip what they
    check and who they ask: a review file, whose ``status`` a direct edit
    could flip on a human's comment with no card. Enforced beside the
    credential paths (``session/secret_paths.py``, which says exactly where
    and how far), and reading such a file stays allowed. ``*`` and ``?``
    match within one path segment; ``install`` refuses an absolute pattern
    or one with an empty, ``.`` or ``..`` segment. **Known limitation:** only
    the Claude backend enforces it (omp has no file or shell tools of its
    own); the Codex backend's own shell and patch tools write inside the
    workspace without asking, so a product that relies on this must not run
    Codex (Annealage Loom refuses it).
``cli_command``
    How the human starts the product, as a hint in refusals ("annealage-mesh";
    Loom: "loom-review --design <id>"). ``None`` (the default) is
    ``distribution``. Read through ``run_command``; the workspace-trust
    refusal appends ``--trust-project-config`` to it.
``doctor_command``
    The command that reports what this machine has, offered when a backend
    CLI fails to start. ``None`` (the default) is ``<distribution> doctor``;
    ``""`` says the product has none, and the hint is left out. Read through
    ``doctor_hint``.
``codex_install_hint``
    How to install the Codex extra, offered when it is missing. ``None`` (the
    default) is ``pip install <distribution>[codex]``. Read through
    ``codex_install``.
``session_context``
    ``session_context(bus, serve_dir) -> str | None``: a short text the
    agent's session is started with as an addition to its system prompt
    (what the run is about: Loom names the design under review and the skills
    to use). Called once per agent-mode session by ``launch.build_session``,
    after the tool server exists (``bus.tools``, ``bus.review_store``). Every
    backend takes it: Claude as its system prompt (the SDK's default one is
    empty), Codex as the thread's developer instructions, omp through
    ``--append-system-prompt``. ``None`` (the default) adds nothing. The
    ``initialize`` instructions of each remote MCP server the tool server
    reached follow it, each under a heading naming the server.
``codex_bridge_module``
    The module Codex launches as the stdio MCP bridge. Defaults to the agent
    layer's own bridge, which is the only one that speaks its ``/mcp``
    contract; it is a field so the path follows wherever the agent layer is
    installed rather than being written down by hand in a session module.
"""

import dataclasses
from typing import Any, Callable, Mapping, Optional, Tuple

#: The agent layer's own stdio MCP bridge, the module Codex launches with
#: ``python -m``.
CODEX_BRIDGE_MODULE = "annealage_agent.session.codex_mcp_stdio_bridge"


@dataclasses.dataclass(frozen=True, eq=False)
class Product:
    """One product's description of itself; see this module's docstring for
    what each field drives. Compared by identity: two products are the same
    product only if they are the same object."""

    name: str
    title: str
    display_name: str
    distribution: str
    module: str
    version: str
    state_dirname: str
    config_dirname: str
    mcp_server_name: str
    viewer_only_command: str
    build_tools: Optional[Callable[..., Any]] = None
    settings_keys: Tuple[Any, ...] = ()
    events: Tuple[type, ...] = ()
    inbound_frames: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    upload_kinds: Tuple[str, ...] = ()
    write_protected: Tuple[str, ...] = ()
    codex_bridge_module: str = CODEX_BRIDGE_MODULE
    cli_command: Optional[str] = None
    doctor_command: Optional[str] = None
    codex_install_hint: Optional[str] = None
    session_context: Optional[Callable[..., Any]] = None

    @property
    def server_header(self):
        """The ``Server`` response header value, ``<distribution>/<version>``."""
        return "%s/%s" % (self.distribution, self.version)

    @property
    def version_fact(self):
        """The diagnostics key the product's version is reported under."""
        return "%s_version" % self.name

    @property
    def run_command(self):
        """How the human starts the product (``cli_command``)."""
        return self.cli_command if self.cli_command is not None else self.distribution

    @property
    def doctor_hint(self):
        """The doctor command, or ``None`` for a product that has none."""
        if self.doctor_command is None:
            return "%s doctor" % self.distribution
        return self.doctor_command or None

    @property
    def codex_install(self):
        """How to install the Codex extra."""
        if self.codex_install_hint is not None:
            return self.codex_install_hint
        return "pip install %s[codex]" % self.distribution


_installed = None


def install(product):
    """Make ``product`` the one this process runs as, and register its
    settings keys, event kinds and inbound frame types (its upload kinds and
    write-protected patterns are checked, and read off it where they apply).

    Installing the product already installed is a no-op, so every entry point
    of a product may install it without coordinating which one runs first.
    Installing a different one raises ``RuntimeError``.
    """
    global _installed
    if _installed is product:
        return
    if _installed is not None:
        raise RuntimeError(
            "this process already runs %s; installing %s as well would mix two "
            "products' settings keys and frame types in one process"
            % (_installed.distribution, product.distribution)
        )
    # Imported here rather than at the top: each of these modules reads the
    # installed product through ``current`` in turn, and importing them from
    # this module's own top level would make the two import each other.
    from . import protocol, settings
    from .http import routes_chat
    from .session import base, secret_paths

    # Each registration validates before it changes anything, and they are
    # checked in an order that leaves nothing half-registered if a later one
    # refuses: all are validated first, then all applied. Upload kinds and
    # write-protected patterns need no applying: the upload route and the
    # session's tool-call guard read them off the installed product.
    settings.check_product_keys(product.settings_keys)
    protocol.check_product_frames(product.inbound_frames)
    base.check_product_events(product.events)
    routes_chat.check_product_upload_kinds(product.upload_kinds)
    secret_paths.check_write_protected(product.write_protected)
    settings.register_product_keys(product.settings_keys)
    protocol.register_product_frames(product.inbound_frames)
    base.register_product_events(product.events)
    _installed = product


def current():
    """The product this process runs as. Raises ``RuntimeError`` if none has
    been installed, naming what to do, rather than letting generic code fall
    back to some product's name."""
    if _installed is None:
        raise RuntimeError(
            "no product is installed in this process; the product's own package "
            "installs one with annealage_agent.product.install before the agent "
            "layer is used"
        )
    return _installed


def reset():
    """Uninstall the current product and clear its registrations.

    For tests alone: a real process installs one product and keeps it. See
    this package's ``tests/conftest.py``, whose ``swap_product`` fixture calls
    this, installs the product a test asks for, and restores the suite's own
    toy product afterwards.
    """
    global _installed
    from . import protocol, settings
    from .session import base

    settings.register_product_keys(())
    protocol.register_product_frames({})
    base.register_product_events(())
    _installed = None
