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
    refused at startup.
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
    The product's own ``AgentEvent`` subclasses (Mesh: ``callouts_changed``,
    ``models_changed``), registered by ``install`` so a kind that collides with
    a generic one is refused before anything is published under it.
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
    codex_bridge_module: str = CODEX_BRIDGE_MODULE

    @property
    def server_header(self):
        """The ``Server`` response header value, ``<distribution>/<version>``."""
        return "%s/%s" % (self.distribution, self.version)

    @property
    def version_fact(self):
        """The diagnostics key the product's version is reported under."""
        return "%s_version" % self.name


_installed = None


def install(product):
    """Make ``product`` the one this process runs as, and register its
    settings keys, event kinds and inbound frame types (its upload kinds are
    checked, and read off it by the upload route).

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
    from .session import base

    # Each registration validates before it changes anything, and they are
    # checked in an order that leaves nothing half-registered if a later one
    # refuses: all are validated first, then all applied. Upload kinds need no
    # applying: the upload route reads them off the installed product.
    settings.check_product_keys(product.settings_keys)
    protocol.check_product_frames(product.inbound_frames)
    base.check_product_events(product.events)
    routes_chat.check_product_upload_kinds(product.upload_kinds)
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
