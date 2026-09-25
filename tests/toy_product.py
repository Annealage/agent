"""The product this package's own suite runs as.

The agent layer knows nothing about what a product shows or does
(``annealage_agent/product.py``), so its tests need some product installed,
and using a real one (Mesh) would both drag a 3D package into this suite and
make "read from the product" indistinguishable from "hard-coded to that
product's values". This one is small and deliberately unlike any real product:
every name it supplies is distinct from the agent layer's own, so an assertion
that sees ``.toy`` or ``mcp__toy__`` proves the value came from here.

It supplies one of everything the Product contract has:

- tools: ``TOY_GRADING`` grades five tools; two read (one of them asks the
  browser, as a product's read tools may), one view (drives the browser
  through the bus, so the pause gate applies), two write (leave a file behind,
  so they reach the permission broker);
- a settings key, ``units``, with choices, laid out in the Viewer section;
- an event, ``NotesChanged`` (``notes_changed``);
- an inbound frame, ``view``, whose ``view`` object may carry ``zoom`` and
  ``selection``;
- an upload kind, ``markup``;
- a page, ``TOY_PAGE``, with one inline script (the import map) for the
  Content-Security-Policy to hash, served at ``/`` by ``register_toy_routes``.

Tool handlers import the Claude Agent SDK only when ``build_toy_tools`` is
called, as a real product's must: the agent layer calls it in agent mode only.
"""

import dataclasses
import json
from pathlib import Path
from typing import ClassVar, Optional

from annealage_agent import product, protocol, settings
from annealage_agent.session.base import AgentEvent

TOY_PAGE = Path(__file__).resolve().parent / "toy_page.html"

#: The toy's tools, by grade (``annealage_agent.tools.Grading``'s fields).
TOY_READ = ("list_notes", "get_view")
TOY_VIEW = ("set_view",)
TOY_WRITE = ("add_note", "clear_notes")

#: The file the toy's write tools change, under the served directory.
NOTES_FILE = "notes.json"

#: What a pause-gated toy tool answers while the human has paused.
TOY_PAUSED_MESSAGE = "The toy view is paused by the human; read tools still work."


@dataclasses.dataclass(frozen=True)
class NotesChanged(AgentEvent):
    """The toy's product event: its notes file changed."""

    kind: ClassVar[str] = "notes_changed"
    viewer: Optional[str] = None


UNITS_KEY = settings.Key(
    name="units",
    type_name='"mm" or "in"',
    default="mm",
    layers=(settings.USER,),
    effect="load",
    description="Which units the toy view labels lengths in: mm or in.",
    py_type=str,
    choices=("mm", "in"),
    section=settings.VIEWER_SECTION,
)


def _check_view(frame):
    return protocol.object_error(frame.get("view"), {"zoom", "selection"}, set(), "view.view")


#: The toy page's report of its own view.
VIEW_FRAME = protocol.FrameSpec({"view"}, {"view"}, _check_view)


def _read_notes(serve_dir):
    path = Path(serve_dir) / NOTES_FILE
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_toy_tools(bus, serve_dir, session_id=None):
    """The toy's ``Product.build_tools``: a ``ToolServer`` over five tools
    graded by ``TOY_READ``/``TOY_VIEW``/``TOY_WRITE``."""
    from claude_agent_sdk import tool

    from annealage_agent import files
    from annealage_agent.tools import Grading, ToolServer, ok

    @tool("list_notes", "Read the notes saved in the served directory.", {})
    async def list_notes(args):
        return ok({"notes": _read_notes(serve_dir)})

    @tool("get_view", "Read what the toy view is showing.", {})
    async def get_view(args):
        return ok(await bus.call("toy.get_view"))

    @tool("set_view", "Zoom the toy view.", {"zoom": int})
    async def set_view(args):
        return ok(await bus.call("toy.set_view", {"zoom": args["zoom"]}))

    @tool("add_note", "Save a note in the served directory.", {"text": str})
    async def add_note(args):
        notes = _read_notes(serve_dir)
        notes.append(args["text"])
        files.atomic_replace(Path(serve_dir) / NOTES_FILE, json.dumps(notes).encode("utf-8"))
        return ok({"id": len(notes)})

    @tool("clear_notes", "Delete every saved note.", {})
    async def clear_notes(args):
        files.atomic_replace(Path(serve_dir) / NOTES_FILE, b"[]")
        return ok({"cleared": True})

    return ToolServer(
        [list_notes, get_view, set_view, add_note, clear_notes],
        grading=Grading(read=TOY_READ, view=TOY_VIEW, write=TOY_WRITE),
        bus=bus,
        paused_message=TOY_PAUSED_MESSAGE,
    )


def register_toy_routes(app, allowed_origins):
    """The toy's ``register_routes``: its page at ``/``."""
    from annealage_agent.http import Response

    @app.get("/")
    async def _page(req):
        return Response(TOY_PAGE.read_bytes(), headers={"Content-Type": "text/html; charset=utf-8"})


TOY = product.Product(
    name="toy",
    title="Toy",
    display_name="Annealage Toy",
    distribution="annealage-toy",
    module="annealage_toy",
    version="1.2.3",
    state_dirname=".toy",
    config_dirname="annealage-toy",
    mcp_server_name="toy",
    viewer_only_command="annealage-toy view",
    build_tools=build_toy_tools,
    settings_keys=(UNITS_KEY,),
    events=(NotesChanged,),
    inbound_frames={"view": VIEW_FRAME},
    upload_kinds=("markup",),
)
