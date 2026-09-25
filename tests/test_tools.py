"""Tests for a product's tool server (``tools.py``), driven against a fake
``ViewerBus`` through the toy product's tools (``tests/toy_product.py``).

Three things are being pinned here, and they are different in kind.

**The grading**, because it is the whole permission design for a product's
tools, and because its two derived sets are deliberately not the same set: what
prompts is the write grade, while what the pause switch refuses is that plus
the view grade. A test that checked only one of those would pass with the other
silently wrong, so both are asserted, exhaustively. The expected tuples below
are written out by hand rather than imported from the toy product, for the same
reason ``tests/test_sdk_session.py`` writes out the allow list: a test that
derives its expectation from the thing it is testing cannot notice that thing
changing.

**The failure mapping**, because the four ways a viewer call can fail mean four
different things to a model. A model told "it timed out" retries; one told "no
viewer is connected" asks the human to open the page; one told "the viewer
refused it" reads the reason. Collapsing them would be invisible in any test
that only checked ``is_error``, so each is asserted on its wording.

**The pause gate**, exhaustively over every gated toy tool rather than over a
sample, and over every tool that is not gated.

Every handler here is reached through the ``ToolServer`` the toy builds, never
called directly, so what is under test includes the wrapper that applies both
policies.
"""

import asyncio
import json

import pytest
from toy_product import NOTES_FILE, TOY_PAUSED_MESSAGE, build_toy_tools

from annealage_agent.tools import Grading, ToolServer, namespaced
from annealage_agent.viewers import CallError, NoViewerConnected, ViewerGone

pytestmark = pytest.mark.asyncio


# The toy's three grades, written out by hand.
EXPECTED_READ_CLASS = ("list_notes", "get_view")
EXPECTED_VIEW_CLASS = ("set_view",)
EXPECTED_WRITE_CLASS = ("add_note", "clear_notes")

# Never prompts: nothing here reaches the broker, so nothing here interrupts.
EXPECTED_PRE_ALLOWED = EXPECTED_READ_CLASS + EXPECTED_VIEW_CLASS
# Refused while paused: everything that changes anything, screen or disk.
EXPECTED_PAUSE_GATED = EXPECTED_VIEW_CLASS + EXPECTED_WRITE_CLASS

# The smallest arguments each toy tool accepts, so the pause tests can drive
# every one of them without each needing its own call.
ARGS = {
    "list_notes": {},
    "get_view": {},
    "set_view": {"zoom": 2},
    "add_note": {"text": "here"},
    "clear_notes": {},
}

VIEWER_URL = "http://127.0.0.1:8765/#t=testtoken"


class FakeBus:
    """A recorder with the two members every tool handler depends on.

    The real ``ViewerBus`` exposes ``paused`` as a read-only property set
    through ``set_paused``; here it is a plain attribute, because a test setting
    it directly is the point. ``call`` records and then either raises whatever
    ``raises`` holds or returns the reply registered for that method, defaulting
    to an empty object rather than None, since a viewer that answers a call
    always answers with something.
    """

    def __init__(self, replies=None, raises=None):
        self.paused = False
        self.calls = []
        self.replies = dict(replies or {})
        self.raises = raises

    async def call(self, method, params=None, *, timeout=None):
        self.calls.append((method, params or {}))
        if self.raises is not None:
            raise self.raises
        return self.replies.get(method, {})


@pytest.fixture
def project(tmp_path):
    (tmp_path / NOTES_FILE).write_text('["first"]', encoding="utf-8")
    return tmp_path


def tools_for(bus, serve_dir):
    """``{name: handler}`` for one built server, wrapper included."""
    return {t.name: t.handler for t in build_toy_tools(bus, serve_dir).tools}


def text_of(result):
    """The text of a result's first text block, whatever else it carries."""
    for item in result["content"]:
        if item["type"] == "text":
            return item["text"]
    return ""


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


async def test_the_three_grades_are_what_they_are_meant_to_be(project):
    grading = build_toy_tools(FakeBus(), project).grading
    assert grading == Grading(
        read=EXPECTED_READ_CLASS, view=EXPECTED_VIEW_CLASS, write=EXPECTED_WRITE_CLASS
    )
    # The two derived sets, which are the ones the code actually acts on, and
    # which are different from each other on purpose.
    assert grading.pre_allowed == EXPECTED_PRE_ALLOWED
    assert grading.pause_gated == EXPECTED_PAUSE_GATED
    assert set(grading.pre_allowed) != set(grading.pause_gated)


async def test_every_classified_tool_exists_and_every_built_tool_is_classified(project):
    built = set(tools_for(FakeBus(), project))
    assert built == (
        set(EXPECTED_READ_CLASS) | set(EXPECTED_VIEW_CLASS) | set(EXPECTED_WRITE_CLASS)
    )


async def test_the_pre_allowed_names_are_exactly_what_the_session_pre_allows(project):
    """The two lists are one list, and this is the seam where a divergence
    would show up as a pre-allowed name matching nothing (fact 1): the tool
    server's ``pre_allowed`` is what ``launch.py`` hands the Claude session as
    its allow list, namespaced under the server's own name, which is the
    installed product's ``mcp_server_name``."""
    tools = build_toy_tools(FakeBus(), project)
    assert tools.pre_allowed == tuple(namespaced("toy", name) for name in EXPECTED_PRE_ALLOWED)
    assert list(tools.mcp_servers) == ["toy"]


async def test_no_write_class_tool_is_pre_allowed(project):
    """The one assertion that keeps the approval card. A write-class name in
    ``allowed_tools`` would silently stop the broker being consulted for it
    (fact 2), and nothing else in the suite would notice."""
    pre_allowed = build_toy_tools(FakeBus(), project).pre_allowed
    for name in EXPECTED_WRITE_CLASS:
        assert namespaced("toy", name) not in pre_allowed


async def test_every_view_class_tool_is_pre_allowed_and_gated(project):
    """The decision that separates the two derived sets: these prompt for
    nothing, because the human is watching the screen they change, and the
    pause switch is what stops them instead. Both halves are asserted here,
    because either one alone would be a different design: pre-allowed and
    ungated is a view nothing can stop, and gated and prompting is the card
    per view change this deliberately does not do."""
    tools = build_toy_tools(FakeBus(), project)
    for name in EXPECTED_VIEW_CLASS:
        assert namespaced("toy", name) in tools.pre_allowed
        assert name in tools.grading.pause_gated


async def test_building_refuses_a_tool_that_was_never_classified(project):
    """A tool added to a product's tool set without being classified must fail
    loudly at startup, because every default is wrong for something: read
    removes the human's card and the pause switch's hold on it, view removes the
    card alone, and write is a tool nobody can reach through the allow list."""
    from claude_agent_sdk import tool

    real = build_toy_tools(FakeBus(), project)

    @tool("wander_off", "unclassified", {})
    async def wander_off(args):
        return {"content": []}

    with pytest.raises(RuntimeError, match="wander_off"):
        ToolServer(
            list(real.tools) + [wander_off],
            grading=real.grading,
            bus=FakeBus(),
            paused_message=TOY_PAUSED_MESSAGE,
        )


# ---------------------------------------------------------------------------
# The viewer round trip and its four failures
# ---------------------------------------------------------------------------


async def test_no_viewer_connected_reaches_the_model_with_the_url_to_open(project):
    bus = FakeBus(
        raises=NoViewerConnected("no viewer connected; ask the human to open %s" % VIEWER_URL)
    )
    result = await tools_for(bus, project)["set_view"]({"zoom": 2})
    assert result["is_error"] is True
    # Passed through unedited, because the URL is the actionable part.
    assert text_of(result) == ("no viewer connected; ask the human to open %s" % VIEWER_URL)


async def test_a_viewer_that_closed_is_reported_as_not_having_happened(project):
    bus = FakeBus(raises=ViewerGone("viewer connection closed"))
    result = await tools_for(bus, project)["set_view"]({"zoom": 2})
    assert result["is_error"] is True
    assert "did not happen" in text_of(result)
    assert "set_view" in text_of(result)


async def test_a_timeout_does_not_claim_either_outcome(project):
    """The one failure where the tool genuinely does not know: the frame was
    sent, so the browser may have acted on it. Telling the model it failed
    would be as wrong as telling it it worked."""
    bus = FakeBus(raises=asyncio.TimeoutError())
    result = await tools_for(bus, project)["set_view"]({"zoom": 2})
    assert result["is_error"] is True
    assert "may or may not have happened" in text_of(result)


async def test_a_viewer_refusal_carries_its_code_and_reason(project):
    bus = FakeBus(raises=CallError({"code": "bad_zoom", "message": "the viewer cannot zoom to 99"}))
    result = await tools_for(bus, project)["set_view"]({"zoom": 99})
    assert result["is_error"] is True
    assert "bad_zoom" in text_of(result)
    assert "cannot zoom to 99" in text_of(result)


async def test_an_unexpected_failure_is_reported_as_this_packages_bug(project):
    bus = FakeBus(raises=RuntimeError("something in the toy broke"))
    result = await tools_for(bus, project)["get_view"]({})
    assert result["is_error"] is True
    assert "bug rather than anything you did" in text_of(result)
    assert "get_view" in text_of(result)


# ---------------------------------------------------------------------------
# The pause gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_PAUSE_GATED)
async def test_every_tool_that_changes_anything_refuses_while_paused(project, name):
    bus = FakeBus()
    bus.paused = True
    result = await tools_for(bus, project)[name](ARGS[name])
    assert result["is_error"] is True
    assert result == {
        "content": [{"type": "text", "text": TOY_PAUSED_MESSAGE}],
        "is_error": True,
    }
    # Refused before anything ran, not after: a paused ``add_note`` that had
    # already written the file would be a refusal in name only.
    assert bus.calls == []
    assert json.loads((project / NOTES_FILE).read_text(encoding="utf-8")) == ["first"]


@pytest.mark.parametrize("name", EXPECTED_READ_CLASS)
async def test_no_tool_that_changes_nothing_is_gated_by_pause(project, name):
    """Pausing exists so the human can work without the view moving. A model
    that keeps reading while paused does no harm and is better informed when
    the pause lifts, so none of these may refuse."""
    bus = FakeBus()
    bus.paused = True
    result = await tools_for(bus, project)[name](ARGS[name])
    assert "is_error" not in result, text_of(result)
