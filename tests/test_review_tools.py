"""The shared review tools (``review/tools.py``) over the native store, as the
toy product builds them.

Two things are defended here beyond ordinary argument handling. What the
model gets wrong reaches it as a tool error it can act on (an anchor off the
card, a callout past the limit), with nothing written. And resolving one of
the human's comments reaches the human as a permission card every time, on
every path a backend calls a tool by, exactly once, whatever grade the
product gave the tool: the handler asks, the tool is pre-allowed so no
transport asks first (Claude never consults ``can_use_tool`` for a
pre-allowed tool, and ``/mcp`` and omp only gate write-grade ones), and the
broker never remembers the answer. The omp path is exercised in
``test_omp_session.py``.
"""

import asyncio
import json

import pytest
from conftest import make_test_client
from microdot import Microdot
from toy_product import TOY_PAUSED_MESSAGE, build_toy_tools, toy_review_store

from annealage_agent import launch, sessions
from annealage_agent import settings as settings_module
from annealage_agent.http.routes_mcp import register_mcp_routes
from annealage_agent.review import Capabilities
from annealage_agent.review.tools import (
    AddCallout,
    DeleteCallout,
    ListComments,
    ResolveComment,
    review_tools,
)
from annealage_agent.session.base import PermissionRequest
from annealage_agent.session.permissions import Decision, PermissionBroker
from annealage_agent.tools import Grading, ToolServer, namespaced

pytestmark = pytest.mark.asyncio

AGENT_TOKEN = "review-tools-agent-token"
RESOLVE = namespaced("toy", "resolve_comment")
GRADES = ("read", "view", "write")


class FakeBus:
    """What the review tools read off the app's bus: the pause switch, the
    review store and the session's broker."""

    def __init__(self, store, broker=None):
        self.paused = False
        self.review_store = store
        self.broker = broker

    async def call(self, method, params=None, *, timeout=None):
        return {}


class CountingBroker:
    """Records every ``ask`` and answers each with ``decision``."""

    def __init__(self, allow=True, message=""):
        self.calls = []
        self.decision = Decision(allow=allow, message=message)

    async def ask(self, tool_name, input_data, context):
        self.calls.append((tool_name, input_data))
        return self.decision


@pytest.fixture
def store(tmp_path):
    return toy_review_store(tmp_path)


def _toy_server(store, broker=None):
    """The toy product's own tool server: resolve_comment graded read."""
    return build_toy_tools(FakeBus(store, broker), store.path.parent, "sess-1")


def _graded_server(store, broker=None, grade="read"):
    """The review tools with resolve_comment graded ``grade`` by the product,
    and the rest as the toy grades them."""
    names = {"read": ["list_comments"], "view": ["add_callout"], "write": ["delete_callout"]}
    names[grade].append("resolve_comment")
    grading = Grading(*(tuple(names[g]) for g in GRADES))
    bus = FakeBus(store, broker)
    return ToolServer(
        review_tools(store, bus=bus), grading=grading, bus=bus, paused_message=TOY_PAUSED_MESSAGE
    )


def _handlers(server):
    return {t.name: t.handler for t in server.tools}


def _text(result):
    return result["content"][0]["text"]


def _payload(result):
    return json.loads(_text(result))


def _human_comment(store, text="the wall is too thin"):
    return store.add_comment(
        anchor={"card": "front", "x": 10, "y": 10}, text=text, author="human"
    ).comment


# --- the surface a product gets by default -------------------------------------


async def test_a_store_gets_exactly_the_tools_its_capabilities_support(store):
    assert [t.name for t in review_tools(store, bus=None)] == [
        "list_comments",
        "add_callout",
        "resolve_comment",
        "delete_callout",
    ]
    store.capabilities = Capabilities(max_open_model_callouts=5)
    assert [t.name for t in review_tools(store, bus=None)] == [
        "list_comments",
        "add_callout",
    ]


async def test_add_callout_offers_the_anchor_space_s_fields(store):
    table = _toy_server(store).tool_table()
    schema = table["add_callout"].schema
    assert list(schema["properties"]) == ["card", "x", "y", "text", "ref"]
    assert schema["required"] == ["card", "x", "y", "text"]
    assert table["list_comments"].schema["properties"]["status"]["enum"] == [
        "open",
        "resolved",
        "all",
    ]


async def test_a_resolve_tool_over_a_store_without_status_is_refused_at_startup(store):
    store.capabilities = Capabilities(can_delete_own=True)
    with pytest.raises(ValueError, match="keeps no status"):
        review_tools(store, bus=None, tools=(ResolveComment(),))


async def test_a_verbatim_schema_must_declare_what_its_handler_reads(store):
    """A schema that hid a field the handler reads, or offered one it
    ignores, would silently drop what the model said."""
    schema = {"type": "object", "properties": {"card": {}, "x": {}, "y": {}, "note": {}}}
    with pytest.raises(ValueError, match="declares"):
        review_tools(store, bus=None, tools=(AddCallout(schema=schema),))


# --- what the model gets wrong reaches it as a tool error -----------------------


@pytest.mark.parametrize(
    "args, fragment",
    [
        ({"card": "side", "x": 1, "y": 1, "text": "t"}, "no card 'side'"),
        ({"card": "front", "x": 1, "y": 900, "text": "t"}, "off the card"),
        ({"card": "front", "y": 1, "text": "t"}, "x must be a number"),
        ({"card": "front", "x": 1, "y": 1, "text": "   "}, "text must say what you mean"),
        ({"card": "front", "x": 1, "y": 1, "text": "t", "ref": "box-z"}, "no box 'box-z'"),
        # The anchor is reported before the text: one mistake at a time, in
        # the order the fields are read.
        ({"card": "side", "x": 1, "y": 1, "text": " "}, "no card 'side'"),
    ],
)
async def test_a_bad_callout_is_a_tool_error_and_writes_nothing(store, args, fragment):
    result = await _handlers(_toy_server(store))["add_callout"](args)
    assert result["is_error"] is True
    assert fragment in _text(result)
    assert not store.path.exists()


async def test_add_callout_records_what_is_at_the_point_or_what_the_model_names(store):
    handlers = _handlers(_toy_server(store))
    under = _payload(await handlers["add_callout"]({"card": "front", "x": 60, "y": 5, "text": "a"}))
    named = _payload(
        await handlers["add_callout"](
            {"card": "front", "x": 60, "y": 5, "text": "b", "ref": "box-a"}
        )
    )
    assert under["added"]["ref"] == "box-b"
    assert named["added"]["ref"] == "box-a"
    assert under["added"]["author"] == "model"


async def test_the_open_callout_limit_reaches_the_model(tmp_path):
    store = toy_review_store(tmp_path, max_open_model_callouts=1)
    add = _handlers(_toy_server(store))["add_callout"]
    assert "is_error" not in await add({"card": "back", "x": 1, "y": 1, "text": "one"})
    result = await add({"card": "back", "x": 2, "y": 2, "text": "two"})
    assert result["is_error"] is True
    assert "1 of your callouts are open" in _text(result)
    assert len(store.list_comments().comments) == 1


async def test_list_comments_filters_by_status(store):
    _human_comment(store, "one")
    _human_comment(store, "two")
    store.resolve_comment(1, "done")
    handlers = _handlers(_toy_server(store))
    assert [c["text"] for c in _payload(await handlers["list_comments"]({}))["comments"]] == ["two"]
    resolved = _payload(await handlers["list_comments"]({"status": "resolved"}))["comments"]
    assert [(c["text"], c["resolution"]) for c in resolved] == [("one", "done")]
    assert _payload(await handlers["list_comments"]({"status": "all"}))["count"] == 2
    bad = await handlers["list_comments"]({"status": "closed"})
    assert bad["is_error"] is True


async def test_a_file_the_store_refuses_reaches_the_model_as_what_to_do(store):
    store.path.write_text("{ broken", encoding="utf-8")
    result = await _handlers(_toy_server(store))["list_comments"]({})
    assert result["is_error"] is True
    assert "fix it by hand" in _text(result)


@pytest.mark.parametrize(
    "args, fragment",
    [
        ({"id": 1}, "human's"),
        ({"id": 9}, "no comment #9"),
        ({"id": True}, "id must be a callout id, as reported by list_comments"),
    ],
)
async def test_delete_callout_removes_only_the_model_s_own(store, args, fragment):
    _human_comment(store)
    result = await _handlers(_toy_server(store))["delete_callout"](args)
    assert result["is_error"] is True
    assert fragment in _text(result)
    assert [c.id for c in store.list_comments().comments] == [1]


# --- resolving: the one approval a product cannot grade away ---------------------


@pytest.mark.parametrize("grade", GRADES)
async def test_resolve_is_pre_allowed_and_ungated_whatever_the_product_grades(store, grade):
    """Graded anything, the tool is on the list Claude pre-allows (so its
    ``can_use_tool`` is never consulted), ``tool_table`` does not mark it
    write (so neither ``/mcp`` nor omp gates it) and the pause switch does not
    refuse it: the handler is the one place that asks. The other review
    tools keep the product's grades, and the broker is told never to
    remember the resolve tool under either name it can be asked by."""
    server = _graded_server(store, grade=grade)
    assert RESOLVE in server.pre_allowed
    assert server.tool_table()["resolve_comment"].write is False
    assert "resolve_comment" not in server.grading.pause_gated
    assert server.tool_table()["delete_callout"].write is True
    assert set(server.never_remembered) == {RESOLVE, "resolve_comment"}


@pytest.mark.parametrize("grade", GRADES)
async def test_resolving_the_model_s_own_callout_asks_nobody(store, grade):
    broker = CountingBroker()
    store.add_comment(anchor={"card": "back", "x": 1, "y": 1}, text="mine", author="model")
    result = await _handlers(_graded_server(store, broker, grade))["resolve_comment"]({"id": 1})
    assert _payload(result)["resolved"]["status"] == "resolved"
    # No card was shown, so the result claims no approval.
    assert "approved" not in _payload(result)
    assert broker.calls == []


@pytest.mark.parametrize("grade", GRADES)
async def test_resolving_a_human_s_comment_asks_the_human_once(store, grade):
    """The in-process path Claude calls a pre-allowed tool by: the handler
    alone, which asks exactly once, with the comment on the card."""
    broker = CountingBroker(allow=True)
    _human_comment(store)
    result = await _handlers(_graded_server(store, broker, grade))["resolve_comment"](
        {"id": 1, "note": " made it 2 mm "}
    )
    assert _payload(result)["resolved"]["resolution"] == "made it 2 mm"
    # The result says the human already approved it, so the model does not
    # go on to ask them to approve it again.
    assert "human approved" in _payload(result)["approved"]
    assert len(broker.calls) == 1
    name, card = broker.calls[0]
    assert name == RESOLVE
    # The card shows the human which comment, where, and what the model says
    # it did about it.
    assert card == {
        "id": 1,
        "note": "made it 2 mm",
        "comment": {
            "id": 1,
            "anchor": {"card": "front", "x": 10.0, "y": 10.0},
            "ref": "box-a",
            "text": "the wall is too thin",
            "author": "human",
            "status": "open",
        },
    }


async def test_a_refused_resolve_changes_nothing_and_tells_the_model_why(store):
    broker = CountingBroker(allow=False, message="not yet, the fillet is still wrong")
    _human_comment(store)
    before = store.path.read_bytes()
    result = await _handlers(_toy_server(store, broker))["resolve_comment"]({"id": 1})
    assert result["is_error"] is True
    assert _text(result) == "not yet, the fillet is still wrong"
    assert store.path.read_bytes() == before


async def test_with_no_broker_a_human_s_comment_is_not_resolved(store):
    """Fail closed: a session with nobody to ask is not a session that may
    change the human's review unasked."""
    _human_comment(store)
    result = await _handlers(_toy_server(store, broker=None))["resolve_comment"]({"id": 1})
    assert result["is_error"] is True
    assert "no permission broker" in _text(result)
    assert store.list_comments(status="open").comments[0].id == 1


async def test_an_already_resolved_comment_is_not_asked_about_again(store):
    broker = CountingBroker()
    _human_comment(store)
    store.resolve_comment(1, "done")
    result = await _handlers(_toy_server(store, broker))["resolve_comment"]({"id": 1})
    assert "approved" not in _payload(result)
    assert broker.calls == []


async def test_the_pause_switch_does_not_refuse_a_resolve(store):
    """Not pause-gated, because every resolve that changes the human's
    review asks them anyway."""
    broker = CountingBroker(allow=True)
    bus = FakeBus(store, broker)
    server = build_toy_tools(bus, store.path.parent, "sess-1")
    bus.paused = True
    _human_comment(store)
    result = await _handlers(server)["resolve_comment"]({"id": 1})
    assert "is_error" not in result
    assert len(broker.calls) == 1


async def _mcp_call(server, broker, name, arguments):
    async def current_broker():
        return broker

    app = Microdot()
    register_mcp_routes(app, tools=server, current_broker=current_broker, agent_token=AGENT_TOKEN)
    res = await make_test_client(app).post(
        "/mcp?t=%s" % AGENT_TOKEN,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"method": "tools/call", "params": {"name": name, "arguments": arguments}}),
    )
    assert res.status_code == 200
    return json.loads(res.body.decode("utf-8"))["result"]


@pytest.mark.parametrize("grade", GRADES)
async def test_through_mcp_the_human_is_asked_exactly_once(store, grade):
    broker = CountingBroker(allow=True)
    server = _graded_server(store, broker, grade)
    _human_comment(store)
    result = await _mcp_call(server, broker, "resolve_comment", {"id": 1, "note": "done"})
    assert result.get("isError") is not True
    assert [name for name, _ in broker.calls] == [RESOLVE]
    assert store.list_comments(status="resolved").comments[0].resolution == "done"


async def test_through_mcp_a_refusal_leaves_the_comment_open(store):
    broker = CountingBroker(allow=False, message="no")
    _human_comment(store)
    result = await _mcp_call(_toy_server(store, broker), broker, "resolve_comment", {"id": 1})
    assert result["isError"] is True
    assert store.list_comments(status="open").comments[0].id == 1


async def test_through_mcp_a_bad_anchor_is_a_tool_error(store):
    result = await _mcp_call(
        _toy_server(store), None, "add_callout", {"card": "side", "x": 1, "y": 1, "text": "t"}
    )
    assert result["isError"] is True
    assert "no card 'side'" in result["content"][0]["text"]


async def _next_request(events, seen):
    for _ in range(200):
        fresh = [e for e in events[seen:] if isinstance(e, PermissionRequest)]
        if fresh:
            return fresh[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no permission request was made")


async def test_always_allow_does_not_stand_in_for_the_next_human_comment(store):
    """The whole round trip with the broker every backend shares, built as
    ``launch.py`` builds it: the card says it cannot be remembered, an
    "always allow" sent anyway lets this one resolve and no other, and the
    next human comment is asked about afresh."""
    events = []
    bus = FakeBus(store)
    server = build_toy_tools(bus, store.path.parent, "sess-1")
    bus.broker = broker = PermissionBroker(
        events.append, no_viewer_grace=0, never_remembered=server.never_remembered
    )
    broker.viewer_connected()
    resolve = _handlers(server)["resolve_comment"]
    _human_comment(store, "one")
    _human_comment(store, "two")

    first = asyncio.ensure_future(resolve({"id": 1, "note": "thicker"}))
    request = await _next_request(events, 0)
    assert (request.tool, request.rememberable) == (RESOLVE, False)
    assert request.to_wire()["rememberable"] is False
    assert store.list_comments(status="open").comments[0].id == 1, "nothing lands first"
    await broker.decide(request.request_id, "allow_always")
    assert _payload(await first)["resolved"]["resolution"] == "thicker"

    seen = len(events)
    second = asyncio.ensure_future(resolve({"id": 2}))
    again = await _next_request(events, seen)
    assert again.tool == RESOLVE
    await broker.decide(again.request_id, "deny", "not this one")
    assert _text(await second) == "not this one"
    assert [c.id for c in store.list_comments(status="open").comments] == [2]


async def test_a_grant_already_in_permissions_toml_is_ignored(store, tmp_path):
    """Written before the name was never-rememberable, or by hand: the
    broker ``launch.py`` builds ignores it under either name, so a human
    comment is still asked about."""
    session_id = sessions.create_session(tmp_path)
    (sessions.state_dir(tmp_path) / "permissions.toml").write_text(
        'allow_always_tools = ["%s", "resolve_comment", "mcp__toy__add_note"]\n' % RESOLVE,
        encoding="utf-8",
    )
    events = []
    bus = FakeBus(store)
    bus.url = "http://127.0.0.1:8765/"
    bus.tools = build_toy_tools(bus, tmp_path, session_id)
    launch.build_session(
        "claude",
        events.append,
        bus=bus,
        serve_dir=tmp_path,
        session_id=session_id,
        resumed=False,
        settings=settings_module.resolve(tmp_path),
        mcp_host="127.0.0.1",
        mcp_port=8765,
        agent_token="agent",
    )
    broker = bus.broker
    broker.viewer_connected()
    # An ordinary grant in the same file still stands.
    assert (await broker.ask("mcp__toy__add_note", {}, None)).allow is True
    assert events == []

    _human_comment(store)
    call = asyncio.ensure_future(_handlers(bus.tools)["resolve_comment"]({"id": 1}))
    request = await _next_request(events, 0)
    assert request.tool == RESOLVE
    await broker.decide(request.request_id, "allow")
    await call
    seen = len(events)
    bare = asyncio.ensure_future(broker.ask("resolve_comment", {}, None))
    assert (await _next_request(events, seen)).tool == "resolve_comment"
    broker.shutdown()
    assert (await bare).allow is False


# --- a product's own surface ----------------------------------------------------


async def test_a_product_names_and_presents_its_own_tools(store):
    """A product whose review tools are already a published surface keeps
    it: names, text field, extra fields, schema and output are its own,
    while validation and policy stay the agent layer's."""

    def present(store, written):
        return {"content": [{"type": "text", "text": "pinned #%d" % written.comment.id}]}

    tools = (
        ListComments(name="list_notes_on_cards", author="human", schema={}),
        AddCallout(
            name="pin_note",
            text_field="body",
            ref_field=None,
            extra_fields={"colour": {"type": "string"}},
            present=present,
        ),
        DeleteCallout(name="unpin_note", schema={"id": int}),
    )
    grading = Grading(read=("list_notes_on_cards",), view=(), write=("pin_note", "unpin_note"))
    bus = FakeBus(store)
    server = ToolServer(
        review_tools(store, bus=bus, tools=tools),
        grading=grading,
        bus=bus,
        paused_message=TOY_PAUSED_MESSAGE,
    )
    handlers = _handlers(server)
    result = await handlers["pin_note"](
        {"card": "back", "x": 5, "y": 5, "body": "here", "colour": " red ", "ignored": 1}
    )
    assert _text(result) == "pinned #1"
    (comment,) = store.list_comments().comments
    assert (comment.text, comment.extra) == ("here", {"colour": "red"})
    empty = await handlers["pin_note"]({"card": "back", "x": 5, "y": 5, "body": ""})
    assert "body must say what you mean" in _text(empty)
    bad_id = await handlers["unpin_note"]({"id": "1"})
    assert "as reported by the list tool" in _text(bad_id)
    assert json.loads(_text(await handlers["list_notes_on_cards"]({}))) == {
        "count": 0,
        "comments": [],
    }
