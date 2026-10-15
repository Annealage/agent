"""Datum proxy policy, annotation discovery, direct-call guard and uploads.
The fake lists its expanded tool set only after ``getting_started`` runs.
"""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import mcp.types as types
import pytest
import uvicorn
from mcp.server import Server
from toy_product import build_toy_tools

from annealage_agent import datum, uploads
from annealage_agent.uploads import check_upload_actions

# The tools Datum lists, written out by hand rather than read from the module
# under test (including the request tools discovered 2026-10-09).
LOOKUPS = (
    "getting_started",
    "search_parts",
    "get_part_by_mpn",
    "list_parts_by_project_tag",
    "get_datasheet_content",
    "semantic_search",
    "get_datasheet_pages",
    "get_datasheet_outline",
    "get_document_relations",
    "get_job_status",
    "get_request_status",
    "get_evidence_usage",
    "get_evidence_result",
    "list_boards",
    "trace_board_rail_or_net",
    "get_board_bom",
    "find_boards_using_part",
    "diff_board_versions",
    "find_board_parts_by_function",
    "locate_board_component",
    "list_vault_categories",
    "search_vault_components",
    "get_vault_component",
    "semantic_search_vault_components",
    "check_board_bom_against_vault",
)
WRITES = ("submit_datasheet", "request_datasheet")
VIEWS = ("submit_evidence_job",)
WITHHELD = (
    "publish_reference",
    "relate_documents",
    "unrelate_documents",
    "cancel_evidence_job",
)
ADVERTISED = ("getting_started", "search_parts", "semantic_search", "list_boards")


class FakeDatum:
    def __init__(self, extra=(), delay=0, new_tools=True):
        self.delay = delay
        self.calls = []
        primed = set()
        additions = ("new_read_tool", "new_write_tool") if new_tools else ()
        names = (*LOOKUPS, *WRITES, *VIEWS, *WITHHELD, *additions, *extra)

        def listed(name):
            read_only = name in LOOKUPS or name in ("new_read_tool", "cancel_evidence_job")
            return types.Tool(
                name=name,
                description="Datum's %s." % name,
                inputSchema={"type": "object"},
                annotations=types.ToolAnnotations(readOnlyHint=read_only),
            )

        async def list_tools(ctx, params):
            here = ctx.request.headers["mcp-session-id"] in primed
            return types.ListToolsResult(tools=[listed(n) for n in (names if here else ADVERTISED)])

        async def call_tool(ctx, params):
            self.calls.append((params.name, params.arguments))
            if self.delay:
                await asyncio.sleep(self.delay)
            if params.name == "getting_started":
                primed.add(ctx.request.headers["mcp-session-id"])
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text="%s %s" % (params.name, json.dumps(params.arguments)),
                    )
                ]
            )

        server = Server(
            "datasheet-wiki",
            instructions="Call getting_started first.",
            on_list_tools=list_tools,
            on_call_tool=call_tool,
        )

        app = server.streamable_http_app()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert self._thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = "http://127.0.0.1:%d/mcp" % port

    def stop(self):
        self._server.should_exit = True
        self._thread.join(10)


@pytest.fixture
def fake():
    remote = FakeDatum(extra=("brand_new_tool",))
    yield remote
    remote.stop()


def _tools(tmp_path, url, *, view=()):
    from annealage_agent.remote_actions import RemoteActionStore

    bus = SimpleNamespace(paused=False)
    bus.remote_action_store = RemoteActionStore(
        tmp_path / "remote-actions.json", project_id=str(tmp_path), session="sess-1"
    )
    return build_toy_tools(bus, tmp_path, "sess-1", remote=(datum.remote(url, view=view),))


def test_the_remote_is_datum_under_the_ds_wiki_key():
    remote = datum.remote("https://example.invalid/mcp")
    assert remote.name == "ds-wiki" == datum.SERVER
    assert remote.url == "https://example.invalid/mcp"
    assert remote.prime == ("getting_started", {})
    assert remote.excluded == datum.DENY
    assert remote.denied == datum.DENY
    assert remote.annotation_grading
    assert datum.EVIDENCE_SUBMIT_TOOL in remote.product_only
    assert remote.headers is None


def test_annotation_grading_keeps_only_explicit_write_names():
    grading = datum.remote("http://x/mcp").grading
    assert grading.read == ()
    assert grading.view == ()
    assert grading.write == WRITES


def test_the_published_default_is_the_hosted_datum():
    assert datum.DEFAULT_URL == "https://ds.story-kettle.ts.net/mcp"


def test_a_tool_server_proxies_annotated_reads_and_explicit_writes(fake, tmp_path, capsys):
    tools = _tools(tmp_path, fake.url)
    proxied = tools.remote_tables()["ds-wiki"]
    # Priming reveals the expanded tool listing.
    assert set(proxied) == {*LOOKUPS, "new_read_tool", *WRITES}
    assert [name for name, spec in proxied.items() if spec.write] == list(WRITES)
    assert not set(WITHHELD) & set(proxied)
    assert "submit_evidence_job" not in proxied
    pre = {
        name.removeprefix("mcp__ds-wiki__")
        for name in tools.pre_allowed
        if name.startswith("mcp__ds-wiki__")
    }
    assert pre == {*LOOKUPS, "new_read_tool"}
    err = capsys.readouterr().err
    assert "new_write_tool" in err and "brand_new_tool" in err
    for name in WITHHELD:
        assert name not in err


def test_the_product_only_job_is_view_grade_only_when_opted_in(fake, tmp_path):
    tools = _tools(tmp_path, fake.url, view=(datum.EVIDENCE_SUBMIT_TOOL,))
    spec = tools.remote_tables()["ds-wiki"][datum.EVIDENCE_SUBMIT_TOOL]
    assert not spec.write
    assert "mcp__ds-wiki__submit_evidence_job" in tools.pre_allowed


def test_the_withheld_tools_are_not_warned_about_when_datum_lists_nothing_new(tmp_path, capsys):
    plain = FakeDatum(new_tools=False)
    try:
        _tools(tmp_path, plain.url)
    finally:
        plain.stop()
    assert "ds-wiki" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_lookup_reaches_datum_and_primes_it(fake, tmp_path):
    proxied = _tools(tmp_path, fake.url).remote_tables()["ds-wiki"]
    result = await proxied["search_parts"].handler({"query": "RP2040"})
    assert result["content"][0]["text"] == 'search_parts {"query": "RP2040"}'
    assert fake.calls[-1] == ("search_parts", {"query": "RP2040"})
    new_read = await proxied["new_read_tool"].handler({})
    assert new_read["content"][0]["text"] == "new_read_tool {}"
    assert fake.calls[-1] == ("new_read_tool", {})


@pytest.mark.asyncio
async def test_direct_calls_are_read_guarded_and_named_writes_are_allowed(fake, tmp_path):
    from annealage_agent import remote
    from annealage_agent.remote_actions import RemoteActionStore

    server = datum.remote(fake.url)
    refused = await remote.call(server, "request_datasheet", {}, read_only=True)
    assert refused["is_error"]
    assert "not read-only" in refused["content"][0]["text"]
    assert fake.calls == []

    store = RemoteActionStore(
        tmp_path / "actions.json", project_id=str(tmp_path), session="direct-call"
    )
    allowed = await remote.call(
        server,
        "request_datasheet",
        {"mpn": "X"},
        action_store=store,
        allowed_writes=("request_datasheet",),
    )
    assert not allowed.get("is_error", False)
    assert fake.calls[-1] == ("request_datasheet", {"mpn": "X"})

    denied = await remote.call(
        server,
        "cancel_evidence_job",
        {},
        allowed_writes=("cancel_evidence_job",),
    )
    assert denied["is_error"]
    assert "denied" in denied["content"][0]["text"]


@pytest.mark.asyncio
async def test_remote_call_uses_its_timeout_override():
    from annealage_agent import remote

    slow = FakeDatum(delay=0.2)
    try:
        result = await remote.call(
            datum.remote(slow.url),
            "get_evidence_result",
            {"job_id": "job"},
            read_only=True,
            timeout=0.05,
        )
        assert result["is_error"]
        assert "no answer came back" in result["content"][0]["text"]
        assert slow.calls == [("get_evidence_result", {"job_id": "job"})]
    finally:
        slow.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "write"),
    (
        ("request_datasheet", {"mpn": "TPS54302"}, True),
        ("get_request_status", {"request_id": 123}, False),
        ("get_evidence_result", {"result_id": "r"}, False),
        ("get_evidence_usage", {}, False),
    ),
)
async def test_request_and_polling_tools_are_discovered_graded_and_forwarded(
    fake, tmp_path, name, arguments, write
):
    tools = _tools(tmp_path, fake.url)
    spec = tools.remote_tables()["ds-wiki"][name]
    assert spec.write is write
    assert ("mcp__ds-wiki__" + name in tools.pre_allowed) is (not write)
    result = await spec.handler(arguments)
    assert result["content"][0]["text"] == "%s %s" % (name, json.dumps(arguments))
    assert fake.calls[-1] == (name, arguments)



def test_datum_upload_action_requires_an_exact_source_url():
    action = datum.upload_action("some-tag")
    assert isinstance(action, uploads.UploadAction)
    assert action.tool == ("ds-wiki", "submit_datasheet")
    assert action.build_args(None, None) == {
        "project_tag": "some-tag",
        "doc_type_hint": "datasheet",
    }
    assert action.to_wire()["source_url_label"]

    source = "https://manufacturer.example/part.pdf?revision=2"
    args = uploads.call_args(action, None, None, source_url=source)
    assert args["original_url"] == source
    assert args["project_tag"] == "some-tag"
    prefix = "https://manufacturer.example/"
    boundary = prefix + "x" * (2048 - len(prefix))
    assert uploads.call_args(action, None, None, source_url=boundary)["original_url"] == boundary
    for invalid in (None, "file:///tmp/part.pdf", "https://", "https://maker.example/" + "x" * 2040):
        with pytest.raises(ValueError, match="original absolute HTTP"):
            uploads.call_args(action, None, None, source_url=invalid)


def test_datum_upload_action_declares_its_remote(fake, tmp_path):
    tools = _tools(tmp_path, fake.url)
    action = datum.upload_action("t")
    assert check_upload_actions((action,), tools)
    assert tools.remote_tables()["ds-wiki"]["submit_datasheet"].write
