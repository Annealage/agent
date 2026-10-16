"""Datum no-approval discovery, pause gating, direct-call guard and uploads.
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
from toy_product import TOY_PAUSED_MESSAGE, build_toy_tools

from annealage_ember import datum, uploads
from annealage_ember.uploads import check_upload_actions

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
FORMERLY_DENIED = (
    "publish_reference",
    "relate_documents",
    "unrelate_documents",
    "cancel_evidence_job",
)
ADVERTISED = ("getting_started", "search_parts", "semantic_search", "list_boards")


class FakeDatum:
    def __init__(self, extra=(), delay=0):
        self.delay = delay
        self.calls = []
        primed = set()
        additions = ("new_read_tool", "new_write_tool")
        names = (*LOOKUPS, *WRITES, *VIEWS, *FORMERLY_DENIED, *additions, *extra)

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


def _tools(tmp_path, url, *, bus=None):
    from annealage_ember.remote_actions import RemoteActionStore

    bus = bus if bus is not None else SimpleNamespace(paused=False)
    bus.remote_action_store = RemoteActionStore(
        tmp_path / "remote-actions.json", project_id=str(tmp_path), session="sess-1"
    )
    return build_toy_tools(bus, tmp_path, "sess-1", remote=(datum.remote(url),))


@pytest.mark.asyncio
async def test_every_discovered_capability_runs_without_an_approval_card(fake, tmp_path, capsys):
    tools = _tools(tmp_path, fake.url)
    proxied = tools.remote_tables()["ds-wiki"]
    names = {
        *LOOKUPS, *WRITES, *VIEWS, *FORMERLY_DENIED,
        "new_read_tool", "new_write_tool", "brand_new_tool",
    }
    # The HTTP server reveals these capabilities only after getting_started.
    assert fake.calls == [("getting_started", {})]
    assert set(proxied) == names
    assert all(not spec.write for spec in proxied.values())
    assert {
        name.removeprefix("mcp__ds-wiki__")
        for name in tools.pre_allowed
        if name.startswith("mcp__ds-wiki__")
    } == names
    assert "ds-wiki" not in capsys.readouterr().err

    for name in sorted(names):
        arguments = {"probe": name}
        result = await proxied[name].handler(arguments)
        assert not result.get("is_error", False), (name, result)
        assert result["content"][0]["text"] == "%s %s" % (name, json.dumps(arguments))
        assert fake.calls[-1] == (name, arguments)
    assert fake.calls[1:] == [(name, {"probe": name}) for name in sorted(names)]


@pytest.mark.asyncio
async def test_pause_gates_datum_view_operations_but_not_annotated_reads(fake, tmp_path):
    bus = SimpleNamespace(paused=False)
    proxied = _tools(tmp_path, fake.url, bus=bus).remote_tables()["ds-wiki"]
    bus.paused = True
    # cancel_evidence_job deliberately advertises readOnlyHint=True in the fake:
    # discovery follows annotations, not a historical list of denied names.
    reads = {*LOOKUPS, "new_read_tool", "cancel_evidence_job"}
    for name in sorted(set(proxied) - reads):
        refused = await proxied[name].handler({})
        assert refused["is_error"] is True
        assert refused["content"][0]["text"] == TOY_PAUSED_MESSAGE
    assert fake.calls == [("getting_started", {})]

    for name in sorted(reads):
        result = await proxied[name].handler({})
        assert not result.get("is_error", False)
        assert result["content"][0]["text"] == name + " {}"
        assert fake.calls[-1] == (name, {})

    bus.paused = False
    resumed = await proxied["new_write_tool"].handler({"resume": True})
    assert not resumed.get("is_error", False)
    assert fake.calls[-1] == ("new_write_tool", {"resume": True})


@pytest.mark.asyncio
async def test_direct_calls_honor_read_only_without_mutation_opt_in(fake, tmp_path):
    from annealage_ember import remote
    from annealage_ember.remote_actions import RemoteActionStore

    server = datum.remote(fake.url)
    for name in (*WRITES, *VIEWS, *FORMERLY_DENIED):
        refused = await remote.call(server, name, {}, read_only=True)
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
    )
    assert not allowed.get("is_error", False)
    assert fake.calls[-1] == ("request_datasheet", {"mpn": "X"})

    cancelled = await remote.call(
        server,
        "cancel_evidence_job",
        {"job_id": "job"},
        action_store=store,
    )
    assert not cancelled.get("is_error", False)
    assert cancelled["content"][0]["text"] == 'cancel_evidence_job {"job_id": "job"}'
    assert fake.calls == [
        ("request_datasheet", {"mpn": "X"}),
        ("cancel_evidence_job", {"job_id": "job"}),
    ]


@pytest.mark.asyncio
async def test_remote_call_uses_its_timeout_override():
    from annealage_ember import remote

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
    ("name", "arguments"),
    (
        ("request_datasheet", {"mpn": "TPS54302"}),
        ("get_request_status", {"request_id": 123}),
        ("get_evidence_result", {"result_id": "r"}),
        ("get_evidence_usage", {}),
    ),
)
async def test_request_and_polling_tools_run_without_approval(fake, tmp_path, name, arguments):
    tools = _tools(tmp_path, fake.url)
    spec = tools.remote_tables()["ds-wiki"][name]
    assert not spec.write
    assert "mcp__ds-wiki__" + name in tools.pre_allowed
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


@pytest.mark.asyncio
async def test_datum_upload_action_reaches_its_remote_without_approval(fake, tmp_path):
    tools = _tools(tmp_path, fake.url)
    action = datum.upload_action("t")
    assert check_upload_actions((action,), tools)
    spec = tools.remote_tables()["ds-wiki"]["submit_datasheet"]
    assert not spec.write
    assert "mcp__ds-wiki__submit_datasheet" in tools.pre_allowed
    arguments = uploads.call_args(
        action, None, None, source_url="https://manufacturer.example/part.pdf"
    )
    result = await spec.handler(arguments)
    assert not result.get("is_error", False)
    assert result["content"][0]["text"] == "submit_datasheet " + json.dumps(arguments)
    assert fake.calls[-1] == ("submit_datasheet", arguments)
