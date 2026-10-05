"""Annealage Datum as a remote (``datum.py``): what each class of its tools is
graded as, what is left out, the priming that reveals the rest, and the
upload action's fields.

The remote is a Datum-shaped fake: it lists a few tools until
``getting_started`` has been called on a connection and all of them after,
including the three the agent must not get and one nobody has graded.
"""

import json
import threading
import time
from types import SimpleNamespace

import mcp.types as types
import pytest
import uvicorn
from mcp.server import Server
from toy_product import build_toy_tools

from annealage_agent import datum
from annealage_agent.uploads import UploadAction, check_upload_actions

# The tools Datum lists, written out by hand rather than read from the module
# under test (probed 2026-09-26).
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
WITHHELD = ("publish_reference", "relate_documents", "unrelate_documents")
ADVERTISED = ("getting_started", "search_parts", "semantic_search", "list_boards")


class FakeDatum:
    def __init__(self, extra=()):
        self.calls = []
        primed = set()
        names = (*LOOKUPS, *WRITES, *WITHHELD, *extra)

        def listed(name):
            return types.Tool(
                name=name, description="Datum's %s." % name, inputSchema={"type": "object"}
            )

        async def list_tools(ctx, params):
            here = ctx.request.headers["mcp-session-id"] in primed
            return types.ListToolsResult(tools=[listed(n) for n in (names if here else ADVERTISED)])

        async def call_tool(ctx, params):
            self.calls.append((params.name, params.arguments))
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


def _tools(tmp_path, url):
    return build_toy_tools(
        SimpleNamespace(paused=False), tmp_path, "sess-1", remote=(datum.remote(url),)
    )


def test_the_remote_is_datum_under_the_ds_wiki_key():
    remote = datum.remote("https://example.invalid/mcp")
    assert remote.name == "ds-wiki" == datum.SERVER
    assert remote.url == "https://example.invalid/mcp"
    assert remote.prime == ("getting_started", {})
    assert remote.excluded == WITHHELD
    assert remote.headers is None


def test_every_lookup_is_read_grade_and_submitting_is_the_only_write():
    grading = datum.remote("http://x/mcp").grading
    assert grading.read == LOOKUPS
    assert grading.view == ()
    assert grading.write == WRITES
    # Nothing the agent must not have is graded, so nothing can be both.
    assert not set(WITHHELD) & {*grading.read, *grading.view, *grading.write}
    assert set(grading.pause_gated) == set(WRITES)
    assert set(grading.pre_allowed) == set(LOOKUPS)


def test_the_published_default_is_the_hosted_datum():
    assert datum.DEFAULT_URL == "https://ds.story-kettle.ts.net/mcp"


def test_a_tool_server_proxies_the_graded_tools_and_drops_the_withheld_ones(fake, tmp_path, capsys):
    tools = _tools(tmp_path, fake.url)
    proxied = tools.remote_tables()["ds-wiki"]
    # Priming revealed the tools Datum only lists after getting_started.
    assert set(proxied) == {*LOOKUPS, *WRITES}
    assert [name for name, spec in proxied.items() if spec.write] == list(WRITES)
    assert not set(WITHHELD) & set(proxied)
    pre = {
        name.removeprefix("mcp__ds-wiki__")
        for name in tools.pre_allowed
        if name.startswith("mcp__ds-wiki__")
    }
    assert pre == set(LOOKUPS)
    # The withheld tools are left out silently; a tool nobody graded is warned about.
    err = capsys.readouterr().err
    assert "brand_new_tool" in err
    for name in WITHHELD:
        assert name not in err


def test_the_withheld_tools_are_not_warned_about_when_datum_lists_nothing_new(tmp_path, capsys):
    plain = FakeDatum()
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


def test_the_upload_action_files_a_datasheet_under_the_products_tag():
    action = datum.upload_action("some-tag")
    assert isinstance(action, UploadAction)
    assert action.name == "datum"
    assert action.label == "Submit to Datum"
    assert action.tool == ("ds-wiki", "submit_datasheet")
    assert action.accepts == "application/pdf"
    assert action.build_args(None, None) == {
        "project_tag": "some-tag",
        "doc_type_hint": "datasheet",
    }
    assert action.to_wire() == {
        "name": "datum",
        "label": "Submit to Datum",
        "accepts": "application/pdf",
    }
    assert datum.upload_action("other").build_args(None, None)["project_tag"] == "other"


def test_the_upload_action_names_a_remote_the_datum_tool_server_declares(fake, tmp_path):
    tools = _tools(tmp_path, fake.url)
    assert check_upload_actions((datum.upload_action("t"),), tools)
    submit = tools.remote_tables()["ds-wiki"][datum.upload_action("t").tool[1]]
    assert submit.write
