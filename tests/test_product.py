"""Tests for the Product contract (``product.py``): the agent layer takes its
identity and its registrations from the installed product, and from nowhere
else.

Most tests here run as a second, made-up product (``swap_product`` from
``conftest.py``), because running as the toy product alone cannot tell "read
from the product" apart from "hard-coded to the toy's values". The fixture
restores the toy after each test, which the last test checks on its own.
"""

import dataclasses
from typing import ClassVar, Optional

import pytest
from conftest import DEFAULT_PORT, TEST_AUTHORITY, product_swapper
from toy_product import TOY

from annealage_agent import lock, net, product, protocol, sessions, settings
from annealage_agent.session import events
from annealage_agent.session.base import AgentEvent, TurnEnd


@dataclasses.dataclass(frozen=True)
class SheetChanged(AgentEvent):
    kind: ClassVar[str] = "sheet_changed"
    viewer: Optional[str] = None


def _other_tools(bus, serve_dir, session_id):
    """The other product's one tool, graded read, built the way a real
    product builds its tool server."""
    from claude_agent_sdk import tool

    from annealage_agent.tools import Grading, ToolServer, ok

    @tool("peek_sheet", "Read the sheet the human is looking at.", {})
    async def peek_sheet(args):
        return ok({"sheet": 1})

    return ToolServer(
        [peek_sheet],
        grading=Grading(read=("peek_sheet",), view=(), write=()),
        bus=bus,
        paused_message="paused",
    )


_OTHER = product.Product(
    name="loom",
    title="Loom",
    display_name="Annealage Loom",
    distribution="annealage-loom",
    module="annealage_loom",
    version="9.9",
    state_dirname=".loom",
    config_dirname="annealage-loom",
    mcp_server_name="loom",
    viewer_only_command="annealage-loom view",
    build_tools=_other_tools,
    settings_keys=(
        settings.Key(
            name="sheet_zoom",
            type_name="int",
            default=100,
            layers=(settings.USER,),
            effect="load",
            description="Zoom.",
            py_type=int,
        ),
    ),
    events=(SheetChanged,),
    inbound_frames={"sheet": protocol.FrameSpec({"sheet"}, {"sheet"})},
)


def _other(**overrides):
    return dataclasses.replace(_OTHER, **overrides)


def test_installing_a_second_product_into_one_process_raises():
    assert product.current() is TOY
    product.install(TOY)  # the same one again is a no-op
    with pytest.raises(RuntimeError, match="already runs annealage-toy"):
        product.install(_other())
    assert product.current() is TOY


def test_generic_helpers_take_their_identity_from_the_installed_product(swap_product, tmp_path):
    swap_product(_other())

    assert sessions.state_dir(tmp_path) == tmp_path / ".loom"
    assert settings.project_config_path(tmp_path) == tmp_path / ".loom" / "config.toml"
    assert settings.user_settings_path().parent.name == "annealage-loom"
    assert str(lock.LockHeld(1, 2)).startswith("annealage-loom is already running")
    assert net.format_banner(net.resolve_bind(None), 9, "t").startswith("Annealage Loom is serving")
    heading = events._render_markdown([], "text", None, None, None).splitlines()[0]
    assert heading == "# Loom transcript"


def test_registrations_follow_the_installed_product(swap_product):
    swap_product(_other())

    names = [key.name for key in settings.SETTING_KEYS]
    assert "sheet_zoom" in names and "units" not in names
    assert protocol.validate_inbound({"v": 1, "type": "sheet", "sheet": {}}) == (
        True,
        {"v": 1, "type": "sheet", "sheet": {}},
    )
    ok, reason = protocol.validate_inbound({"v": 1, "type": "view", "view": {}})
    assert not ok and "unknown frame type" in reason
    assert protocol.is_product_frame("sheet") and not protocol.is_product_frame("view")


@pytest.mark.parametrize(
    "overrides",
    [
        {"settings_keys": (settings.KEYS_BY_NAME["port"],)},
        {"inbound_frames": {"turn": protocol.FrameSpec(set(), set())}},
        {"events": (TurnEnd,)},
        {"events": (object,)},
        {"upload_kinds": ("upload",)},
        {"upload_kinds": ("../sheet",)},
        {"upload_kinds": ("sheet\n",)},
        {"upload_kinds": ("sheet", "sheet")},
        {"upload_kinds": "draw"},
    ],
    ids=[
        "generic-setting",
        "generic-frame",
        "generic-event-kind",
        "not-an-event",
        "generic-upload-kind",
        "upload-kind-not-a-slug",
        "upload-kind-trailing-newline",
        "upload-kind-twice",
        "upload-kinds-a-bare-string",
    ],
)
def test_a_registration_that_collides_with_the_agent_layer_installs_nothing(
    swap_product, overrides
):
    """Each refusal leaves nothing half-registered: install validates every
    registration before applying any of them."""
    with pytest.raises(ValueError):
        swap_product(_other(**overrides))
    with pytest.raises(RuntimeError, match="no product is installed"):
        product.current()
    assert "sheet_zoom" not in settings.KEYS_BY_NAME
    assert not protocol.is_product_frame("sheet")


def _agent_mode_app(tmp_path):
    from annealage_agent import app as agent_app
    from annealage_agent.session.fake import FakeSession

    page = tmp_path / "page.html"
    page.write_text("<html></html>", encoding="utf-8")
    return agent_app.create_app(
        tmp_path,
        page_html=page,
        port=DEFAULT_PORT,
        token="browser",
        agent_token="agent",
        session_id=sessions.create_session(tmp_path),
        build_session=lambda on_event, *, bus: FakeSession(on_event),
    )


@pytest.mark.asyncio
async def test_the_tool_server_comes_from_the_installed_products_builder(swap_product, tmp_path):
    """One source of truth: create_app builds the tools from
    ``product.current().build_tools``, named under the product's own server,
    and /mcp serves exactly those."""
    from microdot.test_client import TestClient

    swap_product(_OTHER)
    app = _agent_mode_app(tmp_path)
    assert app.agent_tools.pre_allowed == ("mcp__loom__peek_sheet",)
    client = TestClient(app, host=TEST_AUTHORITY)
    res = await client.post(
        "/mcp?t=agent",
        headers={"Content-Type": "application/json"},
        body=b'{"method": "tools/list"}',
    )
    assert [t["name"] for t in res.json["result"]["tools"]] == ["peek_sheet"]


@pytest.mark.asyncio
async def test_upload_kinds_follow_the_installed_product(swap_product, tmp_path):
    """/upload takes the generic kind and the installed product's own, and not
    a kind only another product declared: running as the other product, the
    toy's ``markup`` is refused and the other product's kind names the
    written file."""
    from microdot.test_client import TestClient

    swap_product(_other(upload_kinds=("sheet",)))
    client = TestClient(_agent_mode_app(tmp_path), host=TEST_AUTHORITY)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    refused = await client.post("/upload?t=browser&kind=markup", body=png)
    assert refused.status_code == 400
    assert refused.json["error"] == "kind must be one of: upload, sheet"

    res = await client.post("/upload?t=browser&kind=sheet", body=png)
    assert res.status_code == 200
    assert res.json["path"].startswith("images/sheet-")


def test_a_product_key_is_laid_out_in_the_section_it_declares(swap_product):
    """The settings window's layout comes from the keys: a product key with a
    section of its own gets that section, after the generic ones listed before
    it, and one naming a generic section's title joins it, ahead of the
    generic keys listed after the product's."""

    def key(name, section):
        return settings.Key(
            name=name,
            type_name="int",
            default=1,
            layers=(settings.USER,),
            effect="load",
            description=name,
            py_type=int,
            section=section,
        )

    swap_product(
        _other(settings_keys=(key("sheet_zoom", "Sheet"), key("grid", settings.VIEWER_SECTION)))
    )
    assert settings.sections() == [
        {"title": "Server", "keys": ["host", "port", "open_browser"]},
        {"title": "Agent", "keys": ["model", "effort", "permission_mode", "backend"]},
        {"title": "Sheet", "keys": ["sheet_zoom"]},
        {"title": "Viewer", "keys": ["grid", "tool_cards_collapsed"]},
    ]


def test_an_agent_mode_app_for_a_product_without_tools_is_refused_at_startup(
    swap_product, tmp_path
):
    swap_product(_other(build_tools=None))
    with pytest.raises(RuntimeError, match="no tool builder"):
        _agent_mode_app(tmp_path)


def test_the_toy_is_installed_again_after_a_swap():
    """The restore itself, independent of test order: swap in the other
    product, then after the block the toy and its registrations are back and
    the other product's are gone."""
    with product_swapper() as swap:
        swap(_other())
        assert product.current().name == "loom"
        assert "units" not in settings.KEYS_BY_NAME
    assert product.current() is TOY
    assert "sheet_zoom" not in settings.KEYS_BY_NAME
    assert not protocol.is_product_frame("sheet")
    assert "units" in settings.KEYS_BY_NAME
    assert protocol.is_product_frame("view")
