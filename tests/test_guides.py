"""Guides (``guides.py``): the sections a guide is cut into, the descriptions
read off a skill's front matter, the extras a product adds, what each refusal
tells the model, and the ``read_guide`` tool over a set.

The skills here are written to a temporary directory, so the expectations are
written out beside the text they are read from.
"""

import pytest

from annealage_agent import guides
from annealage_agent.guides import GuideSet, read_guide_tool
from annealage_agent.tools import Grading, ToolServer

WORKFLOW = """---
name: workflow
description: >-
  Take a project from a brief
  to a finished board.   Use at
  the start.
---
# Workflow

Intro text.

## Requirements

Write requirements.md.

```markdown
## Not a heading
- inside a fence
```

More requirements text.

## Architecture

Blocks and a power tree.

### Detail

A subsection stays with its parent.

## Review checklist

Ticks.
"""

PLAIN = "# Plain\n\nNo front matter here.\n\n## Only part\n\nText.\n"

QUOTED = '---\ndescription: "Quoted: one line"\n---\nBody.\n'


@pytest.fixture
def skills(tmp_path):
    for name, text in (("workflow", WORKFLOW), ("plain", PLAIN), ("quoted", QUOTED)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def full(skills):
    return GuideSet(
        skills,
        names=("workflow", "plain", "quoted"),
        extra={
            "rules": ("The house rules.", lambda: "# Rules\n\n## One\n\nA.\n\n## Two\n\nB.\n"),
            "format": ("The file format.", lambda: "Just a docstring.\n"),
        },
    )


def test_sections_are_the_level_two_headings_outside_fences():
    assert guides.sections(WORKFLOW) == [
        "Requirements",
        "Architecture",
        "Review checklist",
    ]


def test_a_section_runs_to_the_next_heading_of_its_level_or_above():
    requirements = guides.section(WORKFLOW, "Requirements")
    assert requirements.startswith("## Requirements\n")
    assert requirements.endswith("More requirements text.\n")
    # The fenced "## Not a heading" is part of the text, not the end of it.
    assert "## Not a heading\n- inside a fence" in requirements
    assert "## Architecture" not in requirements
    # A deeper heading stays inside its parent.
    architecture = guides.section(WORKFLOW, "architecture")
    assert architecture.endswith("A subsection stays with its parent.\n")
    assert "Review checklist" not in architecture


def test_a_fenced_heading_cannot_be_asked_for():
    with pytest.raises(ValueError, match="no sections match 'Not a heading'; the sections are: "):
        guides.section(WORKFLOW, "Not a heading")


def test_a_section_is_found_by_exact_title_then_prefix_then_substring():
    text = "## Power\n\na\n\n## Power tree\n\nb\n\n## The clock\n\nc\n"
    assert guides.section(text, "power") == "## Power\n\na\n"
    assert guides.section(text, "## power t").startswith("## Power tree")
    assert guides.section(text, "clo").startswith("## The clock")
    with pytest.raises(
        ValueError, match="several sections match 'pow'; they are: Power; Power tree"
    ):
        guides.section(text, "pow")


def test_descriptions_come_from_front_matter_folded_to_one_line(full):
    listing = full.index()
    assert (
        "workflow: Take a project from a brief to a finished board. Use at the start.\n"
        "  sections: Requirements; Architecture; Review checklist"
    ) in listing
    assert "quoted: Quoted: one line\n  sections: (one part)" in listing


def test_a_skill_without_front_matter_is_whole_with_no_description(full):
    assert "\nplain: \n  sections: Only part\n" in full.index()
    assert full.read("plain") == PLAIN


def test_a_skill_is_read_without_its_front_matter(full):
    text = full.read("workflow")
    assert text.startswith("# Workflow\n")
    assert "description:" not in text


def test_the_index_lists_skills_in_name_order_then_extras(full):
    listing = full.index()
    assert listing.startswith(
        "Guides (read_guide(name) for one whole, read_guide(name, section) for one ## section):\n\n"
    )
    assert listing.endswith("\n")
    order = [
        listing.index("\n%s: " % name)
        for name in ("workflow", "plain", "quoted", "rules", "format")
    ]
    assert order == sorted(order)
    assert "rules: The house rules.\n  sections: One; Two" in listing
    assert "format: The file format.\n  sections: (one part)" in listing


def test_extras_are_read_whole_or_by_section(full):
    assert full.read("format") == "Just a docstring.\n"
    assert full.read("rules", "two") == "## Two\n\nB.\n"
    assert full.names == ("workflow", "plain", "quoted", "rules", "format")


def test_no_name_is_the_index_and_a_section_needs_a_name(full):
    assert full.read() == full.index()
    with pytest.raises(ValueError, match="a section needs the guide's name too"):
        full.read(None, "One")


def test_an_unknown_guide_lists_the_ones_there_are(full):
    with pytest.raises(ValueError) as exc:
        full.read("nope")
    assert (
        str(exc.value) == "no guide 'nope'; the guides are workflow, plain, quoted, rules, format"
    )


def test_an_unknown_section_lists_the_sections_there_are(full):
    with pytest.raises(ValueError) as exc:
        full.read("workflow", "nothing")
    assert str(exc.value) == (
        "no sections match 'nothing'; the sections are: Requirements; Architecture; Review checklist"
    )


def test_a_guide_missing_from_the_install_is_refused_and_left_out_of_the_index(skills):
    def gone():
        return (skills / "missing.md").read_text(encoding="utf-8")

    gs = GuideSet(
        skills,
        names=("workflow", "absent"),
        extra={"rules": ("Rules.", gone), "format": ("Format.", lambda: "x\n")},
    )
    for name in ("absent", "rules"):
        with pytest.raises(ValueError, match="not installed") as exc:
            gs.read(name)
        assert "read_guide() lists the ones that are" in str(exc.value)
    listing = gs.index()
    assert "workflow:" in listing and "format:" in listing
    assert "absent" not in listing and "rules:" not in listing
    # Still a name the tool offers: the set doesn't know the install is short.
    assert gs.names == ("workflow", "absent", "rules", "format")


def test_malformed_front_matter_is_a_refusal_and_leaves_the_guide_out_of_the_index(skills):
    (skills / "broken").mkdir()
    (skills / "broken" / "SKILL.md").write_text(
        "---\ndescription: [unclosed\nname: x: y\n---\nBody.\n", encoding="utf-8"
    )
    gs = GuideSet(skills, names=("workflow", "broken", "plain"))
    with pytest.raises(ValueError, match="front matter is malformed") as exc:
        gs.read("broken")
    assert str(skills / "broken" / "SKILL.md") in str(exc.value)
    listing = gs.index()
    assert "broken" not in listing
    assert "workflow:" in listing and "plain:" in listing


def test_a_name_given_twice_is_refused(skills):
    with pytest.raises(ValueError, match="unique, not workflow"):
        GuideSet(skills, names=("workflow",), extra={"workflow": ("Again.", lambda: "")})
    with pytest.raises(ValueError, match="unique, not plain"):
        GuideSet(skills, names=("plain", "plain"))


# --- the tool ------------------------------------------------------------------------------


def _text(result):
    return "".join(item["text"] for item in result["content"])


def test_the_tool_offers_the_guides_by_name_with_the_products_own_words(full):
    tool = read_guide_tool(full, "Read the demo's guides.")
    assert tool.name == guides.TOOL_NAME == "read_guide"
    assert tool.description == "Read the demo's guides."
    assert tool.input_schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["workflow", "plain", "quoted", "rules", "format"]},
            "section": {"type": "string", "description": "a ## heading of that guide"},
        },
    }


@pytest.mark.asyncio
async def test_the_tool_reads_a_guide_a_section_and_the_index(full):
    handler = read_guide_tool(full).handler
    assert _text(await handler({})) == full.index()
    assert _text(await handler({"name": "", "section": ""})) == full.index()
    assert _text(await handler({"name": "format"})) == "Just a docstring.\n"
    assert _text(await handler({"name": "rules", "section": "one"})) == "## One\n\nA.\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args, message",
    [
        ({"section": "One"}, "a section needs the guide's name too"),
        ({"name": 5}, "name must be a non-empty string"),
        ({"name": "rules", "section": ["One"]}, "section must be a non-empty string"),
        ({"name": "nope"}, "no guide 'nope'"),
    ],
)
async def test_the_tool_rejects_with_a_message_for_the_model(full, args, message):
    with pytest.raises(ValueError, match=message):
        await read_guide_tool(full).handler(args)


@pytest.mark.asyncio
async def test_the_tool_runs_in_a_tool_server_as_a_read_tool_that_fails_as_text(full):
    """Graded read, it runs without a card and while paused, and a refusal
    reaches the model as a failed call carrying the message."""
    from types import SimpleNamespace

    bus = SimpleNamespace(paused=True)
    server = ToolServer(
        [read_guide_tool(full)],
        grading=Grading(read=("read_guide",), view=(), write=()),
        bus=bus,
        paused_message="paused",
    )
    assert server.pre_allowed == ("mcp__toy__read_guide",)
    handler = server.tool_table()["read_guide"].handler
    assert _text(await handler({"name": "format"})) == "Just a docstring.\n"
    refused = await handler({"name": "nope"})
    assert refused["is_error"] and "no guide 'nope'" in _text(refused)
