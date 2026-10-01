"""Guides: the text a product's agent reads on demand with a ``read_guide`` tool.

A product ships guidance (workflow notes, house rules, format references)
that is too long for the system prompt and that the agent should read from
the product's own install, never from the served directory, so a project
anywhere gets the same text and a project cannot plant its own. ``GuideSet``
is that set, behind one tool:

``GuideSet(skills_dir, names=(...), extra={name: (description, loader)})``

* A **skill** is ``<skills_dir>/<name>/SKILL.md``, for each of ``names``. Its
  description is the ``description`` key of the file's YAML front matter (folded
  to one line), and the guide's text is the file after the front matter. A
  file without front matter is a guide with no description.
* An **extra** is a guide that is not a skill file: a rules document, or a
  module's docstring. ``description`` is its one-line listing entry and
  ``loader()`` returns its text. A loader reports a file that is not there by
  raising ``FileNotFoundError`` (what ``Path.read_text`` does), which becomes
  the "not installed" refusal below.

Skills list first, in the order of ``names``, then the extras in dict order.
``index()`` lists every guide with its description and ``##`` sections;
``read(name)`` is one guide whole and ``read(name, part)`` one ``##`` section.
``read_guide_tool(guides)`` is the ``@tool`` a product registers; grade it
read (``TOOL_NAME`` is its name).

A section is a ``##`` heading and everything under it up to the next ``##``
(or ``#``) heading. Headings inside code fences do not count, so a sample
that shows a ``## Heading`` cannot be mistaken for one.

Every refusal is a ``ValueError`` whose message is written for the model and
says what does exist: an unknown guide, an unknown or ambiguous section, a
guide whose file is missing from this install. A guide that is missing is
left out of ``index()`` rather than failing it.
"""

import asyncio
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import yaml

#: The name a product's ``read_guide`` tool is registered and graded under.
TOOL_NAME = "read_guide"

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

_DEFAULT_DESCRIPTION = (
    "Read one of this product's guides. With no name, lists the guides and their ## "
    "sections; with a name, the whole guide; with a name and a section (its ## heading), "
    "that section only."
)


def _headings(text: str) -> List[Tuple[int, int, str]]:
    """(line index, level, title) of every heading outside code fences."""
    out = []
    fenced = False
    for i, line in enumerate(text.splitlines()):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        m = None if fenced else _HEADING.match(line)
        if m:
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def sections(text: str) -> List[str]:
    """The ``##`` headings of ``text``, in order."""
    return [title for _i, level, title in _headings(text) if level == 2]


def section(text: str, wanted: str) -> str:
    """The ``##`` section of ``text`` whose heading is ``wanted`` (case aside),
    or else the one heading that starts with or contains it. ``ValueError``
    when none or several match, listing the candidates."""
    heads = _headings(text)
    titles = [(i, title) for i, level, title in heads if level == 2]
    key = wanted.strip().lstrip("#").strip().casefold()
    match = [(i, t) for i, t in titles if t.casefold() == key]
    match = match or [(i, t) for i, t in titles if t.casefold().startswith(key)]
    match = match or [(i, t) for i, t in titles if key in t.casefold()]
    if len(match) != 1:
        listed = "; ".join(t for _i, t in (match or titles)) or "none"
        raise ValueError(
            "%s sections match %r; %s: %s"
            % (
                "several" if match else "no",
                wanted,
                "they are" if match else "the sections are",
                listed,
            )
        )
    start = match[0][0]
    end = next((i for i, level, _t in heads if i > start and level <= 2), None)
    lines = text.splitlines()
    return "\n".join(lines[start:end]).rstrip() + "\n"


def _skill_file(path: Path) -> Tuple[str, str]:
    """``(description, text)`` of the skill file at ``path``."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        front, sep, body = text[4:].partition("\n---\n")
        if sep:
            try:
                meta = yaml.safe_load(front)
            except yaml.YAMLError as exc:
                reason = getattr(exc, "problem", None) or "not valid YAML"
                raise ValueError(
                    "that guide's front matter is malformed (%s: %s); read_guide() lists the "
                    "ones that are readable" % (path, reason)
                ) from None
            description = meta.get("description", "") if isinstance(meta, dict) else ""
            return " ".join(str(description).split()), body.lstrip("\n")
    return "", text


class GuideSet:
    """The guides of one product; see the module docstring."""

    def __init__(
        self,
        skills_dir,
        names: Tuple[str, ...] = (),
        extra: Optional[Mapping[str, Tuple[str, Callable[[], str]]]] = None,
    ):
        self.skills_dir = Path(skills_dir)
        extra = dict(extra or {})
        every = [*names, *extra]
        if len(set(every)) != len(every):
            repeated = sorted({n for n in every if every.count(n) > 1})
            raise ValueError("guide names must be unique, not %s" % ", ".join(repeated))
        self._extra = extra
        self.names = tuple(every)

    def _load(self, name: str) -> Tuple[str, str]:
        """``(description, text)`` of guide ``name``, which must be one of
        ``names``."""
        try:
            if name in self._extra:
                description, loader = self._extra[name]
                return description, loader()
            return _skill_file(self.skills_dir / name / "SKILL.md")
        except FileNotFoundError as exc:
            raise ValueError(
                "that guide is not installed (%s is missing); read_guide() lists the "
                "ones that are" % (exc.filename or name)
            ) from None

    def index(self) -> str:
        """Every installed guide with its description and ``##`` sections."""
        entries = []
        for name in self.names:
            try:
                description, text = self._load(name)
            except ValueError:
                continue  # not installed
            entries.append(
                "%s: %s\n  sections: %s"
                % (name, description, "; ".join(sections(text)) or "(one part)")
            )
        return (
            "Guides (read_guide(name) for one whole, read_guide(name, section) for one "
            "## section):\n\n" + "\n\n".join(entries) + "\n"
        )

    def read(self, name: Optional[str] = None, part: Optional[str] = None) -> str:
        """No name: the index. A name: that guide whole, or with ``part`` the
        one ``##`` section."""
        if not name:
            if part:
                raise ValueError("a section needs the guide's name too")
            return self.index()
        if name not in self.names:
            raise ValueError("no guide %r; the guides are %s" % (name, ", ".join(self.names)))
        _description, text = self._load(name)
        return section(text, part) if part else text


def _optional_string(args: Dict, key: str) -> Optional[str]:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("%s must be a non-empty string" % key)
    return value or None


def read_guide_tool(guides: GuideSet, description: Optional[str] = None):
    """The ``read_guide`` ``@tool`` over ``guides``, for a product's
    ``build_tools`` to register and grade read (``TOOL_NAME``). ``description``
    is what the model is told it is for; give it the product's own words. The
    file read runs off the event loop."""
    from claude_agent_sdk import tool

    from .tools import ok

    @tool(
        TOOL_NAME,
        description or _DEFAULT_DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(guides.names)},
                "section": {"type": "string", "description": "a ## heading of that guide"},
            },
        },
    )
    async def read_guide(args):
        name = _optional_string(args, "name")
        part = _optional_string(args, "section")
        return ok(text=await asyncio.to_thread(guides.read, name, part))

    return read_guide
