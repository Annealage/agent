"""Which agent backends this machine can run, and picking one when no setting
names it.

No backend is the default. A person who has only one of the three installed
should not have to say which; a person with more than one is asked, once, and
can save the answer with ``--save-default``. "Installed" means the CLI a
person signs in to is on ``PATH`` and the Python client this package drives
it through is importable: the CLI on ``PATH`` is the evidence that this person actually
uses that backend, and the client is what agent mode needs to reach it.
"""

import importlib.util
import shutil

from . import product
from .settings import BACKENDS

# Per backend: the CLI on PATH, and the Python package this one talks to it with.
_REQUIREMENTS = {
    "claude": ("claude", "claude_agent_sdk"),
    "codex": ("codex", "openai_codex"),
    "omp": ("omp", "omp_rpc"),
}


class NoBackend(Exception):
    """No backend could be chosen; ``str(exc)`` is written to be shown as-is."""


def detect(*, which=shutil.which, find_spec=importlib.util.find_spec):
    """The backends usable on this machine, in ``BACKENDS`` order."""
    found = []
    for name in BACKENDS:
        binary, module = _REQUIREMENTS[name]
        if _call(which, binary) and _call(find_spec, module):
            found.append(name)
    return tuple(found)


def _call(fn, arg):
    try:
        return fn(arg)
    except Exception:
        return None


# What each backend needs, as the refusal with none installed names it.
_NEEDS = {
    "claude": "claude (Claude Code)",
    "codex": "codex (OpenAI Codex CLI, plus the openai-codex package)",
    "omp": "omp (Oh My Pi, plus the omp-rpc package)",
}


def _listing(items):
    if len(items) <= 2:
        return " or ".join(items)
    return "%s, or %s" % (", ".join(items[:-1]), items[-1])


def choose(available, *, interactive, ask=None, write=None, supported=BACKENDS):
    """``(backend, save)`` for a run whose settings name no backend.

    One available backend is used without asking. More than one is a
    question for the person at the terminal, followed by whether to keep the
    answer; with nobody at the terminal it is an error naming the ways to
    choose, rather than a guess. ``supported`` is the backends the product
    runs, which the refusal with none available names (a product that
    refuses one passes the rest; ``available`` is the caller's to filter).
    """
    ask = ask or input
    write = write or print
    if not available:
        raise NoBackend(
            "no agent backend found. Agent mode needs one of these on PATH, "
            "signed in: %s. Or run the viewer alone: %s"
            % (
                _listing([_NEEDS[name] for name in supported]),
                product.current().viewer_only_command,
            )
        )
    if len(available) == 1:
        return available[0], False
    if not interactive:
        raise NoBackend(
            "more than one agent backend found (%s) and no setting says which "
            "to use. Pass --backend NAME for this run, or --backend NAME "
            "--save-default to keep it for every project" % ", ".join(available)
        )
    write("More than one agent backend is installed. Which one should this session use?")
    for index, name in enumerate(available, 1):
        write("  %d) %s" % (index, name))
    choice = None
    while choice is None:
        try:
            answer = ask("Backend [1-%d]: " % len(available)).strip()
        except EOFError:
            raise NoBackend("no backend chosen") from None
        if answer in available:
            choice = answer
        elif answer.isdigit() and 1 <= int(answer) <= len(available):
            choice = available[int(answer) - 1]
    try:
        save = ask("Use %s by default from now on? [y/N]: " % choice).strip().lower()
    except EOFError:
        save = ""
    return choice, save in ("y", "yes")
