"""Refusing tool calls that reach for the credentials on this machine.

The sandbox this project runs the agent's shell in restricts writes and network
but not reads (plan section 2, fact 19), so without something else a contained
command can read anything its user can: SSH private keys, cloud access keys, the
GPG secret keyring, the token this very agent authenticates with. That matters
more here than in a general-purpose agent, because a product's agent reads
untrusted input (Mesh's: STL comments and filenames) while holding a shell.

This module is that something else. It decides, for one tool call, whether the
call names a path that must not be read or written, and the decision is enforced
by a ``PreToolUse`` hook, which fact 25 verified is upstream of both a settings
file's allow rules and the sandbox's own auto-approval. A settings ``deny`` rule
would be the other candidate and is weaker: it is invisible to this process, and
a settings file in the served directory can shadow it.

**What this does and does not claim.** For a tool that names a file in an
argument, the check is exact: the path is expanded and resolved, so a symlink
planted inside the project that points at ``~/.ssh`` is refused along with the
direct spelling. For ``Bash`` it is textual and therefore partial: a command
that spells the path in a variable, builds it from a glob, or decodes it from
base64 is not caught. That is worth having anyway, because bash is the agent's
default path and the common cases are the direct ones, but it raises a floor
rather than drawing a boundary. Anyone reasoning about a determined agent should
assume the shell can still read these files.

The list is deliberately short: every entry is somewhere whose theft cannot be
undone by changing a password, and nothing on it is a place legitimate work in a
product's project ever reaches. A list broad enough to cover, say, all of
``~/.config`` would also refuse things the agent has real reasons to read, and a
control that fires on ordinary work teaches people to turn it off.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

#: Paths whose contents are refused, relative to the user's home directory.
#: A trailing entry naming a directory covers everything beneath it.
DENIED_HOME_PATHS: Tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".kube",
    ".gnupg",
    ".netrc",
    ".docker/config.json",
    ".config/gh",
    ".claude/.credentials.json",
)

#: Tool arguments that name a single filesystem path. Covers the file-reading
#: and file-writing tools alike: a write into ``~/.ssh/authorized_keys`` is a
#: worse outcome than a read of it, and refusing both costs nothing extra.
PATH_ARGUMENTS: Tuple[str, ...] = (
    "file_path",
    "path",
    "notebook_path",
    "filePath",
)

#: The tool whose argument is a command line rather than a path, checked
#: textually. Named explicitly rather than inferred, so a future tool with a
#: ``command`` argument does not silently inherit a check written for this one.
COMMAND_TOOL = "Bash"

COMMAND_ARGUMENTS: Tuple[str, ...] = ("command",)


def home() -> Path:
    """The user's home directory, read at call time.

    Not cached, so a test that redirects ``HOME`` is honoured, and so a process
    whose environment changes cannot go on enforcing a stale list.
    """
    return Path(os.path.expanduser("~"))


def denied_roots() -> Tuple[Path, ...]:
    """The denied paths, resolved to absolute paths on this machine.

    A root that does not exist is kept rather than dropped: it can be created
    later in the run, and refusing a path that holds nothing costs nothing.
    ``realpath`` is applied so that a home directory reached through a symlink,
    which is how ``/home`` is arranged on macOS and on some managed Linux
    setups, compares equal to the same directory named directly.
    """
    base = home()
    return tuple(Path(os.path.realpath(base / rel)) for rel in DENIED_HOME_PATHS)


def _within(candidate: Path, root: Path) -> bool:
    """Whether ``candidate`` is ``root`` or sits beneath it.

    Compared case-insensitively, because macOS and Windows both resolve
    ``~/.SSH`` and ``~/.ssh`` to one directory and a check that missed the
    former would be a check that a spelling defeats. On a case-sensitive
    filesystem this costs a false positive only for a path deliberately named
    to differ from one of these entries by case alone, which is not a shape
    real work produces.
    """
    parts = [p.casefold() for p in candidate.parts]
    root_parts = [p.casefold() for p in root.parts]
    return parts[: len(root_parts)] == root_parts


def resolve_argument(value: str, cwd) -> Optional[Path]:
    """The absolute, symlink-resolved path ``value`` names, or None.

    ``cwd`` is the served directory, so a relative path is resolved the way the
    tool itself would resolve it. ``realpath`` is what catches a symlink planted
    inside the project that points somewhere on the denied list, and it works on
    a path that does not exist, which matters because a write names one.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.join(str(cwd), expanded)
    return Path(os.path.realpath(expanded))


def command_spellings(root: Path) -> Tuple[str, ...]:
    """The ways a shell command might write ``root`` as text.

    The absolute form plus the two abbreviations a shell expands itself. This is
    the whole of the textual check, and its incompleteness is the documented
    limit of ``Bash`` coverage rather than an oversight to be patched with more
    spellings: the next spelling along is a variable holding the path, which no
    amount of pattern listing reaches.
    """
    absolute = str(root)
    base = str(home())
    if absolute.startswith(base):
        tail = absolute[len(base) :]
        return (absolute, "~" + tail, "$HOME" + tail, "${HOME}" + tail)
    return (absolute,)


def refusal(tool_name: str, tool_input: dict, cwd) -> Optional[str]:
    """The reason this call is refused, or None to express no opinion.

    Written for the model, since a hook's deny reason reaches it verbatim (plan
    section 2a, fact 15), and written to stop it retrying: a refusal it reads as
    transient produces the same call again with a different spelling, which is
    the one outcome worse than the refusal itself.
    """
    if not isinstance(tool_input, dict):
        return None
    roots = denied_roots()

    for name in PATH_ARGUMENTS:
        target = resolve_argument(tool_input.get(name), cwd)
        if target is None:
            continue
        for root in roots:
            if _within(target, root):
                return (
                    "Refused: %s names %s, which is inside %s. That directory holds "
                    "credentials, and this tool refuses to read or write anything "
                    "under it regardless of who asked. Nothing about the project "
                    "you are working on is in there, so do not look for another "
                    "way to reach it; if you genuinely believe you need it, say so "
                    "to the human and let them fetch it themselves."
                    % (tool_name, tool_input.get(name), root)
                )

    if tool_name == COMMAND_TOOL:
        for name in COMMAND_ARGUMENTS:
            command = tool_input.get(name)
            if not isinstance(command, str):
                continue
            folded = command.casefold()
            for root in roots:
                for spelling in command_spellings(root):
                    if spelling.casefold() in folded:
                        return (
                            "Refused: that command names %s, which holds "
                            "credentials. This refusal is on the path, not on the "
                            "command, so rewriting the command to reach the same "
                            "place is not an answer to it; tell the human what you "
                            "were trying to do instead." % root
                        )
    return None
