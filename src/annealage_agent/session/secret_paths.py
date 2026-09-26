"""Refusing tool calls that reach for the credentials on this machine, or
write a file only the product's own tools may write.

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

**Write-protected files.** The same hook also refuses writes to the files a
product names in ``Product.write_protected`` (``protected_refusal``): files
that only the product's own tools may change, because a change made around
them skips what those tools check and who they ask (Annealage Loom's review
files, whose ``status`` a direct edit could flip on a human's comment with no
permission card). The check has the same two halves and the same honest limit
as the credential list: exact for a file tool naming the file (any tool but the
read-only ones, which may still read it), textual for ``Bash``, which catches a
command that spells the file's name or the protected pattern itself (``cat
x.review.json``, ``sed -i ... designs/src/*.review.json``), whatever the
command does with it (its text cannot say whether it writes), and not one that
builds the name at run time or globs it more broadly (``*.json``).

**Where this is enforced.** Both checks run in the Claude backend's
``PreToolUse`` hook (``session/sdk.py``), for its own file tools and its
``Bash``. The omp backend runs with no tools of its own (``session/omp.py``),
so its only tools are the product's. The Codex backend's shell and patch
tools run under Codex's own workspace sandbox, whose approval requests carry
no path this module could check, so neither list reaches them.
"""

from __future__ import annotations

import fnmatch
import os
import re
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

#: The file tools that only read. A tool naming a write-protected file in one
#: of its ``PATH_ARGUMENTS`` is refused unless it is one of these, so a
#: file-writing tool the agent's CLI adds later is refused rather than let
#: through for want of being listed.
READ_ONLY_TOOLS: Tuple[str, ...] = ("Read", "Grep", "Glob", "LS", "NotebookRead")

# One word of a command line, for the textual check of write-protected names:
# split at whitespace, quotes and the shell's own punctuation, so a name
# written as a redirect target, inside a quoted script or after ``--flag=``
# is still one word ending in it.
_COMMAND_WORD_RE = re.compile(r"[^\s'\"`<>|;&()]+")


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


def check_write_protected(patterns) -> None:
    """Raise ``ValueError`` for a ``Product.write_protected`` value this
    module cannot enforce: anything but a tuple or list of strings, or a
    pattern that is empty, uses backslashes, or has an empty, ``.`` or ``..``
    segment (an absolute pattern has an empty first one). Each of those would
    match nothing under the served directory, or reach outside it, and a
    protection that silently protects nothing is worse than a refusal at
    startup."""
    if isinstance(patterns, str) or not isinstance(patterns, (tuple, list)):
        raise ValueError("write_protected must be a tuple of glob patterns, not %r" % (patterns,))
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(
                "a write_protected pattern must be a non-empty string: %r" % (pattern,)
            )
        if "\\" in pattern or any(seg in ("", ".", "..") for seg in pattern.split("/")):
            raise ValueError(
                "write_protected pattern %r must be relative to the served directory, "
                "with / separators and no empty, . or .. segment" % pattern
            )


def _segments_match(parts, pattern: str) -> bool:
    segments = pattern.casefold().split("/")
    return len(parts) == len(segments) and all(
        fnmatch.fnmatchcase(part, segment) for part, segment in zip(parts, segments, strict=True)
    )


def protected_pattern(target: Path, cwd, patterns) -> Optional[str]:
    """The pattern in ``patterns`` that the resolved path ``target`` falls
    under, or None.

    Patterns are relative to ``cwd``, the served directory, resolved the way
    ``target`` was, and compared segment by segment, so ``*`` never matches
    across a ``/``; case-insensitively, for the reason ``_within`` gives. A
    target outside the served directory matches nothing.
    """
    root = [p.casefold() for p in Path(os.path.realpath(str(cwd))).parts]
    parts = [p.casefold() for p in target.parts]
    if parts[: len(root)] != root:
        return None
    relative = parts[len(root) :]
    for pattern in patterns:
        if _segments_match(relative, pattern):
            return pattern
    return None


def protected_refusal(tool_name: str, tool_input: dict, cwd, patterns) -> Optional[str]:
    """The reason this call is refused for writing a file ``patterns``
    protects (``Product.write_protected``), or None to express no opinion.

    A tool naming such a file in a path argument is refused unless it only
    reads (``READ_ONLY_TOOLS``); the file is resolved like any other path
    argument, so a symlink elsewhere in the project that points at it is
    refused too. ``Bash`` is refused when a word of its command ends in a
    name matching a pattern's last segment, which is textual, like the
    credential check, and for the same reason refuses a command that only
    reads the file: its text cannot say which it does. The message sends the
    model to the tools that do write the file.
    """
    if not patterns or not isinstance(tool_input, dict):
        return None
    if tool_name not in READ_ONLY_TOOLS:
        for name in PATH_ARGUMENTS:
            target = resolve_argument(tool_input.get(name), cwd)
            if target is None:
                continue
            pattern = protected_pattern(target, cwd, patterns)
            if pattern is not None:
                return (
                    "Refused: %s would change %s, and files matching %s are written "
                    "only through the tools this session provides for them, which "
                    "check each change and ask the human where it is theirs to "
                    "decide. Use those tools instead; reading the file with Read is "
                    "fine." % (tool_name, tool_input.get(name), pattern)
                )
    if tool_name == COMMAND_TOOL:
        names = [pattern.rsplit("/", 1)[-1].casefold() for pattern in patterns]
        for argument in COMMAND_ARGUMENTS:
            command = tool_input.get(argument)
            if not isinstance(command, str):
                continue
            for word in _COMMAND_WORD_RE.findall(command):
                base = word.rstrip("/").rsplit("/", 1)[-1].casefold()
                for pattern, name in zip(patterns, names, strict=True):
                    if fnmatch.fnmatchcase(base, name):
                        return (
                            "Refused: that command names %s, and files matching %s "
                            "are written only through the tools this session "
                            "provides for them. A command's text cannot say whether "
                            "it writes, so any command naming one is refused, "
                            "however it is spelled; read the file with the Read "
                            "tool and change it with those tools." % (word, pattern)
                        )
    return None
