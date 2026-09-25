"""Whether the served directory's executable configuration may take effect.

A directory can carry configuration that the `claude` CLI obeys: `.claude/settings.json`, `.claude/settings.local.json`, scripts under `.claude/hooks/`, and `.mcp.json`. Settings files may declare hooks, hooks are shell commands, and a `SessionStart` hook runs when the session opens, before any prompt is sent and before any permission callback exists to consult. A `.mcp.json` names server commands to spawn. So configuration in this directory is executable code, and the directory is not necessarily the human's own work: pointing a product at an unpacked download is an ordinary thing to do.

The CLI's own defence against this is its workspace trust dialog, and it states in `--print`'s help that the dialog "is skipped when Claude is run in non-interactive mode", which is the mode every SDK session runs in, adding: "Only use this in directories you trust." That makes the trust decision the embedding application's to make, and this module is how the Annealage products make it.

Two controls are built from what is here, and they answer different threats.

**The startup gate** (the product's CLI; Mesh's ``cli.py``) refuses agent mode when this directory carries configuration whose current content the human has not accepted. It has to run before the session opens, because a `SessionStart` hook executing is precisely what it prevents, and nothing inside a running session is early enough.

**The in-session tripwire** (``session/sdk.py``) recomputes ``config_digest`` on every tool call and denies the call if it no longer matches what the gate accepted. It exists because the CLI re-reads these files while the session runs: a settings file written mid-session takes effect immediately, so a directory that was trustworthy at startup does not stay trustworthy by itself. The tripwire watches the files rather than the ways they might be written, so it is indifferent to whether a write arrived through Write, Edit, a shell redirect, or a `git checkout`.

Trust records live in the user's own configuration directory, never in the served directory, because a record stored alongside the thing it vouches for could be shipped inside the download it is meant to vouch for.

**Git's configuration is guarded too, and conditionally.** A repository's `.git/config` can name commands git runs (`core.fsmonitor`, `core.pager`, an alias beginning with `!`, a filter or textconv driver), and `.git/hooks/` holds scripts it runs directly. Both execute the moment the agent runs an ordinary git command, and git runs outside the sandbox by design, because it has to see the real filesystem to work on the project's own repository. Measured, so the scope is exact: `git rev-parse` and `git config user.email`, which are the commands a product itself runs against a directory it has not yet trusted (Mesh's does), execute none of this, while `git status` and `git add`, which are the agent's, execute `core.fsmonitor`. So the exposure is the agent's git rather than the product's own scaffolding.

Unlike the four Claude paths, these two are guarded only when they carry something executable. Nearly every real project is a git repository, so gating on `.git/config` existing would ask about all of them and teach the human to accept without reading; gating on it naming a command asks about the ones that can act. An ordinary config from `git init` or `git clone` names none. The same conditional keeps the in-session tripwire quiet while the agent does ordinary git work: adding a remote does not change the digest, while adding an alias does.

What is deliberately not gated is `CLAUDE.md`. It is instructions rather than executable configuration, so a hostile one is a prompt-injection vector and not a code-execution one, and every tool call it might provoke still reaches the sandbox and the human. Gating it would prompt about nearly every real project and teach the human to accept without reading, which costs more than it buys.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .. import product

# Every path, relative to the served directory, whose content the CLI may
# execute or obey. Files and directories both: `.claude/hooks` holds scripts a
# trusted settings file may invoke by name, so a change to one of those scripts
# changes what the directory does without changing any settings file.
GUARDED_PATHS = (
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/hooks",
    ".mcp.json",
)

# Git's own configuration is executable too, and it is reached by a different
# route: the agent's git commands. A repository's `.git/config` can name
# commands git runs, and `.git/hooks/` holds scripts it runs directly, so a
# folder that is an unpacked repository can execute code the moment the agent
# runs `git status`. That path is not sandboxed either, because
# `SANDBOX_SETTINGS` excludes git deliberately so it can work on the project's
# own repository.
#
# These two paths are guarded *conditionally*, unlike the four above, and the
# condition is what makes the control usable: nearly every real project is a git
# repository, so gating on `.git/config` existing would ask about all of them and
# teach the human to accept without reading. Gating on it naming something
# executable asks about the ones that can actually do something. An ordinary
# config, written by `git init` or `git clone`, holds none of these keys.
GIT_CONFIG_REL = ".git/config"
GIT_HOOKS_REL = ".git/hooks"

# Config keys whose value is a command, or a path to something holding commands.
# Matched case-insensitively against the fully-qualified key name.
_GIT_EXECUTABLE_KEYS = frozenset(
    {
        "core.fsmonitor",
        "core.pager",
        "core.editor",
        "core.askpass",
        "core.sshcommand",
        "core.hookspath",
        "init.templatedir",
        "sequence.editor",
        "diff.external",
        "uploadpack.packobjectshook",
        "web.browser",
        "include.path",
    }
)

# Whole families of keys, every member of which names a command. `alias` is here
# because an alias beginning with `!` is a shell command, and telling those apart
# from the safe kind is not worth the risk of getting it wrong.
_GIT_EXECUTABLE_PREFIXES = (
    "alias.",
    "pager.",
    "filter.",
    "difftool.",
    "mergetool.",
    "browser.",
    "trailer.",
    "man.",
    "gpg.",
    "credential.",
    "includeif.",
)

# Key endings that name a command whatever section they appear in.
_GIT_EXECUTABLE_SUFFIXES = (
    ".textconv",
    ".command",
    ".driver",
    ".cmd",
    ".clean",
    ".smudge",
    ".process",
)

# How far a chain of git config includes is followed. Git's own limit is ten,
# and matching it means this sees exactly what git would: a chain longer than
# git follows is one git refuses too.
_MAX_INCLUDE_DEPTH = 10

# Hook files git ships as inert examples. They are not executable as shipped and
# git ignores them by name, so counting them would make every freshly initialised
# repository look like it carried hooks.
_GIT_HOOK_SAMPLE_SUFFIX = ".sample"

# Beyond this many bytes in one guarded file, the digest covers the file's size
# rather than its content. Reached only by something that is not configuration,
# and the size still changes when the file does, so the gate and the tripwire
# both still notice a rewrite.
_MAX_DIGEST_BYTES = 1 << 20

# The digest of a directory carrying no guarded configuration at all. Named so
# callers can say "nothing to trust" without recomputing it, and distinct from
# any real digest because no file list produces an empty hash input.
EMPTY_DIGEST = "empty"

_TRUST_FILE = "trusted-projects"


def git_config_executes(path: Path) -> bool:
    """Whether the git config at ``path``, or anything it includes, names
    something git would run.

    Parsed for key names alone; the values are never inspected, except for the
    include directives themselves, because the question is whether a command is
    configured at all rather than what it says. An unreadable file counts as
    executing: a config this cannot inspect is one whose contents are unknown,
    and unknown has to mean gated.

    Includes are followed because git follows them: a config whose only content
    is ``[include] path = ../elsewhere`` configures whatever that file
    configures.
    """
    for member in config_chain(path):
        try:
            text = member.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True
        if any(_key_executes(key) for key in _git_config_keys(text)):
            return True
    return False


def _include_targets(text: str, base_dir: Path) -> Tuple[Path, ...]:
    """Every path an ``include`` or ``includeIf`` directive in ``text`` names.

    ``includeIf`` conditions are not evaluated. Deciding whether one applies
    means reproducing git's own predicate logic (``gitdir``, ``onbranch``,
    ``hasconfig``) against the state of the moment, and being wrong in the
    permissive direction would mean not watching a file git does read. Watching
    a file git ignores costs nothing but a digest that changes more often.

    Relative paths resolve against the including file's directory, and ``~``
    against the user's home, which is what git does.
    """
    targets = []
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("["):
            header = line[1 : line.find("]") if "]" in line else len(line)]
            section = header.split(None, 1)[0].strip().strip('"').casefold()
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if section not in ("include", "includeif") or key.strip().casefold() != "path":
            continue
        raw_path = value.strip().strip('"')
        if not raw_path:
            continue
        expanded = Path(os.path.expanduser(raw_path))
        targets.append(expanded if expanded.is_absolute() else base_dir / expanded)
    return tuple(targets)


def config_chain(path: Path) -> Tuple[Path, ...]:
    """``path`` followed by every config it pulls in, breadth first.

    Bounded by ``_MAX_INCLUDE_DEPTH`` and by a seen set, so a config that
    includes itself, or two that include each other, terminate rather than
    looping. Paths are resolved before being compared, so two spellings of one
    file count once.
    """
    start = Path(path)
    chain = [start]
    seen = {os.path.realpath(start)}
    frontier = [(start, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= _MAX_INCLUDE_DEPTH:
            continue
        try:
            text = current.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for target in _include_targets(text, current.parent):
            resolved = os.path.realpath(target)
            if resolved in seen:
                continue
            seen.add(resolved)
            chain.append(target)
            frontier.append((target, depth + 1))
    return tuple(chain)


def _key_executes(key: str) -> bool:
    folded = key.casefold()
    if folded in _GIT_EXECUTABLE_KEYS:
        return True
    if folded.startswith(_GIT_EXECUTABLE_PREFIXES):
        return True
    return folded.endswith(_GIT_EXECUTABLE_SUFFIXES)


def _git_config_keys(text: str) -> Iterable[str]:
    """Yield ``section.key`` and ``section.subsection.key`` names from a git config.

    A deliberately small reader rather than a full git-config parser: it tracks
    the current section header and takes the name to the left of each ``=``. It
    errs toward yielding more than git would, which is the safe direction, since
    a name yielded in error costs one unnecessary question while a name missed
    costs the gate.
    """
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("["):
            header = line[1 : line.find("]") if "]" in line else len(line)]
            parts = header.split(None, 1)
            name = parts[0].strip().strip('"')
            if len(parts) > 1:
                section = "%s.%s" % (name, parts[1].strip().strip('"'))
            else:
                section = name
            continue
        if "=" not in line:
            # A valueless key is still a set key as far as git is concerned.
            yield "%s.%s" % (section, line) if section else line
            continue
        key = line.split("=", 1)[0].strip()
        yield "%s.%s" % (section, key) if section else key


def executable_git_hooks(root) -> Tuple[Path, ...]:
    """Hook scripts under ``.git/hooks`` that git would actually run.

    Executable, and not one of the ``.sample`` files git ships. A clone carries
    no hooks, so this finds them in the case that matters: a repository unpacked
    from an archive, which carries whatever the archive's author put there.
    """
    hooks = Path(root) / GIT_HOOKS_REL
    if not hooks.is_dir():
        return ()
    found = []
    try:
        children = sorted(hooks.iterdir())
    except OSError:
        return ()
    for child in children:
        if child.name.endswith(_GIT_HOOK_SAMPLE_SUFFIX):
            continue
        try:
            if child.is_file() and os.access(child, os.X_OK):
                found.append(child)
        except OSError:
            continue
    return tuple(found)


def guarded_paths(root) -> Tuple[str, ...]:
    """Every path guarded for ``root``, relative, in a stable order.

    The four Claude paths always, plus git's two when git's configuration can
    execute. Computed per directory rather than fixed, because whether the git
    entries are guarded depends on what they contain.
    """
    root = Path(root)
    paths = list(GUARDED_PATHS)
    config = root / GIT_CONFIG_REL
    if config.exists() and git_config_executes(config):
        paths.append(GIT_CONFIG_REL)
    if executable_git_hooks(root):
        paths.append(GIT_HOOKS_REL)
    return tuple(paths)


def present(root) -> Tuple[Path, ...]:
    """The guarded paths that exist in ``root``, in ``guarded_paths`` order.

    Uses ``lstat`` semantics via ``Path.exists`` on the entry itself, so a
    symlink that dangles counts as absent while one that resolves counts as
    present, matching what the CLI would find when it tried to read it.
    """
    root = Path(root)
    return tuple(root / rel for rel in guarded_paths(root) if (root / rel).exists())


def config_digest(root) -> str:
    """A digest of every guarded path's identity and content under ``root``.

    ``EMPTY_DIGEST`` when the directory carries none of them, which is the
    common case for a plain project folder and the case that needs no
    trust decision at all.

    Covers three things beyond file content, each because changing it changes
    what the directory does while leaving content equal: which paths exist, so
    deleting one is a change; whether each is a symlink and to where, so
    re-pointing a link at different configuration is a change; and, for
    ``.claude/hooks``, every file beneath it by relative path, so adding a
    script is a change.

    Unreadable is recorded as its own state rather than skipped: a file the
    CLI might read and this cannot has to compare unequal to one that was
    read, or the gate would accept a directory it never actually inspected.
    """
    root = Path(root)
    parts = []
    for rel in guarded_paths(root):
        path = root / rel
        if not path.exists():
            continue
        for name, blob in _entries(path, rel):
            parts.append(b"%s\0%d\0%s" % (name.encode("utf-8"), len(blob), blob))
    if not parts:
        return EMPTY_DIGEST
    digest = hashlib.sha256()
    for part in parts:
        digest.update(b"%d\0" % len(part))
        digest.update(part)
    return digest.hexdigest()


def _entries(path: Path, rel: str) -> Iterable[Tuple[str, bytes]]:
    """One ``(name, bytes)`` pair per file the CLI could read at ``path``.

    A symlink contributes its target rather than its target's content, so the
    link and the file it points at are distinguishable, and the target's own
    content arrives through the read that follows it.
    """
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except OSError as exc:
            yield rel + " (symlink)", b"unreadable:%r" % (exc,)
            return
        yield rel + " (symlink)", target.encode("utf-8", "surrogateescape")
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            yield "%s/%s" % (rel, child.relative_to(path).as_posix()), _read(child)
        return
    if rel == GIT_CONFIG_REL:
        # Every config in the include chain, named by the resolved path rather
        # than by a relative name: an included file usually sits outside the
        # served directory, and re-pointing an include at a different file has
        # to compare unequal even when both files hold the same bytes.
        for member in config_chain(path):
            yield "%s <- %s" % (rel, os.path.realpath(member)), _read(member)
        return
    yield rel, _read(path)


def _read(path: Path) -> bytes:
    try:
        size = path.stat().st_size
        if size > _MAX_DIGEST_BYTES:
            return b"oversized:%d" % size
        return path.read_bytes()
    except OSError as exc:
        return b"unreadable:%r" % (exc,)


# ---------------------------------------------------------------------------
# The trust record.
# ---------------------------------------------------------------------------


def config_home() -> Path:
    """The user's configuration directory, per platform convention."""
    override = os.environ.get("XDG_CONFIG_HOME")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata)
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".config"


def trust_store_path() -> Path:
    return config_home() / product.current().config_dirname / _TRUST_FILE


class TrustStore:
    """Which directories' configuration the human has accepted, and as of what content.

    One record per line, ``<digest> <absolute path>``, split once from the
    left so a path containing spaces survives a round trip. A line this cannot
    parse is dropped rather than raising, and a missing or unreadable file
    reads as no records: every failure here costs one re-acceptance, which is
    the direction that asks more rather than less.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else trust_store_path()

    def _records(self):
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            sys.stderr.write(
                "warning: could not read %s: %r; the served directory's Claude "
                "configuration will need accepting again\n" % (self.path, exc)
            )
            return {}
        records = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, sep, target = line.partition(" ")
            if sep and target.strip():
                records[target.strip()] = digest
        return records

    def accepted(self, root, digest: str) -> bool:
        """Whether ``root``'s configuration is recorded as accepted at exactly ``digest``."""
        return self._records().get(str(Path(root).resolve())) == digest

    def accept(self, root, digest: str) -> None:
        """Record ``root``'s configuration as accepted at ``digest``, replacing any earlier record for it."""
        records = self._records()
        records[str(Path(root).resolve())] = digest
        lines = [
            "# Directories whose Claude configuration you have accepted (%s)."
            % product.current().distribution,
            "# One record per line: <digest> <absolute path>. Deleting a line asks again.",
        ]
        lines.extend("%s %s" % (records[target], target) for target in sorted(records))
        lines.append("")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# The gate, as a decision the CLI renders.
# ---------------------------------------------------------------------------


def _is_git_entry(path: Path) -> bool:
    """Whether ``path`` is one of the two conditionally guarded git entries."""
    parts = path.parts
    return ".git" in parts and (path.name == "config" or "hooks" in parts)


def refusal_message(root, entries: Iterable[Path]) -> str:
    """What to tell a human whose directory carries configuration they have not accepted.

    Names the files, says what they can do rather than only what they contain,
    and offers both ways forward, because a refusal that does not say how to
    proceed gets worked around by whatever the human tries next.

    The two kinds of configuration are explained separately when both are
    present, because they execute by different routes and a human reading a
    single sentence about hooks would not think to look at a git config.
    """
    entries = tuple(entries)
    listed = "\n".join("    %s" % path for path in entries)
    git_named = any(_is_git_entry(path) for path in entries)
    claude_named = any(not _is_git_entry(path) for path in entries)
    why = ""
    if claude_named:
        why += (
            "  The .claude and .mcp entries can declare hooks, which are shell commands the\n"
            "  agent's CLI runs, and a session-start hook runs before any prompt is sent.\n"
        )
    if git_named:
        why += (
            "  The git entries listed name commands git itself runs: a config key such as\n"
            "  core.fsmonitor or an alias, or a hook script under .git/hooks. Those run as\n"
            "  soon as the agent runs an ordinary git command, and git deliberately runs\n"
            "  outside the sandbox so it can work on this project's own repository.\n"
        )
    return (
        "error: %s carries configuration that this run has not been told to trust.\n"
        "%s\n"
        "%s"
        "  Review them, then:\n"
        "    accept them for this directory : %s --trust-project-config\n"
        "    or run the viewer with no agent : %s\n"
        "  Acceptance is recorded per directory against the exact content reviewed, so\n"
        "  any later change to these files asks again.\n"
        % (
            root,
            listed,
            why,
            product.current().distribution,
            product.current().viewer_only_command,
        )
    )
