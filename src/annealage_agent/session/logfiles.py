"""Where each agent backend keeps its own files for one session, and the end
of one read back: what ``AgentSession.backend_logs`` lists, what
``GET /agent/logs/<name>`` (``http/routes_logs.py``) serves, and what the
diagnostics block names.

Every path here is decided by this process, from where each backend is known
to keep its files; none is taken from a request. A file is listed only when it
is a regular file, not a symlink, whose real location is inside the directory
it is expected in (``contained_file``), and it is opened without following a
symlink (``read_tail``). That matters because the backends report some of
these names themselves (omp's conversation file, and the ids the others'
files are named after), and omp's conversations sit inside the project,
where a sandboxed shell writes freely: a name that resolves anywhere else is
dropped rather than followed, so the route serving these can never be made to
read an arbitrary file.

Where each backend keeps them, read from each one's own code at the versions
this package is verified against:

- **Claude Code** keeps a transcript per session at
  ``<config>/projects/<slug>/<session id>.jsonl``. ``<config>`` is
  ``$CLAUDE_CONFIG_DIR`` or ``~/.claude``, and ``<slug>`` is the working
  directory as the CLI's process sees it (symlinks resolved) with every UTF-16
  code unit that is not an ASCII letter or digit replaced by ``-``. A slug over
  200 characters is cut to 200 and given a ``-<hash>`` suffix the CLI computes
  itself, so that case is matched by the prefix.
- **Codex** keeps a rollout per thread at
  ``<home>/sessions/YYYY/MM/DD/rollout-<time>-<thread id>.jsonl``, ``<home>``
  being ``$CODEX_HOME`` or ``~/.codex``.
- **omp** logs each process to ``<root>/logs/omp.<local date>.<pid>.log``
  (``.log.1``, ``.log.2`` once one passes 10 MB, and a new date after
  midnight), one JSON object per line with a ``level``. ``<root>`` is
  ``$HOME/$PI_CONFIG_DIR`` (``~/.omp`` by default), or ``$XDG_STATE_HOME/omp``
  where that exists and the agent directory is omp's default one. omp's named
  profiles (``OMP_PROFILE``) move the root again and are not followed here.
  Its conversation file is under the session directory this package gives it.
"""

from __future__ import annotations

import errno
import glob
import os
import re
import stat
import sys
from typing import Optional

from .. import lock, sessions
from .base import BackendLog

#: How much of a log one request is sent: its end, which is where a failure
#: is, and small enough that a 10 MB omp log or a long transcript does not
#: become a 10 MB response.
TAIL_BYTES = 256 * 1024

# An id a backend reported, used as part of a file name. Letters, digits and
# ``.``/``_``/``-`` only, starting with a letter or digit: no separator and no
# ``..``, so it cannot name anything outside the directory it is joined to,
# and no glob character either.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_OMP_LOG_RE = re.compile(r"^omp\.\d{4}-\d{2}-\d{2}\.(\d+)\.log(?:\.\d+)?$")

# The Claude CLI's cap on a project slug before it adds a hash suffix.
_CLAUDE_SLUG_MAX = 200


def contained_file(path, root) -> Optional[str]:
    """``path``'s real location, if it is a regular file (not itself a
    symlink) that resolves to somewhere inside ``root``; otherwise ``None``.
    Never raises."""
    if not path or not root:
        return None
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        real = os.path.realpath(path)
        real_root = os.path.realpath(root)
        if real == real_root or os.path.commonpath((real, real_root)) != real_root:
            return None
    except (OSError, ValueError):
        return None
    return real


def claude_transcript(cwd, session_id, environ=None) -> Optional[str]:
    """The Claude CLI's transcript of ``session_id`` for a session run in
    ``cwd``, or ``None`` if there is none (yet)."""
    if not cwd or not session_id or not _ID_RE.match(session_id):
        return None
    environ = os.environ if environ is None else environ
    config = environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    projects = os.path.join(config, "projects")
    name = session_id + ".jsonl"
    # The CLI's own process cwd is the physical path; the one this process was
    # given is tried as well, for a platform whose CLI keeps it as it came.
    for directory in dict.fromkeys((os.path.realpath(str(cwd)), str(cwd))):
        slug = _claude_slug(directory)
        if len(slug) <= _CLAUDE_SLUG_MAX:
            candidates = [os.path.join(projects, slug)]
        else:
            prefix = slug[:_CLAUDE_SLUG_MAX] + "-"
            try:
                candidates = [
                    entry.path for entry in os.scandir(projects) if entry.name.startswith(prefix)
                ]
            except OSError:
                candidates = []
        for candidate in candidates:
            found = contained_file(os.path.join(candidate, name), projects)
            if found is not None:
                return found
    return None


def _claude_slug(directory: str) -> str:
    """``directory`` as the Claude CLI names its project folder: JavaScript's
    ``replace(/[^a-zA-Z0-9]/g, "-")``, which works on UTF-16 code units, so a
    character outside the Basic Multilingual Plane becomes two dashes."""
    return "".join(
        c if c.isascii() and c.isalnum() else ("--" if ord(c) > 0xFFFF else "-") for c in directory
    )


def codex_rollout(thread_id, environ=None) -> Optional[str]:
    """Codex's rollout file for ``thread_id``, or ``None`` if there is none
    (yet). The newest, should the same thread appear under two dates."""
    if not thread_id or not _ID_RE.match(thread_id):
        return None
    environ = os.environ if environ is None else environ
    home = environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    root = os.path.join(home, "sessions")
    pattern = os.path.join(glob.escape(root), "*", "*", "*", "rollout-*-%s.jsonl" % thread_id)
    return _newest(path for path in (contained_file(p, root) for p in glob.glob(pattern)) if path)


def omp_logs_dir(config_dir_name=None, agent_dir=None, environ=None) -> str:
    """The directory omp writes its process logs to, for an omp run with
    ``PI_CONFIG_DIR`` = ``config_dir_name`` and ``PI_CODING_AGENT_DIR`` =
    ``agent_dir`` (either ``None`` for what this process's environment says).
    Joined to ``$HOME`` the way omp joins it, whether or not it is absolute."""
    environ = os.environ if environ is None else environ
    home = os.path.expanduser("~")
    name = config_dir_name or environ.get("PI_CONFIG_DIR") or ".omp"
    root = os.path.normpath(home + os.sep + str(name))
    agent_dir = agent_dir or environ.get("PI_CODING_AGENT_DIR")
    default_agent = not agent_dir or os.path.abspath(str(agent_dir)) == os.path.join(root, "agent")
    xdg = environ.get("XDG_STATE_HOME")
    if (
        default_agent
        and sys.platform in ("linux", "darwin")
        and xdg
        and os.path.exists(os.path.join(xdg, "omp"))
    ):
        root = os.path.join(xdg, "omp")
    return os.path.join(root, "logs")


#: How far a log's modification time may sit outside a failed omp start's
#: window and still be its log. The kernel stamps files from a coarse clock
#: that can trail ``time.time()``, so a log written just after the start began
#: can read as a moment before it; and the failure is noted once omp_rpc has
#: seen the process exit, after its last write, so the far end needs only a
#: slow flush's worth. Either way a second, not another run.
_OMP_WINDOW_SLACK_S = 1.0


def omp_process_log(logs_dir, *, pid=None, since=None, until=None) -> Optional[str]:
    """The current log of omp process ``pid`` in ``logs_dir``.

    With no ``pid`` (omp exited before this process could learn it), a guess,
    made only once the start has failed: the newest log last written between
    ``since`` (when the start began) and ``until`` (when it failed), give or
    take ``_OMP_WINDOW_SLACK_S``, by an omp that is no longer running. The
    directory is shared with every omp using the same profile: the human's own
    interactive one, which is alive, and other runs' (a seed script, another
    instance), which a failed start's window excludes unless they died inside
    it. Without both bounds, ``None``: better no log than someone else's.
    """
    if not logs_dir or (pid is None and (since is None or until is None)):
        return None
    best = None
    try:
        entries = list(os.scandir(logs_dir))
    except OSError:
        return None
    for entry in entries:
        match = _OMP_LOG_RE.match(entry.name)
        if match is None:
            continue
        file_pid = int(match.group(1))
        if pid is not None and file_pid != pid:
            continue
        try:
            mtime = entry.stat(follow_symlinks=False).st_mtime
        except OSError:
            continue
        if pid is None and (
            not since - _OMP_WINDOW_SLACK_S <= mtime <= until + _OMP_WINDOW_SLACK_S
            or lock._pid_is_live(file_pid)
        ):
            continue
        if best is None or mtime > best[0]:
            best = (mtime, entry.path)
    return contained_file(best[1], logs_dir) if best is not None else None


def claude_logs(cwd, session_id, stderr: Optional[str] = None) -> list:
    """A Claude session's ``BackendLog`` entries: the CLI's transcript, and
    its stderr when this process kept any (``None`` leaves it out)."""
    entries = []
    transcript = claude_transcript(cwd, session_id)
    if transcript is not None:
        entries.append(BackendLog("Claude transcript", "file", path=transcript, format="jsonl"))
    if stderr is not None:
        entries.append(BackendLog("Claude stderr", "text", text=stderr))
    return entries


def codex_logs(thread_id, stderr: Optional[str] = None) -> list:
    """A Codex session's ``BackendLog`` entries: the thread's rollout, and
    the app-server's stderr when this process kept any."""
    entries = []
    rollout = codex_rollout(thread_id)
    if rollout is not None:
        entries.append(BackendLog("Codex rollout", "file", path=rollout, format="jsonl"))
    if stderr is not None:
        entries.append(BackendLog("Codex stderr", "text", text=stderr))
    return entries


def omp_logs(
    *,
    logs_dir=None,
    pid=None,
    since=None,
    until=None,
    session_file=None,
    session_dir=None,
    stderr: Optional[str] = None,
) -> list:
    """An omp session's ``BackendLog`` entries: its process log
    (``omp_process_log``), the conversation file when it is inside
    ``session_dir``, and omp's stderr when this process kept any."""
    entries = []
    log = omp_process_log(logs_dir, pid=pid, since=since, until=until)
    if log is not None:
        entries.append(BackendLog("omp log", "file", path=log, format="jsonl"))
    conversation = contained_file(session_file, session_dir)
    if conversation is not None:
        entries.append(BackendLog("omp conversation", "file", path=conversation, format="jsonl"))
    if stderr is not None:
        entries.append(BackendLog("omp stderr", "text", text=stderr))
    return entries


def recorded(project_dir, backend) -> list:
    """``backend``'s files for ``project_dir``'s most recent session, found
    from what that session's ``meta.json`` recorded: for a diagnostics report
    made with no session running (a product's ``doctor``). omp's process log is
    not among them, since it is named after a process id nothing records."""
    try:
        sid = sessions.resolve_continue(project_dir)
        info = sessions.get_session_info(project_dir, sid) if sid is not None else None
    except OSError:
        return []
    if info is None:
        return []
    if backend == "claude":
        return claude_logs(str(project_dir), info.sdk_session_id)
    if backend == "codex":
        return codex_logs(info.sdk_session_id)
    if backend == "omp":
        return omp_logs(
            session_file=info.omp_session_file,
            session_dir=sessions.state_dir(project_dir) / "omp",
        )
    return []


def read_tail(entry: BackendLog, limit: int = TAIL_BYTES) -> dict:
    """``{"text", "size", "truncated"}``: at most the last ``limit`` bytes
    of ``entry``, starting at a whole line when anything was cut, decoded as
    UTF-8 with anything undecodable replaced. ``size`` is the whole log's
    length in bytes.

    A file is opened without following a symlink and without blocking on a
    FIFO, and read only if it is a regular file, so a name swapped for
    something else after it was listed is refused here too. Raises ``OSError``
    for a file that has gone or cannot be read.
    """
    if entry.kind == "text":
        data = (entry.text or "").encode("utf-8")
        size = len(data)
        tail = data[size - limit :] if size > limit else data
    else:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(entry.path, flags)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise OSError(errno.EINVAL, "not a regular file", entry.path)
            size = st.st_size
            os.lseek(fd, max(0, size - limit), os.SEEK_SET)
            chunks = []
            remaining = min(size, limit)
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            tail = b"".join(chunks)
        finally:
            os.close(fd)
    truncated = size > len(tail)
    if truncated:
        cut = tail.find(b"\n")
        if cut != -1:
            tail = tail[cut + 1 :]
    return {"text": tail.decode("utf-8", "replace"), "size": size, "truncated": truncated}


def _newest(paths) -> Optional[str]:
    best = None
    for path in paths:
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    return best[1] if best is not None else None
