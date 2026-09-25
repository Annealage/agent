"""``<state dir>/lock``: refuses a second agent-mode instance in one project.

Two SDK clients resuming one session id, or two processes appending to one
``events.jsonl``, is corruption (plan section 3.4), so this module's only
job is to make a second concurrent agent-mode start impossible rather than
merely discouraged. Viewer-only runs never call ``acquire``: several
viewers on one project is the documented, supported case M4 already relies
on, and only one of them may ever also be driving an agent.

The file holds the holder's pid and port as JSON, so a refused second start
can say which process holds the project and where it is listening instead of
a bare refusal with no next step.

It deliberately holds no token. The lock lives inside the served directory,
which the agent's own shell can read, sandboxed or not, and the browser token
is what authorises a permission decision over ``/ws``: a token written here
would let the agent read it back and approve its own permission cards. So a
refused second start can name the running instance's address but not a URL
that logs in; the link with the token is the one the running instance printed
in its own banner. A record left by an older version that still carries a
``token`` field is read without it.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

from . import product

LOCK_FILENAME = "lock"


class LockError(Exception):
    """Base for every way ``acquire`` can fail to hand back a held lock."""


class LockHeld(LockError):
    """A live process already holds the lock.

    ``pid`` and ``port`` are the holder's own, read from the lock file, so the
    caller can say where the running instance is rather than just refusing.
    """

    def __init__(self, pid: int, port: int):
        self.pid = pid
        self.port = port
        super().__init__(
            "%s is already running here (pid %d, port %d)"
            % (product.current().distribution, pid, port)
        )


class LockCorrupt(LockError):
    """The lock file exists but does not hold a valid pid/port record.

    Left in place rather than reclaimed: a file this module cannot parse is
    not proof anything is dead, and removing it on a guess is exactly the
    silent-corruption failure the lock exists to prevent. A human has to
    look at it.
    """


def lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / LOCK_FILENAME


def _pid_is_live(pid: int) -> bool:
    """Whether ``pid`` names a process visible to this user right now.

    ``os.kill(pid, 0)`` sends no signal; the kernel only reports whether the
    target exists. ``ESRCH`` means it does not. Any other outcome, including
    ``EPERM`` (a live process this user does not own), counts as live: the
    lock's purpose is refusing a second start against the same project
    directory, and a pid that exists but is not ours is certainly not dead.
    """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    except Exception:
        # A platform without os.kill(pid, 0) support (unlikely on the
        # platforms this project targets) must not turn a liveness check
        # into a crash; treat as live so the caller reports a conflict
        # instead of silently reclaiming a lock that might not be dead.
        return True
    return True


def _read_record(path: Path):
    """Return ``(pid, port)`` from an existing lock file, or raise
    ``LockCorrupt``. Never called on a path known not to exist."""
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        pid = int(data["pid"])
        port = int(data["port"])
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise LockCorrupt("%s exists but is not a valid lock record (%s)" % (path, exc)) from exc
    return pid, port


class Lock:
    """A held ``<state dir>/lock``, releasable exactly once.

    Returned only by a successful ``acquire``; the file descriptor this holds
    is the one whose inode ``_claim`` linked into place as the sole winner, so
    holding this object is itself the proof of exclusive ownership.
    """

    def __init__(self, path: Path, fd: int):
        self._path = path
        self._fd: Optional[int] = fd
        self.path = path

    def release(self) -> None:
        """Close the descriptor and remove the file. Idempotent: a caller's
        explicit release racing a shutdown path's defensive one must not
        raise on the second call, nor close a descriptor number the OS may
        since have reissued to something else entirely."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "Lock":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def _claim(path: Path, payload: bytes) -> Optional["Lock"]:
    """Try to become the holder of ``path``, returning a held ``Lock`` or None.

    The record is written to a uniquely named sibling first and only then linked
    into place, rather than creating ``path`` empty and filling it in
    afterwards. ``os.link`` is as atomic an arbiter as ``O_CREAT | O_EXCL``, it
    fails with ``FileExistsError`` for exactly the same reason, and it publishes
    the name and its contents in the same instant.

    Creating the final name empty and writing to it afterwards leaves a window,
    microseconds wide but reachable by two starts issued together, in which the
    file exists and holds nothing. A racer that read it there saw an unparseable
    record and reported the lock corrupt, and the advice that accompanies that
    is to check nothing is running and delete the file by hand: advice to delete
    the live lock of the process that had just won the race, which would then
    permit the second server this lock exists to prevent.

    The returned ``Lock`` holds the descriptor opened on the temporary name.
    After the link both names refer to one inode, so it is the same open file
    the holder would have had either way, and releasing it unlinks the name a
    caller can see.
    """
    unique = "%s.%d.%s.tmp" % (path.name, os.getpid(), secrets.token_hex(4))
    tmp = path.with_name(unique)
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
        try:
            os.link(str(tmp), str(path))
        except FileExistsError:
            os.close(fd)
            return None
    except BaseException:
        os.close(fd)
        raise
    finally:
        # The temporary name has done its work whether or not the link landed,
        # and leaving one behind would litter the state directory with a file
        # per lost race.
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
    return Lock(path, fd)


def acquire(state_dir, port: int, *, pid: Optional[int] = None) -> Lock:
    """Create and hold ``lock`` under ``state_dir``, or raise.

    Raises ``LockHeld`` if a live process already holds it, ``LockCorrupt``
    if the file exists but will not parse. A dead holder is reclaimed and
    the attempt retried.

    The reclaim path is race-free by construction rather than by locking
    around the reclaim itself: after unlinking a record whose pid this
    process observed to be dead, the only next step is to retry the claim,
    never to write on the strength of that observation. If a different
    process wins the create in the gap between
    this process's unlink and its retry, that create is what makes the file
    exist again, and this process's retry then fails ``EEXIST`` against
    *that* fresh record, which this function reads and correctly reports as
    held. So the only way this function ever hands back a ``Lock`` is
    holding the file descriptor whose creation the kernel itself arbitrated
    as the sole winner; two processes reclaiming the same stale lock at once
    can each unlink a file that is already gone (harmless: ``os.unlink``
    is tolerant of that below) but can never both believe they created it.
    """
    pid = os.getpid() if pid is None else pid
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = lock_path(state_dir)
    payload = json.dumps({"pid": pid, "port": port}).encode("utf-8")

    while True:
        acquired = _claim(path, payload)
        if acquired is not None:
            return acquired

        try:
            held_pid, held_port = _read_record(path)
        except FileNotFoundError:
            # Released between this loop's failed create and this read
            # (the holder exited and released, or another process's own
            # reclaim already won): retry the create rather than treating
            # a file that is not there as anything to reclaim.
            continue

        if _pid_is_live(held_pid):
            raise LockHeld(held_pid, held_port)

        sys.stderr.write(
            "%s: reclaiming stale lock at %s (pid %d is no longer "
            "running)\n" % (product.current().distribution, path, held_pid)
        )
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass  # already reclaimed by a concurrent start; the retry above settles it
