"""Tests for ``lock.py``: the exclusivity guarantee that stops a second
agent-mode instance from ever running against one project directory.

The lock lives in the installed product's state directory (``.toy/lock`` for
the toy product this suite runs as). ``acquire``/``Lock.release`` are
exercised against a real filesystem, since the whole point of
``O_CREAT | O_EXCL`` is a kernel guarantee that no amount of mocking can stand
in for. What a human running a second instance sees (the exit code, the
message, whether a server started) is the product CLI's to report, and is
tested in that product's suite.
"""

import errno
import json
import os
import stat
import threading

import pytest

from annealage_agent import lock


def test_exclusive_creation_writes_pid_and_port_and_no_token(tmp_path):
    """A fresh ``acquire`` creates the file exactly once, with the caller's
    pid and port and nothing else, mode 0600, and ``release`` removes it
    again. No token: the file sits in the served directory, which the agent's
    own shell can read, and a browser token there would let it approve its
    own permission cards."""
    state_dir = tmp_path / ".toy"
    held = lock.acquire(state_dir, 4242)

    path = lock.lock_path(state_dir)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    record = json.loads(path.read_bytes())
    assert record == {"pid": os.getpid(), "port": 4242}

    held.release()
    assert not path.exists()
    # Idempotent: a second release (a shutdown path racing an explicit one)
    # must not raise on either the already-closed fd or the already-gone file.
    held.release()


def test_live_pid_refused_raises_lock_held_with_holder_details(tmp_path):
    """A lock file naming a pid this process can see raises ``LockHeld``
    carrying the holder's own pid and port, never silently reclaimed. The
    record here is one an older version wrote, still carrying a token: it is
    read without it rather than reported corrupt, and the token goes nowhere."""
    state_dir = tmp_path / ".toy"
    state_dir.mkdir()
    # This test process's own pid is guaranteed live for the test's duration,
    # so no mocking of os.kill is needed to pin this path deterministically.
    lock.lock_path(state_dir).write_bytes(
        json.dumps({"pid": os.getpid(), "port": 9001, "token": "held-tok"}).encode()
    )

    with pytest.raises(lock.LockHeld) as exc:
        lock.acquire(state_dir, 4242)
    assert exc.value.pid == os.getpid()
    assert exc.value.port == 9001
    assert "held-tok" not in str(exc.value)
    # Names the installed product, since that is what the human started twice.
    assert "annealage-toy is already running here" in str(exc.value)
    # Refused, not reclaimed: the file on disk is still the live holder's.
    assert json.loads(lock.lock_path(state_dir).read_bytes())["port"] == 9001


def test_stale_pid_is_reclaimed(tmp_path, monkeypatch, capsys):
    """A lock file naming a pid that no longer exists is unlinked and the
    create retried, so the caller gets back a ``Lock`` rather than a refusal.

    ``os.kill`` is monkeypatched for one specific fake pid rather than
    finding a real dead process, so the "which pid is actually dead right
    now" question never depends on process-table timing; every other pid
    (this test's own, and anything else on the machine) still goes through
    the real syscall.
    """
    dead_pid = 999999

    real_kill = os.kill

    def fake_kill(pid, sig):
        if pid == dead_pid:
            raise OSError(errno.ESRCH, "No such process")
        return real_kill(pid, sig)

    monkeypatch.setattr(lock.os, "kill", fake_kill)

    state_dir = tmp_path / ".toy"
    state_dir.mkdir()
    path = lock.lock_path(state_dir)
    path.write_bytes(json.dumps({"pid": dead_pid, "port": 1, "token": "stale"}).encode())

    held = lock.acquire(state_dir, 4242)
    try:
        record = json.loads(path.read_bytes())
        assert record == {"pid": os.getpid(), "port": 4242}
        assert "annealage-toy: reclaiming stale lock" in capsys.readouterr().err
    finally:
        held.release()


def test_garbage_lock_file_raises_lock_corrupt_and_is_left_in_place(tmp_path):
    """A lock file that exists but does not parse as a pid/port record
    is reported, never guessed at: reclaiming an unreadable file on the
    assumption it must be stale is the exact silent-corruption failure the
    lock exists to prevent."""
    state_dir = tmp_path / ".toy"
    state_dir.mkdir()
    path = lock.lock_path(state_dir)
    path.write_bytes(b"not json at all")

    with pytest.raises(lock.LockCorrupt):
        lock.acquire(state_dir, 4242)
    # Left exactly as it was: no reclaim on a guess.
    assert path.read_bytes() == b"not json at all"


def test_garbage_lock_file_missing_keys_also_raises_lock_corrupt(tmp_path):
    """Valid JSON that is missing a required field is corrupt the same way
    unparseable bytes are: a partial record proves nothing about liveness
    either."""
    state_dir = tmp_path / ".toy"
    state_dir.mkdir()
    lock.lock_path(state_dir).write_bytes(json.dumps({"pid": 1}).encode())

    with pytest.raises(lock.LockCorrupt):
        lock.acquire(state_dir, 4242)


def test_unwritable_state_dir_raises_instead_of_silently_succeeding(tmp_path):
    """A state directory that exists but this user cannot write into
    cannot hold a new lock file; ``acquire`` must surface that as an error
    rather than report success for a lock it never actually created."""
    state_dir = tmp_path / ".toy"
    state_dir.mkdir()
    state_dir.chmod(0o500)
    try:
        with pytest.raises(OSError):
            lock.acquire(state_dir, 4242)
        assert not lock.lock_path(state_dir).exists()
    finally:
        # Restored so pytest's own tmp_path cleanup (which needs to remove
        # this directory) is not the thing that fails instead of the test.
        state_dir.chmod(0o700)


def test_two_threads_racing_one_create_exactly_one_wins(tmp_path):
    """Two callers hitting ``acquire`` at the same instant, simulating two
    processes starting against the same project at once: the kernel's
    ``O_CREAT | O_EXCL`` arbitrates a single winner, and the loser sees that
    winner's record as a live holder, never a corrupt or a doubly-created
    file.

    Real threads, not a mocked interleaving, because the property under
    test is the kernel's own atomicity guarantee on the ``open()`` call, not
    anything this project's code arbitrates itself.

    The race is run repeatedly rather than once. The window this guards is the
    few microseconds between the lock name coming into existence and the record
    inside it being complete, and one attempt hits that window perhaps a quarter
    of the time: often enough to have shown up as an intermittent failure,
    rarely enough to have been mistaken for one. Twenty rounds turn it from a
    coin toss into a test.

    A loser that reports the lock *corrupt* is the specific failure, because the
    advice attached to that is to check nothing is running and delete the file
    by hand, which here would mean deleting the live lock of the process that
    had just won.
    """
    state_dir = tmp_path / ".toy"

    for round_number in range(20):
        barrier = threading.Barrier(2)
        results = [None, None]

        def attempt(i, port, barrier=barrier, results=results):
            # barrier and results are bound as defaults rather than captured:
            # both are rebound on the next iteration of the enclosing loop, and
            # a closure that read them late would sample the following round's.
            barrier.wait()
            try:
                results[i] = ("ok", lock.acquire(state_dir, port))
            except lock.LockHeld as exc:
                results[i] = ("held", exc)
            except lock.LockCorrupt as exc:
                results[i] = ("corrupt", exc)

        t1 = threading.Thread(target=attempt, args=(0, 1111))
        t2 = threading.Thread(target=attempt, args=(1, 2222))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        outcomes = [r[0] for r in results]
        assert outcomes.count("ok") == 1, (
            "round %d: exactly one racer must win the create: got %r" % (round_number, outcomes)
        )
        assert outcomes.count("held") == 1, (
            "round %d: the loser must see the winner as a live holder, never a "
            "corrupt record: got %r" % (round_number, outcomes)
        )

        winner = next(r[1] for r in results if r[0] == "ok")
        loser_exc = next(r[1] for r in results if r[0] == "held")
        # The loser's LockHeld must describe the winner's own record, not a
        # third, phantom holder.
        winner_record = json.loads(lock.lock_path(state_dir).read_bytes())
        assert loser_exc.pid == winner_record["pid"] == os.getpid()
        assert loser_exc.port == winner_record["port"]
        # No temporary file is left behind by the round the loser lost.
        assert sorted(p.name for p in state_dir.iterdir()) == ["lock"]
        winner.release()
    assert loser_exc.port == winner_record["port"]
    winner.release()


def test_the_record_is_complete_before_the_lock_name_exists(tmp_path, monkeypatch):
    """The invariant the racing test can only sample: at the instant the lock
    name is published, the record behind it is already whole.

    Asserted at the publishing call itself rather than by racing threads,
    because the window it guards is microseconds wide and a test that has to
    land inside it catches a regression only some of the time. Creating the
    final name first and writing into it afterwards cannot satisfy this, since
    there is no call at which the name does not yet exist and the content
    already does.
    """
    state_dir = tmp_path / ".toy"
    observed = {}
    real_link = os.link

    def spy(src, dst):
        observed["published_bytes"] = os.stat(src).st_size and open(src, "rb").read()
        observed["name_existed_first"] = os.path.exists(dst)
        return real_link(src, dst)

    monkeypatch.setattr(lock.os, "link", spy)

    held = lock.acquire(state_dir, 4242)
    try:
        assert observed, "the lock name was published without os.link"
        assert observed["name_existed_first"] is False
        assert json.loads(observed["published_bytes"]) == {"pid": os.getpid(), "port": 4242}
    finally:
        held.release()


def test_claiming_never_interprets_an_existing_record(tmp_path):
    """Whoever loses the claim reports what the file says, and the claim itself
    reports only whether it won. Keeping those apart is why a loser can no
    longer describe a record it read too early."""
    state_dir = tmp_path / ".toy"
    state_dir.mkdir()
    path = lock.lock_path(state_dir)
    path.write_bytes(b"")  # the exact state the old create-then-write left behind

    assert lock._claim(path, b'{"pid": 1, "port": 2}') is None
    assert path.read_bytes() == b"", "a lost claim must not touch the holder's file"
    assert sorted(p.name for p in state_dir.iterdir()) == ["lock"]
