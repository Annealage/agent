"""Tests for the review watcher (``review/watcher.py``): the push that tells
every page the review changed, whoever changed it.

Every state-machine test drives ``ReviewWatcher.tick`` directly with the time
it wants to pretend it is, rather than starting ``run`` and sleeping, so the
deferral rule is asserted rather than approximated. The store is the native
one over a real file, written to directly the way a hand edit or an external
agent would, since those are the writers the watcher exists for.

The change signal is a digest of the bytes the store read, not the file's size
and modification time. Two of the tests below only mean anything because of
that choice: a rewrite that preserves both size and mtime is invisible to a
stat-based watcher, and a stat-based readiness rule cannot tell a finished
write from a stalled one.
"""

import asyncio
import json

import pytest
from toy_product import TOY_REVIEW_FILE, toy_review_store

from annealage_agent.review import ReviewWatcher

pytestmark = pytest.mark.asyncio


def _watcher(tmp_path, max_defer=5.0, interval=0.25):
    events = []
    store = toy_review_store(tmp_path)
    return ReviewWatcher(store, events.append, interval=interval, max_defer=max_defer), events


async def _primed(tmp_path, max_defer=5.0):
    """A watcher past its priming sample, which records the state the watcher
    starts in and announces nothing: the page's own fetch on load covers it."""
    watcher, events = _watcher(tmp_path, max_defer=max_defer)
    assert await watcher.tick(-1.0) is False, "the priming tick must not announce"
    return watcher, events


def _write(tmp_path, text):
    (tmp_path / TOY_REVIEW_FILE).write_text(text, encoding="utf-8")


def _doc(ids):
    return json.dumps(
        {
            "version": 2,
            "next_id": 1,
            "comments": [
                {"id": i, "anchor": {}, "text": "t", "author": "human", "status": "open"}
                for i in ids
            ],
        }
    )


async def test_nothing_there_announces_nothing(tmp_path):
    watcher, events = _watcher(tmp_path)
    assert await watcher.tick(0.0) is False
    assert await watcher.tick(1.0) is False
    assert events == []


async def test_a_new_file_is_announced_once_with_no_content(tmp_path):
    watcher, events = await _primed(tmp_path)
    _write(tmp_path, _doc([1]))
    assert await watcher.tick(0.0) is True
    assert [event.to_wire() for event in events] == [{"kind": "review_changed"}]
    for t in (1.0, 2.0):
        assert await watcher.tick(t) is False
    assert len(events) == 1


async def test_rewriting_identical_bytes_is_not_a_change(tmp_path):
    watcher, events = await _primed(tmp_path)
    _write(tmp_path, _doc([1]))
    await watcher.tick(0.0)
    _write(tmp_path, _doc([1]))
    assert await watcher.tick(1.0) is False, "same content, new mtime: nothing to refetch"
    assert len(events) == 1


async def test_a_same_length_edit_is_still_detected(tmp_path):
    watcher, events = await _primed(tmp_path)
    _write(tmp_path, _doc([1]))
    await watcher.tick(0.0)
    _write(tmp_path, _doc([2]))
    assert len(_doc([1])) == len(_doc([2]))
    assert await watcher.tick(1.0) is True
    assert len(events) == 2


async def test_deleting_the_file_is_a_change(tmp_path):
    watcher, events = await _primed(tmp_path)
    _write(tmp_path, _doc([1]))
    await watcher.tick(0.0)
    (tmp_path / TOY_REVIEW_FILE).unlink()
    assert await watcher.tick(1.0) is True, (
        "a deleted review leaves a page showing comments that are gone"
    )


async def test_a_half_written_file_is_not_announced_until_it_parses(tmp_path):
    watcher, events = await _primed(tmp_path)
    _write(tmp_path, '{"version": 2, "comments": [{"id": 1, "te')
    assert await watcher.tick(0.0) is False, "changed but does not parse yet"
    assert await watcher.tick(0.1) is False, "still the same unparseable bytes"
    _write(tmp_path, _doc([1]))
    assert await watcher.tick(0.2) is True
    assert len(events) == 1


async def test_a_file_that_never_parses_is_announced_once_past_max_defer(tmp_path):
    watcher, events = await _primed(tmp_path, max_defer=1.0)
    _write(tmp_path, "{not json 1")
    assert await watcher.tick(10.0) is False
    _write(tmp_path, "{not json 22")
    assert await watcher.tick(10.5) is False
    _write(tmp_path, "{not json 333")
    assert await watcher.tick(11.5) is True, (
        "a file that keeps changing without ever parsing must still be announced "
        "once the deferral bound has passed"
    )
    assert len(events) == 1


async def test_the_deferral_clock_starts_at_the_first_unparseable_sample_and_resets(tmp_path):
    watcher, events = await _primed(tmp_path, max_defer=2.0)
    _write(tmp_path, "{partial")
    assert await watcher.tick(100.0) is False
    assert await watcher.tick(101.9) is False, "still inside the deferral window"
    _write(tmp_path, _doc([]))
    assert await watcher.tick(102.0) is True
    # A later unparseable write gets its own full window.
    _write(tmp_path, "{partial again")
    assert await watcher.tick(103.0) is False
    assert await watcher.tick(104.5) is False
    assert await watcher.tick(105.0) is True
    assert len(events) == 2


async def test_a_change_through_the_store_is_announced_at_once(tmp_path):
    """The store's listener wakes the loop, so a tool's write is announced
    without waiting out the interval, here far longer than the test."""
    watcher, events = _watcher(tmp_path, interval=60.0)
    task = asyncio.ensure_future(watcher.run())
    try:
        await asyncio.sleep(0.05)  # past the priming sample
        store = watcher._store
        loop = asyncio.get_running_loop()
        # From an executor thread, as a tool handler's store call is.
        await loop.run_in_executor(
            None,
            lambda: store.add_comment(
                anchor={"card": "back", "x": 1, "y": 1}, text="t", author="model"
            ),
        )
        for _ in range(100):
            if events:
                break
            await asyncio.sleep(0.01)
        assert [event.kind for event in events] == ["review_changed"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_run_samples_on_its_interval_and_can_be_cancelled(tmp_path):
    """A write around the store (a hand edit) is seen by the poll alone."""
    watcher, events = _watcher(tmp_path, interval=0.01)
    task = asyncio.ensure_future(watcher.run())
    await asyncio.sleep(0.05)
    _write(tmp_path, _doc([1]))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if events:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(events) == 1
    # Cancelled, it has let go of its loop: a later write through the store
    # has nothing to wake and must not fail.
    watcher._store.add_comment(anchor={"card": "back", "x": 1, "y": 1}, text="t", author="model")
