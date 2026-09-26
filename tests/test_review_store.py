"""The native review store's invariants (``review/native.py``), against the
toy product's anchor space.

What this file defends is what a human's words depend on: an id is never
handed out twice, so a reply naming #4 cannot land on another comment; a file
this store cannot read is never replaced, since it holds what the human wrote;
two writers in one process cannot lose a comment between them; and a write
that fails part-way leaves the previous file whole.
"""

import json
import os
import threading

import pytest
from toy_product import TOY_REVIEW_FILE, ToyAnchorSpace, toy_review_store

from annealage_agent.review import JsonReviewStore, ReviewError
from annealage_agent.review.native import FORMAT_VERSION

FRONT_A = {"card": "front", "x": 10, "y": 10}
BACK = {"card": "back", "x": 20, "y": 30}


@pytest.fixture
def store(tmp_path):
    return toy_review_store(tmp_path)


def _file(tmp_path):
    return tmp_path / TOY_REVIEW_FILE


def _document(tmp_path):
    return json.loads(_file(tmp_path).read_text(encoding="utf-8"))


def _add(store, author="model", anchor=BACK, text="look here", **kwargs):
    return store.add_comment(anchor=anchor, text=text, author=author, **kwargs)


# --- the file format ----------------------------------------------------------


def test_a_comment_is_written_in_the_documented_format(store, tmp_path):
    _add(store, author="human", anchor={"card": "front", "x": 10.004, "y": 20}, text=" thin ")
    assert _document(tmp_path) == {
        "version": FORMAT_VERSION,
        "next_id": 2,
        "comments": [
            {
                "id": 1,
                # Nested, normalised by the anchor space, ref from ref_at.
                "anchor": {"card": "front", "x": 10.0, "y": 20.0},
                "ref": "box-a",
                "text": "thin",
                "author": "human",
                "status": "open",
            }
        ],
    }


def test_the_product_s_extra_fields_survive_a_round_trip(store, tmp_path):
    _add(store, extra={"colour": "red"})
    document = _document(tmp_path)
    document["comments"][0]["pinned_by"] = "a hand edit"
    document["reviewer"] = "someone"
    _file(tmp_path).write_text(json.dumps(document), encoding="utf-8")

    _add(store, text="second")
    (first, _second) = store.list_comments().comments
    assert first.extra == {"colour": "red", "pinned_by": "a hand edit"}
    rewritten = _document(tmp_path)
    assert rewritten["reviewer"] == "someone"
    assert rewritten["comments"][0]["pinned_by"] == "a hand edit"


def test_an_extra_field_may_not_shadow_a_comment_s_own(store):
    with pytest.raises(ReviewError, match="status"):
        _add(store, extra={"status": "resolved"})


# --- ids ----------------------------------------------------------------------


def test_ids_are_never_reused_after_a_delete_or_a_resolve(store):
    assert [_add(store).comment.id for _ in range(3)] == [1, 2, 3]
    store.delete_callout(3)
    store.resolve_comment(2, "done")
    assert _add(store).comment.id == 4, "deleting the newest comment must not free its id"
    assert [c.id for c in store.list_comments().comments] == [1, 2, 4]


def test_ids_stay_unique_when_every_comment_has_been_deleted(store, tmp_path):
    _add(store)
    _add(store)
    store.delete_callout(1)
    store.delete_callout(2)
    assert _document(tmp_path)["next_id"] == 3
    assert _add(store).comment.id == 3


def test_a_next_id_behind_the_highest_id_is_not_trusted(store, tmp_path):
    """A hand edit can leave next_id stale; an id already in the file must
    still never be handed out again."""
    _add(store)
    _add(store)
    document = _document(tmp_path)
    document["next_id"] = 1
    _file(tmp_path).write_text(json.dumps(document), encoding="utf-8")
    assert _add(store).comment.id == 3


def test_two_stores_adding_at_once_lose_nothing_and_reuse_no_id(tmp_path):
    """Two stores over one file, as a tool server's and a route's are, each
    adding from several threads: every comment lands and no id repeats. The
    process-wide lock per file is what makes this hold; without it, two
    read-modify-writes interleave and one replaces the other's comment."""
    stores = [toy_review_store(tmp_path, max_open_model_callouts=None) for _ in range(2)]
    per_thread = 15
    errors = []

    def add_many(store, label):
        try:
            for n in range(per_thread):
                _add(store, text="%s-%d" % (label, n))
        except Exception as exc:  # reported below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=add_many, args=(stores[i % 2], "t%d" % i)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    comments = stores[0].list_comments().comments
    assert len(comments) == 8 * per_thread
    assert sorted(c.id for c in comments) == list(range(1, 8 * per_thread + 1))
    assert len({c.text for c in comments}) == 8 * per_thread


def test_reading_while_another_thread_writes_never_reports_a_broken_file(tmp_path):
    """Every read takes the same per-file lock as a write. Unlocked, a read
    can open the file, see an in-process atomic replace swap the name to a
    new inode, fail the name-to-inode recheck and report a healthy file as
    one to "fix by hand": to the model, to the page as a 409, and to the
    watcher as a spurious change."""
    writer = toy_review_store(tmp_path, max_open_model_callouts=None)
    reader = toy_review_store(tmp_path)
    writer.add_comment(anchor=BACK, text="first", author="model")
    done = threading.Event()
    errors = []

    def write_many():
        try:
            for n in range(300):
                writer.add_comment(anchor=BACK, text="w%d" % n, author="model")
        finally:
            done.set()

    thread = threading.Thread(target=write_many)
    thread.start()
    reads = 0
    while not done.is_set():
        for read in (reader.list_comments, lambda: reader.get_comment(1)):
            try:
                read()
            except ReviewError as exc:
                errors.append(str(exc))
        if reader.state()[0] is None:
            errors.append("state() read a present file as absent")
        reads += 1
    thread.join()
    assert reads > 10, "the reader barely overlapped the writer"
    assert errors == []


# --- files this store will not replace ---------------------------------------


@pytest.mark.parametrize(
    "content, fragment",
    [
        ("{ this is not json", "does not parse"),
        ('{"comments": {}}', 'no "comments" list'),
        (
            '{"version": 1, "next_id": 2, "comments": [{"id": 1, "sheet": "root", '
            '"x_mm": 1, "y_mm": 2, "text": "t", "author": "human", "status": "open"}]}',
            "version 1",
        ),
        (
            '{"version": 2, "comments": [{"id": 1, "anchor": {}, "text": "a", '
            '"author": "human"}, {"id": 1, "anchor": {}, "text": "b", "author": "model"}]}',
            "same id",
        ),
        (
            '{"version": 2, "comments": [{"id": 1, "anchor": {}, "text": "a", "author": "x"}]}',
            "author",
        ),
    ],
)
def test_an_unreadable_file_is_refused_and_never_replaced(store, tmp_path, content, fragment):
    _file(tmp_path).write_text(content, encoding="utf-8")
    for attempt in (
        lambda: _add(store),
        lambda: store.list_comments(),
        lambda: store.resolve_comment(1, "x"),
        lambda: store.delete_callout(1),
    ):
        with pytest.raises(ReviewError, match=fragment):
            attempt()
    assert _file(tmp_path).read_text(encoding="utf-8") == content


def test_a_symlinked_review_file_is_neither_read_nor_replaced(store, tmp_path):
    outside = tmp_path.parent / ("outside-%s.json" % tmp_path.name)
    outside.write_text('{"version": 2, "next_id": 1, "comments": []}', encoding="utf-8")
    _file(tmp_path).symlink_to(outside)
    with pytest.raises(ReviewError, match="single-linked"):
        _add(store)
    assert _file(tmp_path).is_symlink()
    assert outside.read_text(encoding="utf-8") == '{"version": 2, "next_id": 1, "comments": []}'


def test_a_write_that_fails_part_way_leaves_the_previous_file_whole(store, tmp_path, monkeypatch):
    """The atomic replace: the new bytes go to a temporary file that only
    replaces the review once complete, so a failure before that point leaves
    the old review readable and no temporary file behind. A store that
    rewrote the file in place would already have truncated it."""
    _add(store, text="kept")
    before = _file(tmp_path).read_bytes()

    def failing_fsync(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(ReviewError, match="could not write"):
        _add(store, text="lost")
    monkeypatch.undo()
    assert _file(tmp_path).read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [TOY_REVIEW_FILE]
    assert [c.text for c in store.list_comments().comments] == ["kept"]


# --- the operations ------------------------------------------------------------


def test_every_call_reads_the_file_afresh(store, tmp_path):
    """A hand edit or a git checkout is seen by the next call, with no cache
    in the way."""
    _add(store, text="before")
    document = _document(tmp_path)
    document["comments"][0]["text"] = "edited by hand"
    _file(tmp_path).write_text(json.dumps(document), encoding="utf-8")
    assert store.list_comments().comments[0].text == "edited by hand"


def test_list_filters_by_author_and_status(store):
    _add(store, author="human", text="h1")
    _add(store, author="model", text="m1")
    _add(store, author="human", text="h2")
    store.resolve_comment(1, "fixed")
    assert [c.text for c in store.list_comments(author="human").comments] == ["h1", "h2"]
    assert [c.text for c in store.list_comments(status="open").comments] == ["m1", "h2"]
    assert [c.text for c in store.list_comments(status="resolved").comments] == ["h1"]


def test_resolving_keeps_the_product_s_extras_and_unknown_keys(store, tmp_path):
    """Loom resolves through exactly this path, so a resolve that rebuilt
    the record from the comment's own fields alone would silently drop the
    product's fields and anything a person added to the human's file."""
    _add(store, author="human", extra={"colour": "red"})
    document = _document(tmp_path)
    document["comments"][0]["pinned_by"] = "a hand edit"
    _file(tmp_path).write_text(json.dumps(document), encoding="utf-8")

    written = store.resolve_comment(1, "done")
    assert written.comment.extra == {"colour": "red", "pinned_by": "a hand edit"}
    record = _document(tmp_path)["comments"][0]
    assert (record["status"], record["colour"], record["pinned_by"]) == (
        "resolved",
        "red",
        "a hand edit",
    )


def test_resolving_records_the_resolution_and_resolving_again_changes_nothing(store, tmp_path):
    _add(store, author="human")
    written = store.resolve_comment(1, "  widened the wall  ")
    assert (written.comment.status, written.comment.resolution) == ("resolved", "widened the wall")
    before = _file(tmp_path).read_bytes()
    heard = []
    store.add_listener(lambda: heard.append(True))
    again = store.resolve_comment(1, "a different note")
    assert again.comment.resolution == "widened the wall"
    assert _file(tmp_path).read_bytes() == before
    assert heard == [], "a resolve that changed nothing is not a change"


def test_a_human_s_comment_is_never_deleted(store, tmp_path):
    _add(store, author="human")
    before = _file(tmp_path).read_bytes()
    with pytest.raises(ReviewError, match="human's"):
        store.delete_callout(1)
    assert _file(tmp_path).read_bytes() == before


def test_the_open_callout_limit_counts_only_the_model_s_open_callouts(tmp_path):
    store = toy_review_store(tmp_path, max_open_model_callouts=2)
    _add(store)
    _add(store)
    _add(store, author="human")
    with pytest.raises(ReviewError, match="2 of your callouts are open"):
        _add(store)
    store.resolve_comment(1, "answered")
    assert _add(store).comment.id == 4, "a resolved callout frees a place"
    with pytest.raises(ReviewError, match="limit"):
        _add(store)
    assert _add(store, author="human").comment.author == "human", "the human is never limited"


def test_the_anchor_is_validated_before_anything_is_written(store, tmp_path):
    with pytest.raises(ReviewError, match="no card 'side'"):
        _add(store, anchor={"card": "side", "x": 1, "y": 1})
    with pytest.raises(ReviewError, match="off the card"):
        _add(store, anchor={"card": "front", "x": 101, "y": 1})
    assert not _file(tmp_path).exists()


def test_listeners_hear_each_landed_write_and_not_a_refused_one(store, tmp_path):
    heard = []
    store.add_listener(lambda: heard.append("change"))
    _add(store)
    store.resolve_comment(1, "done")
    _add(store)
    store.delete_callout(2)
    assert heard == ["change"] * 4
    _file(tmp_path).write_text("{ broken", encoding="utf-8")
    with pytest.raises(ReviewError):
        _add(store)
    assert heard == ["change"] * 4


def test_state_changes_with_the_bytes_and_says_whether_they_parse(store, tmp_path):
    assert store.state() == (None, True)
    _add(store)
    digest, settled = store.state()
    assert digest is not None and settled
    _file(tmp_path).write_text('{"version": 2, "comm', encoding="utf-8")
    half_digest, settled = store.state()
    assert half_digest != digest and settled is False


def test_the_store_is_loom_ready_over_any_path(tmp_path):
    """The native store is not tied to a served directory: Loom keeps
    ``designs/src/<id>.review.json`` beside the sketch."""
    target = tmp_path / "designs" / "src"
    target.mkdir(parents=True)
    store = JsonReviewStore(target / "psu.review.json", ToyAnchorSpace())
    store.add_comment(anchor=FRONT_A, text="R1?", author="human")
    assert json.loads((target / "psu.review.json").read_text())["comments"][0]["ref"] == "box-a"
