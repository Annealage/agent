"""Tests for ``session/turn_images.py``, the dual delivery plan section 2's
verified fact 6 requires, driven through ``SdkSession`` and a fake
``Transport`` the same way ``tests/test_sdk_session.py`` does (fact 11).

``SdkSession.submit_turn`` is the one place a turn's blocks leave this
process; every assertion here reads ``transport.written``, the JSON line the
SDK actually wrote, rather than a private helper's return value, so a defect
in how ``_expand_turn_blocks``'s result is threaded through ``query()`` would
still be caught here even if the expansion itself were correct in isolation.

This file duplicates ``test_sdk_session.py``'s small ``FakeTransport``/
``EventRecorder`` harness rather than importing it: each test module in this
suite is self-contained, sharing only what ``conftest.py`` and
``toy_product.py`` provide, instead of reaching into a sibling test module.
"""

import asyncio
import json

import pytest
from claude_agent_sdk import Transport

from annealage_agent import files
from annealage_agent.session import turn_images
from annealage_agent.session.base import AGENT_READY, AgentError
from annealage_agent.session.sdk import SdkSession

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_real_transport(monkeypatch):
    """Fail loudly, rather than spawning a real ``claude`` subprocess, if
    anything in this module's code path ever falls through to the SDK's own
    ``SubprocessCLITransport`` instead of the fake one a test injected."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "a real SubprocessCLITransport was constructed; the fake "
            "Transport passed to SdkSession was not used"
        )

    monkeypatch.setattr(subprocess_cli, "SubprocessCLITransport", _forbidden)


class FakeTransport(Transport):
    """A ``Transport`` backed by one ``asyncio.Queue``, with no process and
    no socket. See ``test_sdk_session.py``'s copy of this class for the full
    rationale; this one is trimmed to what turn submission needs:
    ``write()`` records every outbound line, parsed from JSON, and answers
    the SDK's own ``initialize`` control request so ``connect()`` unblocks.
    """

    def __init__(self):
        self.written = []
        self.connected = False
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        self.connected = True

    def is_ready(self) -> bool:
        return self.connected and not self.closed

    async def write(self, data: str) -> None:
        obj = json.loads(data)
        self.written.append(obj)
        if obj.get("type") == "control_request":
            subtype = obj["request"].get("subtype")
            if subtype == "initialize":
                self._queue.put_nowait(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": obj["request_id"],
                            "response": {},
                        },
                    }
                )

    async def read_messages(self):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(None)

    async def end_input(self) -> None:
        pass


class EventRecorder:
    """Collects every ``AgentEvent`` an ``SdkSession`` emits, in order."""

    def __init__(self):
        self.all = []

    def __call__(self, event) -> None:
        self.all.append(event)


async def _started_session(serve_dir):
    """A connected ``SdkSession`` over a fresh ``FakeTransport``, rooted at
    ``serve_dir``, with its events going to a fresh ``EventRecorder``."""
    transport = FakeTransport()
    recorder = EventRecorder()
    session = SdkSession(recorder, cwd=str(serve_dir), session_id="toy-sess-1", transport=transport)
    await session.start()
    assert session.agent_status() == AGENT_READY
    return session, transport, recorder


def _last_turn_content(transport):
    """The ``content`` list of the most recent ``user`` message written to
    the transport."""
    for obj in reversed(transport.written):
        if obj.get("type") == "user":
            return obj["message"]["content"]
    raise AssertionError("no user message was written to the transport")


def _png_bytes(payload=b"pixels"):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00" + payload


def _jpeg_bytes(payload=b"pixels"):
    return b"\xff\xd8\xff" + b"\x00" * 9 + payload


def _write_image(serve_dir, name, data):
    """Write ``data`` to ``serve_dir/images/<name>``, creating ``images/`` if
    needed, and return the ``"images/<name>"`` string an ``image_path``
    block carries for it."""
    images_dir = serve_dir / "images"
    images_dir.mkdir(exist_ok=True)
    (images_dir / name).write_bytes(data)
    return "images/%s" % name


# --- expand_turn_blocks called directly --------------------------------------
#
# The rest of this file goes through SdkSession and a fake Transport, which is
# what proves the expansion's result is actually what leaves the process. These
# few call it straight, because the reordering and the refusals are decisions
# about a list of dicts and need no client, no options and no transport to
# assert; a seam that can be tested without those is the reason this code sits
# in a module of its own. They are still written ``async def``, like every other
# test in a file carrying this module's asyncio mark: the mark applies to every
# test here, and on a synchronous one it does nothing except warn, so writing
# one would leave the file half claiming something it does not mean.


async def test_expansion_is_a_pure_function_of_blocks_and_a_directory(tmp_path):
    rel = _write_image(tmp_path, "pic.png", _png_bytes())
    out = turn_images.expand_turn_blocks(
        [{"type": "text", "text": "before"}, {"type": "image_path", "path": rel}], str(tmp_path)
    )
    assert [b["type"] for b in out] == ["image", "text", "text"]
    assert out[1]["text"] == "attached image: %s" % rel
    assert out[2]["text"] == "before"


async def test_a_text_only_list_comes_back_unchanged(tmp_path):
    blocks = [{"type": "text", "text": "no pictures here"}]
    assert turn_images.expand_turn_blocks(blocks, str(tmp_path)) == blocks


@pytest.mark.parametrize(
    "path",
    [
        "notes.json",
        "/etc/passwd",
        "images/../../etc/passwd",
        "images/",
        "images/sub/pic.png",
        None,
        12,
    ],
)
async def test_a_path_that_is_not_a_single_name_under_images_never_becomes_an_image(tmp_path, path):
    """Every one of these is refused on the string alone or by resolve_asset,
    and each becomes a note rather than an image, so a caller cannot turn an
    attachment reference into a read of something else."""
    out = turn_images.expand_turn_blocks([{"type": "image_path", "path": path}], str(tmp_path))
    assert all(b["type"] == "text" for b in out), out


# --- the pairing and its ordering (requirements 1 and 2) ---------------------


async def test_image_path_expands_to_an_image_block_and_path_text(tmp_path):
    """One accepted attachment becomes exactly two blocks: the inline base64
    image, and text naming the path the model can hand to its own tools."""
    rel = _write_image(tmp_path, "pic.png", _png_bytes())
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": rel},
                {"type": "text", "text": "look at this"},
            ]
        )
        content = _last_turn_content(transport)
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[1] == {"type": "text", "text": "attached image: %s" % rel}
        assert content[2] == {"type": "text", "text": "look at this"}
        assert len(content) == 3
    finally:
        await session.close()


async def test_the_image_block_precedes_the_text_that_refers_to_it(tmp_path):
    """Fact 6's ordering requirement, checked directly on the index of each
    block type rather than inferred from the pairing test above: an image
    block placed after the text that refers to it is the observed failure,
    where the model ignores the attachment and goes hunting the filesystem
    for a file matching the trailing words instead of looking at what it was
    handed."""
    rel = _write_image(tmp_path, "pic.png", _png_bytes())
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": rel},
                {"type": "text", "text": "what is this a picture of?"},
            ]
        )
        content = _last_turn_content(transport)
        image_index = next(i for i, b in enumerate(content) if b["type"] == "image")
        text_index = next(
            i
            for i, b in enumerate(content)
            if b["type"] == "text" and b["text"] == "what is this a picture of?"
        )
        assert image_index < text_index
    finally:
        await session.close()


async def test_two_attachments_both_expand_before_the_trailing_text(tmp_path):
    rel_a = _write_image(tmp_path, "a.png", _png_bytes(b"aaa"))
    rel_b = _write_image(tmp_path, "b.png", _png_bytes(b"bbb"))
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": rel_a},
                {"type": "image_path", "path": rel_b},
                {"type": "text", "text": "compare these two"},
            ]
        )
        content = _last_turn_content(transport)
        kinds = [b["type"] for b in content]
        assert kinds == ["image", "text", "image", "text", "text"]
        assert content[1] == {"type": "text", "text": "attached image: %s" % rel_a}
        assert content[3] == {"type": "text", "text": "attached image: %s" % rel_b}
        assert content[4] == {"type": "text", "text": "compare these two"}
    finally:
        await session.close()


# --- containment (requirement 3) --------------------------------------------


async def test_a_path_outside_images_is_refused_as_a_note_with_no_image_block(tmp_path):
    """An ``image_path`` that does not even carry the ``images/`` prefix is
    refused on the string alone, with no filesystem access."""
    session, transport, recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": "notes.json"},
                {"type": "text", "text": "hi"},
            ]
        )
        content = _last_turn_content(transport)
        assert not any(b["type"] == "image" for b in content)
        note = content[0]
        assert note["type"] == "text"
        assert "images/" in note["text"]
        assert content[-1] == {"type": "text", "text": "hi"}
        assert not [e for e in recorder.all if isinstance(e, AgentError)]
    finally:
        await session.close()


async def test_a_traversal_attempt_is_refused(tmp_path):
    """A second path component after ``images/`` (what
    ``images/../../etc/passwd`` becomes once the prefix is stripped) is
    refused as malformed before ``files.resolve_asset`` is even called."""
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": "images/../../etc/passwd"},
                {"type": "text", "text": "hi"},
            ]
        )
        content = _last_turn_content(transport)
        assert not any(b["type"] == "image" for b in content)
        assert "images/../../etc/passwd" in content[0]["text"]
        assert content[-1] == {"type": "text", "text": "hi"}
    finally:
        await session.close()


async def test_an_absent_file_is_noted_and_the_turn_still_sent(tmp_path):
    """Requirement 7: an absent file must not fail the turn."""
    (tmp_path / "images").mkdir()
    session, transport, recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": "images/missing.png"},
                {"type": "text", "text": "hi"},
            ]
        )
        content = _last_turn_content(transport)
        assert not any(b["type"] == "image" for b in content)
        assert content[-1] == {"type": "text", "text": "hi"}
        # The turn was sent and the session is still up: an unreadable
        # attachment is not an agent failure.
        assert session.agent_status() == AGENT_READY
        assert not [e for e in recorder.all if isinstance(e, AgentError)]
    finally:
        await session.close()


# --- caps (requirement 4) ----------------------------------------------------


async def test_default_caps_match_their_documented_values():
    """Pins the constants ``_expand_turn_blocks`` reads against the ceilings
    their comments justify them by.

    The inline cap is the one that matters most and is the least obvious: it
    comes from what the API accepts in one inline block, not from what may be
    stored, so tying it to the upload cap (which is larger on purpose) would
    silently reintroduce a turn the API refuses.
    """
    assert turn_images.MAX_TURN_IMAGES == 4
    assert turn_images.MAX_TURN_IMAGE_BYTES == 2 * files.MAX_INLINE_IMAGE_BYTES
    assert files.MAX_INLINE_IMAGE_BYTES < files.MAX_IMAGE_BYTES


async def test_an_image_over_the_inline_cap_is_named_rather_than_inlined(tmp_path):
    """A photograph larger than may be sent inline is stored, served and named,
    and only its pixels are withheld.

    Inlining it would put a base64 block past what the API accepts into the
    turn, which fails the whole turn rather than one attachment; dropping it
    silently would lose a file the human deliberately attached. The note names
    the path and the tool that can read it, so the model has somewhere to go.
    """
    big = _png_bytes(b"x" * (files.MAX_INLINE_IMAGE_BYTES + 1))
    rel = _write_image(tmp_path, "huge.png", big)
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": rel},
                {"type": "text", "text": "why is this wall thin?"},
            ]
        )
        content = _last_turn_content(transport)
        assert [b["type"] for b in content] == ["text", "text"]
        assert rel in content[0]["text"]
        assert "Read tool" in content[0]["text"]
        assert content[1] == {"type": "text", "text": "why is this wall thin?"}
    finally:
        await session.close()


async def test_an_image_at_the_inline_cap_is_still_inlined(tmp_path):
    """The boundary is inclusive, so a file exactly at the cap is a picture."""
    at_cap = _png_bytes()
    at_cap = at_cap + b"y" * (files.MAX_INLINE_IMAGE_BYTES - len(at_cap))
    assert len(at_cap) == files.MAX_INLINE_IMAGE_BYTES
    rel = _write_image(tmp_path, "exact.png", at_cap)
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn([{"type": "image_path", "path": rel}])
        content = _last_turn_content(transport)
        assert content[0]["type"] == "image"
    finally:
        await session.close()


async def test_a_blank_text_block_never_reaches_the_transport(tmp_path):
    """The API refuses an empty or whitespace-only text block and fails the
    request that carried it, so one reaching the transport costs the human the
    whole turn, image included. The composer sends an attachment with no typed
    message as exactly that shape, and a hand-assembled frame can too."""
    rel = _write_image(tmp_path, "pic.png", _png_bytes())
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": rel},
                {"type": "text", "text": ""},
                {"type": "text", "text": "   \n  "},
            ]
        )
        content = _last_turn_content(transport)
        assert all(b["type"] != "text" or b["text"].strip() for b in content), content
        assert content[0]["type"] == "image"
    finally:
        await session.close()


async def test_a_turn_of_nothing_but_blank_text_still_says_something(tmp_path):
    """An empty content array is refused exactly as an empty block is, so
    dropping every blank block cannot be allowed to leave nothing behind."""
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn([{"type": "text", "text": "  "}])
        content = _last_turn_content(transport)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"].strip()
    finally:
        await session.close()


async def test_the_overflow_note_is_bounded_by_what_it_names(tmp_path):
    """The note that reports dropped attachments must not itself become the
    amplification the caps exist to prevent.

    A ``turn`` frame is only checked for block shape, so a hand-built one can
    carry thousands of long paths inside the frame ceiling; a note quoting all
    of them lands in the single JSON line the turn is written as, and then in
    the API request. Long paths are truncated for the same reason, and because
    this is the one place an inbound string is copied into the model's context.
    """
    long_name = "z" * 4000
    blocks = [
        {"type": "image_path", "path": "images/%s-%d.png" % (long_name, i)} for i in range(400)
    ]
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(blocks)
        content = _last_turn_content(transport)
        rendered = json.dumps(content)
        assert len(rendered) < 20000, len(rendered)
        assert (
            "and %d more" % (400 - turn_images.MAX_TURN_IMAGES - turn_images._PATHS_NAMED_IN_NOTE)
            in rendered
        )
    finally:
        await session.close()


async def test_the_attachment_count_cap_drops_the_extras_with_one_note(tmp_path):
    rels = [
        _write_image(tmp_path, "img%d.png" % i, _png_bytes(str(i).encode()))
        for i in range(turn_images.MAX_TURN_IMAGES + 1)
    ]
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [{"type": "image_path", "path": rel} for rel in rels]
            + [{"type": "text", "text": "all of these"}]
        )
        content = _last_turn_content(transport)
        image_blocks = [b for b in content if b["type"] == "image"]
        assert len(image_blocks) == turn_images.MAX_TURN_IMAGES

        # The dropped attachment is named in a trailing note, ahead of the
        # human's own text (images-group-first still holds for the note).
        dropped_note = content[-2]
        assert dropped_note["type"] == "text"
        assert rels[-1] in dropped_note["text"]
        assert "1" in dropped_note["text"]
        assert content[-1] == {"type": "text", "text": "all of these"}
    finally:
        await session.close()


async def test_the_total_byte_cap_drops_an_attachment_that_would_exceed_it(tmp_path, monkeypatch):
    # A tiny budget, so this test writes bytes rather than megabytes: the
    # code path exercised (a running total checked in encounter order) does
    # not depend on the constant's real-world size, which is pinned
    # separately above.
    monkeypatch.setattr(turn_images, "MAX_TURN_IMAGE_BYTES", 20)
    rel_a = _write_image(tmp_path, "a.png", _png_bytes(b""))  # 12 bytes
    rel_b = _write_image(tmp_path, "b.png", _png_bytes(b""))  # 12 bytes
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn(
            [
                {"type": "image_path", "path": rel_a},
                {"type": "image_path", "path": rel_b},
                {"type": "text", "text": "both please"},
            ]
        )
        content = _last_turn_content(transport)
        image_blocks = [b for b in content if b["type"] == "image"]
        # 12 bytes fits the 20 byte budget; a second 12 bytes on top of that
        # (24 total) does not, so only the first is inlined.
        assert len(image_blocks) == 1
        path_notes = [b["text"] for b in content if b["type"] == "text"]
        assert any(rel_a in t and "attached image" in t for t in path_notes)
        assert any(rel_b in t and "budget" in t for t in path_notes)
        assert content[-1] == {"type": "text", "text": "both please"}
    finally:
        await session.close()


# --- media type from bytes, never from the suffix (requirement 5) -----------


async def test_media_type_comes_from_the_bytes_not_the_suffix(tmp_path):
    """A PNG saved with a ``.jpg`` name must still be reported as
    ``image/png``: the suffix is this project's own naming choice (for an
    upload) or nobody's choice at all (for a file placed by hand), and
    either way it is the weaker source next to the bytes themselves."""
    rel = _write_image(tmp_path, "actually_a_png.jpg", _png_bytes())
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn([{"type": "image_path", "path": rel}])
        content = _last_turn_content(transport)
        image_block = next(b for b in content if b["type"] == "image")
        assert image_block["source"]["media_type"] == "image/png"
    finally:
        await session.close()


async def test_jpeg_bytes_named_png_are_still_reported_as_jpeg(tmp_path):
    rel = _write_image(tmp_path, "actually_a_jpeg.png", _jpeg_bytes())
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn([{"type": "image_path", "path": rel}])
        content = _last_turn_content(transport)
        image_block = next(b for b in content if b["type"] == "image")
        assert image_block["source"]["media_type"] == "image/jpeg"
    finally:
        await session.close()


# --- the common case is untouched (requirement mirrors fact 6's scope) ------


async def test_a_text_only_turn_passes_through_unchanged(tmp_path):
    """No ``image_path`` block among ``blocks`` means nothing here has work
    to do: the text blocks reach the transport exactly as given, in their
    original order, which is what keeps the common case untouched by this
    expansion."""
    session, transport, _recorder = await _started_session(tmp_path)
    try:
        await session.submit_turn([{"type": "text", "text": "please look"}])
        content = _last_turn_content(transport)
        assert content == [{"type": "text", "text": "please look"}]
    finally:
        await session.close()


# --- no server-side record of a turn's blocks (requirement 6) ---------------


async def test_submitting_an_image_turn_emits_no_event(tmp_path):
    """There is no server-side record of a user turn's blocks to leak
    base64 into: ``ws.py`` never broadcasts an inbound ``turn`` frame to any
    connection, and no ``AgentEvent`` carries one either (see
    ``session/base.py``'s catalogue). ``submit_turn`` hands the expanded
    blocks only to ``ClaudeSDKClient.query()``, i.e. the transport, and
    calls ``self._emit`` for nothing else; this asserts that directly, by
    checking that a submitted turn produced no event at all before any
    response arrived."""
    rel = _write_image(tmp_path, "pic.png", _png_bytes())
    session, _transport, recorder = await _started_session(tmp_path)
    try:
        before = len(recorder.all)  # start() already emitted its own AgentStatus
        await session.submit_turn([{"type": "image_path", "path": rel}])
        assert recorder.all[before:] == []
    finally:
        await session.close()
