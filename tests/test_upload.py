"""In-process HTTP contract tests for ``POST /upload``.

Exercises the route registered by ``http/routes_chat.py`` through microdot's
``TestClient``. Covers the token gate ``/upload`` shares with ``/ws``; the
``kind`` whitelist that keeps a client's own naming choices out of the
written filename; the streaming Content-Length checks a non-buffered body
requires (missing, over the image cap, and shorter than declared); sniffing
that decides an upload's type from its bytes rather than its header or its
declared kind; and the containment and collision-retry behaviour
``create_unique_image_file`` and ``create_image_file`` enforce on
``images/``.
"""

import asyncio
import os
from pathlib import Path

import pytest
from conftest import DEFAULT_PORT, TEST_AUTHORITY, TEST_HOST, create_toy_app, make_test_client
from microdot import Request

from annealage_agent import files
from annealage_agent.http import routes_chat

pytestmark = pytest.mark.asyncio

TOKEN = "the-real-upload-token-Value_123"


@pytest.fixture
def make_client(served_dir):
    """Build a ``TestClient`` for ``served_dir``, with a token of the caller's
    choosing. A factory rather than a fixture returning one client, so a test
    of the no-token-configured case can ask for ``token=None`` without a
    second fixture."""

    def _make(*, token=TOKEN):
        return make_test_client(
            create_toy_app(served_dir, token=token, host=TEST_HOST, port=DEFAULT_PORT)
        )

    return _make


def _upload_url(token=TOKEN, kind=None):
    query = "t=%s" % token
    if kind is not None:
        query += "&kind=%s" % kind
    return "/upload?%s" % query


def _png_bytes(payload=b"pixels"):
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00" + payload


def _jpeg_bytes(payload=b"pixels"):
    return b"\xff\xd8\xff" + b"\x00" * 9 + payload


def _webp_bytes(payload=b"pixels"):
    return b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + payload


# --- token gate --------------------------------------------------------------


async def test_upload_refused_when_no_token_configured(make_client):
    """A server started with no token must refuse /upload, not treat an absent check as open."""
    client = make_client(token=None)
    res = await client.post("/upload", body=_png_bytes())
    assert res.status_code == 403
    assert res.body == b"forbidden"


async def test_upload_refused_with_no_token_supplied(make_client):
    """A configured token refuses a request that supplies none at all."""
    client = make_client()
    res = await client.post("/upload", body=_png_bytes())
    assert res.status_code == 403


async def test_upload_refused_with_wrong_token(make_client):
    """A wrong token is refused the same way a missing one is."""
    client = make_client()
    res = await client.post(_upload_url(token="wrong-value"), body=_png_bytes())
    assert res.status_code == 403


async def test_upload_accepted_with_correct_token(make_client):
    """The configured token is actually accepted, so the three refusal tests
    above are not passing merely because every request is refused."""
    client = make_client()
    res = await client.post(_upload_url(), body=_png_bytes())
    assert res.status_code == 200
    assert res.json["ok"] is True


# --- the kind query parameter is a whitelist, never a filename ---------------


@pytest.mark.parametrize("kind", ["../evil", ".hidden", "not-a-real-kind"])
async def test_upload_kind_whitelist_refuses_traversal_and_dotfile_and_junk(
    make_client, served_dir, kind
):
    """kind is checked against a fixed whitelist before any byte is read, so
    neither a traversal string nor a dotfile string ever reaches the name
    generator, and nothing is written for any of them."""
    client = make_client()
    res = await client.post(_upload_url(kind=kind), body=_png_bytes())
    assert res.status_code == 400
    assert res.json["ok"] is False
    assert "kind must be one of: %s" % ", ".join(routes_chat.upload_kinds()) == res.json["error"]
    assert not (served_dir / "images").exists()


async def test_upload_refuses_an_unknown_query_parameter(make_client, served_dir):
    """Only ``t`` and ``kind`` may appear.

    The whitelist is what keeps a parameter naming the written file from being
    introduced later: a ``filename`` a client supplied would be exactly the
    input plan section 3.10 item 6 exists to exclude, and a route that ignored
    unknown parameters would accept one silently the day someone added it to a
    caller.
    """
    client = make_client()
    res = await client.post(_upload_url() + "&filename=../../etc/passwd", body=_png_bytes())
    assert res.status_code == 400
    assert res.json["ok"] is False
    assert "filename" in res.json["error"]
    assert not (served_dir / "images").exists()


async def test_upload_kind_given_twice_refused(make_client):
    """kind supplied more than once is refused outright rather than resolved
    to one of the two values by whichever parser happens to look first."""
    client = make_client()
    res = await client.post(_upload_url(kind="markup") + "&kind=upload", body=_png_bytes())
    assert res.status_code == 400
    assert res.json["ok"] is False


# --- Content-Length checks the non-buffered body now requires ----------------


async def test_upload_with_no_content_length_refused_with_json_error(make_client, served_dir):
    """A request with no declared length is refused with a clear JSON error
    rather than the route attempting to read a stream with no known end."""
    client = make_client()
    res = await client.post(_upload_url(), body=b"")
    assert res.status_code == 411
    assert res.json["ok"] is False
    assert "Content-Length" in res.json["error"]
    assert not (served_dir / "images").exists()


async def test_upload_over_max_image_bytes_refused_with_nothing_on_disk(make_client, served_dir):
    """A declared Content-Length over the image cap is refused before any
    byte is read or any file created, whichever layer (this route's own
    check or microdot's process-wide one) answers it."""
    client = make_client()
    oversized = files.MAX_IMAGE_BYTES + 1
    res = await client.post(
        _upload_url(), headers={"Content-Length": str(oversized)}, body=_png_bytes()
    )
    assert res.status_code == 413
    assert not (served_dir / "images").exists()


async def test_upload_over_its_own_image_cap_refused_by_the_route_itself(
    make_client, served_dir, monkeypatch
):
    """/upload's own Content-Length check fires on its own terms, not only
    when it happens to coincide with microdot's process-wide ceiling."""
    monkeypatch.setattr(files, "MAX_IMAGE_BYTES", 32)
    client = make_client()
    res = await client.post(_upload_url(), headers={"Content-Length": "1000"}, body=_png_bytes())
    assert res.status_code == 413
    assert res.json["ok"] is False
    assert "byte image limit" in res.json["error"]
    assert not (served_dir / "images").exists()


async def test_upload_with_lying_content_length_refused_by_the_counter(make_client, served_dir):
    """A declared Content-Length longer than what the body actually delivers
    is caught by counting bytes as they arrive, not trusted from the header,
    and the file created for the (genuine) image bytes already received is
    removed rather than left behind half-written."""
    client = make_client()
    body = _png_bytes(payload=b"short")
    res = await client.post(_upload_url(), headers={"Content-Length": "1000"}, body=body)
    assert res.status_code == 400
    assert res.json["ok"] is False
    assert "%d of 1000 bytes" % len(body) in res.json["error"]
    images_dir = served_dir / "images"
    assert not images_dir.exists() or list(images_dir.iterdir()) == []


# --- sniffing decides the type, never the header or the declared kind --------


@pytest.mark.parametrize(
    "make_bytes,suffix,media_type",
    [
        (_png_bytes, ".png", "image/png"),
        (_jpeg_bytes, ".jpg", "image/jpeg"),
        (_webp_bytes, ".webp", "image/webp"),
    ],
    ids=["png", "jpeg", "webp"],
)
async def test_upload_accepts_each_image_type_named_by_its_own_bytes(
    make_client, served_dir, make_bytes, suffix, media_type
):
    """Each of the three accepted formats is written with the extension and
    media type its own magic bytes name, is fetchable back through /asset
    with that Content-Type, carries mode 0644, and is named by the server
    alone (a "upload-" slug, never any part of a client-supplied name)."""
    client = make_client()
    data = make_bytes()
    res = await client.post(_upload_url(), body=data)
    assert res.status_code == 200
    body = res.json
    assert body["ok"] is True
    assert body["media_type"] == media_type
    assert body["path"].startswith("images/upload-")
    assert body["path"].endswith(suffix)
    assert body["url"] == "/asset/" + Path(body["path"]).name

    on_disk = served_dir / body["path"]
    assert on_disk.read_bytes() == data
    assert on_disk.stat().st_mode & 0o777 == 0o644

    fetched = await client.get(body["url"])
    assert fetched.status_code == 200
    assert fetched.headers.get("Content-Type") == media_type
    assert fetched.body == data


async def test_upload_refuses_a_body_whose_bytes_do_not_match_its_claimed_type(make_client):
    """A Content-Type header claiming an image is never trusted; only the
    bytes decide, so a mislabelled body is refused."""
    client = make_client()
    res = await client.post(
        _upload_url(),
        headers={"Content-Type": "image/png"},
        body=b"not actually a png, just claiming to be one",
    )
    assert res.status_code == 415
    assert res.json["ok"] is False
    assert "PNG" in res.json["error"] and "JPEG" in res.json["error"]


@pytest.mark.parametrize(
    "body",
    [
        b"<script>alert(document.cookie)</script>",
        b"<svg onload='alert(1)'><rect/></svg>",
    ],
    ids=["html", "svg"],
)
async def test_upload_refuses_html_or_svg_bodies(make_client, served_dir, body):
    """A body that is actual markup, not an image mislabelled as one, is
    refused the same way, and nothing lands in images/."""
    client = make_client()
    res = await client.post(_upload_url(), body=body)
    assert res.status_code == 415
    assert not (served_dir / "images").exists()


# --- images/ handling ----------------------------------------------------------


async def test_upload_creates_images_directory_when_absent(make_client, served_dir):
    """images/ is created on demand by a first upload, not required to
    pre-exist."""
    assert not (served_dir / "images").exists()
    client = make_client()
    res = await client.post(_upload_url(), body=_png_bytes())
    assert res.status_code == 200
    assert (served_dir / "images").is_dir()


async def test_upload_refuses_a_symlinked_images_directory(
    make_client, served_dir, tmp_path_factory
):
    """A symlink at images/ is refused outright, matching /asset's own rule,
    rather than being written through to wherever it points."""
    real_images = tmp_path_factory.mktemp("real_images")
    (served_dir / "images").symlink_to(real_images)

    client = make_client()
    res = await client.post(_upload_url(), body=_png_bytes())
    assert res.status_code == 500
    assert res.json["ok"] is False
    assert res.json["error"] == (
        "refusing to write: images/ must be a real directory this process can create a file in"
    )
    assert list(real_images.iterdir()) == []


async def test_upload_retries_past_a_name_collision_rather_than_overwriting(
    make_client, served_dir, monkeypatch
):
    """A name already taken under images/ makes create_unique_image_file try
    again with a fresh random suffix, producing a second file rather than
    overwriting the first."""
    monkeypatch.setattr(files.time, "strftime", lambda fmt: "20260101-000000")
    hex_values = iter(["deadbeef", "c0ffee11"])
    monkeypatch.setattr(files.secrets, "token_hex", lambda n: next(hex_values))

    colliding_name = "upload-20260101-000000-deadbeef.png"
    fd, target = files.create_image_file(served_dir, colliding_name)
    os.write(fd, b"already here, must survive untouched")
    os.close(fd)

    client = make_client()
    res = await client.post(_upload_url(), body=_png_bytes())
    assert res.status_code == 200
    assert res.json["path"] == "images/upload-20260101-000000-c0ffee11.png"
    assert target.read_bytes() == b"already here, must survive untouched"
    assert (served_dir / res.json["path"]).read_bytes() == _png_bytes()


# --- refusals that must leave nothing behind ---------------------------------


class _ResetAfter:
    """A reader that yields a request head and then raises ``exc``.

    ``Request.create`` reads the head line by line and, once the body is not
    buffered, the route reads the body from this same object, so raising on a
    later read is exactly what a peer disappearing mid-upload does to the
    route: the request has already been dispatched and a file already exists.
    """

    def __init__(self, head, exc):
        self._buffer = head
        self._exc = exc
        self.reads_after_head = 0

    async def readline(self):
        index = self._buffer.find(b"\r\n")
        if index == -1:
            line, self._buffer = self._buffer, b""
            return line
        line, self._buffer = self._buffer[: index + 2], self._buffer[index + 2 :]
        return line

    async def read(self, n=-1):
        if self._buffer:
            chunk = self._buffer if n in (-1, None) else self._buffer[:n]
            self._buffer = self._buffer[len(chunk) :]
            return chunk
        self.reads_after_head += 1
        raise self._exc

    async def readexactly(self, n):
        chunk = self._buffer[:n]
        self._buffer = self._buffer[n:]
        if len(chunk) < n:
            raise EOFError
        return chunk


def _open_descriptor_count():
    """How many descriptors this process holds, or skip where that is unknowable.

    ``/dev/fd`` is the one path both Linux and macOS expose for this; Windows
    exposes no equivalent, and a leak check that cannot count descriptors has
    nothing to assert, so it skips rather than passing vacuously.
    """
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        pytest.skip("no /dev/fd on this platform, so open descriptors cannot be counted")


class _NullWriter:
    def write(self, data):
        pass

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass

    def get_extra_info(self, name, default=None):
        return default


async def _dispatch_aborted_upload(app, served_dir, exc):
    """Drive one /upload request whose body stops partway with ``exc`` raised.

    Goes through ``Request.create`` and ``dispatch_request`` rather than
    ``TestClient``, because the client always delivers the body it was given
    and the failure under test is a body that stops arriving.
    """
    declared = 5000
    head = (
        b"POST " + _upload_url().encode() + b" HTTP/1.1\r\n"
        b"Host: " + TEST_AUTHORITY.encode() + b"\r\n"
        b"Content-Length: " + str(declared).encode() + b"\r\n"
        b"\r\n" + _png_bytes(b"a" * 100)
    )
    reader = _ResetAfter(head, exc)
    req = await Request.create(app, reader, _NullWriter(), ("127.0.0.1", 1234))
    # microdot catches Exception and answers 500; a CancelledError is a
    # BaseException and reaches this caller instead. Which of the two happens is
    # microdot's business, and neither says anything about the descriptor or the
    # file, which is what this returns for the caller to check.
    try:
        await app.dispatch_request(req)
    except BaseException as raised:
        assert isinstance(raised, type(exc)), raised
    return reader


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionResetError("peer went away"),
        asyncio.CancelledError(),
    ],
)
async def test_an_upload_that_dies_mid_body_leaves_no_file_and_no_descriptor(
    make_client, served_dir, exc
):
    """A tab closed, a fetch aborted or a link dropped mid-upload raises out of
    the stream read, past every refusal the route decides for itself.

    Left unhandled it leaks one write descriptor and one truncated image per
    occurrence: the descriptors end at EMFILE, after which no upload succeeds
    again, and the images stay in a git-tracked directory that /asset serves
    and the model can attach later. Both counted here, because either one alone
    passing would still be a leak.
    """
    client = make_client()
    before = _open_descriptor_count()
    reader = await _dispatch_aborted_upload(client.app, served_dir, exc)

    assert reader.reads_after_head == 1, "the route never reached the failing read"
    assert list((served_dir / "images").glob("*")) == [], "a truncated image was left under images/"
    assert _open_descriptor_count() == before, "the write descriptor was not closed"


# --- Origin, which /ws also enforces ----------------------------------------


async def test_upload_refuses_a_cross_origin_post(make_client):
    """A POST carrying a raw body is a CORS simple request, so no preflight
    stands in front of this route; without its own Origin check a page on any
    other origin holding the token could write files into the human's project
    through the one route that writes at all."""
    client = make_client()
    res = await client.post(
        _upload_url(), headers={"Origin": "https://evil.example"}, body=_png_bytes()
    )
    assert res.status_code == 403


async def test_upload_accepts_the_viewers_own_origin(make_client, served_dir):
    """The check must admit the origin the viewer is actually served from, or
    it would refuse every real upload."""
    client = make_client()
    res = await client.post(
        _upload_url(), headers={"Origin": "http://" + TEST_AUTHORITY}, body=_png_bytes()
    )
    assert res.status_code == 200
    assert (served_dir / res.json["path"]).is_file()


# --- what a refusal costs ---------------------------------------------------


async def test_unrecognised_bytes_are_refused_without_reading_the_whole_body(
    make_client, served_dir
):
    """The sniffer looks at twelve bytes and no more, so a body declared at the
    cap whose head is already not an image must be refused on that head.

    Reading to the end first would let any holder of the token make this
    process buffer the whole cap per connection with junk, and nothing here
    caps concurrent connections or times out a slow read.
    """
    app = create_toy_app(served_dir, token=TOKEN, host=TEST_HOST, port=DEFAULT_PORT)
    declared = files.MAX_IMAGE_BYTES
    head = (
        b"POST " + _upload_url().encode() + b" HTTP/1.1\r\n"
        b"Host: " + TEST_AUTHORITY.encode() + b"\r\n"
        b"Content-Length: " + str(declared).encode() + b"\r\n"
        b"\r\n" + b"<html>not an image at all</html>"
    )
    # Raises if the route reads past the head, which is the failure under test.
    reader = _ResetAfter(head, AssertionError("read past the sniff window"))
    req = await Request.create(app, reader, _NullWriter(), ("127.0.0.1", 1234))
    res = await app.dispatch_request(req)
    assert res.status_code == 415
    assert reader.reads_after_head == 0
    assert list((served_dir / "images").glob("*")) == []


async def test_a_directory_that_cannot_be_written_answers_json(make_client, served_dir):
    """A read-only images/ (and equally a full filesystem or a spent quota)
    must answer the JSON shape every other failure here answers, not microdot's
    plain-text 500 with a traceback: the pane parses JSON, and a caller that
    gets none can only say the upload failed for no stated reason."""
    images = served_dir / files.IMAGES_DIRNAME
    images.mkdir()
    images.chmod(0o500)
    try:
        client = make_client()
        res = await client.post(_upload_url(), body=_png_bytes())
        assert res.status_code == 500
        assert res.json["ok"] is False
        assert "could not create the file" in res.json["error"]
    finally:
        images.chmod(0o700)
