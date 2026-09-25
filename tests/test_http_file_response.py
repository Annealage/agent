"""Unit tests for the shared file-streaming response builder.

Exercises ``http.file_response`` directly, independent of any route, since
the open-before-headers and declared-length-matches-delivered-bytes
properties it provides are relied on by every route in this package that
serves file bytes.
"""

import pytest

from annealage_agent.http import file_response

pytestmark = pytest.mark.asyncio


async def _drain(res):
    chunks = []
    async for chunk in res.body:
        chunks.append(chunk)
    return b"".join(chunks)


async def test_file_response_streams_exact_bytes(tmp_path):
    target = tmp_path / "widget.stl"
    target.write_bytes(b"solid widget\nendsolid widget\n")

    res = await file_response(target, "application/vnd.ms-pki.stl")
    assert res.status_code == 200
    assert res.headers["Content-Length"] == str(target.stat().st_size)
    body = await _drain(res)
    assert body == target.read_bytes()


async def test_file_response_404_when_file_is_missing(tmp_path):
    res = await file_response(tmp_path / "does-not-exist.stl", "application/vnd.ms-pki.stl")
    assert res.status_code == 404
    assert "Content-Length" not in res.headers


async def test_file_response_404_when_file_is_unreadable(tmp_path):
    # A file that vanishes or becomes unreadable between a route's existence
    # check and this call (a background regeneration, or a permissions
    # change) must fail before any header is written, not after a 200 has
    # already been committed to the client.
    target = tmp_path / "noperm.stl"
    target.write_bytes(b"solid x\nendsolid x\n")
    target.chmod(0o000)
    try:
        res = await file_response(target, "application/vnd.ms-pki.stl")
    finally:
        target.chmod(0o644)
    assert res.status_code == 404


async def test_file_response_head_does_not_open_the_file(tmp_path, monkeypatch):
    # microdot's Response.write only iterates and closes the body when the
    # request is not HEAD, so a HEAD response built the same way as a GET
    # response would leave its async generator (and the file descriptor its
    # finally-block would have closed) unstarted, to be closed only once the
    # interpreter gets around to finalising the unread file object. Proven
    # here by making the file open itself an assertion failure.
    target = tmp_path / "widget.stl"
    target.write_bytes(b"solid widget\nendsolid widget\n")

    def _must_not_open(*args, **kwargs):
        raise AssertionError("file_response must not open the file for a HEAD request")

    monkeypatch.setattr("builtins.open", _must_not_open)

    res = await file_response(target, "application/vnd.ms-pki.stl", method="HEAD")
    assert res.status_code == 200
    assert res.body == b""
    assert res.headers["Content-Length"] == str(target.stat().st_size)


async def test_file_response_body_never_exceeds_the_declared_content_length(tmp_path):
    target = tmp_path / "growing.stl"
    target.write_bytes(b"a" * 100)

    res = await file_response(target, "application/vnd.ms-pki.stl")
    assert res.headers["Content-Length"] == "100"

    # The file grows, through a second file descriptor, after this response
    # has already committed to a length taken from the first. The body must
    # stop at that length regardless of what is now on disk.
    target.write_bytes(b"a" * 100 + b"b" * 900)
    body = await _drain(res)
    assert len(body) == 100
