"""In-process HTTP contract tests for ``GET /asset/<rel>``
(``http/routes_chat.py``): files under the served directory's ``images/``,
which is where uploads and the pane's screenshots land.

Unlike an indexed route, ``/asset`` resolves each request against the
filesystem (``files.resolve_asset``), so what is pinned here is that
resolve-and-contain check: every traversal shape, a symlink at a file or at
``images/`` itself, dot-prefixed components, the narrowed content types that
keep an uploaded file from being labelled as active content, and 404s that
never disclose a resolved path or which cause produced them.
"""

import pytest

pytestmark = pytest.mark.asyncio


async def test_asset_route_serves_from_images(client, served_dir):
    images = served_dir / "images"
    images.mkdir()
    (images / "photo.png").write_bytes(b"\x89PNG fake bytes")

    res = await client.get("/asset/photo.png")
    assert res.status_code == 200
    assert res.body == b"\x89PNG fake bytes"
    assert res.headers.get("Content-Type") == "image/png"


async def test_asset_route_404_when_images_dir_absent(client):
    res = await client.get("/asset/photo.png")
    assert res.status_code == 404


async def test_asset_route_refuses_file_outside_images(client, served_dir):
    (served_dir / "images").mkdir()
    (served_dir / "sibling.png").write_bytes(b"outside images/")

    res = await client.get("/asset/../sibling.png")
    assert res.status_code == 404


# --- containment: every traversal shape and a symlinked file ----------------


async def test_asset_route_refuses_every_traversal_shape(client, served_dir):
    # /asset/<rel> does a real filesystem resolve-and-contain check
    # (files.safe_join); these payloads exercise that check directly rather
    # than relying on an allowlist miss.
    images = served_dir / "images"
    images.mkdir()
    (images / "legit.png").write_bytes(b"legit image bytes")
    (served_dir / "secret.txt").write_text("do not leak")

    payloads = [
        "/asset/../secret.txt",  # ".." traversal
        "/asset/%2e%2e/secret.txt",  # URL-encoded traversal
        "/asset//../secret.txt",  # doubled slash
        "/asset/..\\secret.txt",  # backslash
        "/asset//" + str(served_dir / "secret.txt"),  # absolute path
    ]
    for path in payloads:
        res = await client.get(path)
        assert res.status_code == 404, path
        assert b"do not leak" not in (res.body or b"")


async def test_asset_route_refuses_symlink_outside_directory(client, served_dir, tmp_path_factory):
    outside_dir = tmp_path_factory.mktemp("outside")
    outside_secret = outside_dir / "outside-secret.png"
    outside_secret.write_bytes(b"outside png bytes")

    images = served_dir / "images"
    images.mkdir()
    (images / "escape.png").symlink_to(outside_secret)

    res = await client.get("/asset/escape.png")
    assert res.status_code == 404
    assert b"outside png bytes" not in (res.body or b"")


# --- percent-decoded paths ------------------------------------------------------


async def test_asset_route_decodes_space_in_filename(client, served_dir):
    images = served_dir / "images"
    images.mkdir()
    (images / "my photo.png").write_bytes(b"space photo bytes")

    res = await client.get("/asset/my%20photo.png")
    assert res.status_code == 200
    assert res.body == b"space photo bytes"


# --- images/ as a symlink -------------------------------------------------------


async def test_asset_route_refuses_a_symlinked_images_directory(client, served_dir):
    # A symlink at images/ cannot be made safe by checking where it points,
    # because containment is satisfied by the served directory itself: an
    # "images -> ." link passes that test and then becomes the base every
    # /asset request is joined against, which restores the serve-anything
    # fallback this route replaces. Pointing it at a subdirectory is no
    # better, since nothing there was indexed.
    real_images = served_dir / "real_images"
    real_images.mkdir()
    (real_images / "photo.png").write_bytes(b"\x89PNG real bytes")
    (served_dir / "images").symlink_to(real_images)

    res = await client.get("/asset/photo.png")
    assert res.status_code == 404
    assert b"real bytes" not in (res.body or b"")


async def test_asset_route_refuses_images_symlinked_to_the_served_dir(client, served_dir):
    # The specific shape that defeats a containment-only check.
    (served_dir / "secret.txt").write_text("TOPSECRET-FLAG-ASSET")
    (served_dir / "images").symlink_to(served_dir)

    res = await client.get("/asset/secret.txt")
    assert res.status_code == 404
    assert b"TOPSECRET-FLAG-ASSET" not in (res.body or b"")


async def test_asset_route_refuses_images_symlink_outside_served_dir(
    client, served_dir, tmp_path_factory
):
    # An images/ symlink whose target is outside the served directory (a
    # shape that arrives inside a zip, a tarball or a git clone, not only
    # by an operator's own hand) must not turn every /asset request into a
    # read of anything under that other location.
    outside_dir = tmp_path_factory.mktemp("outside")
    (outside_dir / "secret.png").write_bytes(b"outside png bytes")
    (served_dir / "images").symlink_to(outside_dir)

    res = await client.get("/asset/secret.png")
    assert res.status_code == 404
    assert b"outside png bytes" not in (res.body or b"")


# --- content-type restriction and dotdir exclusion -----------------------------


async def test_asset_route_serves_non_image_extensions_as_octet_stream(client, served_dir):
    # images/ can contain whatever a reviewed bundle happened to ship. A
    # file saved with an ".html" or ".svg" extension must never be labelled
    # as active content on this server's own origin, regardless of what it
    # actually contains, since that label is what would let a browser run
    # it as script.
    images = served_dir / "images"
    images.mkdir()
    (images / "evil.html").write_text("<script>alert(1)</script>")
    (images / "evil.svg").write_text("<svg onload='alert(1)'></svg>")

    for name in ("evil.html", "evil.svg"):
        res = await client.get("/asset/" + name)
        assert res.status_code == 200
        assert res.headers.get("Content-Type") == "application/octet-stream"


async def test_asset_route_excludes_dot_prefixed_path_components(client, served_dir):
    hidden = served_dir / "images" / "sub" / ".secretdir"
    hidden.mkdir(parents=True)
    (hidden / "x.txt").write_text("do not leak")

    res = await client.get("/asset/sub/.secretdir/x.txt")
    assert res.status_code == 404
    assert b"do not leak" not in (res.body or b"")


# --- 404s do not disclose the resolved path or existence ------------------------


async def test_asset_404_does_not_disclose_resolved_path_or_existence(client, served_dir):
    # An absent name and a present-but-unreadable one must produce the same
    # template ("not found: <the name the client asked for>"), so neither
    # echoes the server's absolute filesystem path, and the only thing that
    # varies between the two responses is the name the client itself
    # supplied, not a signal of which cause produced the 404.
    images = served_dir / "images"
    images.mkdir()
    unreadable = images / "noperm.png"
    unreadable.write_bytes(b"\x89PNG bytes")
    unreadable.chmod(0o000)
    try:
        res_unreadable = await client.get("/asset/noperm.png")
    finally:
        unreadable.chmod(0o644)
    res_absent = await client.get("/asset/does-not-exist.png")

    assert res_unreadable.status_code == 404
    assert res_absent.status_code == 404
    assert res_unreadable.body == b"not found: noperm.png"
    assert res_absent.body == b"not found: does-not-exist.png"
    assert str(unreadable) not in (res_unreadable.text or "")
