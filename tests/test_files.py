"""Unit tests for the static scan and file creation in ``files.py``,
independent of any HTTP route.

These exercise ``files.scan_static``, ``files.StaticIndex``,
``files.create_image_file`` and ``files.atomic_replace`` directly against a
real filesystem tree, so a defect in any of them is pinned at the layer it
lives in rather than only visible through a route's response.
"""

import os

import pytest

from annealage_agent import files

# --- scan_static: extension allowlist, symlinks, the vendor/LICENSE carve-out --


def test_scan_static_indexes_only_allowed_extensions(tmp_path):
    (tmp_path / "app.css").write_text("body {}")
    (tmp_path / "main.js").write_text("console.log(1);")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "notes.bak").write_text("not servable")
    (tmp_path / "source.map").write_text("not servable either")

    entries, truncated = files.scan_static(tmp_path)
    rels = {e["rel"] for e in entries}
    assert rels == {"app.css", "main.js", "data.json", "index.html"}
    assert truncated is False


def test_scan_static_license_requires_a_vendor_ancestor_directory(tmp_path):
    (tmp_path / "LICENSE").write_text("bare license, no vendor dir")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "LICENSE").write_text("vendored license")

    entries, _ = files.scan_static(tmp_path)
    rels = {e["rel"] for e in entries}
    assert rels == {"vendor/LICENSE"}


def test_scan_static_excludes_a_dotdir(tmp_path):
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "file.js").write_text("console.log(1);")

    entries, _ = files.scan_static(tmp_path)
    assert entries == []


def test_scan_static_refuses_a_symlink(tmp_path):
    real = tmp_path / "real.js"
    real.write_text("console.log('real');")
    (tmp_path / "evil.js").symlink_to(real)

    entries, _ = files.scan_static(tmp_path)
    rels = {e["rel"] for e in entries}
    assert rels == {"real.js"}


def test_scan_static_does_not_refuse_a_hardlink(tmp_path):
    # Deliberately the opposite of the rule for files taken from the served
    # directory: uv and pip install package files by hardlinking out of a
    # wheel cache, so a legitimately installed asset routinely has more than
    # one link, and refusing it would make a normal install unservable.
    one = tmp_path / "one.js"
    one.write_text("console.log(1);")
    try:
        os.link(one, tmp_path / "two.js")
    except OSError as exc:
        pytest.skip("cannot hardlink within this directory: %s" % exc)

    entries, truncated = files.scan_static(tmp_path)
    rels = {e["rel"] for e in entries}
    assert rels == {"one.js", "two.js"}
    assert truncated is False


def test_scan_static_cap_truncates_and_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(files, "MAX_STATIC_FILES", 2)
    for i in range(5):
        (tmp_path / ("f%d.js" % i)).write_text("console.log(%d);" % i)

    entries, truncated = files.scan_static(tmp_path)
    assert truncated is True
    assert len(entries) == 2
    assert "MAX_STATIC_FILES" in capsys.readouterr().err


# --- StaticIndex ------------------------------------------------------------


def test_static_index_content_type_of_known_and_extensionless_files():
    assert files.StaticIndex.content_type_of("app.css") == "text/css; charset=utf-8"
    assert files.StaticIndex.content_type_of("main.js") == "text/javascript; charset=utf-8"
    assert files.StaticIndex.content_type_of("data.json") == "application/json"
    assert files.StaticIndex.content_type_of("vendor/LICENSE") == "text/plain; charset=utf-8"


def test_static_index_by_rel_and_identity_of_absent_key(tmp_path):
    idx = files.build_static_index(tmp_path)
    assert idx.by_rel("does-not-exist.js") is None
    assert idx.identity_of("does-not-exist.js") is None


def test_build_static_index_resolves_present_files(tmp_path):
    (tmp_path / "main.js").write_text("console.log(1);")

    idx = files.build_static_index(tmp_path)
    assert idx.by_rel("main.js") == tmp_path / "main.js"
    assert idx.identity_of("main.js") is not None


# --- create_image_file: the one path that creates a file from a model's input --


def test_create_image_file_writes_into_images_and_makes_the_directory(tmp_path):
    fd, target = files.create_image_file(tmp_path, "front.png")
    try:
        os.write(fd, b"bytes")
    finally:
        os.close(fd)
    assert target == tmp_path / files.IMAGES_DIRNAME / "front.png"
    assert target.read_bytes() == b"bytes"
    assert oct(target.stat().st_mode)[-3:] == "644"


@pytest.mark.parametrize(
    "name",
    [
        "../escape.png",  # traversal
        "sub/front.png",  # a directory component
        ".hidden.png",  # a dotfile, which the scan excludes anyway
        "front.svg",  # an extension /asset would not serve
        "front",  # no extension at all
        "",  # nothing
        "front.png\x00.txt",  # a NUL, in case a lower layer truncates at it
    ],
)
def test_create_image_file_refuses_a_name_it_would_not_serve_back(tmp_path, name):
    """The whole containment check for a name that may come from the model, so
    it is a whitelist: a name accepted here is one /asset can hand back, and
    anything else is refused rather than sanitised into something adjacent."""
    assert files.create_image_file(tmp_path, name) is None


def test_create_image_file_refuses_a_symlinked_images_directory(tmp_path):
    """A link named images/ makes this a writer into whatever it points at,
    which is a way to have this process create files anywhere its user can."""
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / files.IMAGES_DIRNAME).symlink_to(outside, target_is_directory=True)

    assert files.create_image_file(project, "front.png") is None
    assert list(outside.iterdir()) == []


def test_create_image_file_raises_rather_than_overwriting(tmp_path):
    """O_EXCL, so the caller has to choose another name. Every image here is
    evidence of what the work looked like at some moment, and a silent
    overwrite loses one."""
    fd, _target = files.create_image_file(tmp_path, "front.png")
    os.close(fd)
    with pytest.raises(FileExistsError):
        files.create_image_file(tmp_path, "front.png")


# --- atomic_replace ---------------------------------------------------------


def test_atomic_replace_keeps_an_existing_files_mode(tmp_path):
    """mkstemp creates 0600 and os.replace carries the mode across, so without
    the chmod a deliberate mode would silently narrow on every write."""
    target = tmp_path / "notes.json"
    target.write_text("[]")
    os.chmod(target, 0o640)

    files.atomic_replace(target, b'["first"]')
    assert oct(target.stat().st_mode)[-3:] == "640"
    assert target.read_text() == '["first"]'


def test_atomic_replace_gives_a_new_file_the_default_mode(tmp_path):
    target = tmp_path / "fresh.json"
    files.atomic_replace(target, b"[]")
    assert oct(target.stat().st_mode)[-3:] == "644"


def test_atomic_replace_leaves_no_temporary_file_behind_on_failure(tmp_path):
    target = tmp_path / "record.json"
    with pytest.raises(TypeError):
        files.atomic_replace(target, "a str, not bytes")
    assert list(tmp_path.iterdir()) == []
