"""Packaging contract tests: what actually ships in the published release.

Exercises plain ``uv build`` against the real project tree: it builds an sdist
and then builds the wheel from that sdist, not from the working tree. A
``--wheel``-only build exercises a different hatchling code path and would not
catch the sdist target missing the artifact override the wheel target has, so
both built files are inspected here rather than parsing ``pyproject.toml``,
since the property under test is hatchling's own VCS-ignore filtering of
package contents, not the config file's syntax.
"""

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "src" / "annealage_agent" / "static"

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")


def _build(out_dir):
    """Build the sdist and, from it, the wheel; return (sdist_names, wheel_names).

    ``sdist_names`` are ``tarfile`` member names with the
    ``<project>-<version>/`` directory hatchling wraps the sdist contents in
    stripped off, so they are paths relative to the project root.
    ``wheel_names`` are ``zipfile`` member names, unprefixed.
    """
    subprocess.run(
        ["uv", "build", "-o", str(out_dir), str(REPO_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    sdists = list(Path(out_dir).glob("*.tar.gz"))
    wheels = list(Path(out_dir).glob("*.whl"))
    assert len(sdists) == 1, sdists
    assert len(wheels) == 1, wheels
    with tarfile.open(sdists[0]) as tf:
        sdist_names = [name.partition("/")[2] for name in tf.getnames()]
    with zipfile.ZipFile(wheels[0]) as zf:
        wheel_names = zf.namelist()
    return sdist_names, wheel_names


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("dist"))


def _static_files():
    """Every file of the front end, relative to ``src/``."""
    files = sorted(
        path.relative_to(STATIC_DIR.parent.parent).as_posix()
        for path in STATIC_DIR.rglob("*")
        if path.is_file()
    )
    # Guards the check itself: a moved tree would otherwise make the tests
    # below pass over zero files.
    assert "annealage_agent/static/chat.js" in files
    assert "annealage_agent/static/agent.css" in files
    return files


def test_wheel_ships_every_static_file(built):
    # The chat pane is a tree of ES modules and a stylesheet, and a product's
    # page that loads it is inert without every one of them: a missing module
    # fails its importer at load time, with no test noticing until someone
    # opened the page in a browser.
    _, wheel_names = built
    for rel in _static_files():
        assert rel in wheel_names, rel


def test_sdist_ships_every_static_file_and_the_license(built):
    sdist_names, _ = built
    for rel in _static_files():
        assert "src/" + rel in sdist_names, rel
    assert "LICENSE" in sdist_names


def test_sdist_and_wheel_ship_a_static_file_under_a_gitignored_directory_name(tmp_path):
    # "lib/" is one of this repo's .gitignore entries (the standard
    # Python-template line for a venv's lib/ directory). A front-end module
    # living under static/lib/ must still ship despite that name collision, in
    # both the sdist and the wheel built from it. Writes a throwaway probe
    # file into the real static/ tree for the duration of one build, since the
    # property under test is hatchling's filtering against the repo's actual
    # .gitignore, not a copy of it.
    probe_dir = STATIC_DIR / "lib"
    probe_dir.mkdir()
    probe_file = probe_dir / "probe.js"
    probe_file.write_text("// packaging test probe, safe to delete\n")
    try:
        sdist_names, wheel_names = _build(tmp_path / "dist")
    finally:
        probe_file.unlink()
        probe_dir.rmdir()
    assert "src/annealage_agent/static/lib/probe.js" in sdist_names
    assert "annealage_agent/static/lib/probe.js" in wheel_names
