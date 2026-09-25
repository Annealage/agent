"""Tests for the credential-path refusal in ``session/secret_paths.py``.

Three groups, and the third is the point of the file as much as the first two.

What must be refused: the direct spelling, a relative path, a symlink planted
inside the project, and a bash command naming the path in any of the forms a
shell expands itself. Writes as well as reads, since a write into
``~/.ssh/authorized_keys`` is the worse outcome of the two.

What must not be refused: everything else. A control that fires on ordinary work
gets turned off, so the tests that pin non-interference are load-bearing rather
than decorative, and they include the paths that merely look similar.

What is known not to be covered: the bash check is textual, so a glob, an
encoding, or a command that walks to the directory before naming the file gets
through. A path held in a variable happens to be caught when the assignment
spells it out, which is incidental rather than an understanding of shell
variables. Those cases are asserted as *allowed*, so
the limit is written down in the suite rather than only in a docstring, and so a
later change that claims to close the gap has to update a test that says it is
open.

``HOME`` is redirected per test, so nothing here reads or refuses against the
developer's own credential directories.
"""

import os
from pathlib import Path

import pytest

from annealage_agent.session import secret_paths


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """A home directory under ``tmp_path``, with the denied entries present.

    Present rather than absent because the interesting comparisons involve
    ``realpath``, which resolves what exists; the separate case of a denied root
    that does not exist on this machine has its own test.
    """
    home = tmp_path / "home" / "someone"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("private", encoding="utf-8")
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("keys", encoding="utf-8")
    (home / ".config" / "gh").mkdir(parents=True)
    (home / ".claude").mkdir()
    (home / ".claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    (home / ".netrc").write_text("login", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture
def project(tmp_path):
    """A served directory, which is what relative paths resolve against."""
    work = tmp_path / "part"
    work.mkdir()
    (work / "notes.json").write_text('["first"]', encoding="utf-8")
    return work


# --- what must be refused ----------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/id_ed25519",
        ".ssh",
        ".aws/credentials",
        ".config/gcloud/credentials.db",
        ".kube/config",
        ".gnupg/secring.gpg",
        ".netrc",
        ".docker/config.json",
        ".config/gh/hosts.yml",
        ".claude/.credentials.json",
    ],
)
def test_reading_a_denied_path_is_refused(fake_home, project, relative):
    reason = secret_paths.refusal("Read", {"file_path": str(fake_home / relative)}, project)
    assert reason is not None
    assert "credential" in reason


def test_the_refusal_names_the_directory_and_tells_the_model_not_to_route_around_it(
    fake_home, project
):
    """A hook's deny reason reaches the model verbatim, so it is the only place
    to say "this is about the path, not the spelling"; without that a refusal
    reads as transient and the next call is the same call rewritten."""
    reason = secret_paths.refusal(
        "Read", {"file_path": str(fake_home / ".ssh" / "id_ed25519")}, project
    )
    assert str(fake_home / ".ssh") in reason
    assert "do not look for another way" in reason
    assert "let them fetch it themselves" in reason


def test_a_tilde_path_is_refused(fake_home, project):
    reason = secret_paths.refusal("Read", {"file_path": "~/.ssh/id_ed25519"}, project)
    assert reason is not None


def test_a_relative_path_climbing_out_of_the_project_is_refused(fake_home, project):
    """Resolved against the served directory, the way the tool itself would."""
    relative = os.path.relpath(fake_home / ".ssh" / "id_ed25519", start=project)
    reason = secret_paths.refusal("Read", {"file_path": relative}, project)
    assert reason is not None


def test_a_symlink_inside_the_project_pointing_at_a_denied_path_is_refused(fake_home, project):
    """The case a string comparison on the argument would miss, and the one an
    attacker with write access to the served directory can actually arrange."""
    (project / "notes").symlink_to(fake_home / ".ssh")
    reason = secret_paths.refusal(
        "Read", {"file_path": str(project / "notes" / "id_ed25519")}, project
    )
    assert reason is not None


@pytest.mark.parametrize("argument", ["file_path", "path", "notebook_path", "filePath"])
def test_every_path_argument_is_checked(fake_home, project, argument):
    reason = secret_paths.refusal("Grep", {argument: str(fake_home / ".aws")}, project)
    assert reason is not None


@pytest.mark.parametrize("tool", ["Read", "Write", "Edit", "NotebookEdit", "Grep", "Glob"])
def test_the_check_is_on_the_path_not_on_which_tool_asked(fake_home, project, tool):
    """Writing into ``~/.ssh/authorized_keys`` is worse than reading a key, so
    the same list covers both directions."""
    reason = secret_paths.refusal(
        tool, {"file_path": str(fake_home / ".ssh" / "authorized_keys")}, project
    )
    assert reason is not None


def test_a_denied_root_that_does_not_exist_is_still_refused(fake_home, project):
    """``~/.kube`` is absent from this fixture's home. It can be created later in
    the run, and refusing a path that holds nothing costs nothing."""
    assert not (fake_home / ".kube").exists()
    reason = secret_paths.refusal(
        "Read", {"file_path": str(fake_home / ".kube" / "config")}, project
    )
    assert reason is not None


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.ssh/id_ed25519",
        "cat $HOME/.ssh/id_ed25519",
        "cat ${HOME}/.ssh/id_ed25519",
        "grep -r secret ~/.aws",
        "cp ~/.claude/.credentials.json /tmp/x",
        # Caught, though it reaches the file through a variable: the check is on
        # the text of the command, and the path is spelled out in the assignment.
        # Nothing here understands shell variables, so this is incidental rather
        # than a claim about indirection generally.
        "P=~/.ssh; cat $P/id_ed25519",
    ],
)
def test_a_bash_command_naming_a_denied_path_is_refused(fake_home, project, command):
    reason = secret_paths.refusal("Bash", {"command": command}, project)
    assert reason is not None
    assert "not an answer to it" in reason


def test_the_absolute_spelling_is_refused_in_bash(fake_home, project):
    reason = secret_paths.refusal(
        "Bash", {"command": "cat %s/id_ed25519" % (fake_home / ".ssh")}, project
    )
    assert reason is not None


# --- what must not be refused ------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "notes.json",
        "models/part.stl",
        "CLAUDE.md",
        ".toy/config.toml",
        "images/markup-1.png",
    ],
)
def test_ordinary_project_paths_are_not_refused(fake_home, project, path):
    assert secret_paths.refusal("Read", {"file_path": path}, project) is None


@pytest.mark.parametrize(
    "relative",
    [
        ".sshfoo/notes.txt",
        ".config/annealage-toy/settings.toml",
        ".config/ghost/config",
        ".claude/CLAUDE.md",
        ".awsome/file",
        "documents/netrc-notes.md",
    ],
)
def test_paths_that_only_resemble_a_denied_one_are_not_refused(fake_home, project, relative):
    """The list is prefix-matched on path components, not on substrings, so
    ``.sshfoo`` is not ``.ssh`` and ``.claude/CLAUDE.md`` is not the credentials
    file next to it. A product's own settings live under ``~/.config`` (the
    toy's in ``annealage-toy``) and must stay readable, which is the concrete
    reason the narrow list stops where it does."""
    assert secret_paths.refusal("Read", {"file_path": str(fake_home / relative)}, project) is None


@pytest.mark.parametrize(
    "command",
    [
        "python build.py && ls models",
        "git status",
        "openscad -o models/part.stl part.scad",
        "echo 'ssh is not mentioned here as a path'",
    ],
)
def test_ordinary_commands_are_not_refused(fake_home, project, command):
    assert secret_paths.refusal("Bash", {"command": command}, project) is None


def test_a_command_argument_on_another_tool_is_not_treated_as_a_command(fake_home, project):
    """Only ``Bash`` gets the textual check. A future tool with a ``command``
    argument must not silently inherit a check written for a shell."""
    assert (
        secret_paths.refusal("SomeOtherTool", {"command": "cat ~/.ssh/id_ed25519"}, project) is None
    )


def test_a_missing_or_odd_input_is_no_opinion_rather_than_a_refusal(fake_home, project):
    assert secret_paths.refusal("Read", {}, project) is None
    assert secret_paths.refusal("Read", {"file_path": ""}, project) is None
    assert secret_paths.refusal("Read", {"file_path": None}, project) is None
    assert secret_paths.refusal("Read", None, project) is None


# --- what is known not to be covered ----------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.s*/id_ed25519",
        "cat $(echo fkg2c3NoL2lkX2VkMjU1MTkK | base64 -d)",
        "cd ~ && cd .ssh && cat id_ed25519",
    ],
)
def test_the_bash_check_does_not_catch_an_indirect_spelling(fake_home, project, command):
    """Asserted as allowed, deliberately. The ``Bash`` half of this control is
    textual, so it raises a floor against accidents and direct attempts and is
    not a boundary against a determined agent; the sandbox does not restrict
    reads either (Annealage Mesh's plan section 2, fact 19). Writing that limit down as tests is
    what stops the docstring's honesty drifting from the code's behaviour, and
    what forces a later change claiming to close the gap to come here and say
    so.
    """
    assert secret_paths.refusal("Bash", {"command": command}, project) is None


def test_the_path_check_is_not_defeated_by_that_indirection(fake_home, project):
    """The contrast that makes the limit specific to ``Bash``: whatever the
    argument looks like, a path is resolved before it is compared."""
    awkward = str(fake_home / "." / ".." / "someone" / ".ssh" / "id_ed25519")
    assert secret_paths.refusal("Read", {"file_path": awkward}, project) is not None


# --- the resolved list ------------------------------------------------------


def test_denied_roots_are_absolute_and_under_the_current_home(fake_home):
    roots = secret_paths.denied_roots()
    assert len(roots) == len(secret_paths.DENIED_HOME_PATHS)
    for root in roots:
        assert root.is_absolute()
        assert str(root).startswith(str(Path(os.path.realpath(fake_home))))


def test_the_list_is_read_at_call_time_rather_than_cached(tmp_path, monkeypatch, project):
    """A cached list would keep enforcing a home directory the process no longer
    has, and would make every test here order-dependent."""
    first = tmp_path / "home-one"
    (first / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(first))
    assert secret_paths.refusal("Read", {"file_path": str(first / ".ssh" / "k")}, project)

    second = tmp_path / "home-two"
    (second / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(second))
    assert (
        secret_paths.refusal("Read", {"file_path": str(second / ".ssh" / "k")}, project) is not None
    )
    assert secret_paths.refusal("Read", {"file_path": str(first / ".ssh" / "k")}, project) is None
