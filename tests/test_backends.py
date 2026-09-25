"""Choosing an agent backend when no setting names one.

No backend is a default. What is installed decides, a person with several is
asked, and nobody at the terminal means an error rather than a guess.
"""

import pytest

from annealage_agent import backends, settings
from annealage_agent.backends import detect as real_detect


def _answers(*replies):
    replies = list(replies)

    def ask(_prompt):
        if not replies:
            raise EOFError
        return replies.pop(0)

    return ask


def test_detect_needs_the_cli_on_path_and_its_python_client():
    on_path = {"claude", "omp"}
    importable = {"claude_agent_sdk", "openai_codex"}
    found = real_detect(
        which=lambda name: "/bin/" + name if name in on_path else None,
        find_spec=lambda module: object() if module in importable else None,
    )
    # codex has its client but no CLI; omp has its CLI but no client.
    assert found == ("claude",)


def test_no_backend_installed_is_an_error_naming_all_three():
    with pytest.raises(backends.NoBackend) as exc:
        backends.choose((), interactive=True, ask=_answers())
    assert all(name in str(exc.value) for name in ("claude", "codex", "omp"))


def test_one_backend_installed_is_used_without_asking():
    assert backends.choose(("omp",), interactive=True, ask=_answers()) == ("omp", False)


def test_several_installed_with_nobody_at_the_terminal_refuses_to_guess():
    with pytest.raises(backends.NoBackend) as exc:
        backends.choose(("claude", "codex"), interactive=False)
    assert "--backend" in str(exc.value) and "--save-default" in str(exc.value)


def test_several_installed_asks_and_reasks_until_the_answer_is_valid():
    ask = _answers("9", "banana", "2", "")
    choice = backends.choose(
        ("claude", "codex", "omp"), interactive=True, ask=ask, write=lambda _: None
    )
    assert choice == ("codex", False)


def test_the_answer_can_be_a_name_and_can_be_saved():
    ask = _answers("omp", "y")
    choice = backends.choose(("claude", "omp"), interactive=True, ask=ask, write=lambda _: None)
    assert choice == ("omp", True)


def test_a_config_file_still_saying_local_is_refused(tmp_path):
    path = settings.project_config_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('backend = "local"\n')
    with pytest.raises(settings.SettingsError) as exc:
        settings.resolve(tmp_path)
    assert "'omp'" in str(exc.value)
