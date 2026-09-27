"""Tests for the three-layer settings resolution in ``settings.py``.

``tests/conftest.py`` sets ``XDG_CONFIG_HOME`` to a fresh directory for every
test (autouse), so ``settings.user_settings_path()`` is already isolated per
test with no extra fixture here; a project layer is isolated the ordinary
way, by writing under ``tmp_path``.
"""

import pytest

from annealage_agent import settings


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- shape of the table itself ----------------------------------------


def test_setting_keys_and_keys_by_name_agree():
    assert len(settings.SETTING_KEYS) == 12
    assert set(settings.KEYS_BY_NAME) == {key.name for key in settings.SETTING_KEYS}
    assert all(settings.KEYS_BY_NAME[key.name] is key for key in settings.SETTING_KEYS)


def test_user_settings_path_shape():
    path = settings.user_settings_path()
    assert path.name == "settings.toml"
    assert "annealage-toy" in str(path)


def test_project_config_path_shape(tmp_path):
    assert settings.project_config_path(tmp_path) == tmp_path / ".toy" / "config.toml"


# --- precedence and provenance, one pair at a time ----------------------


def test_resolve_with_nothing_set_returns_every_default(tmp_path):
    resolved = settings.resolve(tmp_path)
    for key in settings.SETTING_KEYS:
        assert resolved[key.name] == key.default
        assert resolved.provenance(key.name) == settings.DEFAULT


def test_resolve_without_a_project_dir_still_returns_defaults():
    resolved = settings.resolve(None)
    assert resolved["host"] == "127.0.0.1"
    assert resolved.provenance("host") == settings.DEFAULT


def test_user_layer_overrides_default(tmp_path):
    _write(settings.user_settings_path(), 'host = "0.0.0.0"\n')
    resolved = settings.resolve(tmp_path)
    assert resolved["host"] == "0.0.0.0"
    assert resolved.provenance("host") == settings.USER


def test_project_layer_overrides_default(tmp_path):
    _write(settings.project_config_path(tmp_path), 'permission_mode = "plan"\n')
    resolved = settings.resolve(tmp_path)
    assert resolved["permission_mode"] == "plan"
    assert resolved.provenance("permission_mode") == settings.PROJECT


def test_project_layer_overrides_user_layer(tmp_path):
    _write(settings.user_settings_path(), 'model = "user-model"\n')
    _write(settings.project_config_path(tmp_path), 'model = "project-model"\n')
    resolved = settings.resolve(tmp_path)
    assert resolved["model"] == "project-model"
    assert resolved.provenance("model") == settings.PROJECT


def test_flag_overrides_project_layer(tmp_path):
    _write(settings.project_config_path(tmp_path), 'model = "project-model"\n')
    resolved = settings.resolve(tmp_path, flags={"model": "flag-model"})
    assert resolved["model"] == "flag-model"
    assert resolved.provenance("model") == settings.FLAG


def test_flag_overrides_user_layer(tmp_path):
    _write(settings.user_settings_path(), 'host = "0.0.0.0"\n')
    resolved = settings.resolve(tmp_path, flags={"host": "10.0.0.1"})
    assert resolved["host"] == "10.0.0.1"
    assert resolved.provenance("host") == settings.FLAG


def test_flag_overrides_default(tmp_path):
    resolved = settings.resolve(tmp_path, flags={"port": 9000})
    assert resolved["port"] == 9000
    assert resolved.provenance("port") == settings.FLAG


def test_resolve_accepts_a_string_project_dir(tmp_path):
    _write(settings.project_config_path(tmp_path), 'model = "x"\n')
    resolved = settings.resolve(str(tmp_path))
    assert resolved["model"] == "x"


# --- refusals a file can trigger ----------------------------------------


def test_malformed_toml_file_names_the_file_in_the_error(tmp_path):
    path = settings.user_settings_path()
    _write(path, "this is not [ valid = toml\n")
    with pytest.raises(settings.SettingsError) as excinfo:
        settings.resolve(tmp_path)
    assert str(path) in str(excinfo.value)


def test_wrong_typed_value_is_refused(tmp_path):
    _write(settings.user_settings_path(), 'port = "not-a-number"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_bool_typed_key_rejects_a_string(tmp_path):
    _write(settings.user_settings_path(), 'open_browser = "yes"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_bool_typed_key_rejects_the_wrong_int_disguised_as_bool(tmp_path):
    # port is int-typed; a literal boolean must not pass as an integer, since
    # bool is a subclass of int in Python.
    _write(settings.user_settings_path(), "port = true\n")
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_enum_key_rejects_a_value_outside_its_choices(tmp_path):
    _write(settings.user_settings_path(), 'units = "cm"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_an_approval_timeout_under_a_second_is_refused(tmp_path):
    _write(settings.user_settings_path(), "approval_timeout = 0\n")
    with pytest.raises(settings.SettingsError, match="approval_timeout"):
        settings.resolve(tmp_path)
    _write(settings.user_settings_path(), "approval_timeout = 1\n")
    assert settings.resolve(tmp_path)["approval_timeout"] == 1


def test_permission_mode_in_the_user_file_is_refused(tmp_path):
    _write(settings.user_settings_path(), 'permission_mode = "plan"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_host_in_the_project_file_is_refused(tmp_path):
    # host is user-only; a project file setting it is a key in a layer it
    # has no business in, the same class of refusal permission_mode gets in
    # the user file.
    _write(settings.project_config_path(tmp_path), 'host = "0.0.0.0"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_bypass_permissions_in_the_project_file_is_refused(tmp_path):
    _write(settings.project_config_path(tmp_path), 'permission_mode = "bypassPermissions"\n')
    with pytest.raises(settings.SettingsError) as excinfo:
        settings.resolve(tmp_path)
    assert "bypassPermissions" in str(excinfo.value)


def test_bypass_permissions_in_the_user_file_is_refused(tmp_path):
    _write(settings.user_settings_path(), 'permission_mode = "bypassPermissions"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_bypass_permissions_is_allowed_as_a_flag(tmp_path):
    # Never persisted, but legal for a single run: settings.py leaves the
    # printed warning to the caller, not a refusal here.
    resolved = settings.resolve(tmp_path, flags={"permission_mode": "bypassPermissions"})
    assert resolved["permission_mode"] == "bypassPermissions"
    assert resolved.provenance("permission_mode") == settings.FLAG


def test_token_key_in_a_file_is_refused(tmp_path):
    _write(settings.user_settings_path(), 'token = "secret"\n')
    with pytest.raises(settings.SettingsError):
        settings.resolve(tmp_path)


def test_unrecognised_key_is_ignored_by_resolve(tmp_path):
    _write(settings.user_settings_path(), 'custom_thing = "kept"\nhost = "9.9.9.9"\n')
    resolved = settings.resolve(tmp_path)
    assert resolved["host"] == "9.9.9.9"


# --- apply: writing, validation order, and refusals ----------------------


def test_apply_writes_each_key_to_its_declared_layer(tmp_path):
    resolved, written = settings.apply(
        tmp_path,
        {
            "host": "0.0.0.0",
            "port": 9000,
            "model": "claude-x",
            "permission_mode": "plan",
        },
    )

    assert sorted(written["user"]) == ["host", "port"]
    assert sorted(written["project"]) == ["model", "permission_mode"]

    assert resolved["host"] == "0.0.0.0"
    assert resolved["port"] == 9000
    assert resolved["model"] == "claude-x"
    assert resolved["permission_mode"] == "plan"

    user_mapping = settings.tomllib.loads(settings.user_settings_path().read_text())
    assert user_mapping["host"] == "0.0.0.0"
    assert user_mapping["port"] == 9000
    assert "model" not in user_mapping
    assert "permission_mode" not in user_mapping

    project_mapping = settings.tomllib.loads(settings.project_config_path(tmp_path).read_text())
    assert project_mapping["model"] == "claude-x"
    assert project_mapping["permission_mode"] == "plan"
    assert "host" not in project_mapping


def test_apply_refuses_an_unknown_key(tmp_path):
    with pytest.raises(settings.SettingsError):
        settings.apply(tmp_path, {"not_a_real_setting": 1})


def test_apply_refuses_a_token_key(tmp_path):
    with pytest.raises(settings.SettingsError):
        settings.apply(tmp_path, {"token": "secret"})
    assert not settings.user_settings_path().exists()
    assert not settings.project_config_path(tmp_path).exists()


def test_apply_refuses_bypass_permissions(tmp_path):
    with pytest.raises(settings.SettingsError):
        settings.apply(tmp_path, {"permission_mode": "bypassPermissions"})
    assert not settings.project_config_path(tmp_path).exists()


def test_apply_validates_every_change_before_writing_anything(tmp_path):
    user_path = settings.user_settings_path()
    project_path = settings.project_config_path(tmp_path)
    _write(user_path, 'host = "1.1.1.1"\n')
    _write(project_path, 'model = "existing-model"\n')
    user_before = user_path.read_bytes()
    project_before = project_path.read_bytes()

    # port (user layer) is a legal change; permission_mode (project layer)
    # is not, since 123 is not a string. Both files would be touched if the
    # batch were applied change-by-change instead of validated as a whole.
    with pytest.raises(settings.SettingsError):
        settings.apply(tmp_path, {"port": 9000, "permission_mode": 123})

    assert user_path.read_bytes() == user_before
    assert project_path.read_bytes() == project_before


def test_apply_leaves_files_untouched_when_an_existing_key_blocks_serialization(tmp_path):
    project_path = settings.project_config_path(tmp_path)
    _write(project_path, 'custom_list = [1, 2, 3]\nmodel = "old"\n')
    project_before = project_path.read_bytes()
    user_path = settings.user_settings_path()

    # host (user layer) would succeed on its own; it must not be written
    # just because the project layer's pre-existing custom_list cannot be
    # re-emitted by this module's writer.
    with pytest.raises(settings.SettingsError) as excinfo:
        settings.apply(tmp_path, {"model": "new", "host": "9.9.9.9"})

    assert "custom_list" in str(excinfo.value)
    assert project_path.read_bytes() == project_before
    assert not user_path.exists()


def test_unrecognised_key_survives_a_write(tmp_path):
    user_path = settings.user_settings_path()
    _write(user_path, 'custom_thing = "kept"\nhost = "1.1.1.1"\n')

    settings.apply(tmp_path, {"host": "2.2.2.2"})

    mapping = settings.tomllib.loads(user_path.read_text())
    assert mapping["custom_thing"] == "kept"
    assert mapping["host"] == "2.2.2.2"


def test_apply_null_on_a_nullable_key_unsets_it(tmp_path):
    project_path = settings.project_config_path(tmp_path)
    settings.apply(tmp_path, {"model": "claude-x"})
    assert "model" in settings.tomllib.loads(project_path.read_text())

    resolved, written = settings.apply(tmp_path, {"model": None})

    assert written["project"] == ["model"]
    assert "model" not in settings.tomllib.loads(project_path.read_text())
    assert resolved.provenance("model") == settings.DEFAULT


# --- the emitter itself ---------------------------------------------------


def test_emitter_round_trips_every_value_type_in_the_table(tmp_path):
    resolved, _written = settings.apply(
        tmp_path,
        {
            "host": "192.168.1.1",
            "port": 9999,
            "open_browser": False,
            "approval_timeout": 3600,
            "units": "in",
            "tool_cards_collapsed": False,
            "model": "claude-opus-4",
            "effort": "high",
            "permission_mode": "acceptEdits",
            "backend": "codex",
            "omp_base_url": "http://localhost:11434",
            "omp_api_key": "test-key-123",
        },
    )

    user_mapping = settings.tomllib.loads(settings.user_settings_path().read_text())
    project_mapping = settings.tomllib.loads(settings.project_config_path(tmp_path).read_text())

    assert user_mapping["host"] == "192.168.1.1"
    assert isinstance(user_mapping["host"], str)

    assert user_mapping["port"] == 9999
    assert isinstance(user_mapping["port"], int)
    assert not isinstance(user_mapping["port"], bool)

    assert user_mapping["open_browser"] is False
    assert user_mapping["approval_timeout"] == 3600
    assert user_mapping["tool_cards_collapsed"] is False
    assert user_mapping["units"] == "in"

    assert project_mapping["model"] == "claude-opus-4"
    assert project_mapping["effort"] == "high"
    assert project_mapping["permission_mode"] == "acceptEdits"
    assert project_mapping["backend"] == "codex"
    assert project_mapping["omp_base_url"] == "http://localhost:11434"
    assert project_mapping["omp_api_key"] == "test-key-123"

    for key in settings.SETTING_KEYS:
        assert resolved.provenance(key.name) in (settings.USER, settings.PROJECT)


def test_emitter_escapes_strings_that_need_it(tmp_path):
    tricky = 'has "quotes", a \\ backslash, a tab\t and a newline\n'

    resolved, _written = settings.apply(tmp_path, {"model": tricky})

    assert resolved["model"] == tricky
    # A fresh resolve re-parses the file rather than trusting apply()'s
    # in-memory value, so this also proves the bytes on disk are correct
    # TOML, not merely that apply() remembered what it was given.
    reresolved = settings.resolve(tmp_path)
    assert reresolved["model"] == tricky


def test_emitter_quotes_a_key_name_that_is_not_a_bare_key(tmp_path):
    project_path = settings.project_config_path(tmp_path)
    _write(project_path, '"an odd key" = "kept"\nmodel = "old"\n')

    settings.apply(tmp_path, {"model": "new"})

    mapping = settings.tomllib.loads(project_path.read_text())
    assert mapping["an odd key"] == "kept"
    assert mapping["model"] == "new"


# --- to_wire ---------------------------------------------------------------


def test_to_wire_shape(tmp_path):
    resolved = settings.resolve(tmp_path, flags={"port": 9001})
    wire = resolved.to_wire()

    assert set(wire) == {key.name for key in settings.SETTING_KEYS}

    port_entry = wire["port"]
    assert port_entry["value"] == 9001
    assert port_entry["from"] == settings.FLAG
    assert port_entry["effect"] == "restart"
    assert port_entry["editable"] is True
    assert port_entry["type"] == "int"
    assert isinstance(port_entry["description"], str) and port_entry["description"]
