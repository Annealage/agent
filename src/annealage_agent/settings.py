"""Three-layer settings resolution: CLI flag, the project's own config file
(``<state dir>/config.toml``, e.g. Mesh's ``.mesh/config.toml``), user
``settings.toml``, built-in default, highest precedence first.

Provenance is not a debugging aid bolted on after the fact: it is the reason
this module exists at all. A person looking at the settings window needs to
know, for every value shown, whether changing it here will do anything (a
value with ``from == "flag"`` came from the command line for this run only
and cannot be overridden here without a restart) and, if they do change it,
which file just got written. ``Resolved.provenance`` and ``to_wire`` exist so
that answer is a first-class return value, not something a caller has to
reconstruct by re-reading the same files this module already read.

Reading and writing TOML are both hand-written here rather than pulled in as
a dependency. What this package writes is a closed set of scalar settings
keys it defines itself (``SETTING_KEYS`` below): strings, booleans and
integers, never a table, an array or a datetime. A general-purpose TOML
writer earns its complexity handling the cases this module never produces,
and the one case it must get right by hand regardless of what writes the
value, string escaping, is a few lines. Reading is different: TOML syntax a
human might hand-write (multi-line strings, dotted keys, every integer base)
is real complexity worth not re-implementing, so parsing goes through the
standard library's ``tomllib`` (Python 3.11 and later) or the ``tomli``
backport it was vendored from (earlier Python), never through code written
for this module.

A config file already on disk may hold keys this module does not define:
someone hand-edited it, or a future version of this package will add a key
this one has never heard of. Reading such a file skips those keys rather
than raising, and writing to it round-trips them unchanged, provided they
hold a value this module's writer can express. A key whose stored value is a
table, an array or a datetime blocks the whole write with an error naming
that key, rather than silently dropping it: this module has no way to write
that value back, and choosing between destroying it and refusing to write is
not a choice a library gets to make quietly on someone else's data. Comments
in a hand-edited file do not survive a write; the file this module writes
says so at the top.

Every refusal, a malformed file, a value of the wrong type, a key set at a
layer it has no business in, ``permission_mode: "bypassPermissions"`` or a
``token`` key found anywhere persisted, raises ``SettingsError``. Its message
is written to be shown verbatim, to a human on a terminal or as the "error"
field of a JSON response to a browser, and says which file or key is at
fault and, where there is one, what to do instead.

Every write goes through ``files.atomic_replace``, so a crash partway
through never leaves a config file holding half a TOML document; a reader
either sees the file as it was before the write or as it is after, never
something in between.
"""

import re
from pathlib import Path

import platformdirs

from . import files, product, sessions

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomllib is stdlib only from 3.11.
    import tomli as tomllib

USER = "user"
PROJECT = "project"
FLAG = "flag"
DEFAULT = "default"

_BYPASS_PERMISSIONS = "bypassPermissions"
_PERMISSION_MODE_CHOICES = ("default", "acceptEdits", "plan")
_EFFORT_CHOICES = ("low", "medium", "high", "xhigh", "max")

#: Every agent backend, in no order of preference: none is the default.
BACKENDS = ("claude", "codex", "omp")

#: The settings window's section titles for the generic keys (``Key.section``).
#: A product key names one of these to be listed in that section, after the
#: generic keys before it in ``SETTING_KEYS`` order; Mesh's ``up_axis`` joins
#: ``VIEWER_SECTION``.
SERVER_SECTION = "Server"
AGENT_SECTION = "Agent"
VIEWER_SECTION = "Viewer"


class SettingsError(Exception):
    """A settings value, file or requested change was refused.

    Raised for every refusal this module makes: a config file that is not
    valid TOML, a value of the wrong type, a key set at a layer it has no
    business in, ``permission_mode: "bypassPermissions"`` found in a file, a
    ``token`` key found anywhere persisted, or an unknown key offered to
    ``apply``. ``str(exc)`` is written to be shown as-is, on a terminal or as
    the ``"error"`` field of a JSON response, and names the file or key at
    fault.
    """


class Key:
    """One row of the settings table: a name, its type for display, its
    built-in default, which config files may declare it, whether changing it
    takes effect on the next request (``"load"``) or only on the next
    process start (``"restart"``), and a one-line description for the
    settings window.

    ``choices``, when non-empty, is the closed set of legal string values
    beyond ``None``. ``nullable`` allows ``None`` as a value, meaning "no
    override at this layer, fall through to the next one"; since TOML has no
    null literal, a nullable key set to ``None`` through ``apply`` is written
    by removing the key from its file rather than by writing a value.

    ``section`` is the title of the settings window section the key is shown
    in, or ``None`` for a key the window does not offer (it is still resolved,
    reported and writable through ``PUT /settings``). Keys sharing a title
    share a section, in listing order, so a product key can join a generic
    section by naming its title (``sections``).
    """

    __slots__ = (
        "name",
        "type_name",
        "default",
        "layers",
        "effect",
        "description",
        "py_type",
        "nullable",
        "choices",
        "section",
    )

    def __init__(
        self,
        name,
        type_name,
        default,
        layers,
        effect,
        description,
        py_type,
        nullable=False,
        choices=(),
        section=None,
    ):
        self.name = name
        self.type_name = type_name
        self.default = default
        self.layers = layers
        self.effect = effect
        self.description = description
        self.py_type = py_type
        self.nullable = nullable
        self.choices = choices
        self.section = section

    @property
    def write_layer(self):
        """Which single file ``apply`` writes this key to.

        A key declared writable at both layers (``model``, ``effort``) is
        written to the project file: ``apply`` always has a project
        directory to write into, and the project file is the more specific
        of the two, consistent with it also outranking the user file when
        both set the same key. A key writable at only one layer is written
        there.
        """
        return PROJECT if PROJECT in self.layers else USER

    def __repr__(self):
        return "Key(%r)" % self.name


#: The keys every product has: where the server listens, and how the agent
#: session is configured.
_AGENT_KEYS = (
    Key(
        name="host",
        type_name="str",
        default="127.0.0.1",
        layers=(USER,),
        effect="restart",
        description=(
            "The address this process listens on. 127.0.0.1 is "
            "loopback-only; see --host for the other bind modes."
        ),
        py_type=str,
        section=SERVER_SECTION,
    ),
    Key(
        name="port",
        type_name="int",
        default=8765,
        layers=(USER,),
        effect="restart",
        description="The TCP port this process listens on.",
        py_type=int,
        section=SERVER_SECTION,
    ),
    Key(
        name="open_browser",
        type_name="bool",
        default=True,
        layers=(USER,),
        effect="restart",
        description="Whether a browser tab opens automatically on startup.",
        py_type=bool,
        section=SERVER_SECTION,
    ),
    Key(
        name="model",
        type_name="str or null",
        default=None,
        layers=(USER, PROJECT),
        effect="restart",
        description=(
            "The model the agent backend uses. Unset falls back to the backend's own "
            "default. For backend=omp with no omp_base_url set, this is a "
            '"provider/model" reference (e.g. "titan/qwen3.8-27b") against whatever '
            "providers the omp CLI is already configured with, the same form its own "
            "--model flag accepts."
        ),
        py_type=str,
        nullable=True,
        section=AGENT_SECTION,
    ),
    Key(
        name="effort",
        type_name="str or null",
        default=None,
        layers=(USER, PROJECT),
        effect="restart",
        description=(
            "The reasoning effort passed to the agent session: low, "
            "medium, high, xhigh or max. Unset falls back to the CLI's own "
            "default."
        ),
        py_type=str,
        nullable=True,
        choices=_EFFORT_CHOICES,
        section=AGENT_SECTION,
    ),
    Key(
        name="permission_mode",
        type_name="str or null",
        default=None,
        layers=(PROJECT,),
        effect="restart",
        description=(
            "The permission mode passed to the agent session: default, "
            "acceptEdits or plan. Project-scoped only, so a personal "
            "default cannot silently loosen prompting in every project."
        ),
        py_type=str,
        nullable=True,
        choices=_PERMISSION_MODE_CHOICES,
        section=AGENT_SECTION,
    ),
    Key(
        name="backend",
        type_name='"claude" or "codex" or "omp" or null',
        default=None,
        layers=(USER, PROJECT),
        effect="restart",
        description=(
            "Which agent backend this session uses: claude (claude-agent-sdk, "
            "Claude subscription or API billing), codex (OpenAI's openai-codex "
            "SDK, ChatGPT subscription or API billing), or omp (the Oh My Pi CLI "
            "with every provider it is configured for, including local models, "
            "or an arbitrary OpenAI-compatible endpoint via omp_base_url). "
            "Unset: agent mode uses whichever one is on PATH, and asks when "
            "there is more than one."
        ),
        py_type=str,
        nullable=True,
        choices=BACKENDS,
        section=AGENT_SECTION,
    ),
    Key(
        name="omp_base_url",
        type_name="str or null",
        default=None,
        layers=(USER, PROJECT),
        effect="restart",
        description=(
            "The OpenAI-compatible base URL an arbitrary, self-hosted endpoint "
            "backend=omp talks to. Only used when backend is omp, and only "
            "needed for an endpoint the omp CLI does not already know about as a "
            "named provider -- leave unset to use omp exactly as already configured "
            'on this host (see the model key\'s "provider/model" form).'
        ),
        py_type=str,
        nullable=True,
    ),
    Key(
        name="omp_api_key",
        type_name="str or null",
        default=None,
        layers=(USER, PROJECT),
        effect="restart",
        description=(
            "The API key sent to omp_base_url, if that endpoint requires one. "
            "Only used when backend is omp and omp_base_url is set -- a "
            "provider omp already knows about (omp_base_url unset) carries its "
            "own credentials instead."
        ),
        py_type=str,
        nullable=True,
    ),
)

#: The chat pane's own presentation keys. Kept apart from ``_AGENT_KEYS``
#: only so a product's keys are listed between the two, which is where
#: Mesh's ``up_axis`` has always been listed: ``--settings`` prints this
#: order, and it is the order ``to_wire`` builds ``GET /settings`` in.
_PRESENTATION_KEYS = (
    Key(
        name="tool_cards_collapsed",
        type_name="bool",
        default=True,
        layers=(USER,),
        effect="load",
        description=("Whether a tool call's detail card starts collapsed in the chat pane."),
        py_type=bool,
        section=VIEWER_SECTION,
    ),
)

#: Every key this process resolves, generic and product, in listing order.
#: Rebound by ``register_product_keys`` when a product is installed, so it is
#: always read as ``settings.SETTING_KEYS`` rather than imported by name.
SETTING_KEYS = _AGENT_KEYS + _PRESENTATION_KEYS

KEYS_BY_NAME = {key.name: key for key in SETTING_KEYS}


def check_product_keys(keys):
    """Raise ``ValueError`` if ``keys`` cannot be registered: a name the
    generic set already has, a name given twice, or ``token``, which is never
    a setting (see ``_validate_file_mapping``). Changes nothing."""
    generic = {key.name for key in _AGENT_KEYS + _PRESENTATION_KEYS}
    seen = set()
    for key in keys:
        if key.name in generic or key.name in seen or key.name == "token":
            raise ValueError("product settings key %r collides with another key" % key.name)
        seen.add(key.name)


def register_product_keys(keys):
    """Make ``keys`` the product's settings keys, replacing any registered
    before. Called by ``product.install`` (after ``check_product_keys``) and
    ``product.reset`` only."""
    global SETTING_KEYS, KEYS_BY_NAME
    SETTING_KEYS = _AGENT_KEYS + tuple(keys) + _PRESENTATION_KEYS
    KEYS_BY_NAME = {key.name: key for key in SETTING_KEYS}


def sections():
    """``[{"title", "keys"}]``: the settings window's sections, in the order
    their first key is listed, each naming its keys in ``SETTING_KEYS`` order.

    Built from ``Key.section`` rather than written down in the window's own
    code, so a product's key appears in the window by declaring where, and a
    key with no section (``None``) is left out of it entirely.
    """
    ordered = {}
    for key in SETTING_KEYS:
        if key.section is not None:
            ordered.setdefault(key.section, []).append(key.name)
    return [{"title": title, "keys": names} for title, names in ordered.items()]


def user_settings_path():
    """The one ``settings.toml`` shared by every project on this machine."""
    return Path(platformdirs.user_config_dir(product.current().config_dirname)) / "settings.toml"


def project_config_path(project_dir):
    """``<project_dir>/<state dir>/config.toml``, this project's own overrides."""
    return sessions.state_dir(project_dir) / "config.toml"


class Resolved:
    """The outcome of resolving one project's settings across all four
    layers: an effective value and a provenance for every key in
    ``SETTING_KEYS``, immutable once built."""

    def __init__(self, values, sources, flags=None):
        self._values = values
        self._sources = sources
        self._flags = dict(flags or {})

    def __getitem__(self, name):
        return self._values[name]

    def get(self, name, default=None):
        return self._values.get(name, default)

    @property
    def flags(self):
        """The flags this resolution was built from, as a fresh mapping.

        Carried on the result rather than left with the caller because the
        flag layer outranks both files: anything that resolves again later,
        such as a settings window reading what is now on disk, has to supply
        the same flags or it will report a file's value as being in effect
        when this run is still using the flag's.
        """
        return dict(self._flags)

    def provenance(self, name):
        """``"flag"``, ``"project"``, ``"user"`` or ``"default"``: which
        layer supplied ``name``'s effective value."""
        return self._sources[name]

    def to_wire(self):
        """``{name: {"value", "from", "effect", "editable", "type",
        "description", "choices", "nullable"}}`` for every key, the shape
        ``GET /settings`` and the settings window read directly.

        ``editable`` means this key has at least one file it can be written
        to at all, not that a write would change anything visible without a
        restart; ``effect`` next to a ``from`` of ``"flag"`` is what tells a
        reader that changing it here needs a restart to take effect, because
        a flag always outranks whatever gets written to a file. ``choices``
        (empty for a free value) and ``nullable`` are what the window builds a
        select from: the choices, preceded by "not set" when ``None`` is legal.
        """
        wire = {}
        for key in SETTING_KEYS:
            wire[key.name] = {
                "value": self._values[key.name],
                "from": self._sources[key.name],
                "effect": key.effect,
                "editable": bool(key.layers),
                "type": key.type_name,
                "description": key.description,
                "choices": list(key.choices),
                "nullable": key.nullable,
            }
        return wire


def _parse_toml_file(path):
    """The mapping at ``path``, or ``{}`` if it does not exist.

    Raises ``SettingsError`` naming ``path`` if it exists but is not valid
    UTF-8 or not valid TOML: a config file that cannot be parsed is not the
    same thing as a config file that sets nothing, and must not be treated
    as one.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsError("could not read %s: %s" % (path, exc)) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SettingsError("%s is not valid UTF-8: %s" % (path, exc)) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError("%s is not valid TOML: %s" % (path, exc)) from exc


def _check_value(key, value, *, layer, source):
    """Raise ``SettingsError`` if ``value`` is not legal for ``key`` coming
    from ``layer`` (used in the message as ``source``).

    Type checking happens before the ``permission_mode``/``bypassPermissions``
    and ``choices`` checks, both of which assume a string in hand. Booleans
    are checked before integers because ``bool`` is a subclass of ``int`` in
    Python; without that order a stray ``True`` would pass as a legal port
    number.
    """
    if value is None:
        if key.nullable:
            return
        raise SettingsError(
            "%s sets %r to null, but %r must be %s" % (source, key.name, key.name, key.type_name)
        )
    if key.py_type is bool:
        if not isinstance(value, bool):
            raise SettingsError(
                "%s sets %r to %r, which is not a boolean" % (source, key.name, value)
            )
        return
    if key.py_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(
                "%s sets %r to %r, which is not an integer" % (source, key.name, value)
            )
        return
    if not isinstance(value, str):
        raise SettingsError("%s sets %r to %r, which is not a string" % (source, key.name, value))
    if key.name == "permission_mode" and value == _BYPASS_PERMISSIONS:
        if layer == FLAG:
            return
        raise SettingsError(
            "%s sets permission_mode to bypassPermissions; that is accepted "
            "only as a one-off --permission-mode flag and must never be "
            "written to a config file. Remove it from %s" % (source, source)
        )
    if key.choices and value not in key.choices:
        raise SettingsError(
            "%s sets %r to %r, which must be one of %s"
            % (source, key.name, value, ", ".join(repr(choice) for choice in key.choices))
        )


def _validate_file_mapping(mapping, layer, path):
    """Raise ``SettingsError`` for anything in ``mapping`` that ``resolve``
    must refuse to load from ``layer``: a ``token`` key, a known key
    declared for a different layer, a value of the wrong type, or
    ``permission_mode: "bypassPermissions"``.

    A name outside ``SETTING_KEYS`` is left alone here; it is not a setting
    this module reads, and ``apply`` preserves it verbatim on the next
    write. A known key found at a layer it is not declared for is refused
    the same way an entirely unknown key would be if this module tried to
    read it as a setting: it is simply not a legal key in this file, and
    silently ignoring it would hide a config file that does not do what
    whoever wrote it thought it did.
    """
    if "token" in mapping:
        raise SettingsError(
            "%s holds a token key; the per-run token is generated fresh at "
            "every start and must never be written to a config file" % path
        )
    for name, value in mapping.items():
        key = KEYS_BY_NAME.get(name)
        if key is None:
            continue
        if layer not in key.layers:
            raise SettingsError(
                "%s sets %r, which may only be set at the %s layer; remove "
                "it from this file" % (path, name, " or ".join(key.layers))
            )
        _check_value(key, value, layer=layer, source=str(path))


def _validate_flags(flags):
    """Raise ``SettingsError`` for a ``token`` key, an unknown key, or a
    value of the wrong type among ``flags``.

    Flags carry no layer restriction of their own, unlike a file: a
    ``--permission-mode`` flag may legally request ``bypassPermissions`` for
    this run even though the same value can never be written to a file,
    because a flag is never persisted.
    """
    if "token" in flags:
        raise SettingsError(
            "token may not be set through a flag; the per-run token is "
            "generated fresh at every start and is never persisted"
        )
    for name, value in flags.items():
        key = KEYS_BY_NAME.get(name)
        if key is None:
            raise SettingsError("%r is not a %s setting" % (name, product.current().name))
        _check_value(key, value, layer=FLAG, source="a command-line flag for %r" % name)


def resolve(project_dir=None, *, flags=None):
    """Resolve every key in ``SETTING_KEYS`` across all four layers for
    ``project_dir`` (or, if ``None``, without a project layer at all), with
    ``flags`` holding only the keys whose flag was actually given for this
    run.

    Raises ``SettingsError`` for a malformed file, a value of the wrong
    type, a key set at a layer it has no business in,
    ``permission_mode: "bypassPermissions"`` found in a file, or a ``token``
    key found anywhere.
    """
    flags = flags or {}
    _validate_flags(flags)

    user_path = user_settings_path()
    user_mapping = _parse_toml_file(user_path)
    _validate_file_mapping(user_mapping, USER, user_path)

    project_mapping = {}
    if project_dir is not None:
        project_path = project_config_path(project_dir)
        project_mapping = _parse_toml_file(project_path)
        _validate_file_mapping(project_mapping, PROJECT, project_path)

    values = {}
    sources = {}
    for key in SETTING_KEYS:
        if key.name in flags:
            values[key.name] = flags[key.name]
            sources[key.name] = FLAG
        elif PROJECT in key.layers and key.name in project_mapping:
            values[key.name] = project_mapping[key.name]
            sources[key.name] = PROJECT
        elif USER in key.layers and key.name in user_mapping:
            values[key.name] = user_mapping[key.name]
            sources[key.name] = USER
        else:
            values[key.name] = key.default
            sources[key.name] = DEFAULT

    return Resolved(values, sources, flags)


_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _escape_toml_string(text):
    out = []
    for ch in text:
        escape = _TOML_ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _toml_key(name):
    if _BARE_KEY_RE.match(name):
        return name
    return '"%s"' % _escape_toml_string(name)


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"%s"' % _escape_toml_string(value)
    raise TypeError(type(value).__name__)


#: ``%s`` is the product's distribution name.
_HEADER_COMMENT = (
    "# Written by %s (its settings window, or --save-default).\n"
    "# A comment added here by hand does not survive the next save: only\n"
    "# the key/value pairs below are read back and re-emitted.\n"
    "\n"
)


def _serialize_mapping(mapping):
    """The whole of ``mapping`` as a TOML document, bytes ready for
    ``files.atomic_replace``.

    Raises ``SettingsError`` naming the offending key if any value in
    ``mapping`` is not a string, boolean, integer or float: those are the
    only TOML value shapes this writer can express, so a table, an array or
    a datetime blocks the whole write rather than being silently dropped.
    """
    lines = [_HEADER_COMMENT % product.current().distribution]
    for name, value in mapping.items():
        try:
            rendered = _toml_value(value)
        except TypeError:
            raise SettingsError(
                "%r holds a %s, which this settings writer cannot express "
                "in TOML (only strings, booleans and numbers); edit it by "
                "hand instead, or remove it, before saving from here" % (name, type(value).__name__)
            ) from None
        lines.append("%s = %s\n" % (_toml_key(name), rendered))
    return "".join(lines).encode("utf-8")


def apply(project_dir, changes, *, flags=None, layer=None):
    """Validate and write ``changes`` (``{name: value}``), then return
    ``(resolve(project_dir, flags=flags), {"user": [names], "project":
    [names]})`` naming which keys landed in which file.

    Every change is validated, and every file about to be touched is loaded,
    merged and proven writable, before ``files.atomic_replace`` is called
    for any of them: a batch with one good change and one bad one writes
    neither file, and a preserved key in an existing file that this writer
    cannot express blocks the whole call rather than losing whichever change
    would otherwise have gone into that same file.

    Refuses a ``token`` key, an unknown key, a value of the wrong type, or
    ``permission_mode: "bypassPermissions"`` outright: none of those may
    ever be written to a config file.

    ``layer`` writes every change to that one file instead of each key's own
    ``write_layer``: ``--save-default`` uses it to put ``backend`` in the
    user file, where it applies to every project, rather than in this
    project's. A key that may not live at ``layer`` is refused.
    """
    project_dir = Path(project_dir)

    validated = {}
    for name, value in changes.items():
        if name == "token":
            raise SettingsError(
                "token may not be set through apply; the per-run token is "
                "generated fresh at every start and is never persisted"
            )
        key = KEYS_BY_NAME.get(name)
        if key is None:
            raise SettingsError("%r is not a %s setting" % (name, product.current().name))
        target = layer or key.write_layer
        if target not in key.layers:
            raise SettingsError(
                "%r may only be set at the %s layer" % (name, " or ".join(key.layers))
            )
        _check_value(key, value, layer=target, source="the requested change to %r" % name)
        validated[key] = (value, target)

    by_layer = {USER: {}, PROJECT: {}}
    for key, (value, target) in validated.items():
        by_layer[target][key.name] = value

    prepared = {}
    for layer, layer_changes in by_layer.items():
        if not layer_changes:
            continue
        path = project_config_path(project_dir) if layer == PROJECT else user_settings_path()
        mapping = dict(_parse_toml_file(path))
        for name, value in layer_changes.items():
            key = KEYS_BY_NAME[name]
            if value is None and key.nullable:
                mapping.pop(name, None)
            else:
                mapping[name] = value
        prepared[layer] = (path, _serialize_mapping(mapping))

    written = {USER: [], PROJECT: []}
    for layer, (path, data) in prepared.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        files.atomic_replace(path, data)
        written[layer] = sorted(by_layer[layer])

    return resolve(project_dir, flags=flags), written
