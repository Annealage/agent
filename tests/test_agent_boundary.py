"""The agent layer's boundary: nothing under ``src/annealage_agent/`` depends
on a product.

This package is shared by the products built on it (Annealage Mesh was the
first, and the package was extracted from it), so it may import itself, the
standard library and third-party packages, and no product: what it needs to
know about the product it runs in comes through ``product.py``. This is
checked by reading the source rather than by importing it, because an import
that only happens inside a function (the way the backends' session modules are
imported) is as much a dependency as one at the top of a file, and importing
would not see it until that function ran.

Two things are refused:

- an import of a product's module, which is any ``annealage_*`` package other
  than ``annealage_agent`` itself, or ``loom``; a relative import that climbs
  above ``annealage_agent`` (``from ... import paths`` in ``session/sdk.py``),
  which can only be reaching for some other distribution's modules; and a
  string naming a product's module, which is how a module path handed to a
  subprocess would re-couple the two: any string that starts with such a
  module path, including the ``module:attr`` entry-point form and the
  ``"annealage_mesh."`` prefix of a path built by concatenation;
- any reference to the 3D domain (``stl``, ``cad``, ``cadquery``, ``trimesh``,
  ``three``) that Annealage Mesh, the product this layer came out of, works
  in: an imported module name, an identifier (as one ``snake_case`` word of
  it), or a string constant that is not a docstring. Docstrings and comments
  are prose and may still say where an idea came from.

The same boundary holds for the front end under ``static/``, the package's
static files: every JavaScript module there imports only modules of that same
tree, by a relative specifier that resolves inside it (never a product's
``/static/`` file, never ``three`` or any other bare specifier); loads no
script any other way (a worker, ``importScripts``, a URL built from
``import.meta.url``); names no server path but the agent layer's own routes,
so it cannot fetch a product route or a product module by URL, and names each
of those relative to the page through ``url.js``'s ``appUrl`` (so a page
mounted under a front door's ``/p/<id>/`` reaches its own app), never by an
absolute path; and neither a module nor the stylesheet uses a product's
vocabulary (pins, callouts, the camera, the scene, ...) outside a comment.
JavaScript has no parser in the standard library, so ``_lex_js`` is a small
lexer that knows exactly enough of the language to tell a comment from a
string, a template or a regular expression literal, which is all these checks
need.

The last tests feed each checker regressions it must catch, so neither can
pass by checking nothing.
"""

import ast
import functools
import re
import tempfile
from pathlib import Path

import pytest
from conftest import DEFAULT_PORT
from toy_product import TOY_PAGE

AGENT_DIR = Path(__file__).resolve().parents[1] / "src" / "annealage_agent"
AGENT_PACKAGE = "annealage_agent"
# Top-level packages that are a product's own without the ``annealage_``
# prefix every other product's module carries.
UNPREFIXED_PRODUCT_PACKAGES = frozenset({"loom"})
DOMAIN_WORDS = frozenset({"stl", "cad", "cadquery", "trimesh", "three"})
_DOMAIN_TEXT_RE = re.compile(r"\b(stl|cad|cadquery|trimesh|three)\b", re.IGNORECASE)
_MODULE_TEXT_RE = re.compile(r"^(annealage_[A-Za-z0-9_]+|loom)(\.[A-Za-z_][A-Za-z0-9_]*)*\.?(:|$)")


def _agent_modules():
    """``(path, dotted module name, is_package)`` for every module under
    ``src/annealage_agent/``."""
    for path in sorted(AGENT_DIR.rglob("*.py")):
        parts = path.relative_to(AGENT_DIR).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        yield path, ".".join((AGENT_PACKAGE,) + parts), is_package


def _relative_base(node, module_name, is_package):
    """The absolute package the relative ``from`` import ``node`` resolves
    against, or None when it climbs above ``annealage_agent`` itself."""
    package = module_name if is_package else module_name.rpartition(".")[0]
    for _ in range(node.level - 1):
        package = package.rpartition(".")[0]
    return package or None


def _imported_names(node, module_name, is_package):
    """Every absolute module name ``node`` may import. For ``from X import a``
    that is ``X`` and ``X.a``, because ``a`` may itself be a submodule
    (``from .. import files``). A relative import that climbs above the
    package names nothing here; ``violations`` refuses it on its own."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level == 0:
        base = node.module
    else:
        package = _relative_base(node, module_name, is_package)
        if package is None:
            return []
        base = package + ("." + node.module if node.module else "")
    return [base] + ["%s.%s" % (base, alias.name) for alias in node.names if alias.name != "*"]


def _outside_agent(name):
    """Whether module ``name`` is a product's: any ``annealage_*`` package
    other than this one, or an unprefixed product package."""
    top = name.partition(".")[0]
    if top == AGENT_PACKAGE:
        return False
    return top.startswith("annealage_") or top in UNPREFIXED_PRODUCT_PACKAGES


def _module_of(text):
    """The module a string names: the part before an entry point's ``:attr``,
    without the trailing dot a string built by concatenation
    (``"annealage_mesh." + name``) starts with."""
    return text.partition(":")[0].rstrip(".")


def _docstring_nodes(tree):
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    nodes.add(id(body[0].value))
    return nodes


def _identifiers(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, ast.arg):
        yield node.arg
    elif isinstance(node, ast.keyword) and node.arg is not None:
        yield node.arg
    elif isinstance(node, ast.alias):
        yield node.asname or node.name.rpartition(".")[2]


def violations(source, module_name, is_package=False):
    """Every boundary violation in ``source``, read as module ``module_name``."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if (
            isinstance(node, ast.ImportFrom)
            and node.level
            and _relative_base(node, module_name, is_package) is None
        ):
            found.append(
                "line %s imports %s%s from above %s"
                % (line, "." * node.level, node.module or "", AGENT_PACKAGE)
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _imported_names(node, module_name, is_package):
                if _outside_agent(name):
                    found.append("line %s imports %s" % (line, name))
                if DOMAIN_WORDS & set(name.lower().split(".")):
                    found.append("line %s imports the 3D domain: %s" % (line, name))
        for identifier in _identifiers(node):
            if DOMAIN_WORDS & set(identifier.lower().split("_")):
                found.append("line %s names the 3D domain: %s" % (line, identifier))
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            if _MODULE_TEXT_RE.match(node.value) and _outside_agent(_module_of(node.value)):
                found.append("line %s names module %r" % (line, node.value))
            if _DOMAIN_TEXT_RE.search(node.value):
                found.append("line %s mentions the 3D domain: %r" % (line, node.value))
    return found


def test_the_agent_package_is_where_the_checker_looks():
    """Guards the check itself: a moved directory would otherwise make every
    test below pass over zero files."""
    modules = [name for _path, name, _pkg in _agent_modules()]
    assert "annealage_agent.product" in modules
    assert "annealage_agent.session.sdk" in modules
    assert "annealage_agent.http.routes_mcp" in modules


@pytest.mark.parametrize(
    "path,module_name,is_package",
    list(_agent_modules()),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_agent_module_depends_on_no_product(path, module_name, is_package):
    assert violations(path.read_text(encoding="utf-8"), module_name, is_package) == []


@pytest.mark.parametrize(
    "source",
    [
        # The coupling the extraction removed, re-added from sdk.py's home:
        # Annealage Mesh's modules, by a relative import that climbs out of
        # this package, by name, and inside a function.
        "from ...tools import registry\n",
        "from annealage_mesh.tools import registry\n",
        "from ... import paths\n",
        "import annealage_mesh.cli\n",
        "from ... import __version__\n",
        "def f():\n    from ...http.routes_viewer import VIEWER_HTML\n",
        'BRIDGE = "annealage_mesh.session.codex_mcp_stdio_bridge"\n',
        'ENTRY = "annealage_mesh.cli:main"\n',
        'importlib.import_module("annealage_mesh." + name)\n',
        # Any other product's, by either naming.
        "import annealage_loom.sketch\n",
        "from loom import sketch\n",
        'ENTRY = "annealage_canvas.harness:main"\n',
        'importlib.import_module("loom." + name)\n',
        # The 3D domain.
        "from ... import stl\n",
        "import trimesh\n",
        "def load_stl(path):\n    pass\n",
        'CONTENT_TYPES = {".stl": "model/stl"}\n',
    ],
)
def test_the_checker_catches_a_regression(source):
    assert violations(source, "annealage_agent.session.sdk")


@pytest.mark.parametrize(
    "source",
    [
        "from .. import product\n",
        "from ..tools import ToolServer\n",
        "from annealage_agent import files\n",
        "import asyncio\n",
        '"""A docstring may say STL, where the idea came from."""\n',
        "BRIDGE = __package__ + '.session.codex_mcp_stdio_bridge'\n",
        'BRIDGE = "annealage_agent.session.codex_mcp_stdio_bridge"\n',
    ],
)
def test_the_checker_allows_the_agent_layer_itself(source):
    assert violations(source, "annealage_agent.session.sdk") == []


# --- the front end under static/ ------------------------------------------

AGENT_STATIC_DIR = AGENT_DIR / "static"

# Words of the product's domain a generic module has no business using outside
# a comment. Matched against whole words of the code and its string literals,
# with identifiers split at camelCase and snake_case boundaries, so ``upAxis``
# is caught as "axis" and ``callouts_changed`` as "callouts". Every inflection
# is listed rather than matched by prefix, because a prefix match would catch
# words the agent layer does own ("ping" under "pin"). Words the agent layer
# legitimately owns are deliberately absent: "model" (the LLM's), "viewer"
# (the protocol's page), "view" (a tool grade).
FRONT_END_DOMAIN_WORDS = frozenset(
    {
        "three",
        "stl",
        "mesh",
        "meshes",
        "pin",
        "pins",
        "callout",
        "callouts",
        "camera",
        "cameras",
        "scene",
        "scenes",
        "orbit",
        "orbitcontrols",
        "raycast",
        "raycaster",
        "axis",
        "axes",
        "sketch",
        "sketches",
        "measure",
        "measurement",
        "measurements",
        "manifest",
        "geometry",
        "geometries",
    }
)

_REGEX_AFTER_WORDS = frozenset(
    {"return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case"}
    | {"do", "else", "yield", "await"}
)
_REGEX_AFTER_PUNCT = frozenset("(,=:[!&|?{};+-*%<>~^")


def _end_of_regex(source, i):
    """Index just past the regular expression literal starting at ``i``."""
    j = i + 1
    in_class = False
    while j < len(source):
        ch = source[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "\n":
            break
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            j += 1
            while j < len(source) and source[j].isalpha():
                j += 1
            return j
        j += 1
    raise ValueError("unterminated regular expression at offset %d" % i)


def _end_of_quoted(source, i):
    """Index just past the string or template literal starting at ``i``. A
    template's ``${...}`` is code, so strings and templates inside it are
    skipped as literals and its braces counted, rather than its closing
    quote being taken for the template's own."""
    quote = source[i]
    j = i + 1
    while j < len(source):
        ch = source[j]
        if ch == "\\":
            j += 2
            continue
        if ch == quote:
            return j + 1
        if quote != "`" and ch == "\n":
            break
        if quote == "`" and source.startswith("${", j):
            depth = 1
            j += 2
            while depth:
                if j >= len(source):
                    raise ValueError("unterminated template at offset %d" % i)
                if source[j] in "'\"`":
                    j = _end_of_quoted(source, j)
                    continue
                depth += {"{": 1, "}": -1}.get(source[j], 0)
                j += 1
            continue
        j += 1
    raise ValueError("unterminated string at offset %d" % i)


def _lex_js(source):
    """``(code, strings)``: ``source`` with every comment replaced by a space
    and everything else, string, template and regex literals included, kept as
    written; and the contents of every string and template literal (quotes
    stripped), in order.

    A ``/`` starts a regular expression where an expression may start (after
    an operator, an opening bracket, a comma, or a keyword such as
    ``return``) and is division otherwise, which is the rule the language
    itself uses for every case these modules contain.
    """
    out = []
    strings = []
    last = ""
    i = 0
    while i < len(source):
        ch = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end == -1 else end
            out.append(" ")
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                raise ValueError("unterminated comment at offset %d" % i)
            i = end + 2
            out.append(" ")
        elif ch in "'\"`":
            end = _end_of_quoted(source, i)
            out.append(source[i:end])
            strings.append(source[i + 1 : end - 1])
            last, i = "literal", end
        elif ch == "/" and (last == "" or last in _REGEX_AFTER_PUNCT or last in _REGEX_AFTER_WORDS):
            end = _end_of_regex(source, i)
            out.append(source[i:end])
            last, i = "literal", end
        elif ch.isalnum() or ch in "_$":
            end = i
            while end < len(source) and (source[end].isalnum() or source[end] in "_$"):
                end += 1
            last = source[i:end]
            out.append(last)
            i = end
        else:
            out.append(ch)
            if not ch.isspace():
                last = ch
            i += 1
    return "".join(out), strings


# Whitespace is optional everywhere the language allows it to be, so
# ``import{a}from"./x.js"`` and ``import"./x.js"`` are read as the imports
# they are.
_STATIC_IMPORT_RE = re.compile(
    r"(?:^|[\s;}])(?:import|export)\s*(?:[\w$*{}\s,]+?\s*from\s*)?([\"'])([^\"']*)\1"
)
# Every other way a page script loads a script: a worker (whose own imports
# this check could never see), a classic worker's importScripts, and a URL
# resolved against this module's own, which is how a module names another
# file without an import statement.
_OTHER_LOADS_RE = re.compile(r"\b(?:Worker|SharedWorker|importScripts)\s*\(|\bimport\s*\.\s*meta\b")
_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\s*\(")
_LITERAL_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\s*\(\s*([\"'])([^\"']*)\1\s*\)")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# A call of url.js's helper, not its definition; the route is its argument.
_APP_URL_CALL_RE = re.compile(r"(\bfunction\s+)?(?<![\w$.])appUrl\s*\(\s*")
_WORD_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def _domain_words(code):
    """The domain words ``code`` uses, each identifier split into its parts."""
    found = set()
    for word in _WORD_RE.findall(code):
        for part in _WORD_PART_RE.findall(word):
            if part.lower() in FRONT_END_DOMAIN_WORDS:
                found.add(part.lower())
    return found


@functools.cache
def agent_route_prefixes():
    """The literal start of every route the agent layer registers, read off a
    viewer-only app it builds with the toy product's page but none of its
    routes (``register_routes=None``): ``/ws``, ``/agent/static/``,
    ``/session/`` and so on. Derived rather than written down, so a route the
    agent layer adds or renames moves this list with it. (``/mcp`` is
    registered only in agent mode and is the agent's, not the page's, so a
    page naming it would rightly be refused.)"""
    from annealage_agent import app as agent_app

    with tempfile.TemporaryDirectory() as d:
        app = agent_app.create_app(d, page_html=TOY_PAGE, port=DEFAULT_PORT, register_routes=None)
        patterns = {entry[1].url_pattern for entry in app.url_map}
    return tuple(sorted(p.partition("<")[0] for p in patterns))


def _names_an_agent_route(text):
    """Whether ``text`` starts with one of the agent layer's route prefixes,
    as a whole path segment: ``/ws?t=`` does, ``/wsx`` does not."""
    for prefix in agent_route_prefixes():
        if text == prefix or (
            text.startswith(prefix) and (prefix.endswith("/") or text[len(prefix)] in "?/#")
        ):
            return True
    return False


def _app_url_routes(code):
    """``(route, computed)`` for each ``appUrl(...)`` call in ``code``: the
    literal start of its argument (a template's text up to its first
    ``${``), and whether that argument starts with anything but a literal,
    which leaves nothing to check."""
    found = []
    for match in _APP_URL_CALL_RE.finditer(code):
        if match.group(1):
            continue
        start = match.end()
        if start >= len(code) or code[start] not in "'\"`":
            found.append((None, True))
            continue
        literal = code[start + 1 : _end_of_quoted(code, start) - 1]
        found.append((literal.split("${")[0] if code[start] == "`" else literal, False))
    return found


def js_violations(source, path):
    """Every boundary violation in the module ``source``, read as if it lived
    at ``path`` inside the agent static tree."""
    code, strings = _lex_js(source)
    found = []
    if _OTHER_LOADS_RE.search(code):
        found.append("loads a script some way other than an import statement")
    for text in strings:
        if not text.startswith("/"):
            continue
        if _names_an_agent_route(text):
            found.append(
                "names %r by an absolute path, which misses the app under a front door's "
                "prefix; resolve it with appUrl" % text
            )
        else:
            found.append("names %r, which is not one of the agent layer's routes" % text)
    for route, computed in _app_url_routes(code):
        if computed:
            found.append("names a route by a computed expression, which cannot be checked")
        elif not _names_an_agent_route("/" + route):
            found.append("names %r, which is not one of the agent layer's routes" % route)
    specifiers = [m.group(2) for m in _STATIC_IMPORT_RE.finditer(code)]
    specifiers += [m.group(2) for m in _LITERAL_DYNAMIC_IMPORT_RE.finditer(code)]
    if len(_DYNAMIC_IMPORT_RE.findall(code)) != len(_LITERAL_DYNAMIC_IMPORT_RE.findall(code)):
        found.append("imports a module named by a computed expression")
    for spec in specifiers:
        if not spec.startswith(("./", "../")):
            found.append("imports %r, which is not a path relative to this tree" % spec)
            continue
        target = (path.parent / spec).resolve()
        if not target.is_relative_to(AGENT_STATIC_DIR.resolve()):
            found.append("imports %r, which is outside static/" % spec)
        elif not target.is_file():
            found.append("imports %r, which does not exist" % spec)
    for word in sorted(_domain_words(code)):
        found.append("uses the product's vocabulary: %r" % word)
    return found


def css_violations(source):
    """The domain words the stylesheet ``source`` uses outside a comment."""
    code = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return ["uses the product's vocabulary: %r" % w for w in sorted(_domain_words(code))]


def test_the_front_end_tree_is_where_the_checker_looks():
    """Guards the check itself, as for the Python modules above."""
    names = {path.name for path in AGENT_STATIC_DIR.glob("*.js")}
    assert {"chat.js", "ws.js", "store.js", "settings.js", "uploads.js", "ui.js"} <= names
    assert (AGENT_STATIC_DIR / "agent.css").is_file()


@pytest.mark.parametrize("path", sorted(AGENT_STATIC_DIR.rglob("*.js")), ids=lambda path: path.name)
def test_front_end_module_depends_on_nothing_of_the_product(path):
    assert js_violations(path.read_text(encoding="utf-8"), path) == []


@pytest.mark.parametrize(
    "path", sorted(AGENT_STATIC_DIR.rglob("*.css")), ids=lambda path: path.name
)
def test_front_end_stylesheet_uses_no_product_vocabulary(path):
    assert css_violations(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "source",
    [
        # Reaching back into the product's own tree, three ways.
        'import { store } from "../../static/js/store.js";\n',
        'import { initPins } from "/static/js/pins.js";\n',
        'export { store } from "agent/store.js";\n',
        # The renderer, by its bare specifier.
        'import * as THREE from "three";\n',
        # A module that is not there, and one that cannot be checked at all.
        'import { x } from "./no-such-module.js";\n',
        'const m = await import("./" + name);\n',
        # The same, with no whitespace where the language needs none.
        'import"../../static/js/store.js";\n',
        'import{store}from"../../static/js/store.js";\n',
        'export*from"../../static/js/store.js";\n',
        # A script loaded some way other than an import statement.
        'const w = new Worker("./worker.js", { type: "module" });\n',
        'const w = new SharedWorker("./worker.js");\n',
        'importScripts("./worker.js");\n',
        'const u = new URL("../../static/js/store.js", import.meta.url);\n',
        # A product route or file named by URL, however it is then used.
        'const s = document.createElement("script"); s.src = "/static/js/x.js";\n',
        'fetch("/static/js/store.js");\n',
        'fetch("/model/" + rel);\n',
        "fetch(`/callouts`);\n",
        'fetch("/wsx");\n',
        # The same through the helper: a product route, a prefix that is not
        # a whole segment, and a route nothing can check.
        'fetch(appUrl("static/js/store.js"));\n',
        "fetch(appUrl(`submit/${id}`));\n",
        'fetch(appUrl("wsx"));\n',
        "fetch(appUrl(route));\n",
        # An agent route by an absolute path, which misses a mounted app.
        'fetch("/ws?t=" + token);\n',
        "fetch(`/session/${id}/export?t=${t}`);\n",
        # The product's words, in code and in a string, in every inflection.
        "store.setUpAxis(value);\n",
        'toast("Pin #" + id + " added");\n',
        "if (event.kind === `callouts_changed`) refetch();\n",
        "const cameras = [];\n",
        "const r = new Raycaster();\n",
        "const upAxes = 2;\n",
        "const sketches = [];\n",
        "const measurement = 1;\n",
        "const scenes = [];\n",
        "const geometries = [];\n",
        "const orbitcontrols = null;\n",
    ],
)
def test_the_front_end_checker_catches_a_regression(source):
    assert js_violations(source, AGENT_STATIC_DIR / "chat.js")


@pytest.mark.parametrize(
    "source",
    [
        'import { store } from "./store.js";\n',
        'export { toast } from "./ui.js";\n',
        "// Mesh's pins and callouts, the scene and the camera: prose.\nconst a = 1;\n",
        "/* three.js */ const b = 2;\n",
        # A regular expression holding a backtick, which a lexer that took it
        # for division would read as the start of a template running on to
        # swallow the rest of the module.
        'const re = /`([^`]+)`/g;\nconst c = "ok"; // camera\n',
        "const half = total / 2; // pins / 2\n",
        "const t = `a ${`nested ${x}`} b`; // scene\n",
        # The agent layer's own routes, through the helper, and "ping", which
        # a prefix match on "pin" would have caught.
        'fetch(appUrl("ws") + "?t=" + token); fetch(appUrl("agent/static/chat.js"));\n',
        "fetch(appUrl(`session/${id}/export`) + `?t=${t}`);\n",
        'fetch(appUrl("agent/logs/" + encodeURIComponent(name)));\n',
        "export function appUrl(route) {\n  return route;\n}\n",
        'if (frame.type === "ping") return;\n',
    ],
)
def test_the_front_end_checker_allows_the_tree_itself(source):
    assert js_violations(source, AGENT_STATIC_DIR / "chat.js") == []


def test_the_route_prefixes_are_the_agent_layers_own():
    """Guards the derivation: the list is read off real routes, and holds the
    routes a page needs and none of a product's: not the toy's page at ``/``,
    nor the routes Annealage Mesh registers beside this layer's."""
    prefixes = agent_route_prefixes()
    assert {"/ws", "/upload", "/asset/", "/settings", "/session/", "/login"} <= set(prefixes)
    assert "/agent/static/" in prefixes
    assert not {"/", "/static/", "/model/", "/manifest", "/callouts", "/submit"} & set(prefixes)


def test_the_stylesheet_checker_catches_a_regression():
    assert css_violations("#pins .empty { color: red; }\n")
    assert css_violations("/* the camera */ .ok { color: red; }\n") == []
