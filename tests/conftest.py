"""Fixtures shared by this package's suite.

Importing ``toy_product`` and installing ``TOY`` here makes the toy product
(``tests/toy_product.py``) the one every test runs as, so a test module that
imports only agent-layer modules still has a product installed.
``swap_product`` is the one sanctioned way for a test to run as some other
product, and it puts the toy back afterwards.

``create_toy_app`` is the toy product's application: the agent layer's
``create_app`` with the toy's page and ``/`` route plugged in, the shape a
real product's own ``create_app`` has. ``served_dir`` and ``client`` are the
smallest served directory and a viewer-only app over it.
"""

import contextlib

import pytest
from microdot.test_client import TestClient
from toy_product import TOY, TOY_PAGE, register_toy_routes

from annealage_agent import app as agent_app
from annealage_agent import net
from annealage_agent import product as agent_product

agent_product.install(TOY)

# The port an app built by a test is told it listens on. Nothing binds it; it
# only feeds the Origin and Host allowlists the app computes from its bind.
DEFAULT_PORT = 8765

# Every app validates the inbound Host header against the address it is bound
# to, so a test client has to send one that truthfully names this app or every
# request is refused before it reaches a route. microdot's TestClient
# otherwise defaults to "example.com:1234", which is exactly the mismatched
# name that check exists to refuse.
TEST_HOST = "127.0.0.1"
TEST_AUTHORITY = "%s:%d" % (TEST_HOST, DEFAULT_PORT)


def make_test_client(app):
    """A TestClient whose Host header names the bind ``app`` was built for."""
    return TestClient(app, host=TEST_AUTHORITY)


def create_toy_app(
    serve_dir,
    *,
    token=None,
    agent_token=None,
    host=net.DEFAULT_HOST,
    port=DEFAULT_PORT,
    extra_origins=(),
    session_id=None,
    build_session=None,
    settings=None,
    login=None,
):
    """The toy product's app over ``serve_dir``: ``agent_app.create_app`` with
    the toy page, whose inline script the Content-Security-Policy hashes, and
    the toy's ``/`` route. Arguments are ``create_app``'s own."""
    return agent_app.create_app(
        serve_dir,
        page_html=TOY_PAGE,
        port=port,
        token=token,
        agent_token=agent_token,
        host=host,
        extra_origins=extra_origins,
        session_id=session_id,
        build_session=build_session,
        register_routes=register_toy_routes,
        settings=settings,
        login=login,
    )


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path_factory, monkeypatch):
    """Point the user configuration directory at a scratch path for every test.

    The workspace-trust store records which directories' agent configuration a
    human has accepted, and it lives in the user's own configuration directory
    by design, beside the user ``settings.toml``. A test that reached the real
    one would record acceptances against the developer's account, and one that
    read it could pass or fail according to what that developer had accepted
    earlier.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config")))


@pytest.fixture(autouse=True)
def one_backend_installed(monkeypatch):
    """Report exactly one installed agent backend, ``claude``, for every test.

    With no backend in any setting, agent mode picks from what is installed,
    and asks when there is more than one. A test that let the real PATH
    decide would pass on a machine with one CLI and fail on a CI runner with
    none; ``tests/test_backends.py`` covers the choosing itself.
    """
    from annealage_agent import backends

    monkeypatch.setattr(backends, "detect", lambda **_kw: ("claude",))


@pytest.fixture
def served_dir(tmp_path):
    (tmp_path / "notes.json").write_text('["first"]', encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(served_dir):
    return make_test_client(create_toy_app(served_dir, host=TEST_HOST, port=DEFAULT_PORT))


@contextlib.contextmanager
def product_swapper():
    """Yields ``swap(product)``, which installs ``product`` in place of the
    toy; on leaving the block the toy product, and every registration it made,
    is restored however the block ends. ``swap_product`` is this as a fixture;
    ``tests/test_product.py`` uses it directly to check the restore itself."""

    def _swap(product):
        agent_product.reset()
        agent_product.install(product)

    try:
        yield _swap
    finally:
        agent_product.reset()
        agent_product.install(TOY)


@pytest.fixture
def swap_product():
    """``swap_product(product)`` installs ``product`` in place of the toy for
    the rest of one test; the toy is restored afterwards, so nothing leaks
    into the next test (``product_swapper``)."""
    with product_swapper() as swap:
        yield swap
