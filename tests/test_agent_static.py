"""The agent layer's own front end over HTTP: ``GET /agent/static/<rel>``
(``http/static.py``), served from ``static/`` through an index of its own,
apart from a product's ``/static/`` tree; and the machinery every indexed
route shares.

The first half covers what is particular to this route: that it serves the
real tree, that the agent tree and a product's never answer for each other,
and that nothing of the Python package the tree sits inside is reachable
through it. The product tree beside it is a route this file registers the
way a product does (Annealage Mesh's ``/static/``): ``static_tree`` over a
directory of its own.

The second half covers the machinery in ``http/static.py`` that every indexed
route goes through (``static_tree``, ``serve_indexed``,
``maybe_not_modified`` and ``make_index_cache``): the extension allowlist,
symlink refusal, rescan on a replaced file, conditional requests, and one
scan shared by a burst of requests. It is exercised once, here, through this
route, over the real tree where the real files suffice and over a throwaway
tree where a test has to plant or change a file.
"""

import asyncio
import os
import time
from pathlib import Path
from urllib.parse import unquote

import pytest
from conftest import DEFAULT_PORT, TEST_HOST, create_toy_app, make_test_client
from toy_product import TOY_PAGE, register_toy_routes

from annealage_agent import app as agent_app
from annealage_agent import files
from annealage_agent.http import static as agent_static

pytestmark = pytest.mark.asyncio


async def test_every_file_of_the_real_tree_is_served_and_revalidatable(client):
    # Enumerates the tree rather than naming files, so a module added later is
    # covered without anyone remembering to add it here.
    entries, _ = files.scan_static(agent_static.AGENT_STATIC_DIR)
    assert {"chat.js", "store.js", "ws.js", "agent.css"} <= {e["rel"] for e in entries}
    for entry in entries:
        rel = entry["rel"]
        res = await client.get("/agent/static/" + rel)
        assert res.status_code == 200, rel
        assert res.body == Path(entry["path"]).read_bytes(), rel
        assert res.headers.get("Content-Type") == files.StaticIndex.content_type_of(rel), rel
        assert res.headers.get("Cache-Control") == "no-cache", rel
        again = await client.get(
            "/agent/static/" + rel, headers={"If-None-Match": res.headers["ETag"]}
        )
        assert again.status_code == 304, rel


def _toy_routes_with_a_static_tree(product_dir):
    """The toy's ``register_routes`` plus ``GET /static/<rel>`` over
    ``product_dir``, registered the way a product serves its own packaged
    tree: one ``static_tree`` of its own."""
    serve = agent_static.static_tree(product_dir)

    def register(app, allowed_origins):
        register_toy_routes(app, allowed_origins)

        @app.get("/static/<path:rel>")
        async def product_static(req, rel):
            return await serve(req, unquote(rel), rel)

    return register


@pytest.fixture
def two_trees(tmp_path, monkeypatch):
    """A client whose product tree and agent tree are throwaway directories
    holding a file under the same relative name, with different bytes, plus
    one file only each tree has."""
    product_dir = tmp_path / "product_static"
    agent_dir = tmp_path / "agent_static"
    for tree, text in ((product_dir, "product"), (agent_dir, "agent layer")):
        (tree / "js").mkdir(parents=True)
        (tree / "js" / "ui.js").write_text("// the %s's ui.js\n" % text)
    (product_dir / "viewer.html").write_text("<html></html>")
    (agent_dir / "chat.js").write_text("// only the agent layer has this\n")
    # Beside both trees, with an extension either tree would serve.
    (tmp_path / "outside.js").write_text("// OUTSIDE-BOTH-TREES\n")
    monkeypatch.setattr(agent_static, "AGENT_STATIC_DIR", agent_dir)
    served = tmp_path / "served"
    served.mkdir()
    app = agent_app.create_app(
        served,
        page_html=TOY_PAGE,
        host=TEST_HOST,
        port=DEFAULT_PORT,
        register_routes=_toy_routes_with_a_static_tree(product_dir),
    )
    return make_test_client(app), product_dir, agent_dir


async def test_the_agent_tree_and_the_products_never_answer_for_each_other(two_trees):
    client, product_dir, agent_dir = two_trees

    agent_res = await client.get("/agent/static/js/ui.js")
    product_res = await client.get("/static/js/ui.js")
    assert agent_res.body == (agent_dir / "js" / "ui.js").read_bytes()
    assert product_res.body == (product_dir / "js" / "ui.js").read_bytes()
    assert agent_res.headers["ETag"] != product_res.headers["ETag"]

    # A validator one tree issued never confirms the other tree's file of the
    # same name: a browser that cached the product's module must still be
    # sent the agent layer's.
    crossed = await client.get(
        "/agent/static/js/ui.js", headers={"If-None-Match": product_res.headers["ETag"]}
    )
    assert crossed.status_code == 200
    assert crossed.body == agent_res.body
    crossed = await client.get(
        "/static/js/ui.js", headers={"If-None-Match": agent_res.headers["ETag"]}
    )
    assert crossed.status_code == 200
    assert crossed.body == product_res.body

    # And a name only one tree has is reachable only under that tree's prefix.
    assert (await client.get("/static/chat.js")).status_code == 404
    assert (await client.get("/agent/static/viewer.html")).status_code == 404


async def test_no_file_outside_the_agent_tree_is_reachable_through_it(two_trees):
    """Containment is the index's, not the extension allowlist's: a ``.js``
    file one directory up, and the product's own tree beside it, are both
    servable extensions and both unreachable under /agent/static/."""
    client, _product_dir, _agent_dir = two_trees
    payloads = [
        "/agent/static/%2e%2e/outside.js",
        "/agent/static/..%2foutside.js",
        "/agent/static/../outside.js",
        "/agent/static/%2e%2e/product_static/js/ui.js",
        "/agent/static/..%2fproduct_static%2fviewer.html",
    ]
    for path in payloads:
        res = await client.get(path)
        assert res.status_code == 404, path
        assert b"OUTSIDE-BOTH-TREES" not in (res.body or b""), path
        assert b"product's ui.js" not in (res.body or b""), path


async def test_nothing_beside_the_tree_is_reachable_through_it(client):
    # static/ sits inside the Python package, next to its source, and three
    # levels up from it, in a source checkout, is this suite's own toy page,
    # whose extension the allowlist accepts.
    payloads = [
        "/agent/static/../app.py",
        "/agent/static/%2e%2e/app.py",
        "/agent/static/..%2fapp.py",
        "/agent/static/%2e%2e/http/static.py",
        "/agent/static//../product.py",
        "/agent/static/%2e%2e/%2e%2e/%2e%2e/tests/toy_page.html",
        "/agent/static/..%2f..%2f..%2ftests%2ftoy_page.html",
    ]
    for path in payloads:
        res = await client.get(path)
        assert res.status_code == 404, path
        assert b"def " not in (res.body or b""), path
        assert b"import" not in (res.body or b""), path
        assert b"<html" not in (res.body or b""), path


# --- the machinery every indexed route shares -----------------------------------


async def test_static_head_matches_get(client):
    res_get = await client.get("/agent/static/agent.css")
    res_head = await client.request("HEAD", "/agent/static/agent.css")

    assert res_head.status_code == res_get.status_code
    assert res_head.headers.get("Content-Type") == res_get.headers.get("Content-Type")
    assert res_head.headers.get("Content-Length") == res_get.headers.get("Content-Length")
    assert res_head.body is None


@pytest.fixture
def static_client_factory(tmp_path, monkeypatch):
    """Build a client whose /agent/static/<rel> route resolves against a
    throwaway static tree instead of the package's own static/ directory.

    Lets a test plant a symlink or a disallowed file and check it is refused
    without writing into (or risking leaving debris in) the real installed
    static/ tree. Returns ``(client, static_dir)`` so a test can mutate a
    file after the client already exists, for the rescan-and-retry case.
    """

    def make(build):
        static_dir = tmp_path / "fake_static"
        static_dir.mkdir()
        build(static_dir)
        monkeypatch.setattr(agent_static, "AGENT_STATIC_DIR", static_dir)
        served = tmp_path / "served"
        served.mkdir()
        app = create_toy_app(served, host=TEST_HOST, port=DEFAULT_PORT)
        return make_test_client(app), static_dir

    return make


async def test_static_refuses_traversal_shapes(static_client_factory):
    def build(static_dir):
        (static_dir / "chat.js").write_text("// chat")

    client, static_dir = static_client_factory(build)

    secret = static_dir.parent / "secret.txt"
    secret.write_text("TOPSECRET-STATIC-TRAVERSAL")

    payloads = [
        "/agent/static/../secret.txt",  # ".." traversal
        "/agent/static/%2e%2e/secret.txt",  # percent-encoded traversal
        "/agent/static/%252e%252e%2fsecret.txt",  # doubly percent-encoded
        "/agent/static/..%2fsecret.txt",  # percent-encoded separator
        "/agent/static//../secret.txt",  # doubled slash
    ]
    for path in payloads:
        res = await client.get(path)
        assert res.status_code == 404, path
        assert b"TOPSECRET" not in (res.body or b"")


async def test_static_refuses_a_symlink(static_client_factory):
    def build(static_dir):
        real = static_dir / "real.js"
        real.write_text("console.log('real');")
        (static_dir / "evil.js").symlink_to(real)

    client, _ = static_client_factory(build)

    res = await client.get("/agent/static/evil.js")
    assert res.status_code == 404

    res_real = await client.get("/agent/static/real.js")
    assert res_real.status_code == 200


async def test_static_refuses_a_disallowed_extension(static_client_factory):
    def build(static_dir):
        (static_dir / "chat.js").write_text("// chat")
        (static_dir / "notes.bak").write_text("not servable")
        (static_dir / "source.map").write_text("not servable either")

    client, _ = static_client_factory(build)

    for name in ("notes.bak", "source.map"):
        res = await client.get("/agent/static/" + name)
        assert res.status_code == 404, name


async def test_static_serves_a_replaced_file_after_one_rescan_and_retry(static_client_factory):
    # A build step or an editor's write-then-rename leaves a new inode at
    # the same name; the cached index still points at the old one until a
    # request's own identity check misses and forces a rescan.
    def build(static_dir):
        (static_dir / "js").mkdir()
        (static_dir / "js" / "app.js").write_text("console.log('old');")

    client, static_dir = static_client_factory(build)

    res1 = await client.get("/agent/static/js/app.js")
    assert res1.text == "console.log('old');"

    target = static_dir / "js" / "app.js"
    tmp = static_dir / "js" / "app.js.new"
    tmp.write_text("console.log('new');")
    os.replace(tmp, target)

    res2 = await client.get("/agent/static/js/app.js")
    assert res2.status_code == 200
    assert res2.text == "console.log('new');"


# Conditional requests. Packaged assets carry a validator so a reloading
# client can be told "unchanged" instead of refetching every module on every
# page load, while "no-cache" keeps the browser asking rather than reusing a
# cached module without checking.


async def test_static_returns_304_for_a_matching_validator(client):
    first = await client.get("/agent/static/chat.js")
    etag = first.headers["ETag"]

    second = await client.get("/agent/static/chat.js", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert not second.body
    # The 304 repeats the validator, or a client that revalidated once would
    # have nothing to revalidate with next time and would refetch in full.
    assert second.headers["ETag"] == etag
    assert second.headers.get("Cache-Control") == "no-cache"


async def test_static_returns_304_for_a_wildcard_validator(client):
    res = await client.get("/agent/static/chat.js", headers={"If-None-Match": "*"})
    assert res.status_code == 304


async def test_static_ignores_a_validator_from_a_different_file(client):
    other = await client.get("/agent/static/agent.css")
    res = await client.get(
        "/agent/static/chat.js", headers={"If-None-Match": other.headers["ETag"]}
    )
    assert res.status_code == 200
    assert res.headers["ETag"] != other.headers["ETag"]
    assert res.body


async def test_static_validator_changes_when_the_file_is_edited_in_place(static_client_factory):
    # Same inode, new content. The validator is computed from a fresh stat of
    # the file being served, not from what the cached index scan recorded, so
    # an in-place edit must not be affirmed as unchanged.
    def build(static_dir):
        (static_dir / "js").mkdir()
        (static_dir / "js" / "app.js").write_text("console.log('one');")

    client, static_dir = static_client_factory(build)
    target = static_dir / "js" / "app.js"

    first = await client.get("/agent/static/js/app.js")
    etag = first.headers["ETag"]
    inode_before = os.stat(target).st_ino

    with open(target, "r+") as fh:
        fh.write("console.log('two and a bit longer');")
    assert os.stat(target).st_ino == inode_before, "the edit must reuse the inode"

    res = await client.get("/agent/static/js/app.js", headers={"If-None-Match": etag})
    assert res.status_code == 200
    assert "two and a bit longer" in res.text
    assert res.headers["ETag"] != etag


async def test_static_does_not_affirm_a_validator_for_a_name_become_a_symlink(
    static_client_factory,
):
    # A conditional request must not become a way to have a symlink's target
    # validated: lstat sees the link itself, which is not a regular file, so
    # no 304 is issued and the request falls through to the open path that
    # refuses it outright.
    def build(static_dir):
        (static_dir / "app.js").write_text("console.log('real');")

    client, static_dir = static_client_factory(build)

    first = await client.get("/agent/static/app.js")
    etag = first.headers["ETag"]

    secret = static_dir.parent / "secret.txt"
    secret.write_text("TOPSECRET-CONDITIONAL")
    (static_dir / "app.js").unlink()
    (static_dir / "app.js").symlink_to(secret)

    res = await client.get("/agent/static/app.js", headers={"If-None-Match": etag})
    assert res.status_code == 404
    assert b"TOPSECRET" not in (res.body or b"")


async def test_static_head_carries_the_same_validator_as_get(client):
    res_get = await client.get("/agent/static/agent.css")
    res_head = await client.request("HEAD", "/agent/static/agent.css")
    assert res_head.headers["ETag"] == res_get.headers["ETag"]
    assert res_head.headers.get("Cache-Control") == "no-cache"


async def test_concurrent_requests_during_a_cold_window_share_one_scan(served_dir, monkeypatch):
    # A burst of requests arriving while the cache is cold (startup, or just
    # after the TTL expires) must not each launch their own walk; they should
    # all await the one scan already in flight. A short sleep inside the
    # (executor-run) scan widens the race window deterministically, since
    # without it a fast real scan might finish before a second concurrent
    # coroutine even reaches its own cache check.
    calls = []
    real_scan = files.build_static_index

    def slow_counting_scan(static_dir):
        calls.append(static_dir)
        time.sleep(0.05)
        return real_scan(static_dir)

    # Before the app is built: the route's index cache takes the builder when
    # it is registered.
    monkeypatch.setattr(files, "build_static_index", slow_counting_scan)
    client = make_test_client(create_toy_app(served_dir, host=TEST_HOST, port=DEFAULT_PORT))

    results = await asyncio.gather(*[client.get("/agent/static/chat.js") for _ in range(20)])

    assert len(calls) == 1
    assert all(res.status_code == 200 for res in results)
