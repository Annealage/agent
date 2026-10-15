# Releasing Annealage Ember to PyPI

Annealage Ember publishes to PyPI via GitHub Actions trusted publishing
(OIDC), so there's no API token to store. First-time setup, then it's one
release per version.

The version is not written down anywhere: `hatch-vcs` takes it from the git
tag at build time. So the tag is the version, and there is nothing to keep in
step with it.

## The first release comes first

Annealage Mesh and Annealage Loom both depend on this package through a uv
path source to a sibling checkout (`[tool.uv.sources]` in their
`pyproject.toml`). A path source never reaches published metadata, so neither
product can publish a release until this package has one: Mesh's own publish
workflow refuses to build while the path source is there. The order is:

1. Push `main` and its tags to `Annealage/ember` (`git push origin main --tags`)
   before publishing a release. The repository is public, so product CI can
   fetch the pinned `lib/ember` submodule without a private read token.
2. Consumers pin the Ember commit they tested in their `lib/ember` submodule.
   A product's CI must fetch that submodule revision, including its tags where
   the build uses `hatch-vcs` for the package version.
3. Release this package (below).
4. In each product, raise the `annealage-ember` floor to that release and
   delete the path source. A path source to a git submodule of this repository
   is fine for development, but it is still a path source, so a product that
   publishes has to depend on the release.
5. Drop the sibling checkout (or submodule checkout) of this repository from
   Mesh's CI.

## One-time setup

The repository has to exist first (step 1 above).

1. On PyPI, add a "pending publisher" for a new project (Account → Publishing):
   - PyPI project name: `annealage-ember`
   - Owner: `Annealage`
   - Repository: `agent`
   - Workflow: `publish.yml`
   - Environment: leave blank (the workflow doesn't use one)

That's it: PyPI now trusts this repo's `publish.yml` to upload `annealage-ember`.

## Cutting a release

1. Tag it and push the tag: `git tag v0.1.0 && git push origin v0.1.0`.
2. Create a GitHub Release for that tag. The `publish` workflow runs the whole
   test suite against that commit, builds, checks that what it built carries the
   tag's version, and uploads to PyPI.

A version on PyPI cannot be replaced once uploaded, which is why the suite runs
inside the publish workflow rather than being trusted from an earlier run.

Run the suite of every product that depends on this package against the
commit you're about to tag as well. Annealage Mesh's includes the browser suite
that drives this package's front end, which nothing in this repository does.

## Checking a build before tagging

`uv build` writes an sdist and a wheel into `dist/`. Two things are worth
checking by hand when the packaging itself has changed:

    uv build
    uvx twine check dist/*
    # the front end actually shipped, since a gitignore pattern has
    # silently dropped static files before (see the artifacts note in
    # pyproject.toml)
    unzip -l dist/*.whl | grep -c static/

The version in those filenames will be a dev version (`0.1.1.dev4`) unless you
are exactly on a tag, and `0.0.1.devN` before the first tag exists. That is
expected: `local_scheme = "no-local-version"` keeps it uploadable, but only a
tagged build produces a release version.
