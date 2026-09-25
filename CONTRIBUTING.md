# Contributing to Annealage Agent

Thanks for your interest. Read the licensing section before submitting a patch; it sets out the grant you make to the project by submitting.

## Commit messages

```
<Capitalised subject in the imperative, under 72 characters>

Optional body explaining what the change makes true and why, wrapped at
75 characters per line.

Signed-off-by: Your Name <you@example.com>
```

## Pull requests

- Rebase onto current `main` before submitting; do not merge `main` into your branch.
- One logical change per pull request. Small focused PRs are easier to review and revert.
- Include a test or a reproducer in the same PR where it is reasonable to do so.
- Run `uv run --extra dev --extra codex pytest -q` locally before pushing.
- Run the suite of every product that depends on this package as well (Annealage Mesh's includes the browser suite that exercises this package's front end); a change here is a change to each of them.
- Install the pre-commit hook once per checkout (`uv run --extra dev pre-commit install`), so `ruff check --fix` and `ruff format` run automatically on `git commit`. CI's `lint` job runs the identical hooks (`.pre-commit-config.yaml`) and fails a PR where they were skipped.

## Contribution licensing

Annealage Agent is offered under the PolyForm Noncommercial License 1.0.0. Andrew Leech also offers commercial licences to organisations whose use is not permitted under the PolyForm Noncommercial License. For that dual model to work, contributions need a clear licensing grant; otherwise contributed code could not be offered under a commercial licence without re-asking every contributor.

By submitting a contribution (a pull request, patch, or any change) to this project, you agree to the following.

### Developer Certificate of Origin

You certify the contribution under the Developer Certificate of Origin 1.1 (<https://developercertificate.org/>). Sign off each commit with:

```
Signed-off-by: Your Name <your.email@example.com>
```

(`git commit -s` adds this line.) The sign-off certifies that you wrote the contribution or otherwise have the right to submit it under the terms below.

### Licence grant

You grant Andrew Leech a perpetual, worldwide, irrevocable, royalty-free, sublicensable, and transferable licence to use, reproduce, modify, distribute, and relicense your contribution, in whole or in part, under any terms, including the PolyForm Noncommercial License and any commercial licence Andrew Leech offers, now or in future.

You confirm you have the right to grant this licence (the contribution is your own work, or you are authorised to submit it under these terms). The grant is royalty-free: no payment is due to you for it.

### Outbound licence

Your contribution is also made available to the public under the outbound licence of the repository, PolyForm Noncommercial 1.0.0. Your own rights to your contribution are not otherwise affected; you retain copyright in your work.

### Other terms

If you cannot grant the licence above, for example, your employer owns the work and has not authorised this grant, or you require a separate contributor agreement, contact andrew@alelec.net before submitting, so alternative arrangements can be made.
