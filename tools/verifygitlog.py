#!/usr/bin/env python3
"""Check that commit messages carry a Developer Certificate of Origin sign-off.

Cut down from Annealage Canvas's tools/verifygitlog.py (itself MicroPython's)
to the one rule enforced here: the last paragraph of every commit message has
a `Signed-off-by: Name <email>` line, which `git commit -s` adds. The sign-off
is the contributor's DCO certification (CONTRIBUTING.md), the provenance
record the dual-licence model relies on.

Merge commits are skipped: the merge CI creates to test a pull or merge
request has no sign-off, and a merge carries no authored change of its own,
since contributions are rebased rather than merged.

usage:
  verifygitlog.py [git log arguments]  check commits, e.g. origin/main..HEAD
  verifygitlog.py --check-file FILE    check a candidate message (commit-msg hook)
"""

import re
import subprocess
import sys

SIGN_OFF = re.compile(r"^Signed-off-by: \S.* <[^<>@\s]+@[^<>\s]+>$")
# `git commit -v` appends the diff below this line, uncommented.
SCISSORS = "# ------------------------ >8 ------------------------"


def signed_off(message):
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", message.strip()) if p.strip()]
    return bool(paragraphs) and any(
        SIGN_OFF.match(line.rstrip()) for line in paragraphs[-1].splitlines()
    )


def check_file(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(SCISSORS):
                break
            if not line.startswith("#"):
                lines.append(line)
    if signed_off("".join(lines)):
        return []
    return ["commit message"]


def check_commits(git_log_args):
    out = subprocess.run(
        ["git", "log", "--no-merges", "-z", "--format=%h%n%B", *git_log_args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    failures = []
    count = 0
    for entry in filter(None, out.split("\0")):
        sha, _, message = entry.partition("\n")
        count += 1
        if not signed_off(message):
            failures.append("commit " + sha)
    print(f"checked {count} commit(s)")
    return failures


def main(args):
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args else 2
    if args[0] == "--check-file":
        if len(args) != 2:
            print("usage: verifygitlog.py --check-file FILE", file=sys.stderr)
            return 2
        failures = check_file(args[1])
    else:
        try:
            failures = check_commits(args)
        except subprocess.CalledProcessError as e:
            print(f"error: git log failed ({e.returncode})", file=sys.stderr)
            return 1
    for failure in failures:
        print(
            f"error: {failure}: no Signed-off-by: line in the message's last paragraph."
            ' Use "git commit -s" (or "git commit --amend -s").'
        )
    if failures:
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
