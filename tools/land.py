"""
Land a finished branch on main: merge, gate, push, clean up.

    python tools/land.py companion/swe "Merge SWE-bench and SWE-agent companions"

The procedure is the one in .claude/skills/shipping-the-primer. Generated
assets (glossary.js, catalog.js) that conflict are rebuilt by `make docs`
rather than merged by hand. Anything else that needs a person (a conflict in
a lesson, a term both sides defined, a red gate) stops the run with main left
mid-merge or with the merge committed but unpushed, so it can be fixed and
pushed by hand. The branch's worktree and branches are removed only after the
push.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = {"docs/papers/assets/glossary.js", "docs/papers/assets/catalog.js"}
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def conflicts_needing_a_person(paths: list[str]) -> list[str]:
    return [p for p in paths if p not in GENERATED]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=check)


def stop(why: str) -> int:
    print(f"land: stopped: {why}")
    return 1


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    branch, message = sys.argv[1], sys.argv[2]
    merged = git("merge", "--no-ff", branch, "-m", f"{message}\n\n{TRAILER}", check=False)
    if merged.returncode != 0:
        conflicted = git("diff", "--name-only", "--diff-filter=U").stdout.split()
        if not conflicted:
            return stop(merged.stderr.strip() or merged.stdout.strip())
        if people := conflicts_needing_a_person(conflicted):
            return stop(f"conflicts to resolve by hand: {', '.join(people)}")
        git("checkout", "--theirs", *conflicted)
        git("add", *conflicted)
        git("commit", "--no-edit")
    # A term both branches added fails here, before the slow gate.
    glossary = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_glossary.py"], cwd=ROOT)
    if glossary.returncode != 0:
        return stop("tests/test_glossary.py failed (a term defined twice?): fix, commit, then run make gate and push")
    gate = subprocess.run(["make", "gate"], cwd=ROOT)
    if gate.returncode != 0:
        return stop("make gate failed: fix, commit, then push")
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-m", f"Regenerate assets after: {message}\n\n{TRAILER}")
    git("push")
    # The agent's worktree and its two branches (the job branch and the harness's worktree branch).
    for line in git("worktree", "list", "--porcelain").stdout.split("\n\n"):
        if f"branch refs/heads/{branch}" in line:
            path = line.split("\n")[0].removeprefix("worktree ")
            wt_branch = git("-C", path, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
            git("worktree", "remove", "--force", path, check=False)
            name = Path(path).name
            git("branch", "-D", f"worktree-{name}", check=False)
            if wt_branch and wt_branch != branch:
                git("branch", "-D", wt_branch, check=False)
    git("branch", "-D", branch, check=False)
    print(f"land: {branch} merged, gated and pushed: {git('log', '-1', '--oneline').stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
