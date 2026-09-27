"""
Land a finished branch on main: merge, gate, push, clean up.

    python tools/land.py companion/swe "Merge SWE-bench and SWE-agent companions"

The procedure is the one in .claude/skills/shipping-the-primer. Generated
assets (glossary.js, catalog.js) that conflict are rebuilt by `make docs`
rather than merged by hand, and two writers' deletions from LEVELS_PENDING both
win. Anything else that needs a person (a conflict in a lesson, a term both
sides defined, a red gate) stops the run with main left
mid-merge or with the merge committed but unpushed, so it can be fixed and
pushed by hand. The branch's worktree and branches are removed only after the
push.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = {"docs/papers/assets/glossary.js", "docs/papers/assets/catalog.js"}
TRAILER = "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"


def conflicts_needing_a_person(paths: list[str]) -> list[str]:
    return [p for p in paths if p not in GENERATED]


CONFLICT = re.compile(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", re.S)


def resolve_pending_conflict(source: str) -> str | None:
    """primer/curriculum.py with its conflict markers resolved, or None if any conflict lies outside LEVELS_PENDING.

    Writers delete their own lessons from LEVELS_PENDING; when two delete neighbouring groups,
    git shows each side's leftovers as a conflict. A lesson stays only if both sides kept it."""
    start = source.find("LEVELS_PENDING")
    end = source.find("})", start)
    if start < 0 or any(m.start() < start or m.end() > end for m in CONFLICT.finditer(source)):
        return None

    def keep_common(m: re.Match) -> str:
        ours, theirs = m.group(1).splitlines(keepends=True), m.group(2).splitlines(keepends=True)
        return "".join(line for line in ours if line in theirs)

    return CONFLICT.sub(keep_common, source)


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
        curriculum = ROOT / "primer" / "curriculum.py"
        if "primer/curriculum.py" in conflicted:
            resolved = resolve_pending_conflict(curriculum.read_text())
            if resolved is not None:
                curriculum.write_text(resolved)
                conflicted.remove("primer/curriculum.py")
                git("add", "primer/curriculum.py")
        if people := conflicts_needing_a_person(conflicted):
            return stop(f"conflicts to resolve by hand: {', '.join(people)}")
        if conflicted:
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
            git("worktree", "unlock", path, check=False)  # the harness locks agent worktrees
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
