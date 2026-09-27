---
name: shipping-the-primer
description: Merge finished work into main and publish the site: the full gate, one branch at a time, then push and watch the deploy. Use when merging agent or feature branches, or before any push to main.
---

# Shipping the primer

Every push to `main` publishes the site: [`.github/workflows/pages.yml`](../../../.github/workflows/pages.yml) runs
the same gate as below and deploys `docs/html` to GitHub Pages. A red gate
locally is a red deploy, so main only moves when the gate is **green**.

## 1. Merge one branch at a time

```bash
git merge --no-ff <branch> -m "Merge <what it adds>"
```

Conflicts come from shared files several branches extend at once:

- [`primer/glossary.py`](../../../primer/glossary.py) merges as a union (see
  [`.gitattributes`](../../../.gitattributes)), so both sides' new terms stay.
  If both added the same term, [`tests/test_glossary.py`](../../../tests/test_glossary.py) fails on it: keep the
  clearer definition, once.
- Generated files ([`docs/papers/assets/glossary.js`](../../../docs/papers/assets/glossary.js), `catalog.js`): take
  either side, then `make docs` rewrites them from their sources.
- A lesson's `## The papers behind this lesson`: keep both sides' links.

Run `make test` after each merge, before the next one.

**Done when** every finished branch is merged and `make test` is green.

## 2. The gate

Stage everything first (`git add -A`): some checks only see files git knows
about, so a new file that isn't staged passes locally and fails on CI.

```bash
make gate         # test, docs, sitecheck, browsercheck; stops at the first failure
make links        # network; before a push that adds or changes a URL
```

Commit only after `make gate` has passed in the same command, as in
`make gate && git commit ...`, so a red gate can never reach main.

`make links` lists servers that refuse automated checks (dl.acm.org and some
doi.org links answer 403) as "check by hand": confirm each one is the right
paper, and treat a 404 as a real break.

**Done when** all of it passes.

## 3. Publish

```bash
git push
gh run watch "$(gh run list -L 1 --json databaseId -q '.[0].databaseId')" --exit-status
```

Then remove what the merge left behind: `git branch -d <branch>` and
`git worktree remove <path>` for each merged branch.

**Done when** the deploy run is green, and `git branch` and `git worktree list`
show only `main`.
