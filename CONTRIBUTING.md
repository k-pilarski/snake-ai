# Contributing

## Language

Everything that goes to GitHub is written in **English**: code, comments, docstrings, commits, branches, issues, pull requests, milestones, labels and documentation.

## Workflow

- `main` is always working. Direct pushes are not allowed (branch protection).
- Every change starts from an issue and goes through a short-lived branch and a pull request.
- Pull requests are merged with **squash merge only**. The branch is deleted automatically after merge.

## Branch naming

`<type>/<issue-number>-<short-description>`

Example: `feat/12-snake-env`

## Commit messages (Conventional Commits)

`<type>(<scope>): <description> (#<issue-number>)`

Example: `feat(game): add step() and reset() (#12)`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

Rules: imperative mood ("add", not "added"), lowercase, no trailing period.

## Pull requests

- Title follows the commit format, including `(#<issue-number>)`, because it becomes the commit message on `main`.
- The description must contain `Closes #<issue-number>` so the issue is closed and moved to Done on merge.
- Fill in the PR template checklist before requesting review.

## Board automation

| Event | Result |
|---|---|
| Issue created | Added to the project, status: Backlog |
| Branch work started | Status: In progress |
| Pull request opened | Status: In review |
| Pull request merged | Issue closed, status: Done |

## Definition of done

- Acceptance criteria from the issue are met
- Code is tested and runs locally
- README and docs are updated if behavior changed

## Branch protection

Rules for `main` are defined in `.github/branch-protection.sh` and can be re-applied with:

    bash .github/branch-protection.sh

Summary: pull request required, `lint-and-test` check must pass, branch must be up to date, linear history, no force pushes, no deletion.

## Branch automation

Creating a branch named `type/NR-description` runs `.github/workflows/branch-in-progress.yml`, which moves issue NR to **In progress** on the project board. The workflow needs a fine-grained personal access token with *Projects: read and write* stored as the repository secret `PROJECT_TOKEN`. Renew it before it expires.
