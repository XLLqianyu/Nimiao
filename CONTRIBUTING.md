# Contributing to Nimiao

Thanks for taking the time to contribute. This document covers the branch
model, the commit format, and the checks a change has to pass before it lands.

## Branch model

Nimiao follows Git Flow. `main` and `develop` are the only long-lived
branches; everything else is short-lived and deleted after it merges.

| Branch | Branches from | Merges into | Purpose |
| --- | --- | --- | --- |
| `main` | — | — | Released code only. Every merge is tagged `vX.Y.Z`. |
| `develop` | `main` | — | Integration branch. Day-to-day work lands here. |
| `feature/<slug>` | `develop` | `develop` | One feature or change. |
| `release/<x.y.z>` | `develop` | `main` and `develop` | Version bump, changelog, last-minute fixes. |
| `hotfix/<slug>` | `main` | `main` and `develop` | Urgent fix for a released version. |

Name branches in lowercase kebab-case, for example:

```
feature/tool-timeout
feature/openai-streaming
release/0.1.0
hotfix/repl-crash
```

Merge `feature/*` into `develop` with `--no-ff` so the merge point survives.
Merge `release/*` and `hotfix/*` into `main` with `--no-ff`, then tag the
result.

**Pull requests target `develop`, not `main`.** `main` only receives merges
from `release/*` and `hotfix/*`.

## Commit messages

Nimiao uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>)!: <subject>
```

| Part | Rules |
| --- | --- |
| `type` | One of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. |
| `scope` | Optional. A crate name — `protocol`, `sandbox`, `execpolicy`, `tools`, `model`, `engine`, `cli` — or `repo` / `ci`. |
| `!` | Marks a breaking change. Explain it in the body with a `BREAKING CHANGE:` footer. |
| `subject` | Lowercase, imperative, no trailing period, at most 72 characters. |

Examples:

```
feat(tools): add timeout to the bash tool
fix(sandbox): reject device paths on windows
docs: describe the release flow
refactor(engine)!: rename run_turn to step

BREAKING CHANGE: Agent::run_turn is now Agent::step.
```

Merge commits, `Revert ...`, and interactive-rebase `fixup!` / `squash!`
prefixes are exempt.

### Enable the local commit-msg hook

The hook lives in `.githooks/commit-msg` and rejects subjects that do not
match the format above. It is not active until you point Git at it:

```bash
git config core.hooksPath .githooks
```

Run it once per clone. Git for Windows runs the hook through its bundled
`sh`, so no extra tooling is required.

## Before you open a pull request

Run the same checks CI runs:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

If `cargo fmt --check` reports differences, run `cargo fmt --all` and commit
the result.

## Pull request process

1. Branch from `develop`: `git switch -c feature/<slug> develop`.
2. Make focused commits that follow the format above.
3. Fill in every section of the pull request template.
4. Open the pull request against `develop`.
5. Wait for CI. All jobs must pass before review.
6. Once merged, delete the branch.

Keep a pull request to one concern. Split unrelated changes into separate
pull requests.

## Release flow

1. `git switch -c release/x.y.z develop`
2. Bump `version` in the workspace `Cargo.toml`, refresh `Cargo.lock`, and
   update the documentation.
3. Open a pull request into `main` and merge with `--no-ff`.
4. Tag the merge commit: `git tag -a vx.y.z -m "Nimiao x.y.z"`.
5. Push `main` and the tag: `git push origin main && git push origin vx.y.z`.
6. Merge `main` back into `develop` so the version bump is not lost.

Hotfixes follow the same steps starting from `main`, with the branch named
`hotfix/<slug>` and the patch version incremented.

## Repository settings (maintainers)

Recommended configuration on GitHub:

- Protect `main`: require a pull request, require the `test` and `clippy`
  checks, and disallow force pushes.
- Protect `develop`: require a pull request and the same checks.
- Enable "Automatically delete head branches".

## License

By contributing you agree that your contributions are licensed under the
terms in `LICENSE-APACHE` and `LICENSE-COMMONS-CLAUSE`.
