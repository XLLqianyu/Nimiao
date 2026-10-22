# AGENTS.md

Instructions for AI coding agents working on Nimiao.
[CONTRIBUTING.md](CONTRIBUTING.md) is the human-facing policy; this file
restates the parts an agent must not get wrong and points at the code that
enforces them.

## The repository

Nimiao is a small, from-scratch AI agent core in Rust. An OpenAI-compatible
model is bound to four filesystem and shell tools, sandboxed to a workspace
directory, and driven through a bounded agent loop. The binary is `nimiao`,
built from `nimiao-cli`.

## Read before you change

- Read the whole crate you are about to touch. The crates are small; a search
  snippet hides the invariants that make the code correct.
- Keep the dependency direction one-way: `protocol` has no internal
  dependencies; `sandbox` and `execpolicy` are standalone; `tools` builds on
  `protocol`, `sandbox`, and `execpolicy`; `model` builds on `protocol`;
  `engine` composes `model` and `tools`; `nimiao-cli` wires everything
  together. A new edge pointing backwards is a design error, not something to
  work around.
- One crate, one responsibility. Adding a crate needs a reason a reviewer can
  check.

## Checks

Run the same four checks CI runs, from the repository root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

Narrow a run with `-p`, for example `cargo test -p model`. `cargo deny check`
needs `cargo-deny` installed; if it is not, say so rather than skipping it
silently.

## Branches and commits

[CONTRIBUTING.md](CONTRIBUTING.md) has the full model. In short:

- `develop` is the integration branch; `main` holds released code only and is
  reached through `release/*` and `hotfix/*` merges. Work happens on
  `feature/<slug>` branched from `develop`.
- Commit subjects follow Conventional Commits: `<type>(<scope>): <subject>`,
  lowercase imperative subject, no trailing period, 72 characters at most.
- The `commit-msg` hook in `.githooks` enforces that format. Enable it once per
  clone with `git config core.hooksPath .githooks`. Never bypass a hook with
  `--no-verify`.
- Commits carry no co-author or tool-attribution trailers. Do not add them.

## Code

- `unsafe` is limited to the documented Win32 console-mode call in
  `nimiao-cli/src/main.rs`. A new block needs a `// SAFETY:` comment that
  justifies it, and needs to be unavoidable.
- Modules open with a `//!` summary and public items get `///`, as `sandbox`,
  `tools`, and `model::sse` do. Comment why something is done, not what the
  line does; the code already says what.
- Errors are `anyhow::Result` at crate boundaries. Inside a tool the split is
  load-bearing: `Err` means the tool could not run at all (bad arguments, an
  inaccessible path), while `ToolOutput::failure` means it ran and the model
  should see the outcome. A non-zero exit status is a failure, not an error.
- Wire-facing names are camelCase through `#[serde(rename = "...")]`; internal
  names stay snake_case.
- `#[allow(...)]` needs a one-line reason. The two in `execpolicy` show how
  narrow the allowance should be.

## Tests

- Unit tests sit in a `#[cfg(test)] mod tests` block at the bottom of the
  module they test. Integration tests go in `<crate>/tests/`.
- Temporary workspaces are created under `std::env::temp_dir()` with the crate
  name, the process id, and an atomic counter, then canonicalized before being
  handed to `Sandbox`. Copy the helper in `tools/tests/tools.rs`.
- No network in tests. `model/tests/streaming.rs` starts a hand-rolled HTTP/1.1
  server on a loopback port; reuse that harness instead of adding a mock-server
  dependency.
- Tests run on Linux and Windows. Platform-specific cases use
  `#[cfg(windows)]` / `#[cfg(unix)]`, as `sandbox` does.
- A behaviour change needs a test that fails without it. If something genuinely
  cannot be tested, say why in the pull request.

## Safety invariants

Do not weaken these in passing:

- Every model-supplied path goes through `Sandbox::validate_path`. Do not join
  against the workspace root by hand.
- Every shell command goes through `execpolicy::check_command` before it
  reaches a shell.
- Output limits live in `tools/src/truncate.rs` (2000 lines or 50 KB, whichever
  is hit first), the `bash` timeout bounds in `tools/src/bash.rs` (30 seconds
  by default, 600 at most), and the per-turn step cap in `engine/src/lib.rs`
  (`MAX_STEPS_PER_TURN = 32`). Changing any of them is a behaviour change and
  needs a test plus a README update.

## Files that are not part of the repository

- Third-party reference checkouts are kept out of the repository. They are not
  tracked and not workspace members. Do not edit them, stage them, or add them
  to the workspace.
- `target/` is build output.
- `Cargo.lock` is committed and updated by Cargo, never by hand.

## Documentation

`README.md` covers user-facing behaviour, `CONTRIBUTING.md` covers process, and
crate-level modules cover design. When a change alters a flag, a limit, a
rendered output, or the fallback path, update `README.md` in the same change.
Release notes are per version under [Releases](README.md#releases).

## Definition of done

- The four checks pass.
- New behaviour has tests.
- `README.md` and the crate docs match the code.
- Commit subjects are conventional, and the branch targets `develop`.
