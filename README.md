# Nimiao

A small, from-scratch AI agent core in Rust. Nimiao pairs an OpenAI-compatible
model with four filesystem and shell tools, sandboxed to a workspace directory,
and drives them through a bounded agent loop.

[![CI](https://github.com/XLLqianyu/Nimiao/actions/workflows/ci.yml/badge.svg)](https://github.com/XLLqianyu/Nimiao/actions/workflows/ci.yml)

## Quick start

```bash
git clone https://github.com/XLLqianyu/Nimiao.git
cd Nimiao

export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
export OPENAI_MODEL_NAME=gpt-4o                    # optional

cargo run -p nimiao-cli
```

The REPL prints a green `>` prompt and answers with a blue `Agent:` prefix.
Type `exit` or `quit`, or send EOF, to leave.

| Variable | Required | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` |
| `OPENAI_MODEL_NAME` | no | `gpt-4o` |

Any endpoint that speaks the OpenAI chat-completions API works, including local
servers. Point `OPENAI_BASE_URL` at it and set `OPENAI_MODEL_NAME` accordingly.

## How it works

One turn is a loop: the model is asked what to do next, any tool calls it
returns are executed, and the results go back as conversation items. The loop
ends when the model answers without asking for a tool. A turn is capped at 32
tool rounds so a model that never stops calling tools cannot run forever.

```
user input
    │
    ▼
┌─────────────────────────────────────┐
│ complete(history, tool definitions) │◄────┐
└─────────────────────────────────────┘     │
    │                                       │
    ├── no tool calls ──► reply, turn ends  │
    │                                       │
    └── tool calls ──► execute ──► results ─┘
```

Conversation state lives in one `Vec<Item>`. Every user message, assistant
reply, tool call, and tool result is appended to it, so each turn sees
everything before it.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `protocol` | Wire types: `Item`, `ToolCall`, `ToolDefinition` |
| `sandbox` | Workspace-scoped path validation |
| `execpolicy` | Static screening of shell commands |
| `tools` | The four built-in tools |
| `model` | `Model` trait and an OpenAI-compatible implementation |
| `engine` | The `Agent` loop |
| `nimiao-cli` | The REPL binary, `nimiao` |

Dependencies point one way. `protocol` has no internal dependencies.
`sandbox` and `execpolicy` are standalone. `tools` builds on `protocol`,
`sandbox`, and `execpolicy`. `model` builds on `protocol`. `engine` composes
`model` and `tools`, and `nimiao-cli` wires everything together.

## Tools

| Tool | What it does |
| --- | --- |
| `read_file` | Reads a line window from a text file, with `offset` and `limit`. |
| `write_file` | Writes a file, creating parent directories. Content is never truncated. |
| `edit_file` | Applies exact-text replacements, several per call, each matched against the original file. |
| `bash` | Runs a shell command in the workspace root and returns combined stdout and stderr. |

**Limits.** File reads and command output are capped at 2000 lines or 50 KB,
whichever comes first; the model sees a notice telling it how to continue.
`bash` defaults to a 30-second timeout that the model may raise to 600 seconds.
Results longer than the cap are truncated from the tail, keeping the part that
usually matters.

**Editing.** `edit_file` matches `oldText` exactly first. If that fails it
retries against a normalized copy, tolerating smart quotes, en-dashes, and
trailing whitespace. Line endings and a byte-order mark are preserved on write.
Overlapping or non-unique edits are rejected rather than guessed at.

## Safety

Two layers, neither of which is a security boundary on its own:

**Sandbox.** Every model-supplied path is resolved against the workspace root
and rejected if it escapes. The check canonicalizes the deepest existing
ancestor, so `..` traversal and symlink escapes are caught, while paths that do
not exist yet are still accepted for writes. Windows device paths (`\\.\`,
`\\?\GLOBALROOT`, named pipes) are refused outright.

**Command screening.** `execpolicy` tokenizes a command, splits it on shell
separators, unwraps launchers such as `sudo`, `env`, and `xargs`, recurses into
nested `bash -c` payloads, and blocks a short list of catastrophic operations:
recursive deletes aimed at system roots, disk formatting, `dd` to a raw device,
and fork bombs. It is a best-effort static check; it cannot see what a script
does after it starts.

Shell commands run with stdin set to null and are killed as a process tree on
timeout, so nothing is left holding the pipes.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

Enable the commit-message hook once per clone:

```bash
git config core.hooksPath .githooks
```

Branches, commit format, and the release process are described in
[CONTRIBUTING.md](CONTRIBUTING.md). In short: Git Flow, Conventional Commits,
pull requests target `develop`.

### Platform notes

CI runs the test suite on Linux and Windows. The Windows jobs matter: `sandbox`,
`execpolicy`, and the `bash` tool all carry platform-specific code.

On Windows, the `bash` tool looks for Git for Windows in the usual install
locations and on `PATH`, skipping `%SystemRoot%\System32\bash.exe` because that
is the WSL launcher. Install [Git for Windows](https://gitforwindows.org/) if
the tool reports that no shell was found.

## License

Licensed under the Apache License 2.0 with the
[Commons Clause](LICENSE-COMMONS-CLAUSE). In short: use it freely, but do not
sell it. Commercial use requires a separate written license.

See [LICENSE-APACHE](LICENSE-APACHE) and
[LICENSE-COMMONS-CLAUSE](LICENSE-COMMONS-CLAUSE).
