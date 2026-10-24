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

When both input and output are terminals, Nimiao opens a full-screen TUI. Use
`--plain` (or redirect either stream) for the line-oriented REPL. Replies appear
as they are generated in both modes.

```bash
cargo run -p nimiao-cli -- --plain
```

| Variable | Required | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` |
| `OPENAI_MODEL_NAME` | no | `gpt-4o` |

Any endpoint that speaks the OpenAI chat-completions API works, including local
servers. Point `OPENAI_BASE_URL` at it and set `OPENAI_MODEL_NAME` accordingly.

## Interactive client

The TUI keeps the transcript, composer, running status, and queued prompts on
one screen. It renders assistant prose as a small Markdown subset: headings,
lists, quotes, inline code, and fenced code. Tables, syntax highlighting, and
diffs stay plain text.

| Key | Action |
| --- | --- |
| `Enter` | Send, or queue while a turn is running |
| `Tab` | Queue while running; send when idle |
| `Shift+Enter` | Insert a newline |
| `Esc` | Interrupt the current turn |
| `PgUp` / `PgDn` | Scroll the live transcript |
| `Ctrl+Home` / `Ctrl+End` | Jump to the start or latest output |
| `Ctrl+T` | Open or close the detailed transcript |
| `Ctrl+C` | Interrupt when running; quit when idle |
| `Ctrl+D` | Quit when the composer is empty |

Queued prompts start automatically, one at a time, after a successful turn. An
error pauses the queue so it is not sent blindly; press `Tab` on an empty
composer to send the next one. Interrupting keeps the partial reply visible with
an `[interrupted]` marker, but that partial text is not added to model history.
Any queued prompts and the current draft are merged back into the composer.

The detail page scrolls with `↑`/`↓`, `j`/`k`, `PgUp`/`PgDn`, `Space`, `Ctrl+B`/
`Ctrl+F`, and `Ctrl+U`/`Ctrl+D`; `Home`/`End` jump to either end. Close it with
`q`, `Ctrl+C`, or `Ctrl+T`.

Commands are `/help`, `/clear`, and `/quit`. `/clear` clears both the visible
transcript and model history, and only runs while no turn is active. The plain
REPL keeps its original line-oriented behaviour and exits with `exit`, `quit`,
or EOF.

## How it works

One turn is a loop: the model is asked what to do next, any tool calls it
returns are executed, and the results go back as conversation items. The loop
ends when the model answers without asking for a tool. A turn is capped at 32
tool rounds so a model that never stops calling tools cannot run forever.

```
user input
    │
    ▼
┌──────────────────────────────────────────────┐
│ complete_stream(history, tool definitions)   │◄────┐
└──────────────────────────────────────────────┘     │
    │  text deltas render as they arrive             │
    ├── no tool calls ──► reply, turn ends            │
    │                                                 │
    └── tool calls ──► execute ──► results ───────────┘
```

Conversation state lives in one `Vec<Item>`. Every user message, assistant
reply, tool call, and tool result is appended to it, so each turn sees
everything before it.

That flat list is regrouped into the wire format on every request:

| History | Wire message |
| --- | --- |
| `UserMessage` | one `user` message |
| `AssistantMessage` followed by the `ToolCall`s of the same step | **one** `assistant` message carrying the text as `content` and every call in `tool_calls` |
| `ToolResult` | one `tool` message with the matching `tool_call_id` |

Providers reject a step split across several `assistant` messages, so the
grouping is not cosmetic. An `assistant` message with neither text nor calls is
skipped rather than sent empty, and a call that arrives with no text produces
`content: null`.

**History stays answerable.** A provider also rejects a call that is never
answered, so every call the model declares gets exactly one result, in the
order declared. Calls that never run are recorded as failed instead of being
dropped: a cancelled turn answers the in-flight call and the rest of its batch,
a reply cut off at the token limit answers each call it carried, and a call
whose arguments did not parse answers with the parse error. A turn that ends
early — cancelled, capped at 32 rounds, refused by the provider, or failed
mid-way — adds one closing assistant message, so the next user prompt never
follows a `tool` message.

## Streaming

Each step is requested with `stream: true` and read as server-sent events. Prose
appears token by token in the transcript (under the `Agent:` label in plain
mode). A tool call is not rendered as it streams; once the model finishes
deciding, it produces one status line before the tool runs:

```
> bash(cargo test)
```

The preview shows the `command` argument, or the `path` for file tools, on a
single line, cut to 60 characters.

Some OpenAI-compatible proxies do not support streaming. Nimiao handles that
without configuration:

- A `400` or `422` rejection of the request is retried once without `stream`.
- A `200` whose content type is not `text/event-stream` is read whole.
- A response that claims to be a stream but carries plain JSON is read whole.
- A single SSE frame that holds a finished `message` instead of a `delta` is
  used as the reply.

In all four cases the reply is still correct and a gray `note:` line on stderr
says the endpoint did not stream. Other failures, such as an invalid key or a
rate limit, are reported as errors and are not retried.

**Stalled connections.** A connection that cannot be opened fails after 10
seconds, and one that goes silent fails after 120 seconds without a read. The
read bound applies to each read rather than the whole reply, so a long stream
that keeps arriving is never cut off, while a dead endpoint fails the turn
instead of hanging it.

**Stop reasons.** The provider's `finish_reason` is honoured. A reply cut off
at the output token limit with tool calls in it has none of those calls run:
their arguments may be truncated mid-value, so each is recorded as a failed
result telling the model to re-issue it, and the turn ends with an error. A
cut-off reply without tool calls is kept and a note says it was shortened. A
`content_filter` finish is reported as an error naming it, rather than as the
misleading "empty response". A provider that omits `finish_reason` altogether,
as some proxies do, is accepted.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `protocol` | Wire types: `Item`, `ToolCall`, `ToolDefinition` |
| `sandbox` | Workspace-scoped path validation |
| `execpolicy` | Static screening of shell commands |
| `tools` | The four built-in tools |
| `model` | `Model` trait, SSE decoding, and an OpenAI-compatible implementation |
| `engine` | The `Agent` loop |
| `nimiao-cli` | The TUI and plain-REPL binary, `nimiao` |

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

**Malformed tool arguments.** When the model finishes a call whose arguments
text is not a JSON object, that call is not run: it would only fail against a
schema it never composed. The result tells the model the arguments were
invalid and repeats the text it sent, so it can re-issue the call. The request
body still carries a valid `{}` for that call, because providers require
`arguments` to be JSON.

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
and fork bombs. It also looks through the wrappers that hide a command from a
first reading: `$(...)`, backticks and `<( )` substitutions; `find`'s `-delete`
and `-exec` family; and backslashes inside a program name (`r\m`). A pipe that
feeds a shell or an interpreter with no script of its own (`curl … | sh`) is
refused, because the code it runs never appears in the command line. A
redirection that would write outside the workspace (`> /etc/passwd`,
`> ../../x`) is refused as well; `> /dev/null`, `2>&1` and a path inside the
workspace are fine. The comparison is lexical, so a symlink inside the
workspace that points outside it is not caught.

It is a best-effort static check. It does **not** see: a command assembled from
variables, a script stored on disk, an interpreter not on its list, nesting
deeper than 8 levels, shell aliases and functions, or anything an `ssh` session
does on the far side. It cannot see what a script does after it starts.

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

`References/` is ignored through `.gitignore` and holds third-party reference
checkouts used while working on Nimiao. Nothing in it is tracked, built, or
published.

## Releases

**0.0.3.** The interactive client is now a full-screen TUI built with ratatui:
a live transcript, a multi-line composer, a prompt queue, cancellation,
Markdown rendering, scrolling, and a detail page for full tool output.
Redirected streams and `--plain` keep the line-oriented REPL. See
[Interactive client](#interactive-client).

**0.0.2.** Replies stream from an OpenAI-compatible endpoint as server-sent
events, rendered as they arrive. Each tool call prints one status line before it
runs, and endpoints that cannot stream fall back to a one-shot reply with a note
on stderr. See [Streaming](#streaming).

**0.0.1.** The first release: the agent loop, the four built-in tools, the
sandbox, command screening, and the REPL.

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
