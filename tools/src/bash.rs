//! `bash`: run a shell command in the workspace root.
//!
//! Commands are screened by `execpolicy`, executed through `bash -c`, and
//! bounded by a timeout. Output is streamed from both pipes with bounded
//! memory and truncated from the tail, keeping the part that usually matters.

use crate::truncate::{
    format_size, truncate_tail, TruncatedBy, TruncationOptions, DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
};
use crate::{Tool, ToolContext, ToolOutput};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use protocol::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time::Instant;

/// Timeout applied when the model does not ask for one.
const DEFAULT_TIMEOUT_SECONDS: f64 = 30.0;

/// Hard ceiling on the model-chosen timeout.
const MAX_TIMEOUT_SECONDS: f64 = 600.0;

/// How long to keep draining pipes after the shell process exits.
const DRAIN_GRACE: Duration = Duration::from_secs(1);

/// How long to wait for a killed process to be reaped.
const KILL_GRACE: Duration = Duration::from_secs(5);

/// Read chunk size.
const READ_CHUNK: usize = 16 * 1024;

/// Raw output retained while streaming. The snapshot only ever needs
/// `DEFAULT_MAX_BYTES`, so keeping a few times that allows accurate line
/// boundaries without unbounded growth.
const ROLLING_CAPACITY: usize = DEFAULT_MAX_BYTES * 4;

/// Bytes kept when the rolling buffer is trimmed.
const ROLLING_KEEP: usize = DEFAULT_MAX_BYTES * 2;

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    timeout: Option<f64>,
}

/// Runs shell commands.
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: format!(
                "Execute a bash command in the workspace root. Returns combined stdout and stderr. \
                 Output is truncated to the last {} lines or {} (whichever is hit first). \
                 A timeout can be provided in seconds; it defaults to {} and is capped at {}.",
                DEFAULT_MAX_LINES,
                format_size(DEFAULT_MAX_BYTES),
                DEFAULT_TIMEOUT_SECONDS as u64,
                MAX_TIMEOUT_SECONDS as u64
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": format!(
                            "Timeout in seconds (default {}, maximum {})",
                            DEFAULT_TIMEOUT_SECONDS as u64,
                            MAX_TIMEOUT_SECONDS as u64
                        )
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let args: BashArgs = serde_json::from_value(args)
            .map_err(|err| anyhow::anyhow!("Invalid arguments: {err}"))?;

        if args.command.trim().is_empty() {
            bail!("Invalid arguments: 'command' must not be empty");
        }

        let timeout_seconds = match args.timeout {
            None => DEFAULT_TIMEOUT_SECONDS,
            Some(value) => {
                if !value.is_finite() || value.fract() != 0.0 {
                    bail!("Invalid arguments: 'timeout' must be a whole number of seconds");
                }
                if value < 1.0 || value > MAX_TIMEOUT_SECONDS {
                    bail!(
                        "Invalid arguments: 'timeout' must be between 1 and {} seconds",
                        MAX_TIMEOUT_SECONDS as u64
                    );
                }
                value
            }
        };

        // The workspace root lets the screening refuse a redirection that
        // would write outside it, such as `> /etc/passwd`.
        execpolicy::check_command_in(&args.command, Some(ctx.workspace_root()))?;

        let shell = resolve_shell()
            .context("No bash executable found. Install Git for Windows or put bash on PATH")?;

        let mut command = Command::new(&shell);
        command
            .arg("-c")
            .arg(&args.command)
            .current_dir(ctx.workspace_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command
            .spawn()
            .with_context(|| format!("Failed to start {}", shell.display()))?;

        let mut output = OutputBuffer::default();
        let deadline = Duration::from_secs(timeout_seconds as u64);

        let outcome = tokio::time::timeout(deadline, pump(&mut child, &mut output)).await;

        let timed_out = outcome.is_err();
        let status = match outcome {
            Ok(Ok(status)) => Some(status),
            Ok(Err(err)) => {
                reap(&mut child).await;
                return Err(anyhow::Error::new(err).context("Failed to wait for the command"));
            }
            Err(_) => {
                // The future was dropped, closing the pipe handles. The shell
                // itself must be killed explicitly: `kill_on_drop` only fires
                // when `Child` is dropped, and we still need to reap it.
                kill_process_tree(&mut child).await;
                None
            }
        };

        let (body, has_output) = render_output(&output, timed_out);

        if timed_out {
            return Ok(ToolOutput::failure(append_status(
                &body,
                has_output,
                format!("Command timed out after {} seconds", timeout_seconds as u64),
            )));
        }

        let code = status.map(exit_code).unwrap_or(1);
        if code != 0 {
            return Ok(ToolOutput::failure(append_status(
                &body,
                has_output,
                format!("Command exited with code {code}"),
            )));
        }

        Ok(ToolOutput::success(body))
    }
}

/// Kills the shell and waits, with a bound, for it to be reaped.
async fn reap(child: &mut Child) {
    let _ = child.start_kill();
    let _ = tokio::time::timeout(KILL_GRACE, child.wait()).await;
}

/// Kills the shell together with the processes it started.
///
/// Killing only the shell would leave grandchildren running; on Windows they
/// keep the pipe handles open and block reading until they finish on their own.
async fn kill_process_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;

        if let Some(pid) = child.id() {
            let _ = tokio::process::Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid.to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(CREATE_NO_WINDOW)
                .status()
                .await;
        }
    }

    reap(child).await;
}

/// Drains both pipes until they close and the child exits.
///
/// A single task with `select!` avoids two detached reader tasks and the
/// cleanup they would need. Once the child exits, pipes may still be held open
/// by grandchildren, so draining stops after [`DRAIN_GRACE`].
async fn pump(child: &mut Child, output: &mut OutputBuffer) -> std::io::Result<ExitStatus> {
    let mut stdout = child.stdout.take().expect("stdout was configured as piped");
    let mut stderr = child.stderr.take().expect("stderr was configured as piped");

    let mut stdout_buf = vec![0u8; READ_CHUNK];
    let mut stderr_buf = vec![0u8; READ_CHUNK];
    let mut stdout_open = true;
    let mut stderr_open = true;
    let mut status: Option<ExitStatus> = None;
    let mut drain_deadline: Option<Instant> = None;

    loop {
        if !stdout_open && !stderr_open {
            if let Some(status) = status {
                return Ok(status);
            }
        }

        let deadline = drain_deadline.unwrap_or_else(Instant::now);

        tokio::select! {
            biased;

            read = stdout.read(&mut stdout_buf), if stdout_open => {
                match read {
                    Ok(0) => stdout_open = false,
                    Ok(count) => output.append(&stdout_buf[..count]),
                    Err(_) => stdout_open = false,
                }
            }

            read = stderr.read(&mut stderr_buf), if stderr_open => {
                match read {
                    Ok(0) => stderr_open = false,
                    Ok(count) => output.append(&stderr_buf[..count]),
                    Err(_) => stderr_open = false,
                }
            }

            result = child.wait(), if status.is_none() => {
                status = Some(result?);
                drain_deadline = Some(Instant::now() + DRAIN_GRACE);
            }

            _ = tokio::time::sleep_until(deadline), if drain_deadline.is_some() => {
                if let Some(status) = status {
                    return Ok(status);
                }
            }
        }
    }
}

/// Keeps the tail of a byte stream with bounded memory while counting the
/// whole stream.
#[derive(Debug, Default)]
struct OutputBuffer {
    tail: Vec<u8>,
    starts_mid_line: bool,
    trimmed: bool,
    total_bytes: usize,
    completed_lines: usize,
    last_line_bytes: usize,
}

impl OutputBuffer {
    fn append(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        self.total_bytes += data.len();

        let mut newlines = 0usize;
        let mut tail_of_chunk = 0usize;
        for (index, &byte) in data.iter().enumerate() {
            if byte == b'\n' {
                newlines += 1;
                tail_of_chunk = index + 1;
            }
        }
        if newlines == 0 {
            self.last_line_bytes += data.len();
        } else {
            self.completed_lines += newlines;
            self.last_line_bytes = data.len() - tail_of_chunk;
        }

        self.tail.extend_from_slice(data);
        if self.tail.len() > ROLLING_CAPACITY {
            self.trim();
        }
    }

    fn trim(&mut self) {
        let cut = self.tail.len() - ROLLING_KEEP;
        match self.tail[cut..].iter().position(|&byte| byte == b'\n') {
            Some(relative) => {
                self.tail.drain(..cut + relative + 1);
                self.starts_mid_line = false;
            }
            None => {
                self.tail.drain(..cut);
                let mut start = 0;
                while start < self.tail.len() && (self.tail[start] & 0xc0) == 0x80 {
                    start += 1;
                }
                self.tail.drain(..start);
                self.starts_mid_line = true;
            }
        }
        self.trimmed = true;
    }

    /// Decodes the retained bytes and drops a leading partial line.
    fn snapshot(&self) -> String {
        let text = String::from_utf8_lossy(&self.tail);
        if self.starts_mid_line {
            if let Some(index) = text.find('\n') {
                return text[index + 1..].to_string();
            }
        }
        text.into_owned()
    }

    fn total_lines(&self) -> usize {
        self.completed_lines + usize::from(self.last_line_bytes > 0)
    }
}

/// Builds the model-facing text plus whether any output was produced.
fn render_output(output: &OutputBuffer, timed_out: bool) -> (String, bool) {
    let snapshot = output.snapshot();
    if snapshot.is_empty() {
        return (
            if timed_out {
                String::new()
            } else {
                "(no output)".to_string()
            },
            false,
        );
    }

    let result = truncate_tail(&snapshot, TruncationOptions::default());
    let total_lines = output.total_lines();
    let truncated = result.truncated || output.trimmed;

    if !truncated {
        return (result.content, true);
    }

    let output_lines = result.output_lines;
    let end_line = total_lines;
    let start_line = total_lines.saturating_sub(output_lines) + 1;

    let notice = if result.last_line_partial {
        format!(
            "[Showing last {} of line {} (line is {}).]",
            format_size(result.output_bytes),
            end_line,
            format_size(output.last_line_bytes)
        )
    } else {
        match result.truncated_by {
            Some(TruncatedBy::Bytes) if !result.last_line_partial => format!(
                "[Showing lines {}-{} of {} ({} limit).]",
                start_line,
                end_line,
                total_lines,
                format_size(DEFAULT_MAX_BYTES)
            ),
            _ => format!(
                "[Showing lines {}-{} of {}.]",
                start_line, end_line, total_lines
            ),
        }
    };

    (format!("{}\n\n{}", result.content, notice), true)
}

fn append_status(body: &str, has_output: bool, status: impl AsRef<str>) -> String {
    if has_output && !body.is_empty() {
        format!("{}\n\n{}", body, status.as_ref())
    } else {
        status.as_ref().to_string()
    }
}

fn exit_code(status: ExitStatus) -> i32 {
    match status.code() {
        Some(code) => code,
        #[cfg(unix)]
        None => {
            use std::os::unix::process::ExitStatusExt;
            status.signal().map_or(1, |signal| 128 + signal)
        }
        #[cfg(not(unix))]
        None => 1,
    }
}

// ---------------------------------------------------------------------------
// Shell discovery
// ---------------------------------------------------------------------------

/// Locates a bash executable.
///
/// On Windows the usual Git for Windows install locations are checked before
/// PATH. `%SystemRoot%\System32\bash.exe` is skipped: it is the WSL launcher,
/// not Git Bash, and fails outright when WSL is not installed.
fn resolve_shell() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        for candidate in windows_shell_candidates() {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        for candidate in path_candidates("bash.exe") {
            if candidate.is_file() && !is_wsl_launcher(&candidate) {
                return Some(candidate);
            }
        }
        // Git for Windows may live anywhere; derive its layout from the
        // `git.exe` that is already on PATH.
        for git in path_candidates("git.exe") {
            if !git.is_file() {
                continue;
            }
            let Some(root) = git.parent().and_then(Path::parent) else {
                continue;
            };
            for candidate in [
                root.join("bin").join("bash.exe"),
                root.join("usr").join("bin").join("bash.exe"),
            ] {
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    #[cfg(not(windows))]
    {
        let direct = Path::new("/bin/bash");
        if direct.is_file() {
            return Some(direct.to_path_buf());
        }
        for candidate in path_candidates("bash") {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        let sh = Path::new("/bin/sh");
        if sh.is_file() {
            return Some(sh.to_path_buf());
        }
        None
    }
}

#[cfg(windows)]
fn windows_shell_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut push_from = |variable: &str| {
        if let Some(root) = std::env::var_os(variable) {
            if !root.is_empty() {
                candidates.push(Path::new(&root).join("Git").join("bin").join("bash.exe"));
            }
        }
    };
    push_from("ProgramFiles");
    push_from("ProgramFiles(x86)");
    push_from("LocalAppData");
    if let Some(local_app_data) = std::env::var_os("LocalAppData") {
        candidates.push(
            Path::new(&local_app_data)
                .join("Programs")
                .join("Git")
                .join("bin")
                .join("bash.exe"),
        );
    }
    candidates
}

#[cfg(windows)]
fn is_wsl_launcher(candidate: &Path) -> bool {
    let Some(system_root) = std::env::var_os("SystemRoot") else {
        return false;
    };
    let launcher = Path::new(&system_root).join("System32").join("bash.exe");
    candidate
        .to_string_lossy()
        .eq_ignore_ascii_case(&launcher.to_string_lossy())
}

fn path_candidates(executable: &str) -> Vec<PathBuf> {
    let directories = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default();
    directories
        .into_iter()
        .map(|directory| directory.join(executable))
        .collect()
}
