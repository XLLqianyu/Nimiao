//! `read_file`: line-oriented text file reader.

use crate::path_util::normalize_input_path;
use crate::truncate::{
    format_size, truncate_head, TruncatedBy, TruncationOptions, DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
};
use crate::{Tool, ToolContext, ToolOutput};
use anyhow::{Context, Result};
use async_trait::async_trait;
use protocol::ToolDefinition;
use serde::Deserialize;
use serde_json::json;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, BufReader, SeekFrom};

/// Bytes inspected up front to tell text files from binary ones.
const BINARY_SNIFF_BYTES: usize = 8 * 1024;

/// Read buffer size; also the largest single line kept in memory.
const READER_CAPACITY: usize = 64 * 1024;

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    path: String,
    #[serde(default)]
    offset: Option<i64>,
    #[serde(default)]
    limit: Option<i64>,
}

/// Reads text files, one line window at a time.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: format!(
                "Read the contents of a file. Text output is truncated to {} lines or {} (whichever is hit first). \
                 Use offset/limit for large files; continue with offset to read the rest.",
                DEFAULT_MAX_LINES,
                format_size(DEFAULT_MAX_BYTES)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let args: ReadFileArgs = serde_json::from_value(args)
            .map_err(|err| anyhow::anyhow!("Invalid arguments: {err}"))?;
        let display_path = args.path.clone();

        let resolved = ctx
            .sandbox
            .validate_path(normalize_input_path(&args.path))
            .with_context(|| format!("Cannot read '{display_path}'"))?;

        let mut file = File::open(&resolved)
            .await
            .with_context(|| format!("Cannot read '{display_path}'"))?;
        let metadata = file.metadata().await?;
        if metadata.is_dir() {
            anyhow::bail!("'{display_path}' is a directory, not a file");
        }

        let mut sniff = [0u8; BINARY_SNIFF_BYTES];
        let sniffed = file.read(&mut sniff).await?;
        if sniff[..sniffed].contains(&0) {
            anyhow::bail!("'{display_path}' appears to be a binary file");
        }
        file.seek(SeekFrom::Start(0)).await?;

        let offset_given = args.offset.is_some();
        let start_index = match args.offset.unwrap_or(1) {
            value if value > 1 => (value - 1) as usize,
            _ => 0,
        };
        let user_limit = args.limit.map(|value| value.max(0) as usize);
        let line_limit = user_limit.unwrap_or(usize::MAX);

        let mut collector = LineCollector::new(start_index, line_limit);
        let mut reader = BufReader::with_capacity(READER_CAPACITY, file);

        loop {
            let consumed = {
                let buf = reader.fill_buf().await?;
                if buf.is_empty() {
                    break;
                }
                let mut cursor = 0usize;
                while cursor < buf.len() {
                    match buf[cursor..].iter().position(|&byte| byte == b'\n') {
                        Some(relative) => {
                            collector.feed(&buf[cursor..cursor + relative]);
                            collector.end_line();
                            cursor += relative + 1;
                        }
                        None => {
                            collector.feed(&buf[cursor..]);
                            cursor = buf.len();
                        }
                    }
                }
                cursor
            };
            reader.consume(consumed);
        }
        collector.finish();

        let total_lines = collector.total_lines;
        let collected = collector.collected.len();
        let first_line_bytes = collector.first_line_bytes;

        if offset_given && start_index > 0 && start_index >= total_lines {
            anyhow::bail!(
                "Offset {} is beyond end of file ({} lines total)",
                args.offset.unwrap_or(1),
                total_lines
            );
        }

        let mut selected = collector.collected.join("\n");
        // Reproduce the file's trailing newline when the window reaches the end,
        // so the model sees the exact bytes it would be editing.
        let reaches_end = total_lines > 0
            && collector.last_collected_index == Some(total_lines - 1)
            && collector.ends_with_newline;
        if reaches_end {
            selected.push('\n');
        }

        let truncation = truncate_head(&selected, TruncationOptions::default());
        let start_display = start_index + 1;
        let remaining = total_lines.saturating_sub(start_index + collected);
        let user_limited = user_limit.is_some_and(|limit| collected == limit);

        let content = if truncation.first_line_exceeds_limit {
            format!(
                "[Line {} is {}, exceeds {} limit. Use bash: sed -n '{}p' {} | head -c {}]",
                start_display,
                format_size(first_line_bytes),
                format_size(DEFAULT_MAX_BYTES),
                start_display,
                display_path,
                DEFAULT_MAX_BYTES
            )
        } else if !truncation.truncated && user_limited && remaining > 0 {
            let next_offset = start_index + collected + 1;
            with_notice(
                truncation.content,
                format!("[{remaining} more lines in file. Use offset={next_offset} to continue.]"),
            )
        } else if truncation.truncated || remaining > 0 {
            let truncated_by = truncation.truncated_by.unwrap_or({
                if collected >= DEFAULT_MAX_LINES {
                    TruncatedBy::Lines
                } else {
                    TruncatedBy::Bytes
                }
            });
            let shown_lines = truncation.output_lines.max(1);
            let end_display = start_display + shown_lines - 1;
            let next_offset = end_display + 1;
            let notice = match truncated_by {
                TruncatedBy::Lines => format!(
                    "[Showing lines {start_display}-{end_display} of {total_lines}. Use offset={next_offset} to continue.]"
                ),
                TruncatedBy::Bytes => format!(
                    "[Showing lines {start_display}-{end_display} of {total_lines} ({} limit). Use offset={next_offset} to continue.]",
                    format_size(DEFAULT_MAX_BYTES)
                ),
            };
            with_notice(truncation.content, notice)
        } else {
            truncation.content
        };

        Ok(ToolOutput::success(content))
    }
}

fn with_notice(body: String, notice: String) -> String {
    if body.is_empty() {
        notice
    } else {
        format!("{body}\n\n{notice}")
    }
}

/// Scans a file line by line with bounded memory.
///
/// Only the selected window is kept; the rest of the file is still walked so
/// the total line count is accurate.
struct LineCollector {
    start_index: usize,
    line_limit: usize,
    max_line_bytes: usize,
    line_index: usize,
    total_lines: usize,
    collected: Vec<String>,
    collected_bytes: usize,
    last_collected_index: Option<usize>,
    ends_with_newline: bool,
    collecting: bool,
    first_line_bytes: usize,
    line_buf: Vec<u8>,
    line_len: usize,
}

impl LineCollector {
    fn new(start_index: usize, line_limit: usize) -> Self {
        Self {
            start_index,
            line_limit,
            max_line_bytes: DEFAULT_MAX_BYTES + 1,
            line_index: 0,
            total_lines: 0,
            collected: Vec::new(),
            collected_bytes: 0,
            last_collected_index: None,
            ends_with_newline: false,
            collecting: true,
            first_line_bytes: 0,
            line_buf: Vec::new(),
            line_len: 0,
        }
    }

    /// Appends bytes to the current line, keeping at most one byte over the
    /// byte budget so an oversized line can still be recognized.
    fn feed(&mut self, bytes: &[u8]) {
        self.line_len += bytes.len();
        if self.line_buf.len() < self.max_line_bytes {
            let room = self.max_line_bytes - self.line_buf.len();
            let take = room.min(bytes.len());
            self.line_buf.extend_from_slice(&bytes[..take]);
        }
    }

    fn end_line(&mut self) {
        let index = self.line_index;
        self.line_index += 1;
        self.total_lines += 1;

        let should_collect = self.collecting
            && index >= self.start_index
            && self.collected.len() < self.line_limit
            && self.collected.len() < DEFAULT_MAX_LINES
            && self.collected_bytes <= DEFAULT_MAX_BYTES;

        if should_collect {
            let text = String::from_utf8_lossy(&self.line_buf).into_owned();
            if self.collected.is_empty() {
                self.first_line_bytes = self.line_len;
            }
            self.collected_bytes += text.len() + usize::from(!self.collected.is_empty());
            self.last_collected_index = Some(index);
            self.collected.push(text);
        } else if index >= self.start_index {
            self.collecting = false;
        }

        self.line_buf.clear();
        self.line_len = 0;
    }

    /// Records a final line without a terminating newline, or a file whose
    /// last line is empty because it ends with one.
    fn finish(&mut self) {
        if self.line_len > 0 {
            self.end_line();
        } else if self.line_index > 0 {
            self.ends_with_newline = true;
        }
    }
}
