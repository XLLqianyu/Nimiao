//! Byte- and line-bounded truncation for tool output.
//!
//! Two independent limits apply and whichever is hit first wins: a line count
//! and a byte count. Output never ends mid-line, except for the one documented
//! tail case where the final line alone exceeds the byte budget.

/// Default maximum number of lines kept in one tool response.
pub const DEFAULT_MAX_LINES: usize = 2000;

/// Default maximum number of bytes kept in one tool response.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Which limit ended the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Debug, Clone)]
pub struct TruncationResult {
    /// The text that survived truncation.
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    /// Lines in the original text.
    pub total_lines: usize,
    /// Bytes in the original text.
    pub total_bytes: usize,
    /// Lines in `content`.
    pub output_lines: usize,
    /// Bytes in `content`.
    pub output_bytes: usize,
    /// Whether the single kept line was cut at its start (tail truncation only).
    pub last_line_partial: bool,
    /// Whether the first line alone exceeded the byte budget (head truncation only).
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TruncationOptions {
    pub max_lines: usize,
    pub max_bytes: usize,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

impl TruncationOptions {
    pub fn new(max_lines: usize, max_bytes: usize) -> Self {
        Self {
            max_lines,
            max_bytes,
        }
    }
}

/// Formats a byte count as `512B`, `50.0KB`, or `1.2MB`.
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Splits text into lines the way truncation counts them: `\n` separates lines
/// and a trailing newline does not introduce an extra empty line.
fn split_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Keeps the beginning of `content`, up to the first limit that is reached.
///
/// A first line longer than the byte budget yields empty content with
/// `first_line_exceeds_limit` set, because no complete line can be shown.
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines;
    let max_bytes = options.max_bytes;

    let total_bytes = content.len();
    let lines = split_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    let first_line_bytes = lines.first().map_or(0, |line| line.len());
    if first_line_bytes > max_bytes {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines,
            max_bytes,
        };
    }

    let mut kept: Vec<&str> = Vec::new();
    let mut kept_bytes = 0usize;
    let mut truncated_by = TruncatedBy::Lines;

    for (index, line) in lines.iter().enumerate() {
        if index >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(index > 0);
        if kept_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        kept.push(line);
        kept_bytes += line_bytes;
    }

    if kept.len() >= max_lines && kept_bytes <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output = kept.join("\n");
    let output_bytes = output.len();

    TruncationResult {
        content: output,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: kept.len(),
        output_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Keeps the end of `content`, up to the first limit that is reached.
///
/// When the final line alone is larger than the byte budget, its tail is kept
/// as a partial line and `last_line_partial` is set.
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines;
    let max_bytes = options.max_bytes;

    let total_bytes = content.len();
    let lines = split_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    let mut kept: Vec<&str> = Vec::new();
    let mut kept_bytes = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev() {
        if kept.len() >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(!kept.is_empty());
        if kept_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            if kept.is_empty() {
                // Nothing fits: show the end of this line rather than nothing.
                let partial = tail_bytes(line, max_bytes);
                kept_bytes = partial.len();
                kept.push(partial);
                last_line_partial = true;
            }
            break;
        }
        kept.push(line);
        kept_bytes += line_bytes;
    }

    if kept.len() >= max_lines && kept_bytes <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    kept.reverse();
    let output = kept.join("\n");
    let output_bytes = output.len();

    TruncationResult {
        content: output,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: kept.len(),
        output_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Takes at most `max_bytes` from the end of `line`, starting on a UTF-8
/// character boundary.
fn tail_bytes(line: &str, max_bytes: usize) -> &str {
    if line.len() <= max_bytes {
        return line;
    }
    let mut start = line.len() - max_bytes;
    while start < line.len() && !line.is_char_boundary(start) {
        start += 1;
    }
    &line[start..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_short_content_untouched() {
        let result = truncate_head("one\ntwo\n", TruncationOptions::default());
        assert!(!result.truncated);
        assert_eq!(result.content, "one\ntwo\n");
        assert_eq!(result.total_lines, 2);
        assert_eq!(result.total_bytes, 8);
        assert_eq!(result.truncated_by, None);
    }

    #[test]
    fn counts_empty_content_as_no_lines() {
        let result = truncate_head("", TruncationOptions::default());
        assert_eq!(result.total_lines, 0);
        assert_eq!(result.total_bytes, 0);
        assert!(!result.truncated);
    }

    #[test]
    fn truncates_by_line_count() {
        let content = (1..=10)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_head(&content, TruncationOptions::new(3, DEFAULT_MAX_BYTES));
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.content, "1\n2\n3");
        assert_eq!(result.output_lines, 3);
        assert_eq!(result.output_bytes, 5);
        assert_eq!(result.total_lines, 10);
    }

    #[test]
    fn truncates_by_byte_count() {
        let content = "aaaa\nbbbb\ncccc";
        let result = truncate_head(content, TruncationOptions::new(2000, 8));
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(result.content, "aaaa");
    }

    #[test]
    fn reports_a_first_line_over_the_byte_limit() {
        let content = "x".repeat(100) + "\nshort";
        let result = truncate_head(&content, TruncationOptions::new(2000, 50));
        assert!(result.truncated);
        assert!(result.first_line_exceeds_limit);
        assert_eq!(result.content, "");
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    }

    #[test]
    fn tail_keeps_the_last_lines() {
        let content = (1..=10)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_tail(&content, TruncationOptions::new(3, DEFAULT_MAX_BYTES));
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.content, "8\n9\n10");
        assert_eq!(result.total_lines, 10);
    }

    #[test]
    fn tail_cuts_an_oversized_final_line_at_a_char_boundary() {
        let content = format!("short\n{}", "é".repeat(100));
        let result = truncate_tail(&content, TruncationOptions::new(2000, 10));
        assert!(result.truncated);
        assert!(result.last_line_partial);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        // 5 two-byte characters fit into 10 bytes.
        assert_eq!(result.content, "ééééé");
        assert_eq!(result.output_bytes, 10);
    }

    #[test]
    fn tail_byte_limit_drops_leading_lines() {
        let content = "aaaa\nbbbb\ncccc";
        let result = truncate_tail(content, TruncationOptions::new(2000, 8));
        assert_eq!(result.content, "cccc");
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert!(!result.last_line_partial);
    }

    #[test]
    fn formats_sizes_readably() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(50 * 1024), "50.0KB");
        assert_eq!(format_size(1024 * 1024), "1.0MB");
    }
}
