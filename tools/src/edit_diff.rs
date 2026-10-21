//! Text replacement engine for `edit_file`.
//!
//! Edits are matched against the original file, never incrementally. Exact
//! matches win; when none is found the search falls back to a normalized view
//! that tolerates trailing whitespace and typographic look-alikes (smart
//! quotes, dashes, exotic spaces). Fuzzy results are applied line by line so
//! untouched lines keep their original bytes.

use anyhow::{bail, Result};
use serde::Deserialize;

/// One targeted replacement.
#[derive(Debug, Clone, Deserialize)]
pub struct Edit {
    #[serde(rename = "oldText")]
    pub old_text: String,
    #[serde(rename = "newText")]
    pub new_text: String,
}

#[derive(Debug)]
pub struct AppliedEdits {
    /// Content the edits were matched against.
    pub base_content: String,
    /// Content after all replacements.
    pub new_content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    Lf,
    Crlf,
}

/// The dominant line ending, decided by whichever appears first.
pub fn detect_line_ending(content: &str) -> LineEnding {
    let Some(lf) = content.find('\n') else {
        return LineEnding::Lf;
    };
    match content.find("\r\n") {
        Some(crlf) if crlf < lf => LineEnding::Crlf,
        _ => LineEnding::Lf,
    }
}

pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn restore_line_endings(text: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::Lf => text.to_string(),
        LineEnding::Crlf => text.replace('\n', "\r\n"),
    }
}

/// Splits off a leading byte-order mark.
///
/// The mark is invisible, so it is stripped before matching and put back on
/// write.
pub fn split_bom(content: &str) -> (bool, &str) {
    match content.strip_prefix('\u{feff}') {
        Some(rest) => (true, rest),
        None => (false, content),
    }
}

/// Normalizes text for fuzzy comparison.
///
/// Applies: trailing-whitespace removal per line, smart quotes to ASCII,
/// Unicode dashes to `-`, and exotic spaces to a plain space.
pub fn normalize_for_fuzzy(text: &str) -> String {
    let trimmed: String = text
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");

    trimmed
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

#[derive(Debug, Clone, Copy)]
struct LineSpan {
    start: usize,
    end: usize,
}

/// Splits content into lines, keeping each terminating newline attached.
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            lines.push(&content[start..=index]);
            start = index + 1;
        }
    }
    if start < content.len() {
        lines.push(&content[start..]);
    }
    lines
}

fn line_spans(content: &str) -> Vec<LineSpan> {
    let mut spans = Vec::new();
    let mut offset = 0;
    for line in split_lines_with_endings(content) {
        spans.push(LineSpan {
            start: offset,
            end: offset + line.len(),
        });
        offset += line.len();
    }
    spans
}

fn replacement_line_range(lines: &[LineSpan], replacement: &MatchedEdit) -> Result<(usize, usize)> {
    let start = replacement.match_index;
    let end = replacement.match_index + replacement.match_length;

    let Some(start_line) = lines
        .iter()
        .position(|line| start >= line.start && start < line.end)
    else {
        bail!("Replacement range is outside the base content");
    };

    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        bail!("Replacement range is outside the base content");
    }

    Ok((start_line, end_line + 1))
}

/// Applies replacements to `content` in reverse order so offsets stay valid.
fn apply_replacements(content: &str, replacements: &[MatchedEdit], offset: usize) -> String {
    let mut result = content.to_string();
    for replacement in replacements.iter().rev() {
        let start = replacement.match_index - offset;
        let end = start + replacement.match_length;
        result.replace_range(start..end, &replacement.new_text);
    }
    result
}

/// Applies replacements that were matched against a normalized view of the
/// original content.
///
/// Each replacement is widened to the lines it touches; those lines are
/// rewritten from the normalized base, while every other line is copied back
/// from the original text unchanged.
fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[MatchedEdit],
) -> Result<String> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        bail!(
            "Cannot preserve unchanged lines because the base content has a different line count"
        );
    }

    struct Group {
        start_line: usize,
        end_line: usize,
        replacements: Vec<MatchedEdit>,
    }

    let mut sorted = replacements.to_vec();
    sorted.sort_by_key(|replacement| replacement.match_index);

    let mut groups: Vec<Group> = Vec::new();
    for replacement in sorted {
        let (start_line, end_line) = replacement_line_range(&base_lines, &replacement)?;
        match groups.last_mut() {
            Some(current) if start_line < current.end_line => {
                current.end_line = current.end_line.max(end_line);
                current.replacements.push(replacement);
            }
            _ => groups.push(Group {
                start_line,
                end_line,
                replacements: vec![replacement],
            }),
        }
    }

    let mut original_index = 0;
    let mut result = String::new();
    for group in groups {
        for line in &original_lines[original_index..group.start_line] {
            result.push_str(line);
        }

        let group_start = base_lines[group.start_line].start;
        let group_end = base_lines[group.end_line - 1].end;
        result.push_str(&apply_replacements(
            &base_content[group_start..group_end],
            &group.replacements,
            group_start,
        ));
        original_index = group.end_line;
    }
    for line in &original_lines[original_index..] {
        result.push_str(line);
    }

    Ok(result)
}

fn not_found_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    } else {
        format!(
            "Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
    }
}

fn duplicate_error(
    path: &str,
    edit_index: usize,
    total_edits: usize,
    occurrences: usize,
) -> String {
    if total_edits == 1 {
        format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    } else {
        format!(
            "Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn empty_old_text_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!("oldText must not be empty in {path}.")
    } else {
        format!("edits[{edit_index}].oldText must not be empty in {path}.")
    }
}

fn no_change_error(path: &str, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

/// Applies all edits to LF-normalized content.
///
/// Every edit is matched against the same original content. Replacements are
/// applied in reverse order so earlier offsets stay valid. If any edit needs
/// fuzzy matching, the whole operation runs in normalized space and the result
/// is overlaid line by line onto the original content.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEdits> {
    let normalized_edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect();

    for (index, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            bail!(empty_old_text_error(path, index, normalized_edits.len()));
        }
    }

    let used_fuzzy = normalized_edits
        .iter()
        .any(|edit| !normalized_content.contains(&edit.old_text));
    let replacement_base = if used_fuzzy {
        normalize_for_fuzzy(normalized_content)
    } else {
        normalized_content.to_string()
    };

    let mut matched: Vec<MatchedEdit> = Vec::with_capacity(normalized_edits.len());
    for (index, edit) in normalized_edits.iter().enumerate() {
        let needle = if used_fuzzy {
            normalize_for_fuzzy(&edit.old_text)
        } else {
            edit.old_text.clone()
        };
        if needle.is_empty() {
            bail!(not_found_error(path, index, normalized_edits.len()));
        }

        let Some(match_index) = replacement_base.find(&needle) else {
            bail!(not_found_error(path, index, normalized_edits.len()));
        };

        let occurrences = replacement_base.matches(&needle).count();
        if occurrences > 1 {
            bail!(duplicate_error(
                path,
                index,
                normalized_edits.len(),
                occurrences
            ));
        }

        matched.push(MatchedEdit {
            edit_index: index,
            match_index,
            match_length: needle.len(),
            new_text: edit.new_text.clone(),
        });
    }

    matched.sort_by_key(|replacement| replacement.match_index);
    for pair in matched.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        if previous.match_index + previous.match_length > current.match_index {
            bail!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                previous.edit_index,
                current.edit_index
            );
        }
    }

    let new_content = if used_fuzzy {
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &replacement_base,
            &matched,
        )?
    } else {
        apply_replacements(&replacement_base, &matched, 0)
    };

    if normalized_content == new_content {
        bail!(no_change_error(path, normalized_edits.len()));
    }

    Ok(AppliedEdits {
        base_content: normalized_content.to_string(),
        new_content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(old_text: &str, new_text: &str) -> Edit {
        Edit {
            old_text: old_text.to_string(),
            new_text: new_text.to_string(),
        }
    }

    #[test]
    fn replaces_a_single_block() {
        let content = "alpha\nbeta\ngamma\n";
        let result =
            apply_edits_to_normalized_content(content, &[edit("beta", "BETA")], "f.txt").unwrap();
        assert_eq!(result.new_content, "alpha\nBETA\ngamma\n");
    }

    #[test]
    fn applies_multiple_disjoint_edits() {
        let content = "one\ntwo\nthree\nfour\n";
        let result = apply_edits_to_normalized_content(
            content,
            &[edit("one", "1"), edit("three", "3")],
            "f.txt",
        )
        .unwrap();
        assert_eq!(result.new_content, "1\ntwo\n3\nfour\n");
    }

    #[test]
    fn rejects_missing_text() {
        let error =
            apply_edits_to_normalized_content("hello", &[edit("nope", "x")], "f.txt").unwrap_err();
        assert!(error
            .to_string()
            .contains("Could not find the exact text in f.txt"));
    }

    #[test]
    fn rejects_duplicate_matches() {
        let error =
            apply_edits_to_normalized_content("a\na\n", &[edit("a", "b")], "f.txt").unwrap_err();
        assert!(error.to_string().contains("Found 2 occurrences"));
    }

    #[test]
    fn rejects_overlapping_edits() {
        let content = "abcdef";
        let error = apply_edits_to_normalized_content(
            content,
            &[edit("abcd", "x"), edit("cdef", "y")],
            "f.txt",
        )
        .unwrap_err();
        assert!(error.to_string().contains("overlap in f.txt"));
    }

    #[test]
    fn rejects_empty_old_text() {
        let error =
            apply_edits_to_normalized_content("abc", &[edit("", "x")], "f.txt").unwrap_err();
        assert!(error.to_string().contains("oldText must not be empty"));
    }

    #[test]
    fn rejects_identical_replacement() {
        let error =
            apply_edits_to_normalized_content("abc", &[edit("abc", "abc")], "f.txt").unwrap_err();
        assert!(error.to_string().contains("No changes made to f.txt"));
    }

    #[test]
    fn fuzzy_matches_smart_quotes() {
        let content = "let s = \u{201C}hello\u{201D};\n";
        let result =
            apply_edits_to_normalized_content(content, &[edit("\"hello\"", "\"world\"")], "f.txt")
                .unwrap();
        assert_eq!(result.new_content, "let s = \"world\";\n");
    }

    #[test]
    fn fuzzy_preserves_lines_that_were_not_touched() {
        let content = "first   \nsecond\nthird\t\n";
        let result =
            apply_edits_to_normalized_content(content, &[edit("second", "SECOND")], "f.txt")
                .unwrap();
        // Untouched lines keep their trailing whitespace.
        assert_eq!(result.new_content, "first   \nSECOND\nthird\t\n");
    }

    #[test]
    fn detects_crlf() {
        assert_eq!(detect_line_ending("a\r\nb\n"), LineEnding::Crlf);
        assert_eq!(detect_line_ending("a\nb\r\n"), LineEnding::Lf);
        assert_eq!(detect_line_ending("no newline"), LineEnding::Lf);
    }

    #[test]
    fn normalizes_and_restores_line_endings() {
        assert_eq!(normalize_to_lf("a\r\nb\rc"), "a\nb\nc");
        assert_eq!(restore_line_endings("a\nb", LineEnding::Crlf), "a\r\nb");
    }

    #[test]
    fn split_bom_reports_the_mark() {
        let (has_bom, text) = split_bom("\u{feff}hello");
        assert!(has_bom);
        assert_eq!(text, "hello");

        let (has_bom, text) = split_bom("hello");
        assert!(!has_bom);
        assert_eq!(text, "hello");
    }

    #[test]
    fn handles_multibyte_text() {
        let content = "你好世界\n第二行\n";
        let result =
            apply_edits_to_normalized_content(content, &[edit("第二行", "另一个")], "f.txt")
                .unwrap();
        assert_eq!(result.new_content, "你好世界\n另一个\n");
    }

    #[test]
    fn fuzzy_edit_spanning_lines() {
        let content = "a\nb  \nc\nd\n";
        let result =
            apply_edits_to_normalized_content(content, &[edit("b\nc", "x\ny")], "f.txt").unwrap();
        assert_eq!(result.new_content, "a\nx\ny\nd\n");
    }
}
