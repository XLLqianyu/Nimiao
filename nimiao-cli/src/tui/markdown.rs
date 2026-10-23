//! Small, deterministic Markdown renderer for assistant prose.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;

/// Removes terminal control sequences and normalizes control whitespace.
pub(crate) fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\u{1b}' => match chars.next() {
                Some('[') => {
                    for next in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    while let Some(next) = chars.next() {
                        if next == '\u{7}' {
                            break;
                        }
                        if next == '\u{1b}' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                Some(_) | None => {}
            },
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    continue;
                }
            }
            '\n' => out.push('\n'),
            '\t' => out.push_str("    "),
            ch if ch.is_control() => {}
            ch => out.push(ch),
        }
    }
    out
}

/// Renders the supported Markdown subset into wrapped terminal lines.
pub(crate) fn render(text: &str, width: u16) -> Vec<Line<'static>> {
    let text = sanitize(text);
    let width = usize::from(width.max(1));
    let base = Style::default();
    let mut lines = Vec::new();
    let mut in_fence = false;

    for source in text.split('\n') {
        let trimmed = source.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }

        if in_fence {
            let mut spans = vec![Span::styled("  ", Style::default().fg(DIM))];
            spans.push(Span::styled(
                source.to_string(),
                Style::default().fg(Color::Gray),
            ));
            lines.extend(wrap_spans(spans, width));
            continue;
        }

        if source.trim().is_empty() {
            lines.push(Line::default());
            continue;
        }

        if let Some((level, heading)) = heading(source) {
            let style = if level <= 2 {
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().add_modifier(Modifier::BOLD)
            };
            lines.extend(wrap_spans(inline_spans(heading, style, base), width));
            continue;
        }

        if let Some(quoted) = source
            .strip_prefix('>')
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
        {
            let mut spans = vec![Span::styled("│ ", Style::default().fg(DIM))];
            spans.extend(inline_spans(
                quoted,
                base.add_modifier(Modifier::ITALIC),
                base,
            ));
            lines.extend(wrap_spans(spans, width));
            continue;
        }

        if let Some(item) = source
            .strip_prefix("- ")
            .or_else(|| source.strip_prefix("* "))
            .or_else(|| source.strip_prefix("+ "))
        {
            let mut spans = vec![Span::styled("• ", Style::default().fg(ACCENT))];
            spans.extend(inline_spans(item, base, base));
            lines.extend(wrap_spans(spans, width));
            continue;
        }

        if let Some((marker, item)) = ordered_item(source) {
            let mut spans = vec![Span::styled(
                format!("{marker} "),
                Style::default().fg(ACCENT),
            )];
            spans.extend(inline_spans(item, base, base));
            lines.extend(wrap_spans(spans, width));
            continue;
        }

        lines.extend(wrap_spans(inline_spans(source, base, base), width));
    }

    if lines.is_empty() {
        lines.push(Line::default());
    }
    lines
}

/// Renders plain text, preserving hard line breaks and wrapping to width.
pub(crate) fn plain(text: &str, width: u16, style: Style) -> Vec<Line<'static>> {
    let text = sanitize(text);
    let width = usize::from(width.max(1));
    let mut lines = Vec::new();
    for source in text.split('\n') {
        lines.extend(wrap_spans(
            vec![Span::styled(source.to_string(), style)],
            width,
        ));
    }
    if lines.is_empty() {
        lines.push(Line::default());
    }
    lines
}

fn heading(line: &str) -> Option<(usize, &str)> {
    let level = line.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let rest = line[level..].strip_prefix(' ')?;
    Some((level, rest))
}

fn ordered_item(line: &str) -> Option<(&str, &str)> {
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let (marker, rest) = line.split_at(digits);
    let item = rest.strip_prefix(". ")?;
    Some((marker, item))
}

fn inline_spans(text: &str, regular: Style, base: Style) -> Vec<Span<'static>> {
    let code = Style::default()
        .fg(ACCENT)
        .bg(Color::Indexed(236))
        .add_modifier(Modifier::BOLD);
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('`') {
        if start > 0 {
            spans.push(Span::styled(
                rest[..start].to_string(),
                merge(base, regular),
            ));
        }
        let after = &rest[start + 1..];
        let Some(end) = after.find('`') else {
            spans.push(Span::styled(
                rest[start..].to_string(),
                merge(base, regular),
            ));
            return spans;
        };
        spans.push(Span::styled(after[..end].to_string(), merge(base, code)));
        rest = &after[end + 1..];
    }
    if !rest.is_empty() || spans.is_empty() {
        spans.push(Span::styled(rest.to_string(), merge(base, regular)));
    }
    spans
}

fn merge(base: Style, overlay: Style) -> Style {
    base.patch(overlay)
}

#[derive(Clone, Copy)]
struct StyledChar {
    ch: char,
    style: Style,
}

fn wrap_spans(spans: Vec<Span<'static>>, width: usize) -> Vec<Line<'static>> {
    let mut chars = Vec::new();
    for span in &spans {
        for ch in span.content.chars() {
            chars.push(StyledChar {
                ch,
                style: span.style,
            });
        }
    }
    if chars.is_empty() {
        return vec![Line::from(spans)];
    }

    let mut lines = Vec::new();
    let mut rest = chars.as_slice();
    while !rest.is_empty() {
        let mut used = 0usize;
        let mut end = 0usize;
        let mut last_space = None;
        for (index, item) in rest.iter().enumerate() {
            let char_width = UnicodeWidthChar::width(item.ch).unwrap_or(0);
            if used + char_width > width && index > 0 {
                end = last_space.unwrap_or(index);
                break;
            }
            used += char_width;
            if item.ch == ' ' {
                last_space = Some(index);
            }
            end = index + 1;
        }
        if end == 0 {
            end = 1;
        }

        let mut visible = &rest[..end];
        while visible.last().is_some_and(|item| item.ch == ' ') {
            visible = &visible[..visible.len() - 1];
        }
        lines.push(chars_to_line(visible));

        rest = &rest[end..];
        while rest.first().is_some_and(|item| item.ch == ' ') {
            rest = &rest[1..];
        }
    }
    lines
}

fn chars_to_line(chars: &[StyledChar]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for item in chars {
        if let Some(last) = spans.last_mut().filter(|span| span.style == item.style) {
            last.content.to_mut().push(item.ch);
        } else {
            spans.push(Span::styled(item.ch.to_string(), item.style));
        }
    }
    Line::from(spans)
}

/// Display width of a line, used by layout tests and cache measurements.
pub(crate) fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_headings_lists_quotes_and_code() {
        let rendered = render("# Title\n- one\n> quoted\n```\ncode\n```", 80);
        let rendered = text(&rendered);
        assert!(rendered.contains("Title"));
        assert!(rendered.contains("• one"));
        assert!(rendered.contains("│ quoted"));
        assert!(rendered.contains("  code"));
        assert!(!rendered.contains("```"));
    }

    #[test]
    fn strips_control_sequences_and_tabs() {
        assert_eq!(sanitize("a\u{1b}[31mb\tc\u{7}"), "ab    c");
    }

    #[test]
    fn wraps_wide_characters_without_exceeding_width() {
        let rendered = render("喵喵喵喵喵喵", 6);
        assert!(rendered.len() > 1);
        assert!(rendered.iter().all(|line| line_width(line) <= 6));
    }

    #[test]
    fn keeps_unknown_markdown_as_text() {
        let rendered = render("| a | b |\n|---|---|", 20);
        assert!(text(&rendered).contains("| a | b |"));
    }
}
