//! Layout and rendering for the live transcript and detail pager.

use super::app::{CachedLines, Cell, CellBody, ToolStatus, TuiApp, ViewportState};
use super::markdown;
use model::Model;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Widget};
use ratatui::Frame;
use std::time::Duration;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
const ERROR: Color = Color::Red;
const WARN: Color = Color::Yellow;
const SURFACE: Color = Color::Indexed(236);

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub(crate) fn draw<M: Model + 'static>(frame: &mut Frame<'_>, app: &mut TuiApp<M>) {
    if app.detail.is_some() {
        draw_detail(frame, app);
    } else {
        draw_main(frame, app);
    }
}

fn draw_main<M: Model + 'static>(frame: &mut Frame<'_>, app: &mut TuiApp<M>) {
    let area = frame.area();
    let mut footer = u16::from(area.height >= 8);
    let mut status = u16::from((app.active.is_some() || app.queue_paused) && area.height >= 5);
    let mut queue = queue_height(app, area);
    let mut composer = app.composer_height(area.width);
    let minimum = 1u16;

    if queue + status + footer + composer + minimum > area.height {
        queue = 0;
    }
    if queue + status + footer + composer + minimum > area.height {
        footer = 0;
    }
    if queue + status + footer + composer + minimum > area.height {
        status = 0;
    }
    if queue + status + footer + composer + minimum > area.height {
        composer = area.height.saturating_sub(minimum).max(1);
    }
    let transcript = area
        .height
        .saturating_sub(queue + status + composer + footer);

    let rows = Layout::vertical([
        Constraint::Length(transcript),
        Constraint::Length(queue),
        Constraint::Length(status),
        Constraint::Length(composer),
        Constraint::Length(footer),
    ])
    .split(area);

    draw_transcript(
        frame.buffer_mut(),
        rows[0],
        &mut app.cells,
        &mut app.live,
        false,
    );
    if rows[1].height > 0 {
        draw_queue(frame.buffer_mut(), rows[1], app);
    }
    if rows[2].height > 0 {
        draw_status(frame.buffer_mut(), rows[2], app);
    }
    draw_composer(frame, rows[3], app);
    if rows[4].height > 0 {
        draw_footer(frame.buffer_mut(), rows[4], footer_text(app, area.width));
    }
}

fn draw_detail<M: Model + 'static>(frame: &mut Frame<'_>, app: &mut TuiApp<M>) {
    let area = frame.area();
    if area.height < 3 {
        draw_transcript(
            frame.buffer_mut(),
            area,
            &mut app.cells,
            app.detail.as_mut().expect("detail exists"),
            true,
        );
        return;
    }

    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(area);
    let header = Line::from(vec![
        Span::styled(
            ">_ ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("Transcript", Style::default().add_modifier(Modifier::BOLD)),
    ]);
    Paragraph::new(header).render(rows[0], frame.buffer_mut());
    draw_transcript(
        frame.buffer_mut(),
        rows[1],
        &mut app.cells,
        app.detail.as_mut().expect("detail exists"),
        true,
    );
    draw_footer(frame.buffer_mut(), rows[2], detail_footer(area.width));
}

fn queue_height<M: Model>(app: &TuiApp<M>, area: Rect) -> u16 {
    if app.queue.is_empty() || area.height < 12 {
        return 0;
    }
    let items = app.queue.len().min(3) as u16;
    (items + 1).min(area.height / 3)
}

fn draw_queue<M: Model>(buf: &mut Buffer, area: Rect, app: &TuiApp<M>) {
    let width = usize::from(area.width.saturating_sub(2));
    let mut lines = Vec::new();
    let header = if app.queue.len() > 3 {
        format!("{} queued · showing first 3", app.queue.len())
    } else {
        format!("{} queued", app.queue.len())
    };
    lines.push(Line::from(Span::styled(
        header,
        Style::default().fg(ACCENT),
    )));
    for item in app.queue.iter().take(3) {
        let text = clean(item);
        lines.push(Line::from(vec![
            Span::styled("↳ ", Style::default().fg(DIM)),
            Span::styled(
                truncate(&text, width.saturating_sub(2)),
                Style::default().fg(DIM),
            ),
        ]));
    }
    Paragraph::new(lines).render(inner(area), buf);
}

fn draw_status<M: Model>(buf: &mut Buffer, area: Rect, app: &TuiApp<M>) {
    let line = if let Some(active) = &app.active {
        let frame = (active.started.elapsed().as_millis() / 100) as usize % SPINNER.len();
        let mut spans = vec![
            Span::styled(format!("{} ", SPINNER[frame]), Style::default().fg(ACCENT)),
            Span::styled(
                if active.interrupt_requested {
                    "Interrupting"
                } else {
                    "Working"
                },
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ];
        if let Some(tool) = &active.current_tool {
            spans.push(Span::styled(
                format!(" · {}", clean(tool)),
                Style::default().fg(DIM),
            ));
        }
        spans.push(Span::styled(
            format!(" · {}", elapsed(active.started.elapsed())),
            Style::default().fg(DIM),
        ));
        Line::from(spans)
    } else {
        Line::from(Span::styled(
            "Queue paused · Tab sends the next message",
            Style::default().fg(WARN),
        ))
    };
    Paragraph::new(line).render(inner(area), buf);
}

fn draw_composer<M: Model + 'static>(frame: &mut Frame<'_>, area: Rect, app: &mut TuiApp<M>) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    Block::default()
        .style(Style::default().bg(SURFACE))
        .render(area, frame.buffer_mut());
    let text_area = inner(area);
    if text_area.height == 0 || text_area.width == 0 {
        return;
    }
    frame.render_widget(&app.composer, text_area);
    let prompt = Rect {
        x: area.x,
        y: text_area.y,
        width: 1,
        height: 1,
    };
    Block::default()
        .style(
            Style::default()
                .fg(ACCENT)
                .bg(SURFACE)
                .add_modifier(Modifier::BOLD),
        )
        .render(prompt, frame.buffer_mut());
    if let Some(cell) = frame
        .buffer_mut()
        .cell_mut(Position::new(prompt.x, prompt.y))
    {
        cell.set_symbol("›");
    }

    let (top_row, top_col) = app.composer.scroll_offset();
    let cursor = app.composer.screen_cursor();
    let x = text_area
        .x
        .saturating_add(cursor.col.saturating_sub(usize::from(top_col)) as u16)
        .min(area.right().saturating_sub(1));
    let y = text_area
        .y
        .saturating_add(cursor.row.saturating_sub(usize::from(top_row)) as u16)
        .min(area.bottom().saturating_sub(1));
    frame.set_cursor_position((x, y));
}

fn draw_footer(buf: &mut Buffer, area: Rect, text: &str) {
    let text = truncate(text, usize::from(area.width));
    Paragraph::new(Line::from(Span::styled(text, Style::default().fg(DIM)))).render(area, buf);
}

fn footer_text<M: Model>(app: &TuiApp<M>, width: u16) -> &'static str {
    if app.active.is_some() {
        if width < 64 {
            "Enter queue · Esc stop · Ctrl+T details"
        } else {
            "Enter queue · Tab queue · Esc interrupt · Ctrl+T details · PgUp/PgDn scroll"
        }
    } else if app.queue_paused {
        if width < 64 {
            "Tab send queued · Ctrl+T details"
        } else {
            "Tab send next queued · Enter new request · Ctrl+T details"
        }
    } else if width < 56 {
        "Enter send · Shift+Enter newline · Ctrl+T details"
    } else {
        "Enter send · Tab queue · Shift+Enter newline · Esc interrupt · Ctrl+T details · Ctrl+C quit"
    }
}

fn detail_footer(width: u16) -> &'static str {
    if width < 60 {
        "↑↓/jk scroll · PgUp/PgDn page · q close"
    } else {
        "↑↓/jk scroll · PgUp/PgDn/Space page · Ctrl+U/D half · Home/End ends · q/Ctrl+T close"
    }
}

fn inner(area: Rect) -> Rect {
    area.inner(Margin::new(1, 0))
}

fn elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}m {}s", seconds / 60, seconds % 60)
    }
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clean(text: &str) -> String {
    collapse(&markdown::sanitize(text))
}

fn truncate(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut used = 0;
    for ch in text.chars() {
        let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + char_width >= width {
            break;
        }
        out.push(ch);
        used += char_width;
    }
    out.push('…');
    out
}

fn draw_transcript(
    buf: &mut Buffer,
    area: Rect,
    cells: &mut [Cell],
    viewport: &mut ViewportState,
    detail: bool,
) {
    if area.height == 0 || area.width == 0 {
        viewport.resolve(0, 0);
        return;
    }
    let width = area.width;
    let mut total = 0usize;
    for cell in cells.iter_mut() {
        total = total.saturating_add(cached_lines(cell, width, detail).len());
    }
    viewport.resolve(total, usize::from(area.height));

    let start = viewport.top;
    let end = start.saturating_add(usize::from(area.height));
    let mut visible = Vec::new();
    let mut offset = 0usize;
    for cell in cells.iter_mut() {
        let lines = cached_lines(cell, width, detail);
        let cell_start = offset;
        let cell_end = offset.saturating_add(lines.len());
        offset = cell_end;
        if cell_end <= start || cell_start >= end {
            continue;
        }
        let skip = start.saturating_sub(cell_start);
        let take = end.min(cell_end).saturating_sub(start.max(cell_start));
        visible.extend(lines.iter().skip(skip).take(take).cloned());
    }
    Paragraph::new(visible).render(area, buf);
}

fn cached_lines(cell: &mut Cell, width: u16, detail: bool) -> &[Line<'static>] {
    let revision = cell.revision;
    let stale = match if detail {
        cell.detail_cache.as_ref()
    } else {
        cell.main_cache.as_ref()
    } {
        Some(cache) => cache.width != width || cache.revision != revision,
        None => true,
    };
    if stale {
        let lines = build_lines(&cell.body, width, detail);
        let cache = CachedLines {
            width,
            revision,
            lines,
        };
        if detail {
            cell.detail_cache = Some(cache);
        } else {
            cell.main_cache = Some(cache);
        }
    }
    if detail {
        cell.detail_cache
            .as_ref()
            .expect("just cached")
            .lines
            .as_slice()
    } else {
        cell.main_cache
            .as_ref()
            .expect("just cached")
            .lines
            .as_slice()
    }
}

fn build_lines(body: &CellBody, width: u16, detail: bool) -> Vec<Line<'static>> {
    match body {
        CellBody::Header {
            version,
            model,
            workspace,
        } => vec![
            Line::from(vec![
                Span::styled(
                    ">_ ",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("Nimiao v{version}"),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" · {} · {}", clean(model), clean(workspace)),
                    Style::default().fg(DIM),
                ),
            ]),
            Line::default(),
        ],
        CellBody::User(text) => {
            let style = Style::default().fg(Color::White).bg(SURFACE);
            let mut lines = prefixed(
                markdown::plain(text, width.saturating_sub(2), style),
                Span::styled("› ", style.fg(ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled("  ", style),
            );
            for line in &mut lines {
                pad_line(line, width, style);
            }
            lines.push(Line::default());
            lines
        }
        CellBody::Assistant { text, interrupted } => {
            let mut lines = prefixed(
                markdown::render(text, width.saturating_sub(2)),
                Span::styled("• ", Style::default().fg(ACCENT)),
                Span::raw("  "),
            );
            if *interrupted {
                lines.push(Line::from(Span::styled(
                    "  [interrupted]",
                    Style::default().fg(WARN).add_modifier(Modifier::ITALIC),
                )));
            }
            lines.push(Line::default());
            lines
        }
        CellBody::Tool(tool) if !detail => {
            let status = match tool.status {
                ToolStatus::Running => "Running",
                ToolStatus::Ran => "Ran",
                ToolStatus::Failed => "Failed",
                ToolStatus::Interrupted => "Interrupted",
            };
            let color = match tool.status {
                ToolStatus::Running => ACCENT,
                ToolStatus::Ran => Color::Green,
                ToolStatus::Failed => ERROR,
                ToolStatus::Interrupted => WARN,
            };
            let mut text = format!("{status} {} · {}", clean(&tool.name), tool.preview());
            if let Some(duration) = tool.elapsed {
                text.push_str(&format!(" · {}", elapsed(duration)));
            }
            vec![
                Line::from(vec![
                    Span::styled("• ", Style::default().fg(color)),
                    Span::styled(
                        truncate(&text, usize::from(width.saturating_sub(2))),
                        Style::default().fg(color),
                    ),
                ]),
                Line::default(),
            ]
        }
        CellBody::Tool(tool) => tool_detail_lines(tool, width),
        CellBody::Notice(text) => {
            let mut lines = prefixed(
                markdown::plain(text, width.saturating_sub(2), Style::default().fg(DIM)),
                Span::styled("· ", Style::default().fg(DIM)),
                Span::raw("  "),
            );
            lines.push(Line::default());
            lines
        }
        CellBody::Error(text) => {
            let mut lines = prefixed(
                markdown::plain(text, width.saturating_sub(2), Style::default().fg(ERROR)),
                Span::styled(
                    "! ",
                    Style::default().fg(ERROR).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
            );
            lines.push(Line::default());
            lines
        }
        CellBody::Help(text) => {
            let mut lines = prefixed(
                markdown::plain(text, width.saturating_sub(2), Style::default().fg(DIM)),
                Span::styled("? ", Style::default().fg(ACCENT)),
                Span::raw("  "),
            );
            lines.push(Line::default());
            lines
        }
    }
}

fn tool_detail_lines(tool: &super::app::ToolRow, width: u16) -> Vec<Line<'static>> {
    let status = match tool.status {
        ToolStatus::Running => "Running",
        ToolStatus::Ran => "Ran",
        ToolStatus::Failed => "Failed",
        ToolStatus::Interrupted => "Interrupted",
    };
    let color = match tool.status {
        ToolStatus::Running => ACCENT,
        ToolStatus::Ran => Color::Green,
        ToolStatus::Failed => ERROR,
        ToolStatus::Interrupted => WARN,
    };
    let mut lines = vec![Line::from(vec![
        Span::styled("• ", Style::default().fg(color)),
        Span::styled(
            format!("{status} {}", clean(&tool.name)),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            tool.elapsed
                .map(|duration| format!(" · {}", elapsed(duration)))
                .unwrap_or_default(),
            Style::default().fg(DIM),
        ),
    ])];
    lines.push(Line::from(Span::styled(
        "  Arguments",
        Style::default().fg(DIM).add_modifier(Modifier::BOLD),
    )));
    let arguments = serde_json::to_string_pretty(&tool.arguments)
        .unwrap_or_else(|_| tool.arguments.to_string());
    lines.extend(prefixed(
        markdown::plain(
            &arguments,
            width.saturating_sub(2),
            Style::default().fg(Color::Gray),
        ),
        Span::raw("  "),
        Span::raw("  "),
    ));
    lines.push(Line::from(Span::styled(
        "  Result",
        Style::default().fg(DIM).add_modifier(Modifier::BOLD),
    )));
    let content = if tool.content.is_empty() {
        "(no output)"
    } else {
        &tool.content
    };
    lines.extend(prefixed(
        markdown::plain(
            content,
            width.saturating_sub(2),
            Style::default().fg(Color::Gray),
        ),
        Span::raw("  "),
        Span::raw("  "),
    ));
    lines.push(Line::default());
    lines
}

fn prefixed(
    lines: Vec<Line<'static>>,
    first: Span<'static>,
    rest: Span<'static>,
) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            let mut spans = vec![if index == 0 {
                first.clone()
            } else {
                rest.clone()
            }];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect()
}

fn pad_line(line: &mut Line<'static>, width: u16, style: Style) {
    let used = markdown::line_width(line);
    let remaining = usize::from(width).saturating_sub(used);
    if remaining > 0 {
        line.spans.push(Span::styled(" ".repeat(remaining), style));
    }
}
