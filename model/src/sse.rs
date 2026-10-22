//! Incremental decoder for server-sent events.
//!
//! The decoder is transport-agnostic: raw bytes go in, complete `data`
//! payloads come out. It implements the parts of the event-stream framing that
//! matter for chat completions: lines end with `\n` or `\r\n`, consecutive
//! `data:` lines are joined with a newline, a blank line dispatches the event,
//! and comments and other fields are ignored.

use anyhow::bail;

/// Largest accepted payload for a single event: the joined `data` lines.
const MAX_EVENT_BYTES: usize = 1024 * 1024;

/// Largest accepted partial line. A non-SSE JSON body arrives without newlines,
/// so this cap is deliberately looser than the event cap.
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

/// Turns a byte stream into event payloads.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    data: String,
    has_data: bool,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds raw bytes and returns every payload completed by them.
    ///
    /// Bytes that do not yet form a complete line stay buffered until the next
    /// call. A multi-byte UTF-8 character can be split across calls safely:
    /// `0x0A` never occurs inside one, so every line boundary is also a
    /// character boundary.
    pub fn feed(&mut self, bytes: &[u8]) -> anyhow::Result<Vec<String>> {
        self.buffer.extend_from_slice(bytes);

        let mut events = Vec::new();
        let mut start = 0;
        for index in 0..self.buffer.len() {
            if self.buffer[index] != b'\n' {
                continue;
            }
            let mut line = self.buffer[start..index].to_vec();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(event) = self.process_line(&line)? {
                events.push(event);
            }
            start = index + 1;
        }
        if start > 0 {
            self.buffer.drain(..start);
        }
        if self.buffer.len() > MAX_LINE_BYTES {
            bail!("the response sent a line larger than {MAX_LINE_BYTES} bytes");
        }
        Ok(events)
    }

    /// Flushes a trailing event that was not terminated by a blank line.
    pub fn finish(&mut self) -> Option<String> {
        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let _ = self.process_line(&line);
        }
        self.dispatch()
    }

    fn process_line(&mut self, line: &[u8]) -> anyhow::Result<Option<String>> {
        if line.is_empty() {
            return Ok(self.dispatch());
        }
        if line.first() == Some(&b':') {
            return Ok(None);
        }
        let (field, value) = match line.iter().position(|byte| *byte == b':') {
            Some(colon) => {
                let value = &line[colon + 1..];
                let value = value.strip_prefix(b" ").unwrap_or(value);
                (&line[..colon], value)
            }
            None => (line, &b""[..]),
        };
        if field != b"data" {
            return Ok(None);
        }
        let value = String::from_utf8_lossy(value);
        if self.has_data {
            self.data.push('\n');
        }
        self.data.push_str(&value);
        self.has_data = true;
        if self.data.len() > MAX_EVENT_BYTES {
            bail!("the event stream sent an event larger than {MAX_EVENT_BYTES} bytes");
        }
        Ok(None)
    }

    fn dispatch(&mut self) -> Option<String> {
        if !self.has_data {
            return None;
        }
        self.has_data = false;
        Some(std::mem::take(&mut self.data))
    }
}

#[cfg(test)]
mod tests {
    use super::SseDecoder;

    fn feed_all(decoder: &mut SseDecoder, chunks: &[&[u8]]) -> Vec<String> {
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.feed(chunk).unwrap());
        }
        events
    }

    #[test]
    fn dispatches_on_blank_lines() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, &[b"data: one\n\ndata: two\n\n"]);
        assert_eq!(events, ["one", "two"]);
    }

    #[test]
    fn handles_crlf() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, &[b"data: one\r\n\r\n"]);
        assert_eq!(events, ["one"]);
    }

    #[test]
    fn joins_multiple_data_lines() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, &[b"data: a\ndata: b\ndata: c\n\n"]);
        assert_eq!(events, ["a\nb\nc"]);
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(
            &mut decoder,
            &[b": keep-alive\nevent: message\nid: 7\nretry: 1000\ndata: ok\n\n"],
        );
        assert_eq!(events, ["ok"]);
    }

    #[test]
    fn keeps_empty_data_payloads() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, &[b"data:\n\n"]);
        assert_eq!(events, [""]);
    }

    #[test]
    fn blank_line_without_data_dispatches_nothing() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, &[b"\n\n: ping\n\n"]);
        assert!(events.is_empty());
    }

    #[test]
    fn assembles_events_split_across_feeds() {
        let mut decoder = SseDecoder::new();
        let events = feed_all(
            &mut decoder,
            &[b"data: {\"a\":", b"1}\n", b"\ndata: [DONE]", b"\n\n"],
        );
        assert_eq!(events, ["{\"a\":1}", "[DONE]"]);
    }

    #[test]
    fn handles_utf8_split_across_feeds() {
        let payload = "data: 你好\n\n".as_bytes();
        let split = payload.len() - 3; // inside the second character
        let mut decoder = SseDecoder::new();
        let events = feed_all(&mut decoder, &[&payload[..split], &payload[split..]]);
        assert_eq!(events, ["你好"]);
    }

    #[test]
    fn finish_flushes_a_trailing_event() {
        let mut decoder = SseDecoder::new();
        assert!(decoder.feed(b"data: tail").unwrap().is_empty());
        assert_eq!(decoder.finish().as_deref(), Some("tail"));
        assert_eq!(decoder.finish(), None);
    }
}
