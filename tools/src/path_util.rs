//! Normalization of model-supplied path strings.
//!
//! Models sometimes wrap paths in quotes, prefix them with `@`, or paste
//! non-breaking spaces from documentation. Normalizing here keeps the tools
//! free of that ceremony. Relative paths are resolved later, against the
//! workspace root, by the sandbox.

use std::path::PathBuf;

/// Cleans up a raw path string and expands a leading `~`.
pub fn normalize_input_path(raw: &str) -> PathBuf {
    let mut text = raw.trim().to_string();

    // Strip one layer of surrounding quotes.
    if text.len() >= 2 {
        let bytes = text.as_bytes();
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            text = text[1..text.len() - 1].trim().to_string();
        }
    }

    // `@path` is a common way for editors to reference files.
    if let Some(stripped) = text.strip_prefix('@') {
        text = stripped.to_string();
    }

    // Unicode spaces (non-breaking, ideographic, narrow no-break) look like
    // spaces but never compare equal to one.
    let mut normalized = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => {
                normalized.push(' ');
            }
            _ => normalized.push(c),
        }
    }

    expand_home(&normalized)
}

fn expand_home(text: &str) -> PathBuf {
    let rest = if text == "~" {
        Some("")
    } else {
        text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\"))
    };

    let Some(rest) = rest else {
        return PathBuf::from(text);
    };

    let Some(home) = home_dir() else {
        return PathBuf::from(text);
    };

    if rest.is_empty() {
        home
    } else {
        home.join(rest)
    }
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                let mut home = PathBuf::from(drive);
                home.push(path);
                Some(home)
            })
    }

    #[cfg(not(windows))]
    {
        std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_and_strips_quotes() {
        assert_eq!(
            normalize_input_path("  src/main.rs \n"),
            PathBuf::from("src/main.rs")
        );
        assert_eq!(
            normalize_input_path("\"src/main.rs\""),
            PathBuf::from("src/main.rs")
        );
        assert_eq!(
            normalize_input_path("'src/main.rs'"),
            PathBuf::from("src/main.rs")
        );
    }

    #[test]
    fn strips_at_prefix() {
        assert_eq!(
            normalize_input_path("@README.md"),
            PathBuf::from("README.md")
        );
    }

    #[test]
    fn normalizes_unicode_spaces() {
        assert_eq!(
            normalize_input_path("my\u{00A0}file.txt"),
            PathBuf::from("my file.txt")
        );
        assert_eq!(
            normalize_input_path("my\u{3000}file.txt"),
            PathBuf::from("my file.txt")
        );
    }

    #[test]
    fn leaves_ordinary_names_alone() {
        assert_eq!(
            normalize_input_path("src/lib.rs"),
            PathBuf::from("src/lib.rs")
        );
        assert_eq!(
            normalize_input_path(r"C:\temp\x.txt"),
            PathBuf::from(r"C:\temp\x.txt")
        );
    }

    #[test]
    fn expands_home_when_available() {
        if home_dir().is_none() {
            return;
        }
        let expanded = normalize_input_path("~/notes.txt");
        assert!(expanded.is_absolute());
        assert!(expanded.ends_with("notes.txt"));
        assert!(!expanded.to_string_lossy().contains('~'));
    }
}
