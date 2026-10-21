//! Static shell-command screening.
//!
//! [`check_command`] rejects commands that are unambiguously destructive before
//! they reach a shell. It is a best-effort filter, not a security boundary: it
//! cannot evaluate variables, command substitution, scripts on disk, or
//! arbitrarily layered quoting. The real boundary is the workspace sandbox
//! plus OS-level isolation; this module exists to stop obvious accidents and
//! prompt-injected catastrophes.
//!
//! The lexer is hand-written because POSIX tokenizers treat `\` as an escape
//! character, which mangles Windows paths (`C:\Users` becomes `C:Users`) and
//! hides the real command name from the rules below.

use std::collections::VecDeque;

const MAX_NESTING_DEPTH: usize = 8;

/// Screens a shell command. Returns an error when the command is considered
/// unsafe to execute.
pub fn check_command(command: &str) -> anyhow::Result<()> {
    if contains_fork_bomb(command) {
        anyhow::bail!("Shell command blocked: fork bomb pattern detected");
    }
    check_inner(command, 0)
}

fn check_inner(command: &str, depth: usize) -> anyhow::Result<()> {
    if depth > MAX_NESTING_DEPTH {
        anyhow::bail!("Shell command blocked: nested commands exceed the depth limit");
    }

    for segment in split_segments(tokenize(command)?) {
        check_segment(&segment, depth)?;
    }
    Ok(())
}

/// Rejects `:(){ :|:& };:` regardless of spacing around the body.
fn contains_fork_bomb(command: &str) -> bool {
    let compact: String = command.chars().filter(|c| !c.is_whitespace()).collect();
    compact.contains(":(){")
}

fn check_segment(words: &[String], depth: usize) -> anyhow::Result<()> {
    let words = strip_redirections(words);
    let Some(words) = unwrap_launchers(&words) else {
        return Ok(());
    };
    let program = normalize_program(&words[0]);
    let args = &words[1..];

    if let Some(payload) = nested_payload(&program, args) {
        check_inner(&payload, depth + 1)?;
    }

    if is_blocked_program(&program) {
        anyhow::bail!("Shell command blocked: '{program}' is not allowed");
    }

    match program.as_str() {
        "rm" => check_rm(args),
        "rd" | "rmdir" | "del" | "erase" => check_windows_delete(&program, args),
        "remove-item" | "ri" => check_recursive_delete(args),
        "dd" => check_dd(args),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Word(String),
    Separator,
}

/// Splits a command line into words, dropping quotes and splitting on shell
/// control operators.
fn tokenize(command: &str) -> anyhow::Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut chars = command.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                started = true;
                let mut closed = false;
                #[allow(clippy::while_let_on_iterator)]
                while let Some(inner) = chars.next() {
                    if inner == '\'' {
                        closed = true;
                        break;
                    }
                    word.push(inner);
                }
                if !closed {
                    anyhow::bail!("Shell command blocked: unbalanced quotes");
                }
            }
            '"' => {
                started = true;
                let mut closed = false;
                // The escape branch below must consume the following character.
                #[allow(clippy::while_let_on_iterator)]
                while let Some(inner) = chars.next() {
                    match inner {
                        '"' => {
                            closed = true;
                            break;
                        }
                        '\\' => match chars.peek().copied() {
                            Some(next @ ('"' | '\\')) => {
                                word.push(next);
                                chars.next();
                            }
                            _ => word.push('\\'),
                        },
                        _ => word.push(inner),
                    }
                }
                if !closed {
                    anyhow::bail!("Shell command blocked: unbalanced quotes");
                }
            }
            '\\' => match chars.peek().copied() {
                Some(next) if next.is_whitespace() || "'\"\\$`;&|<>()".contains(next) => {
                    word.push(next);
                    chars.next();
                    started = true;
                }
                _ => {
                    // Not an escape outside quotes: keep the backslash so that
                    // Windows paths survive intact.
                    word.push('\\');
                    started = true;
                }
            },
            ';' | '\n' => {
                push_word(&mut tokens, &mut word, &mut started);
                tokens.push(Token::Separator);
            }
            '&' => match chars.peek().copied() {
                Some('&') => {
                    chars.next();
                    push_word(&mut tokens, &mut word, &mut started);
                    tokens.push(Token::Separator);
                }
                Some('>') => {
                    word.push('&');
                    started = true;
                }
                _ => {
                    if word.ends_with('>') || word.ends_with('<') {
                        // `2>&1`, `>&2`
                        word.push('&');
                        started = true;
                    } else {
                        push_word(&mut tokens, &mut word, &mut started);
                        tokens.push(Token::Separator);
                    }
                }
            },
            '|' => {
                if chars.peek() == Some(&'|') {
                    chars.next();
                }
                push_word(&mut tokens, &mut word, &mut started);
                tokens.push(Token::Separator);
            }
            '#' if !started => {
                // Comment: skip to the end of the line.
                while let Some(&next) = chars.peek() {
                    if next == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            c if c.is_whitespace() => {
                push_word(&mut tokens, &mut word, &mut started);
            }
            other => {
                word.push(other);
                started = true;
            }
        }
    }

    push_word(&mut tokens, &mut word, &mut started);
    Ok(tokens)
}

fn push_word(tokens: &mut Vec<Token>, word: &mut String, started: &mut bool) {
    if *started {
        tokens.push(Token::Word(std::mem::take(word)));
        *started = false;
    }
}

fn split_segments(tokens: Vec<Token>) -> Vec<Vec<String>> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    for token in tokens {
        match token {
            Token::Word(word) => current.push(word),
            Token::Separator => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
            }
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

// ---------------------------------------------------------------------------
// Redirections
// ---------------------------------------------------------------------------

enum RedirectKind {
    /// The operator stands alone; the next word is its target.
    ConsumesNext,
    /// The target is embedded in the same token (`2>&1`, `>file`).
    Embedded,
}

fn classify_redirect(word: &str) -> Option<RedirectKind> {
    let bytes = word.as_bytes();
    let mut index = 0;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index >= bytes.len() {
        return None;
    }

    let rest = &word[index..];
    const OPERATORS: [&str; 10] = ["&>>", "&>", "<<<", ">>", "<<", ">&", "<&", ">|", ">", "<"];
    for operator in OPERATORS {
        if let Some(target) = rest.strip_prefix(operator) {
            return Some(if target.is_empty() {
                RedirectKind::ConsumesNext
            } else {
                RedirectKind::Embedded
            });
        }
    }
    None
}

fn strip_redirections(words: &[String]) -> Vec<String> {
    let mut queue: VecDeque<&String> = words.iter().collect();
    let mut kept = Vec::with_capacity(words.len());
    while let Some(word) = queue.pop_front() {
        match classify_redirect(word) {
            Some(RedirectKind::ConsumesNext) => {
                queue.pop_front();
            }
            Some(RedirectKind::Embedded) => {}
            None => kept.push(word.clone()),
        }
    }
    kept
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

struct LauncherSpec {
    value_flags: &'static [&'static str],
    accepts_assignments: bool,
    first_arg_is_duration: bool,
}

fn launcher_spec(program: &str) -> Option<LauncherSpec> {
    let spec = match program {
        "sudo" => LauncherSpec {
            value_flags: &[
                "-u", "-g", "-p", "-U", "-C", "-h", "-r", "-t", "-T", "-D", "-R", "-a",
            ],
            accepts_assignments: false,
            first_arg_is_duration: false,
        },
        "doas" => LauncherSpec {
            value_flags: &["-u", "-C"],
            accepts_assignments: false,
            first_arg_is_duration: false,
        },
        "env" => LauncherSpec {
            value_flags: &["-u", "-C", "-S"],
            accepts_assignments: true,
            first_arg_is_duration: false,
        },
        "command" | "exec" | "nohup" | "setsid" => LauncherSpec {
            value_flags: &["-a"],
            accepts_assignments: false,
            first_arg_is_duration: false,
        },
        "nice" => LauncherSpec {
            value_flags: &["-n"],
            accepts_assignments: false,
            first_arg_is_duration: false,
        },
        "timeout" => LauncherSpec {
            value_flags: &["-s", "--signal", "-k", "--kill-after"],
            accepts_assignments: false,
            first_arg_is_duration: true,
        },
        "xargs" => LauncherSpec {
            value_flags: &[
                "-n", "-I", "-P", "-L", "-s", "-E", "-d", "-a", "-e", "-l", "-i",
            ],
            accepts_assignments: false,
            first_arg_is_duration: false,
        },
        "busybox" | "toybox" => LauncherSpec {
            value_flags: &[],
            accepts_assignments: false,
            first_arg_is_duration: false,
        },
        _ => return None,
    };
    Some(spec)
}

/// Skips launchers such as `sudo`, `env`, or `timeout` so the rules inspect the
/// command that will actually run.
fn unwrap_launchers(words: &[String]) -> Option<Vec<String>> {
    let mut current: &[String] = words;

    for _ in 0..16 {
        let program = normalize_program(current.first()?);
        let Some(spec) = launcher_spec(&program) else {
            return Some(current.to_vec());
        };

        let mut index = 1;
        let mut consumed_duration = false;
        while index < current.len() {
            let word = &current[index];
            if word == "--" {
                index += 1;
                break;
            }
            if word.starts_with('-') && word.len() > 1 {
                if spec.value_flags.contains(&word.as_str()) {
                    index += 2;
                } else {
                    index += 1;
                }
                continue;
            }
            if spec.accepts_assignments && is_assignment(word) {
                index += 1;
                continue;
            }
            if spec.first_arg_is_duration && !consumed_duration {
                consumed_duration = true;
                index += 1;
                continue;
            }
            break;
        }

        if index >= current.len() {
            return None;
        }
        current = &current[index..];
    }

    Some(current.to_vec())
}

fn is_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

// ---------------------------------------------------------------------------
// Nested interpreters
// ---------------------------------------------------------------------------

/// Extracts a command string that an interpreter will evaluate itself, so the
/// rules apply to the payload rather than only to the launcher.
fn nested_payload(program: &str, args: &[String]) -> Option<String> {
    match program {
        "eval" => {
            let payload = args.join(" ");
            (!payload.trim().is_empty()).then_some(payload)
        }
        "sh" | "bash" | "zsh" | "dash" | "ksh" | "mksh" | "ash" | "fish" => {
            let mut index = 0;
            while index < args.len() {
                let arg = &args[index];
                if arg == "--" {
                    index += 1;
                    continue;
                }
                if arg == "--command" {
                    return args.get(index + 1).cloned();
                }
                if let Some(short) = arg.strip_prefix('-') {
                    let is_short_cluster = !short.starts_with('-') && short.contains('c');
                    if is_short_cluster {
                        return args.get(index + 1).cloned();
                    }
                }
                index += 1;
            }
            None
        }
        "su" | "runuser" => {
            let mut index = 0;
            while index < args.len() {
                let arg = &args[index];
                if arg == "-c" || arg == "--command" {
                    return args.get(index + 1).cloned();
                }
                index += 1;
            }
            None
        }
        "cmd" => {
            let mut index = 0;
            while index < args.len() {
                let lower = args[index].to_ascii_lowercase();
                if lower == "/c" || lower == "/k" {
                    let payload = args[index + 1..].join(" ");
                    return (!payload.trim().is_empty()).then_some(payload);
                }
                index += 1;
            }
            None
        }
        "powershell" | "pwsh" => {
            let mut index = 0;
            while index < args.len() {
                let lower = args[index].to_ascii_lowercase();
                if lower == "-c" || lower == "-command" {
                    let payload = args[index + 1..].join(" ");
                    return (!payload.trim().is_empty()).then_some(payload);
                }
                index += 1;
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

const BLOCKED_PROGRAMS: &[&str] = &[
    "mke2fs",
    "wipefs",
    "blkdiscard",
    "fdisk",
    "sfdisk",
    "cfdisk",
    "parted",
    "sgdisk",
    "gdisk",
    "diskpart",
    "bcdedit",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "kexec",
    "format",
    "mkswap",
    "partprobe",
];

fn is_blocked_program(program: &str) -> bool {
    program.starts_with("mkfs") || BLOCKED_PROGRAMS.contains(&program)
}

fn check_rm(args: &[String]) -> anyhow::Result<()> {
    let mut recursive = false;
    let mut no_preserve_root = false;
    let mut end_of_flags = false;
    let mut targets: Vec<&str> = Vec::new();

    for arg in args {
        if !end_of_flags && arg == "--" {
            end_of_flags = true;
            continue;
        }
        if !end_of_flags && arg.starts_with("--") {
            match arg.as_str() {
                "--recursive" => recursive = true,
                "--no-preserve-root" => no_preserve_root = true,
                _ => {}
            }
            continue;
        }
        if !end_of_flags && arg.starts_with('-') && arg.len() > 1 {
            if arg[1..].chars().any(|c| c == 'r' || c == 'R') {
                recursive = true;
            }
            continue;
        }
        targets.push(arg);
    }

    if no_preserve_root {
        anyhow::bail!("Shell command blocked: 'rm --no-preserve-root' is not allowed");
    }
    if !recursive {
        return Ok(());
    }
    for target in targets {
        if is_dangerous_delete_target(target) {
            anyhow::bail!("Shell command blocked: refusing to recursively delete '{target}'");
        }
    }
    Ok(())
}

fn check_windows_delete(program: &str, args: &[String]) -> anyhow::Result<()> {
    let mut recursive = false;
    let mut targets: Vec<&str> = Vec::new();

    for arg in args {
        let lower = arg.to_ascii_lowercase();
        let is_flag = lower.starts_with('/') || (lower.starts_with('-') && lower.len() > 1);
        if is_flag {
            let body = &lower[1..];
            // `/s`, `/r`, `-recurse`, `-r`, and slash-free short forms.
            let recursive_flag = if lower.starts_with('/') {
                body.contains('s') || body.contains('r')
            } else {
                lower == "-recurse" || body.chars().any(|c| c == 'r')
            };
            if recursive_flag {
                recursive = true;
            }
            continue;
        }
        targets.push(arg);
    }

    if !recursive {
        return Ok(());
    }
    for target in targets {
        if is_dangerous_delete_target(target) {
            anyhow::bail!(
                "Shell command blocked: refusing to recursively delete '{target}' via '{program}'"
            );
        }
    }
    Ok(())
}

/// Handles PowerShell's `Remove-Item`, where `-Recurse` is a named parameter.
fn check_recursive_delete(args: &[String]) -> anyhow::Result<()> {
    let recursive = args.iter().any(|arg| {
        let lower = arg.to_ascii_lowercase();
        lower == "-recurse" || lower == "-r" || lower == "/s"
    });
    if !recursive {
        return Ok(());
    }
    for arg in args {
        if arg.starts_with('-') || (arg.starts_with('/') && arg.len() <= 3) {
            continue;
        }
        if is_dangerous_delete_target(arg) {
            anyhow::bail!("Shell command blocked: refusing to recursively delete '{arg}'");
        }
    }
    Ok(())
}

const DANGEROUS_DELETE_TARGETS: &[&str] = &[
    "",
    "/",
    ".",
    "..",
    "~",
    "$home",
    "${home}",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
    "/home",
    "/var",
    "/lib",
    "/lib64",
    "/opt",
    "c:/windows",
    "c:/users",
    "c:/program files",
    "c:/program files (x86)",
    "c:/programdata",
];

fn is_dangerous_delete_target(target: &str) -> bool {
    let mut path = target.replace('\\', "/").to_ascii_lowercase();
    while path.len() > 1 && path.ends_with('/') {
        path.pop();
    }
    let normalized = path.trim_end_matches('*').trim_end_matches('/').to_string();

    if DANGEROUS_DELETE_TARGETS.contains(&normalized.as_str()) {
        return true;
    }

    // Drive roots such as `C:` or `C:\`.
    let bytes = normalized.as_bytes();
    bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn check_dd(args: &[String]) -> anyhow::Result<()> {
    for arg in args {
        let Some(target) = arg.strip_prefix("of=") else {
            continue;
        };
        let normalized = target.replace('\\', "/").to_ascii_lowercase();
        if normalized.starts_with("/dev/") || normalized.starts_with("//./") {
            anyhow::bail!("Shell command blocked: refusing to write directly to '{target}'");
        }
        let bytes = normalized.as_bytes();
        if bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            anyhow::bail!("Shell command blocked: refusing to write directly to drive '{target}'");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Program names
// ---------------------------------------------------------------------------

fn normalize_program(word: &str) -> String {
    let base = word
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(word)
        .trim_matches(['"', '\'']);
    let lower = base.to_ascii_lowercase();
    for extension in [".exe", ".com", ".bat", ".cmd", ".ps1"] {
        if let Some(stripped) = lower.strip_suffix(extension) {
            return stripped.to_string();
        }
    }
    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(command: &str) {
        assert!(
            check_command(command).is_ok(),
            "expected command to be allowed: {command}"
        );
    }

    fn blocked(command: &str) {
        assert!(
            check_command(command).is_err(),
            "expected command to be blocked: {command}"
        );
    }

    #[test]
    fn allows_ordinary_development_commands() {
        allowed("cargo test");
        allowed("cargo test --workspace");
        allowed("echo hi");
        allowed("git commit -m \"a; rm -rf /\"");
        allowed("find . -name target");
        allowed("bash -c \"echo hi\"");
        allowed("sh -lc 'cargo fmt'");
        allowed("ls -la | grep src");
        allowed("echo hi > out.txt 2>&1");
        allowed("timeout 10 cargo test");
    }

    #[test]
    fn allows_non_recursive_removal() {
        allowed("rm build.log");
        allowed("rm -f build.log");
    }

    #[test]
    fn allows_recursive_removal_inside_the_workspace() {
        allowed("rm -rf target");
        allowed("rm -rf ./target");
        allowed("rm -rf /tmp/nimiao-test");
        allowed("rm -rf ~/projects/build");
    }

    #[test]
    fn blocks_catastrophic_removal() {
        blocked("rm -rf /");
        blocked("rm -rf /*");
        blocked("rm -rf ~");
        blocked("rm -rf $HOME");
        blocked("rm -rf ${HOME}");
        blocked("rm -rf .");
        blocked("rm -rf ..");
        blocked("rm -rf /etc");
        blocked("rm -rf C:\\Windows");
        blocked("rm -rf C:\\");
        blocked("rm -fr --no-preserve-root /");
    }

    #[test]
    fn blocks_laundered_commands() {
        blocked("sudo rm -rf /");
        blocked("sudo -u root rm -rf /");
        blocked("env X=1 rm -rf /");
        blocked("nice -n 10 rm -rf /");
        blocked("timeout 30 rm -rf /");
        blocked("echo hi && rm -rf /");
        blocked("echo hi; rm -rf /");
        blocked("echo hi || rm -rf /");
        blocked("echo hi | rm -rf /");
    }

    #[test]
    fn blocks_nested_shells() {
        blocked("bash -c \"rm -rf /\"");
        blocked("sh -c 'rm -rf ~'");
        blocked("bash -lc \"rm -rf /\"");
        blocked("sudo bash -c \"rm -rf /\"");
        blocked("eval \"rm -rf /\"");
        blocked("cmd /c rd /s /q C:\\");
        blocked("powershell -Command \"Remove-Item -Recurse C:/\"");
    }

    #[test]
    fn blocks_destructive_system_tools() {
        blocked("mkfs.ext4 /dev/sda");
        blocked("mke2fs /dev/sda");
        blocked("wipefs -a /dev/sda");
        blocked("dd if=/dev/zero of=/dev/sda");
        blocked("dd if=/dev/zero of=C:");
        blocked("format C:");
        blocked("diskpart");
        blocked("shutdown /s");
        blocked("reboot");
        blocked("parted /dev/sda");
    }

    #[test]
    fn blocks_windows_recursive_deletes() {
        blocked("rd /s /q C:\\");
        blocked("rmdir /s C:\\Windows");
        blocked("del /s C:\\Windows");
    }

    #[test]
    fn blocks_fork_bombs() {
        blocked(":(){ :|:& };:");
        blocked(":(){ :|:& };:  ");
    }

    #[test]
    fn rejects_unbalanced_quotes() {
        blocked("echo \"unterminated");
        blocked("echo 'unterminated");
    }

    #[test]
    fn windows_paths_survive_tokenization() {
        let tokens = tokenize(r"dir C:\Users\Public").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Word("dir".into()),
                Token::Word(r"C:\Users\Public".into())
            ]
        );
    }

    #[test]
    fn redirections_are_not_mistaken_for_commands() {
        let tokens = tokenize("> out.txt").unwrap();
        let segments = split_segments(tokens);
        assert_eq!(segments.len(), 1);
        let stripped = strip_redirections(&segments[0]);
        assert!(stripped.is_empty());
        assert!(check_command("> out.txt").is_ok());
    }
}
