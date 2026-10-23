//! Nimiao command-line entry point.
//!
//! The interactive client is a full-screen TUI. Redirected input or output,
//! and --plain, use the line-oriented streaming REPL instead.

mod plain;
mod tool_preview;
mod tui;

use engine::Agent;
use model::openai::OpenAIModel;
use sandbox::Sandbox;
use std::io::{self, IsTerminal};
use std::process::ExitCode;
use tools::{default_registry, ToolContext};

/// Bright red label for fatal errors.
const ERROR_PREFIX: &str = "\x1b[91merror:\x1b[0m ";

const USAGE: &str = "usage: nimiao [--plain]\n\n  --plain      use the line-oriented REPL\n  -h, --help   show this help";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliMode {
    Tui,
    Plain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliAction {
    Run(CliMode),
    Help,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let action = match parse_args(&args) {
        Ok(action) => action,
        Err(error) => {
            eprintln!("{ERROR_PREFIX}{error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let CliAction::Run(mode) = action else {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    };

    enable_ansi();

    let Some(api_key) = read_env("OPENAI_API_KEY") else {
        eprintln!("{ERROR_PREFIX}OPENAI_API_KEY is not set");
        return ExitCode::FAILURE;
    };
    let base_url = read_env("OPENAI_BASE_URL")
        .unwrap_or_else(|| concat!("https:", "//api.openai.com/v1").to_string());
    let model_name = read_env("OPENAI_MODEL_NAME").unwrap_or_else(|| "gpt-4o".to_string());

    let sandbox = match Sandbox::new(".") {
        Ok(sandbox) => sandbox,
        Err(error) => {
            eprintln!("{ERROR_PREFIX}cannot open the workspace: {error}");
            return ExitCode::FAILURE;
        }
    };

    let ctx = ToolContext::new(sandbox);
    let root = ctx.workspace_root().to_path_buf();
    let registry = default_registry();
    let model = OpenAIModel::new(&api_key, &base_url, &model_name);
    let agent = Agent::new(model, registry, ctx);

    let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
    let use_tui = mode == CliMode::Tui && interactive;
    let result = if use_tui {
        tui::run(agent, model_name, root).await
    } else {
        plain::run(agent, model_name, root).await
    };

    if let Err(error) = result {
        eprintln!("{ERROR_PREFIX}{error:#}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Parses the deliberately small command-line surface.
fn parse_args(args: &[String]) -> Result<CliAction, String> {
    let mut mode = CliMode::Tui;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(CliAction::Help),
            "--plain" => mode = CliMode::Plain,
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(CliAction::Run(mode))
}

/// Reads an environment variable, treating blank values as absent.
fn read_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// Turns on ANSI escape handling for the plain REPL on legacy Windows consoles.
///
/// Crossterm owns this setup in TUI mode.
#[cfg(windows)]
fn enable_ansi() {
    use std::ffi::c_void;

    const STD_OUTPUT_HANDLE: u32 = 0xFFFF_FFF5;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    extern "system" {
        fn GetStdHandle(std_handle: u32) -> *mut c_void;
        fn GetConsoleMode(console_handle: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(console_handle: *mut c_void, mode: u32) -> i32;
    }

    // SAFETY: the calls only inspect the standard output handle owned by this
    // process. Any failure just means the console keeps its default mode.
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        if handle.is_null() {
            return;
        }
        let mut mode = 0u32;
        if GetConsoleMode(handle, &mut mode) != 0 {
            let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}

#[cfg(not(windows))]
fn enable_ansi() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn defaults_to_tui() {
        assert_eq!(parse_args(&[]), Ok(CliAction::Run(CliMode::Tui)));
    }

    #[test]
    fn parses_plain_and_help() {
        assert_eq!(
            parse_args(&args(&["--plain"])),
            Ok(CliAction::Run(CliMode::Plain))
        );
        assert_eq!(parse_args(&args(&["--help"])), Ok(CliAction::Help));
        assert_eq!(parse_args(&args(&["-h"])), Ok(CliAction::Help));
    }

    #[test]
    fn rejects_unknown_arguments() {
        assert!(parse_args(&args(&["--wat"])).is_err());
    }
}
