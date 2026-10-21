//! Nimiao command-line entry point.
//!
//! Reads an OpenAI-compatible endpoint from the environment, wires one model
//! to the built-in tool set, and serves a line-oriented REPL. The whole
//! session lives in one history vector, so each turn sees everything before
//! it.

use engine::Agent;
use model::openai::OpenAIModel;
use protocol::Item;
use sandbox::Sandbox;
use std::io::{self, Write};
use tokio::io::{AsyncBufReadExt, BufReader};
use tools::{default_registry, ToolContext};

/// Bright green prompt.
const PROMPT: &str = "\x1b[92m> \x1b[0m";

/// Bright blue label in front of model replies.
const AGENT_PREFIX: &str = "\x1b[94mAgent:\x1b[0m ";

/// Bright red label for non-fatal errors.
const ERROR_PREFIX: &str = "\x1b[91merror:\x1b[0m ";

#[tokio::main]
async fn main() {
    enable_ansi();

    let Some(api_key) = read_env("OPENAI_API_KEY") else {
        eprintln!("{ERROR_PREFIX}OPENAI_API_KEY is not set");
        std::process::exit(1);
    };
    let base_url =
        read_env("OPENAI_BASE_URL").unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    let model_name = read_env("OPENAI_MODEL_NAME").unwrap_or_else(|| "gpt-4o".to_string());

    let sandbox = match Sandbox::new(".") {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("{ERROR_PREFIX}cannot open the workspace: {err}");
            std::process::exit(1);
        }
    };

    let ctx = ToolContext::new(sandbox);
    let root = ctx.workspace_root().to_path_buf();
    let registry = default_registry();

    let model = OpenAIModel::new(&api_key, &base_url, &model_name);
    let agent = Agent::new(model, registry, ctx);

    println!(
        "\x1b[1;36mNimiao\x1b[0m \x1b[90m·\x1b[0m {model_name} \x1b[90m·\x1b[0m {}",
        root.display()
    );
    println!("\x1b[90mType a request. `exit` or `quit` leaves.\x1b[0m");

    let mut history: Vec<Item> = Vec::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("{PROMPT}");
        if let Err(err) = io::stdout().flush() {
            eprintln!("{ERROR_PREFIX}{err}");
            break;
        }

        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            // EOF: stdin was closed or the user pressed Ctrl+Z / Ctrl+D.
            Ok(None) => break,
            Err(err) => {
                eprintln!("{ERROR_PREFIX}cannot read input: {err}");
                break;
            }
        };

        let request = line.trim();
        if request.is_empty() {
            continue;
        }
        if request.eq_ignore_ascii_case("exit") || request.eq_ignore_ascii_case("quit") {
            break;
        }

        if let Err(err) = agent.run_turn(&mut history, request).await {
            eprintln!("{ERROR_PREFIX}{err:#}");
            continue;
        }

        match latest_assistant_message(&history) {
            Some(reply) => println!("{AGENT_PREFIX}{reply}"),
            None => println!("{AGENT_PREFIX}\x1b[90m(no reply)\x1b[0m"),
        }
    }

    println!("\x1b[90mbye\x1b[0m");
}

/// Reads an environment variable, treating blank values as absent.
fn read_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

/// The model's most recent prose reply, newest first.
fn latest_assistant_message(history: &[Item]) -> Option<&str> {
    history.iter().rev().find_map(|item| match item {
        Item::AssistantMessage(text) => Some(text.as_str()),
        _ => None,
    })
}

/// Turns on ANSI escape handling for the current console.
///
/// Windows Terminal renders escape sequences out of the box, but a legacy
/// console window needs `ENABLE_VIRTUAL_TERMINAL_PROCESSING` first.
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
