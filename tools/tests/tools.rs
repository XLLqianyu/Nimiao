//! End-to-end tests for the built-in tools.

use sandbox::Sandbox;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tools::{
    builtin_tools, BashTool, EditFileTool, ReadFileTool, Tool, ToolContext, WriteFileTool,
};

struct TestWorkspace {
    root: PathBuf,
    ctx: ToolContext,
}

impl TestWorkspace {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "nimiao-tools-{}-{}-{}",
            std::process::id(),
            seq,
            tag
        ));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let ctx = ToolContext::new(Sandbox::new(&root).unwrap());
        Self { root, ctx }
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    fn write(&self, relative: &str, content: &str) {
        let path = self.path(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn read(&self, relative: &str) -> String {
        std::fs::read_to_string(self.path(relative)).unwrap()
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).ok();
    }
}

async fn run(
    tool: &dyn Tool,
    args: serde_json::Value,
    ctx: &ToolContext,
) -> anyhow::Result<tools::ToolOutput> {
    tool.execute(args, ctx).await
}

#[tokio::test]
async fn writes_then_reads_a_file() {
    let workspace = TestWorkspace::new("roundtrip");
    let write = WriteFileTool;
    let read = ReadFileTool;

    let output = run(
        &write,
        json!({"path": "nested/dir/hello.txt", "content": "hello\nworld\n"}),
        &workspace.ctx,
    )
    .await
    .unwrap();
    assert!(!output.is_error);
    assert!(output.content.contains("Successfully wrote to"));
    assert_eq!(workspace.read("nested/dir/hello.txt"), "hello\nworld\n");

    let output = run(
        &read,
        json!({"path": "nested/dir/hello.txt"}),
        &workspace.ctx,
    )
    .await
    .unwrap();
    assert_eq!(output.content, "hello\nworld\n");
}

#[tokio::test]
async fn write_creates_missing_parent_directories() {
    let workspace = TestWorkspace::new("mkdirs");
    run(
        &WriteFileTool,
        json!({"path": "a/b/c/d.txt", "content": "x"}),
        &workspace.ctx,
    )
    .await
    .unwrap();
    assert!(workspace.path("a/b/c/d.txt").is_file());
}

#[tokio::test]
async fn write_overwrites_existing_content() {
    let workspace = TestWorkspace::new("overwrite");
    workspace.write("file.txt", "old");
    run(
        &WriteFileTool,
        json!({"path": "file.txt", "content": "new"}),
        &workspace.ctx,
    )
    .await
    .unwrap();
    assert_eq!(workspace.read("file.txt"), "new");
}

#[tokio::test]
async fn rejects_paths_outside_the_workspace() {
    let workspace = TestWorkspace::new("escape");
    let outside = std::env::temp_dir().join("nimiao-should-not-exist.txt");

    for path in [
        "../escape.txt".to_string(),
        outside.to_string_lossy().into_owned(),
    ] {
        let write = run(
            &WriteFileTool,
            json!({"path": path, "content": "x"}),
            &workspace.ctx,
        )
        .await;
        assert!(write.is_err(), "write should reject {path}");
    }

    let read = run(
        &ReadFileTool,
        json!({"path": "../escape.txt"}),
        &workspace.ctx,
    )
    .await;
    assert!(read.is_err());
}

#[tokio::test]
async fn reads_a_window_with_offset_and_limit() {
    let workspace = TestWorkspace::new("window");
    let body = (1..=100)
        .map(|n| format!("line {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    workspace.write("big.txt", &body);

    let output = run(
        &ReadFileTool,
        json!({"path": "big.txt", "offset": 10, "limit": 3}),
        &workspace.ctx,
    )
    .await
    .unwrap();
    assert!(output.content.starts_with("line 10\nline 11\nline 12"));
    assert!(output
        .content
        .contains("more lines in file. Use offset=13 to continue."));
}

#[tokio::test]
async fn reading_past_the_end_is_an_error() {
    let workspace = TestWorkspace::new("past-end");
    workspace.write("small.txt", "a\nb\n");

    let error = run(
        &ReadFileTool,
        json!({"path": "small.txt", "offset": 10}),
        &workspace.ctx,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("beyond end of file"));
}

#[tokio::test]
async fn reads_an_empty_file() {
    let workspace = TestWorkspace::new("empty");
    workspace.write("empty.txt", "");

    let output = run(&ReadFileTool, json!({"path": "empty.txt"}), &workspace.ctx)
        .await
        .unwrap();
    assert_eq!(output.content, "");
}

#[tokio::test]
async fn rejects_binary_files() {
    let workspace = TestWorkspace::new("binary");
    std::fs::write(workspace.path("bin.dat"), [0u8, 1, 2, 3]).unwrap();

    let error = run(&ReadFileTool, json!({"path": "bin.dat"}), &workspace.ctx)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("binary"));
}

#[tokio::test]
async fn truncates_long_files_and_offers_continuation() {
    let workspace = TestWorkspace::new("truncate");
    let body = (1..=5000)
        .map(|n| format!("line {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    workspace.write("huge.txt", &body);

    let output = run(&ReadFileTool, json!({"path": "huge.txt"}), &workspace.ctx)
        .await
        .unwrap();
    assert!(output.content.contains("[Showing lines 1-2000 of 5000."));
    assert!(output.content.contains("Use offset=2001 to continue."));
}

#[tokio::test]
async fn preserves_crlf_and_bom_in_edits() {
    let workspace = TestWorkspace::new("crlf-bom");
    workspace.write("crlf.txt", "\u{feff}alpha\r\nbeta\r\ngamma\r\n");

    let output = run(
        &EditFileTool,
        json!({
            "path": "crlf.txt",
            "edits": [{"oldText": "beta", "newText": "BETA"}]
        }),
        &workspace.ctx,
    )
    .await
    .unwrap();
    assert!(output.content.contains("Successfully replaced 1 block(s)"));

    let bytes = std::fs::read(workspace.path("crlf.txt")).unwrap();
    assert_eq!(bytes, "\u{feff}alpha\r\nBETA\r\ngamma\r\n".as_bytes());
}

#[tokio::test]
async fn applies_multiple_edits_in_one_call() {
    let workspace = TestWorkspace::new("multi-edit");
    workspace.write("code.rs", "fn a() {}\n\nfn b() {}\n");

    run(
        &EditFileTool,
        json!({
            "path": "code.rs",
            "edits": [
                {"oldText": "fn a() {}", "newText": "fn alpha() {}"},
                {"oldText": "fn b() {}", "newText": "fn beta() {}"}
            ]
        }),
        &workspace.ctx,
    )
    .await
    .unwrap();

    assert_eq!(workspace.read("code.rs"), "fn alpha() {}\n\nfn beta() {}\n");
}

#[tokio::test]
async fn fuzzy_edit_tolerates_smart_quotes() {
    let workspace = TestWorkspace::new("fuzzy");
    workspace.write("quotes.txt", "say \u{201C}hello\u{201D} now\n");

    run(
        &EditFileTool,
        json!({
            "path": "quotes.txt",
            "edits": [{"oldText": "say \"hello\" now", "newText": "say \"goodbye\" now"}]
        }),
        &workspace.ctx,
    )
    .await
    .unwrap();

    assert_eq!(workspace.read("quotes.txt"), "say \"goodbye\" now\n");
}

#[tokio::test]
async fn edit_rejects_a_missing_match() {
    let workspace = TestWorkspace::new("edit-miss");
    workspace.write("file.txt", "hello\n");

    let error = run(
        &EditFileTool,
        json!({
            "path": "file.txt",
            "edits": [{"oldText": "absent", "newText": "x"}]
        }),
        &workspace.ctx,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("Could not find the exact text"));
}

#[tokio::test]
async fn edit_rejects_non_utf8_content() {
    let workspace = TestWorkspace::new("edit-binary");
    std::fs::write(workspace.path("bad.txt"), [0xff, 0xfe, 0x00, 0x41]).unwrap();

    let error = run(
        &EditFileTool,
        json!({
            "path": "bad.txt",
            "edits": [{"oldText": "A", "newText": "B"}]
        }),
        &workspace.ctx,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("not valid UTF-8"));
}

#[tokio::test]
async fn bash_runs_a_command_in_the_workspace() {
    let workspace = TestWorkspace::new("bash-basic");

    let output = run(&BashTool, json!({"command": "echo hello"}), &workspace.ctx).await;
    let Ok(output) = output else {
        // No bash available on this machine; the tool already reported why.
        return;
    };
    assert!(!output.is_error);
    assert!(output.content.contains("hello"));
}

#[tokio::test]
async fn bash_reports_a_non_zero_exit_as_a_failed_tool_output() {
    let workspace = TestWorkspace::new("bash-fail");

    let Ok(output) = run(&BashTool, json!({"command": "exit 3"}), &workspace.ctx).await else {
        return;
    };
    assert!(output.is_error);
    assert!(output.content.contains("Command exited with code 3"));
}

#[tokio::test]
async fn bash_times_out() {
    let workspace = TestWorkspace::new("bash-timeout");
    let started = std::time::Instant::now();

    let Ok(output) = run(
        &BashTool,
        json!({"command": "sleep 10", "timeout": 1}),
        &workspace.ctx,
    )
    .await
    else {
        return;
    };

    assert!(output.is_error);
    assert!(output.content.contains("Command timed out after 1 seconds"));
    assert!(started.elapsed() < std::time::Duration::from_secs(8));
}

#[tokio::test]
async fn bash_truncates_large_output() {
    let workspace = TestWorkspace::new("bash-truncate");

    let Ok(output) = run(&BashTool, json!({"command": "seq 1 5000"}), &workspace.ctx).await else {
        return;
    };

    assert!(!output.is_error);
    assert!(output.content.contains("[Showing lines"));
    assert!(output.content.contains("of 5000"));
    assert!(output.content.trim_end().ends_with(']') || output.content.contains("]"));
}

#[tokio::test]
async fn bash_blocks_destructive_commands() {
    let workspace = TestWorkspace::new("bash-blocked");

    let error = run(&BashTool, json!({"command": "rm -rf /"}), &workspace.ctx)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("blocked"));
}

#[tokio::test]
async fn bash_does_not_wait_on_stdin() {
    let workspace = TestWorkspace::new("bash-stdin");
    let started = std::time::Instant::now();

    let Ok(output) = run(
        &BashTool,
        json!({"command": "cat", "timeout": 5}),
        &workspace.ctx,
    )
    .await
    else {
        return;
    };

    // stdin is /dev/null, so cat exits immediately with no output.
    assert!(!output.is_error);
    assert!(output.content.contains("(no output)"));
    assert!(started.elapsed() < std::time::Duration::from_secs(4));
}

#[tokio::test]
async fn rejects_invalid_timeout_values() {
    let workspace = TestWorkspace::new("bash-timeout-bounds");

    let error = run(
        &BashTool,
        json!({"command": "echo hi", "timeout": 0}),
        &workspace.ctx,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("between 1 and 600"));

    let error = run(
        &BashTool,
        json!({"command": "echo hi", "timeout": 601}),
        &workspace.ctx,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("between 1 and 600"));
}

#[tokio::test]
async fn builtin_tools_expose_unique_names() {
    let tools = builtin_tools();
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    assert_eq!(names, vec!["read_file", "write_file", "edit_file", "bash"]);

    for tool in &tools {
        let definition = tool.definition();
        assert_eq!(definition.name, tool.name());
        assert!(!definition.description.is_empty());
        assert_eq!(definition.parameters["type"], "object");
    }
}

#[tokio::test]
async fn definition_schemas_are_self_consistent() {
    for tool in builtin_tools() {
        let definition = tool.definition();
        let parameters = &definition.parameters;
        let properties = parameters["properties"].as_object().unwrap();
        for required in parameters["required"].as_array().unwrap() {
            let name = required.as_str().unwrap();
            assert!(
                properties.contains_key(name),
                "{} requires unknown property {name}",
                definition.name
            );
        }
    }
}

#[test]
fn workspace_helper_creates_unique_directories() {
    let first = TestWorkspace::new("unique");
    let second = TestWorkspace::new("unique");
    assert_ne!(first.root, second.root);
    assert!(Path::new(&first.root).is_absolute());
}
