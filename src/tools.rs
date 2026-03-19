use anyhow::{Context, Result};
use tokio::process::Command;

// ---------------------------------------------------------------------------
// ToolExecutor trait
// ---------------------------------------------------------------------------

/// Executes agent tools by name.  The trait exists so Phase 4 can swap in a
/// `DockerExecutor` without touching the agentic loop.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, name: &str, args: serde_json::Value) -> Result<String>;
}

// ---------------------------------------------------------------------------
// Local (host) executor
// ---------------------------------------------------------------------------

pub struct LocalExecutor;

#[async_trait::async_trait]
impl ToolExecutor for LocalExecutor {
    async fn execute(&self, name: &str, args: serde_json::Value) -> Result<String> {
        tracing::debug!(tool = %name, "executing tool");
        match name {
            "shell_command" => shell_command(args).await,
            "tmux_capture_pane" => tmux_capture_pane(args).await,
            "tmux_send_keys" => tmux_send_keys(args).await,
            "read_file" => read_file(args).await,
            "edit_file" => edit_file(args).await,
            other => anyhow::bail!("unknown tool: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

/// Run an arbitrary shell command in the background, capturing stdout+stderr.
async fn shell_command(args: serde_json::Value) -> Result<String> {
    let command = args["command"]
        .as_str()
        .context("shell_command: missing string argument 'command'")?;

    tracing::debug!(command = %command, "running shell command");

    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .await
        .with_context(|| format!("spawning shell command: {command}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit_code = output.status.code().unwrap_or(-1);

    let mut result = String::new();

    if !stdout.is_empty() {
        result.push_str(stdout.trim_end());
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n[stderr]\n");
        }
        result.push_str(stderr.trim_end());
    }
    if result.is_empty() {
        result = format!("[exit {exit_code}]");
    } else if !output.status.success() {
        result.push_str(&format!("\n[exit {exit_code}]"));
    }

    Ok(result)
}

/// Capture the visible text of a tmux pane.
async fn tmux_capture_pane(args: serde_json::Value) -> Result<String> {
    // Default to the first pane if no target provided.
    let target = args["target"].as_str().unwrap_or("%0");

    let output = Command::new("tmux")
        .args(["capture-pane", "-t", target, "-p"])
        .output()
        .await
        .context("spawning tmux capture-pane")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux capture-pane failed: {stderr}");
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Send keystrokes to a tmux pane (for interactive programs).
async fn tmux_send_keys(args: serde_json::Value) -> Result<String> {
    let keys = args["keys"]
        .as_str()
        .context("tmux_send_keys: missing string argument 'keys'")?;
    let target = args["target"].as_str().unwrap_or("%0");

    let output = Command::new("tmux")
        .args(["send-keys", "-t", target, keys])
        .output()
        .await
        .context("spawning tmux send-keys")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tmux send-keys failed: {stderr}");
    }

    Ok(format!("sent keys to pane {target}"))
}

/// Read the full contents of a file.
async fn read_file(args: serde_json::Value) -> Result<String> {
    let path = args["path"]
        .as_str()
        .context("read_file: missing string argument 'path'")?;

    tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("read_file: reading '{path}'"))
}

/// Overwrite a file with new content.
async fn edit_file(args: serde_json::Value) -> Result<String> {
    let path = args["path"]
        .as_str()
        .context("edit_file: missing string argument 'path'")?;
    let content = args["content"]
        .as_str()
        .context("edit_file: missing string argument 'content'")?;

    tokio::fs::write(path, content)
        .await
        .with_context(|| format!("edit_file: writing '{path}'"))?;

    Ok(format!("wrote {} bytes to {path}", content.len()))
}

// ---------------------------------------------------------------------------
// Tool schemas (OpenAI function-calling format)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ---- tool_schemas ----------------------------------------------------

    #[test]
    fn tool_schemas_have_expected_names() {
        let schemas = tool_schemas();
        let names: Vec<&str> = schemas
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        for name in [
            "shell_command",
            "tmux_capture_pane",
            "tmux_send_keys",
            "read_file",
            "edit_file",
        ] {
            assert!(names.contains(&name), "missing tool: {name}");
        }
    }

    #[test]
    fn tool_schemas_all_have_type_function() {
        for schema in tool_schemas() {
            assert_eq!(
                schema["type"].as_str().unwrap(),
                "function",
                "unexpected type for: {}",
                schema["function"]["name"]
            );
        }
    }

    #[test]
    fn tool_schemas_all_have_parameters_with_required_array() {
        for schema in tool_schemas() {
            let name = schema["function"]["name"].as_str().unwrap();
            assert!(
                schema["function"]["parameters"]["required"].is_array(),
                "missing required array for {name}"
            );
        }
    }

    // ---- shell_command ---------------------------------------------------

    #[tokio::test]
    async fn shell_command_captures_stdout() {
        let exec = LocalExecutor;
        let out = exec
            .execute(
                "shell_command",
                serde_json::json!({"command": "echo hello"}),
            )
            .await
            .unwrap();
        assert_eq!(out.trim(), "hello");
    }

    #[tokio::test]
    async fn shell_command_appends_exit_code_on_failure() {
        let exec = LocalExecutor;
        let out = exec
            .execute(
                "shell_command",
                serde_json::json!({"command": "echo bad && exit 2"}),
            )
            .await
            .unwrap();
        assert!(out.contains("[exit 2]"), "got: {out}");
        assert!(out.contains("bad"), "stdout should be present: {out}");
    }

    #[tokio::test]
    async fn shell_command_empty_output_shows_exit_zero() {
        let exec = LocalExecutor;
        let out = exec
            .execute("shell_command", serde_json::json!({"command": "true"}))
            .await
            .unwrap();
        assert_eq!(out, "[exit 0]");
    }

    #[tokio::test]
    async fn shell_command_missing_arg_returns_err() {
        let exec = LocalExecutor;
        let result = exec.execute("shell_command", serde_json::json!({})).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("command"),
            "error should mention 'command': {msg}"
        );
    }

    // ---- read_file -------------------------------------------------------

    #[tokio::test]
    async fn read_file_returns_content() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.txt");
        std::fs::write(&path, "file contents").unwrap();

        let exec = LocalExecutor;
        let out = exec
            .execute(
                "read_file",
                serde_json::json!({"path": path.to_str().unwrap()}),
            )
            .await
            .unwrap();
        assert_eq!(out, "file contents");
    }

    #[tokio::test]
    async fn read_file_nonexistent_returns_err() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("does_not_exist.txt");
        let exec = LocalExecutor;
        let result = exec
            .execute(
                "read_file",
                serde_json::json!({"path": path.to_str().unwrap()}),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_file_missing_path_arg_returns_err() {
        let exec = LocalExecutor;
        let result = exec.execute("read_file", serde_json::json!({})).await;
        assert!(result.is_err());
    }

    // ---- edit_file -------------------------------------------------------

    #[tokio::test]
    async fn edit_file_writes_new_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("new.txt");

        let exec = LocalExecutor;
        let out = exec
            .execute(
                "edit_file",
                serde_json::json!({"path": path.to_str().unwrap(), "content": "written"}),
            )
            .await
            .unwrap();
        assert!(
            out.contains("wrote"),
            "return message should mention 'wrote': {out}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "written");
    }

    #[tokio::test]
    async fn edit_file_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("existing.txt");
        std::fs::write(&path, "old").unwrap();

        let exec = LocalExecutor;
        exec.execute(
            "edit_file",
            serde_json::json!({"path": path.to_str().unwrap(), "content": "new"}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    }

    // ---- unknown tool ---------------------------------------------------

    #[tokio::test]
    async fn unknown_tool_returns_descriptive_error() {
        let exec = LocalExecutor;
        let result = exec
            .execute("nonexistent_tool", serde_json::json!({}))
            .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unknown tool"), "got: {msg}");
    }
}

/// Return the JSON schema array to include in every chat request.
pub fn tool_schemas() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "shell_command",
                "description": "Run a shell command (via sh -c) in the background and \
                                return its stdout and stderr. Use this for grep, find, git, \
                                cargo, systemctl, etc. Does NOT interact with the user's \
                                tmux pane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_capture_pane",
                "description": "Capture and return the current visible text of a tmux pane.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "tmux target pane (e.g. '%0', '0:1.0'). \
                                            Defaults to %0."
                        }
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "tmux_send_keys",
                "description": "Send keystrokes to a tmux pane. Use for interactive \
                                programs (e.g. pressing Enter, Ctrl-C). For background \
                                tasks prefer shell_command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "string",
                            "description": "Keys to send, e.g. 'q', 'Enter', 'C-c'."
                        },
                        "target": {
                            "type": "string",
                            "description": "tmux target pane. Defaults to %0."
                        }
                    },
                    "required": ["keys"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full contents of a file on disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Overwrite a file with new content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the file to write."
                        },
                        "content": {
                            "type": "string",
                            "description": "New full content for the file."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
    ]
}
