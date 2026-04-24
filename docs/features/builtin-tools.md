# Built-in Tools

The framework provides a set of high-quality, built-in tools for file manipulation and shell execution, packaged as the `BuiltinToolsSkill`.

## Overview

The `BuiltinToolsSkill` includes six primary tools:
- `read_file`: Read content from the workspace.
- `write_file`: Create or overwrite files.
- `edit_file`: Perform precise, hash-anchored line edits.
- `bash`: Execute shell commands in a sandboxed-ready environment.
- `glob`: Find files matching a glob pattern in the workspace.
- `interactive_bash`: Execute persistent tmux commands for interactive sessions.

Tool results from `BuiltinToolsSkill` are automatically collected into the **One-Shot Context Pool** if enabled, providing immediate context for the next reasoning turn without polluting the permanent conversation history.

## Tool Bundle Behaviour

`BuiltinToolsSkill` sets `is_tool_bundle = True`. This means:

- Its tools are registered on `ToolRegistryComponent` and available for the LLM to call.
- The skill is **not** listed in `SkillComponent` and does **not** appear in the
  `${_installed_skills}` system-prompt placeholder.
- `load_skill_details` cannot be called for it (it has no Tier-2 skill details).
- No `system_prompt()` fragment is injected into the agent's system prompt.

This keeps the agent's skill list clean: `BuiltinToolsSkill` is infrastructure,
not a user-facing capability the LLM should reason about.


## Installation

```python
from ecs_agent import SkillManager
from ecs_agent.tools.builtins import BuiltinToolsSkill

manager = SkillManager()
manager.install(world, agent_entity, BuiltinToolsSkill())
```

## Tool Reference

### `read_file(path: str) -> str`
Reads the content of a file relative to the workspace root.
- **Security**: Prevents path traversal outside the workspace.
- **Output**: Includes line numbers and hashes in `LINE#HASH|content` format to facilitate use with `edit_file`.

### `write_file(path: str, content: str) -> str`
Writes full content to a file. Creates the file if it does not exist.

### `edit_file(file_path, op, pos, end=None, content="", workspace_root="") -> str`
Applies a single precise, hash-anchored edit to a file. This is the preferred way for LLMs to modify code, as it avoids rewriting entire files and handles concurrent modification risks.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | `str` | Yes | Workspace-relative path to the target file. |
| `op` | `"replace" \| "append" \| "prepend"` | Yes | Edit operation. |
| `pos` | `str` | Yes | Start position in `LINE#HASH` format. Obtain from `read_file` output. |
| `end` | `str \| None` | No | End position in `LINE#HASH` format for range replace. Omit for single-line operations. |
| `content` | `str` | No | New content to insert or replace. Use `\n` to separate multiple lines. |

#### Hash-Anchored Editing

The `edit_file` tool uses a `LINE#HASH` format for addressing lines.
- `LINE`: The 1-based line number.
- `HASH`: A 4-character MD5-based hash of the line's content.

Example output from `read_file`:
```
1#a1b2|def my_function():
2#c3d4|    return 42
```

When applying an edit the tool verifies that the hash at the given line number matches the current file. If the hashes do not match (stale anchor), the edit is rejected — preventing hallucinated edits on outdated content.

#### Operations

- `replace`: Replace a single line (no `end`) or a range of lines (`end` provided). Supply multiline `content` with `\n` separators to expand one line into many.
- `append`: Insert `content` after the line at `pos`.
- `prepend`: Insert `content` before the line at `pos`.

#### Examples

Single-line replace:
```python
await edit_file(
    file_path="src/app.py",
    op="replace",
    pos="5#a1b2",
    content="    return result * 2",
    workspace_root=workspace,
)
```

Range replace (lines 3–5 → new content):
```python
await edit_file(
    file_path="src/app.py",
    op="replace",
    pos="3#c3d4",
    end="5#e5f6",
    content="    step_one()\n    step_two()\n    step_three()",
    workspace_root=workspace,
)
```

Append after a line:
```python
await edit_file(
    file_path="README.md",
    op="append",
    pos="10#g7h8",
    content="## New Section",
    workspace_root=workspace,
)
```

### `bash(command: str, timeout: int = 30) -> str`
Executes a shell command and returns its stdout and stderr combined.
- **Security**: Runs within the workspace root.
- **Timeout**: Default 30 seconds to prevent hanging the agent.

### `glob(pattern: str, base_path: str) -> str`
Finds files in the workspace matching the given glob pattern.
- **`pattern`**: A glob pattern like `**/*.py`, `src/*.ts`, `*.md`.
- **`base_path`**: Directory within the workspace to start from (use `.` for root).
- **Output**: Sorted, newline-delimited workspace-relative file paths. Returns empty string if no matches.
- **Security**: Restricted to the workspace root; no absolute paths returned.

Example output for `pattern="**/*.py"`, `base_path="."`:
```
src/agent.py
src/core/world.py
tests/test_agent.py
```

### `interactive_bash(tmux_command: str) -> str`
Executes a tmux subcommand, enabling persistent interactive sessions across multiple tool calls.

Pass the tmux subcommand and its arguments **without** the leading `tmux` prefix:

```python
await interactive_bash("new-session -d -s train \"bash\"")
await interactive_bash("send-keys -t train \"python train.py\" Enter")
await interactive_bash("capture-pane -t train -p")
```

Equivalent shell commands:
```bash
tmux new-session -d -s train "bash"
tmux send-keys -t train "python train.py" Enter
tmux capture-pane -t train -p
```

Use `interactive_bash` when you need:
- Long-running processes (training loops, servers, build jobs).
- Multiple commands that share session state (environment variables, working directory).
- Monitoring output of a running process without blocking.

## Parameter Descriptions in Tool Schema

All tool parameters expose a human-readable `description` field in their JSON schema. This is surfaced to the LLM via the `Annotated` metadata mechanism in `discovery.py`, so the model understands each parameter's purpose without requiring lengthy docstrings.

## Security

All file tools require a `workspace_root` to be configured. The framework validates that all paths are relative to this root and prevents any traversal (e.g., using `..`) that would access files outside the allowed directory.

## Best Practices

1. **Read Before Edit**: Always use `read_file` to get the latest `LINE#HASH` tags before calling `edit_file`.
2. **One Edit Per Call**: Each `edit_file` call applies a single atomic edit. Re-read after each edit before issuing the next one if line numbers may have shifted.
3. **Use Bash for Verification**: Use the `bash` tool to run tests or linting after editing files.
4. **Use Glob for Discovery**: Use `glob` to discover files before reading or editing them, especially in large workspaces.
5. **Persistent Sessions with tmux**: Use `interactive_bash` to create named sessions for long-running tasks; use `capture-pane` to retrieve their output.

## Examples

See [`examples/script_skill_agent.py`](../../examples/script_skill_agent.py) for a demonstration of these tools in a reasoning loop.
