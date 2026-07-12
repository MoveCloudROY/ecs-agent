# Built-in Tools

The framework provides a set of high-quality, built-in tools for file manipulation, shell execution, code search, directory exploration, web fetching, and code execution, packaged as the `BuiltinToolsSkill`.

## Overview

The `BuiltinToolsSkill` includes ten primary tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read content from the workspace with optional line range. |
| `write_file` | Create or overwrite files. |
| `edit_file` | Perform precise, snapshot-protected text edits. |
| `bash` | Execute shell commands in a sandboxed-ready environment. |
| `glob` | Find files matching a glob pattern in the workspace. |
| `interactive_bash` | Execute persistent tmux commands for interactive sessions. |
| `grep` | Search a file for lines matching a regex pattern. |
| `explore` | Display the workspace directory tree up to a given depth. |
| `webfetch` | Fetch content from a URL and return the response body. |
| `code_execution` | Execute a code snippet and return its output. |

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

## Concurrency

The read-only tools — `read_file`, `grep`, `glob`, `explore`, `webfetch` — are
declared `concurrency_safe` in their schemas. When the model batches several of
them in one response, `ToolExecutionSystem` executes them concurrently and lands
the results in the original call order. Mutating tools (`write_file`,
`edit_file`, `bash`, `interactive_bash`, `code_execution`) stay serial and act
as barriers within a batch. See the ToolExecutionSystem section in
[`docs/systems.md`](../systems.md) for the grouping rules and the
`ToolExecutionConfigComponent(max_concurrency=...)` bound.


## Installation

```python
from ecs_agent import SkillManager
from ecs_agent.tools.builtins import BuiltinToolsSkill

manager = SkillManager()
manager.install(world, agent_entity, BuiltinToolsSkill())
```

To bind a workspace root (recommended — hides `workspace_root` from the LLM schema and injects it automatically):

```python
skill = BuiltinToolsSkill().bind_workspace("/path/to/workspace")
manager.install(world, agent_entity, skill)
```

## Tool Reference

### `read_file(file_path, offset=1, limit=0) -> str`
Reads content from a file relative to the workspace root.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | — | Workspace-relative path. |
| `offset` | `int \| str` | `1` | 1-based starting line number. Numeric strings such as `"10"` are accepted. |
| `limit` | `int \| str` | `0` | Maximum lines to return. `0` reads all lines. Numeric strings such as `"20"` are accepted. |

- **Security**: Prevents path traversal outside the workspace.
- **Output**: Clean file content only. The framework records internal line-hash snapshots for later safe edits without exposing hashes, snapshot IDs, or read IDs to the LLM.
- **Best practice**: Always specify `offset` and `limit` for large files to avoid reading oversized content into the context window.

Example:
```python
# Read lines 10–30 of a large file
result = await read_file("src/module.py", workspace_root=workspace, offset=10, limit=20)
```

### `write_file(file_path, content) -> str`
Writes full content to a file. Creates the file (and any parent directories) if it does not exist.
Overwriting an existing file requires a fresh prior `read_file` call; the framework rejects the write if the file changed after that read.

### `edit_file(file_path, op, pos, end=None, content="") -> str`
Applies a precise, snapshot-protected edit to a file. `pos` and `end` are 1-based file line numbers; the framework resolves internal hashes from entity-scoped snapshots and does not expose hash anchors to the LLM.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | `str` | Yes | Workspace-relative path to the target file. |
| `op` | `"replace" \| "append" \| "prepend"` | Yes | Edit operation. |
| `pos` | `int \| str` | Yes | 1-based file line number to edit, e.g. `12` or `"12"`. |
| `end` | `int \| str \| None` | No | End 1-based file line number for range replace, e.g. `20` or `"20"`. Omit for single-line operations. |
| `content` | `str` | No | New content to insert or replace. Use `\n` to separate multiple lines. |

#### Snapshot-Protected Editing

`read_file` returns clean content:
```
def my_function():
    return 42
```

Internally, the framework stores snapshots containing line hashes and a full-file digest. During normal agent execution these snapshots live on `ToolRuntimeStateComponent` under the current ECS entity, reached through the internal `ToolExecutionContext`; they are not function parameters and never appear in tool schemas. This means two agents, or two entities using the same `BuiltinToolsSkill` instance, cannot authorize edits with each other's reads.

`read_file` may record multiple snapshots for the same file, including parallel reads of different ranges. `edit_file` verifies that the file has not changed since a matching read, then resolves the requested line numbers against recent snapshots for that unchanged digest. If the snapshot is stale or the line was not part of the last read range, the edit is rejected and the model should call `read_file` again with a range that includes the target lines.

Direct Python calls outside `ToolExecutionSystem` keep private fallback snapshot state for backward compatibility. That fallback is only used when no `ToolExecutionContext` is active; framework-managed tool calls always prefer ECS runtime state.

### Runtime State Extension Pattern

`ToolRuntimeStateComponent` is the reusable ECS-native place for cross-tool internal state. Tool families should create a typed accessor backed by a namespace, just like file tools use `FileSnapshotState` in the `"file"` namespace. This keeps hidden state reusable for future browser, search, shell, or database tools without adding schema-visible parameters such as `snapshot_store`, `context`, `world`, or `entity_id`.

#### Operations

- Replace a line with `op="replace"`, `pos=12` or `pos="12"`, and `content` set to the replacement line.
- Replace a line range with `op="replace"`, `pos` set to the first line number, `end` set to the last line number, and `content` set to the replacement block.
- Append/prepend with `op="append"` or `op="prepend"`, `pos` set to the target line number, and `content` set to the inserted text.

### `bash(command, timeout) -> str`
Executes a shell command and returns stdout (or combined output on error).
- **Security**: Runs within the workspace root.
- **Timeout**: Required. Prevents hanging the agent.

### `glob(pattern, base_path) -> str`
Finds files matching a glob pattern in the workspace.
- **Output**: Sorted, newline-delimited workspace-relative paths. Empty string if no matches.
- **Security**: Restricted to the workspace root.

### `interactive_bash(tmux_command) -> str`
Executes a tmux subcommand for persistent interactive sessions. Pass the subcommand **without** the leading `tmux` prefix:

```python
await interactive_bash("new-session -d -s train")
await interactive_bash("send-keys -t train 'python train.py' Enter")
await interactive_bash("capture-pane -t train -p")
```

---

### `grep(pattern, file_path) -> str`
Searches a file for lines matching a regular expression and returns matching lines with line numbers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `str` | Regular expression pattern (Python `re` syntax). |
| `file_path` | `str` | Workspace-relative path to the file. |

- **Case-sensitive** by default. Use `(?i)` prefix for case-insensitive matching.
- **Output**: Newline-delimited `LINE: content` entries. Empty string if no matches.
- **Security**: Restricted to workspace root; no path traversal.

Example output for `pattern="^def "`, `file_path="module.py"`:
```
1: def hello():
4: def add(a, b):
```

---

### `explore(path, max_depth) -> str`
Displays the directory tree of a workspace path as ASCII art.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Workspace-relative path (`'.'` for root). |
| `max_depth` | `int` | Maximum tree depth to display (1–5). |

- **Security**: Restricted to workspace root.
- **Output**: ASCII tree format with `├──` and `└──` connectors.

Example output for `path="."`, `max_depth=2`:
```
.
├── src/
│   ├── __init__.py
│   └── module.py
└── tests/
    └── test_module.py
```

---

### `webfetch(url, timeout=30.0) -> str`
Fetches the content of a URL and returns the response body as text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | — | The URL to fetch. |
| `timeout` | `float` | `30.0` | Request timeout in seconds. |

- Follows redirects automatically.
- Raises an error on HTTP 4xx/5xx responses.
- Returns raw text (HTML, JSON, plain text, etc.).

---

### `code_execution(code, language, timeout=30.0) -> str`
Executes a code snippet and returns its stdout output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code` | `str` | — | Source code to execute. |
| `language` | `str` | — | Programming language (`'python'`). |
| `timeout` | `float` | `30.0` | Maximum execution time in seconds. |

- **Supported languages**: `python`
- On non-zero exit, returns `Exit code N\nSTDOUT:\n...\nSTERR:\n...`.
- On timeout, raises `ValueError`.

Example:
```python
result = await code_execution(
    code="print(sum(range(1, 101)))",
    language="python",
)
# result: "5050\n"
```

---

## Parameter Descriptions in Tool Schema

All tool parameters expose a human-readable `description` field in their JSON schema via the `Annotated` metadata mechanism in `discovery.py`, so the LLM understands each parameter's purpose.

## Security

All file and directory tools require a `workspace_root` to be configured (injected automatically via `bind_workspace`). The framework validates that all paths resolve within this root and rejects any traversal attempts (e.g., `..`, absolute paths, symlinks pointing outside).

Tools that do not operate on the workspace (`webfetch`, `code_execution`) do not receive or require `workspace_root`.

## Best Practices

1. **Read Before Edit**: Use `read_file` (with appropriate `offset`/`limit`) before `edit_file` so the framework has a fresh internal snapshot.
2. **One Edit Per Call**: Re-read after each `edit_file` if line numbers may have shifted.
3. **Discover Before Reading**: Use `glob` or `explore` to discover files in unfamiliar workspaces.
4. **Search Before Reading Large Files**: Use `grep` to locate relevant lines, then use `read_file` with a narrow range.
5. **Use Bash for Verification**: Run tests or linting via `bash` after editing files.
6. **Persistent Sessions**: Use `interactive_bash` for long-running tasks; retrieve output with `capture-pane`.

## Examples

See [`examples/script_skill_agent.py`](../../examples/script_skill_agent.py) for a demonstration of these tools in a reasoning loop.
