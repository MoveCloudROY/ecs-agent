# Built-in Tools

The framework provides a set of high-quality, built-in tools for file manipulation and shell execution, packaged as the `BuiltinToolsSkill`.

## Overview

The `BuiltinToolsSkill` includes four primary tools:
-   `read_file`: Read content from the workspace.
-   `write_file`: Create or overwrite files.
-   `edit_file`: Perform precise, hash-anchored line edits.
-   `bash`: Execute shell commands in a sandboxed-ready environment.

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
-   **Security**: Prevents path traversal outside the workspace.
-   **Output**: Includes line numbers and hashes in `LINE#ID|content` format to facilitate use with `edit_file`.

### `write_file(path: str, content: str) -> str`
Writes full content to a file. Creates the file if it does not exist.

### `edit_file(file_path: str, edits_json: str) -> str`
Applies a list of precise edits to a file. This is the preferred way for LLMs to modify code, as it avoids rewriting entire files and handles concurrent modification risks.

#### Hash-Anchored Editing
The `edit_file` tool uses a `LINE#ID` format for addressing lines.
-   `LINE`: The 1-based line number.
-   `ID`: A 4-character hash of the line's content.

Example: `10#A1B2|def my_function():`

When applying an edit, the tool verifies that the hash for the specified line number matches the current file content. If the hashes do not match, the edit is rejected, preventing "hallucinated" edits on stale content.

#### Operations
-   `replace`: Replace a single line or a range of lines.
-   `append`: Insert lines after a specific line.
-   `prepend`: Insert lines before a specific line.

#### Edit Schema
```json
[
  {
    "op": "replace",
    "pos": "10#A1B2",
    "end": "12#C3D4",
    "lines": ["new line 1", "new line 2"]
  }
]
```

### `bash(command: str, timeout: int = 30) -> str`
Executes a shell command and returns its stdout and stderr combined.
-   **Security**: Runs within the workspace root.
-   **Timeout**: Default 30 seconds to prevent hanging the agent.

## Security

All file tools require a `workspace_root` to be configured. The framework validates that all paths are relative to this root and prevents any traversal (e.g., using `..`) that would access files outside the allowed directory.

## Best Practices

1.  **Read Before Edit**: Always use `read_file` to get the latest `LINE#ID` tags before calling `edit_file`.
2.  **Small Edits**: Prefer multiple small edits over one massive replacement to reduce the chance of collision or error.
3.  **Use Bash for Verification**: Use the `bash` tool to run tests or linting after editing files.

## Examples

See [`examples/skill_agent.py`](../../examples/skill_agent.py) for a demonstration of these tools in a reasoning loop.
