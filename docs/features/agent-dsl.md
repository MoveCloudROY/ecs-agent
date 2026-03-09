# Agent DSL

Define AI agents declaratively using JSON or Markdown configuration files. The Agent DSL (Domain-Specific Language) allows you to specify agent roles, models, system prompts, and tool permissions without writing boilerplate setup code.

## Overview

The Agent DSL provides a clean separation between agent configuration and application logic. Instead of manually creating entities and attaching components, you define your agents in structured files.

Key benefits:
- **Declarative Configuration**: Define agents in simple JSON or Markdown.
- **Role-Based Modeling**: Easily switch between `primary` and `subagent` roles.
- **Reusable Prompts**: Reference external prompt files using the `{file:...}` syntax.
- **Granular Permissions**: Map tool names to boolean flags for secure access control.
- **Deterministic Loading**: Stable `last-one-wins` resolution for multiple configuration sources.
- **Fail-Fast Validation**: Strict schema checking with clear error messages.

## JSON DSL Format

The JSON format is ideal for defining multiple agents in a single file or for programmatic generation.

### Schema

A JSON file should contain a root dictionary where each key is an **agent name** and the value is its **configuration**.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mode` | `string` | Yes | Either `"primary"` (main agent) or `"subagent"` (delegated helper). |
| `model` | `string` | Yes | The LLM model identifier (e.g., `"gpt-4o"`). |
| `prompt` | `string` | Yes | The system prompt or a `{file:path}` reference. |
| `tools` | `dict[str, bool]` | No | Mapping of tool names to enabled/disabled status. |
| `metadata` | `dict` | No | Arbitrary key-value pairs for custom application logic. |

### Example: `agents.json`

```json
{
  "coordinator": {
    "mode": "primary",
    "model": "gpt-4o",
    "prompt": "You are a coordinator agent. Delegate tasks to the researcher.",
    "tools": {
      "delegate": true,
      "bash": false
    }
  },
  "researcher": {
    "mode": "subagent",
    "model": "gpt-4o-mini",
    "prompt": "{file:prompts/researcher_system.txt}",
    "tools": {
      "web_search": true,
      "read_file": true
    }
  }
}
```

## Markdown DSL Format

The Markdown format is preferred for human-authored prompts, allowing you to use rich text for instructions while keeping configuration in a YAML frontmatter.

### Syntax

- **Name**: Derived from the **filename** (e.g., `analyst.md` becomes the "analyst" agent).
- **Configuration**: Defined in a YAML frontmatter block (between `---` lines).
- **Prompt**: The entire Markdown body after the frontmatter becomes the system prompt.

### Example: `analyst.md`

```markdown
---
mode: primary
model: gpt-4o
tools:
  read_file: true
  edit_file: true
---
# Data Analyst Specialist

You are an expert data analyst. Your role is to:
1. Read data files provided by the user.
2. Perform statistical analysis.
3. Suggest improvements to the data structure.

## Guidelines
- Always verify data types before analysis.
- Use the `read_file` tool for inspection.
```

## Loading & Compilation

The loading pipeline follows a deterministic four-step workflow:

1. **Discovery**: Find all `*.json` and `*.md` files in a directory.
2. **Loading**: Parse files into normalized `AgentSpec` objects.
3. **Resolution**: Apply `last-one-wins` conflict resolution for duplicate agent names.
4. **Compilation**: Transform specs into a runnable ECS `World` and primary `EntityId`.

### Basic Workflow

```python
from ecs_agent import (
    discover_agent_sources,
    load_json_agents,
    load_markdown_agent,
    resolve_agent_specs,
    compile_agent_specs
)

# 1. Discover sources
paths = discover_agent_sources("./agents")

# 2. Load all specs
all_specs = []
for p in paths:
    if p.suffix == ".json":
        all_specs.extend(load_json_agents(p))
    else:
        all_specs.append(load_markdown_agent(p))

# 3. Resolve conflicts (last-one-wins)
resolved = resolve_agent_specs(all_specs)

# 4. Compile to World
def provider_factory(model, prompt):
    return OpenAIProvider(api_key="...", model=model)

primary_id, world = compile_agent_specs(resolved, provider_factory)
```

## Conflict Resolution

Agent names must be unique within a single `World`. If multiple files define an agent with the same name, the **last one** discovered overrides previous definitions. 

Discovery ordering is lexicographical (alphabetical by path), ensuring that conflict resolution is 100% deterministic and reproducible across different environments.

## Prompt File References

Instead of inlining long prompts, you can use the `{file:relative/path}` syntax in the `prompt` field.

### Security Features
- **Path Traversal Protection**: References like `{file:../../etc/passwd}` are strictly rejected.
- **Absolute Path Restriction**: Only relative paths within the agent's directory are allowed.
- **Symlink Validation**: Resolved paths must remain within the source directory boundaries.
- **UTF-8 Enforcement**: All prompt files must be valid UTF-8.

## Permission Mapping

The `tools` dictionary in the DSL maps directly to the `PermissionComponent` at runtime.

- Only tools explicitly set to `true` are added to the `allowed_tools` list.
- Tools set to `false` or omitted are excluded.
- If the `tools` block is missing, no `PermissionComponent` is attached, falling back to the default system behavior (usually allow-all).
- If a `tools` block exists but all are `false`, an empty allowlist is created, effectively denying all tool usage.

## Error Handling

The Agent DSL follows a **fail-fast policy**. Any configuration error will raise a standard Python exception immediately to prevent undefined behavior.

| Scenario | Exception |
|----------|-----------|
| Missing directory during discovery | `FileNotFoundError` |
| Missing required fields (`mode`, `model`, `prompt`) | `ValueError` |
| Invalid mode (not `primary` or `subagent`) | `ValueError` |
| Multiple primary agents in one compile | `ValueError` |
| Path traversal in prompt reference | `ValueError` |
| Missing prompt file reference | `FileNotFoundError` |

## API Reference

The following exports are available from `ecs_agent`:

### `AgentSpec`
A dataclass representing a normalized agent definition.
```python
@dataclass(slots=True)
class AgentSpec:
    mode: Literal["primary", "subagent"]
    model: str
    prompt: str
    tools: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""
```

### `validate_agent_spec(data: dict, source_name: str = "") -> AgentSpec`
Validates a raw dictionary against the DSL schema.

### `discover_agent_sources(directory: Path | str) -> list[Path]`
Returns a sorted list of JSON and Markdown files in the target directory.

### `load_json_agents(path: Path | str) -> list[AgentSpec]`
Parses a JSON file into a list of specs.

### `load_markdown_agent(path: Path | str) -> AgentSpec`
Parses a Markdown file (filename becomes agent name).

### `resolve_agent_specs(specs: list[AgentSpec]) -> dict[str, AgentSpec]`
Resolves naming conflicts using a deterministic last-one-wins policy.

### `compile_agent_specs(specs: dict[str, AgentSpec], provider_factory: Callable[[str, str], LLMProvider]) -> tuple[EntityId, World]`
Compiles resolved specs into a runnable ECS World. Requires a factory function to create LLM providers.

### `resolve_prompt_file(prompt_spec: str, source_dir: Path) -> str`
Resolves `{file:path}` syntax to UTF-8 file content with security checks.
