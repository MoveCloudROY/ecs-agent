# Agent DSL

Define AI agents declaratively using JSON or Markdown configuration files with deterministic loading and fail-fast validation.

## Overview

The Agent DSL (Domain-Specific Language) allows you to define agent configurations in JSON or Markdown files instead of programmatically creating entities and components. The DSL compiler transforms these configurations into ECS entities with properly attached components.

**Key Features:**
- **Dual Format Support**: JSON (multi-agent dict) or Markdown (YAML frontmatter + body)
- **Deterministic Loading**: Sorted file discovery ensures reproducible conflict resolution
- **Fail-Fast Validation**: Strict schema validation with detailed error messages
- **Security**: Path traversal protection for prompt file references
- **Last-One-Wins**: Predictable conflict resolution for duplicate agent names
- **Permission Mapping**: Boolean tool dictionaries compile to `PermissionComponent` allowlists
- **Skill Installation**: Declare SKILL.md-based skills to auto-install onto the primary agent at compile time

## JSON DSL Format

### Schema

Agent specifications use the following schema:

```python
@dataclass
class AgentSpec:
    mode: Literal["primary", "subagent"]  # Required
    model: str                             # Required
    prompt: str                            # Required
    tools: dict[str, bool] = {}            # Optional
    metadata: dict[str, Any] = {}          # Optional
    name: str = ""                              # Optional
    placeholders: list[dict[str, str]] = []    # Optional (primary only)
    triggers: list[dict[str, str | int]] = []  # Optional (primary only)
```

**Required Fields:**
- `mode`: Either `"primary"` (runnable main entity) or `"subagent"` (config template)
- `model`: LLM model identifier (e.g., `"gpt-4"`, `"claude-3-opus"`)
- `prompt`: System prompt or instruction text

**Optional Fields:**
- `tools`: Tool permission mapping (`{tool_name: true/false}`)
- `metadata`: Arbitrary user-defined metadata
- `name`: Agent name (overridden by dict key in JSON)
- `placeholders`: List of `{name, value}` dicts declaring `${name}` template variables in `prompt`
- `triggers`: List of trigger dicts enabling `UserPromptNormalizationSystem` (see Triggers section)
- `skills`: List of skill path dicts (`[{"path": "relative/dir"}]`); installs SKILL.md-based skills on compile (primary only)

### JSON Example

```json
{
  "assistant": {
    "mode": "primary",
    "model": "qwen3.5-flash",
    "prompt": "You are a manager agent. When given a complex question, use the 'subagent' tool to delegate work to background workers. After receiving the results, synthesize them into a concise summary.\n\nAvailable tools:\n${_installed_tools}\n\nAvailable subagents:\n${_installed_subagents}\n\nSession: ${session_label}",
    "placeholders": [
      {"name": "session_label", "value": "subagent-delegation-demo"}
    ]
  },
  "researcher": {
    "mode": "subagent",
    "model": "qwen3.5-flash",
    "prompt": "You are a research sub-agent. Investigate the given topic thoroughly and report your findings back to the manager."
  }
}
```

**Key Points:**
- Root must be `dict[str, dict]` (agent_name → config)
- Dictionary key becomes the agent's `name` field
- Multiple agents allowed, but exactly ONE must have `mode: "primary"`

## Markdown DSL Format

### Structure

Markdown files use YAML frontmatter for configuration and markdown body for the prompt:

```markdown
---
mode: primary
model: gpt-4
tools:
  read_file: true
  write_file: true
---

# System Prompt

You are a helpful assistant specialized in file operations.

## Guidelines

- Always verify file paths before operations
- Use read_file before write_file to check existing content
- Provide clear error messages
```

### Markdown Example

Filename: `assistant.md`

```markdown
---
mode: primary
model: qwen3.5-flash
placeholders:
  - name: session_label
    value: subagent-delegation-demo
---

You are a manager agent. When given a complex question, use the 'subagent' tool to delegate work to background workers. After receiving the results, synthesize them into a concise summary.

Available tools:
${_installed_tools}

Available subagents:
${_installed_subagents}

Session: ${session_label}
```

**Key Points:**
- Filename (without `.md`) becomes agent name
- YAML frontmatter contains same fields as JSON schema
- Entire markdown body becomes the `prompt` field
- Uses `yaml.safe_load()` for security

## Loading & Compilation

### Workflow

The DSL loading pipeline has four stages:

```python
from ecs_agent.dsl import (
    discover_agent_sources,
    load_json_agents,
    load_markdown_agent,
    resolve_agent_specs,
    compile_agent_specs,
)

# 1. Discover sources (sorted for determinism)
sources = discover_agent_sources("./agents")
# Returns: [Path('agents/assistant.json'), Path('agents/researcher.md')]

# 2. Load from files
specs = []
for source in sources:
    if str(source).endswith('.json'):
        specs.extend(load_json_agents(source))
    else:
        specs.append(load_markdown_agent(source))

# 3. Resolve conflicts (last-one-wins)
resolved = resolve_agent_specs(specs)
# Returns: {'assistant': AgentSpec(...), 'researcher': AgentSpec(...)}

# 4. Compile to ECS World
def model_factory(model: str, system_prompt: str):
    return Model(
        model,
        base_url="https://api.openai.com/v1",
        api_key="...",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )

primary_entity, world = compile_agent_specs(resolved, model_factory)
# Returns: (EntityId, World) with components attached
```

### Stage Details

**1. discover_agent_sources(directory: Path) → list[Path]**
- Glob patterns: `*.json` and `*.md` (flat directory only)
- Returns sorted paths for deterministic ordering
- Raises `FileNotFoundError` if directory missing

**2. load_json_agents(path: Path) → list[AgentSpec]**
- Parses multi-agent dict from JSON file
- Dict key becomes agent name
- Validates each agent spec
- Raises `ValueError` for malformed JSON or invalid schema

**3. load_markdown_agent(path: Path) → AgentSpec**
- Parses YAML frontmatter + markdown body
- Filename (sans `.md`) becomes agent name
- Body content becomes prompt
- Raises `ValueError` for invalid YAML or missing frontmatter

**4. resolve_agent_specs(specs: list[AgentSpec]) → dict[str, AgentSpec]**
- Implements last-one-wins conflict resolution
- Later specs override earlier ones with same name
- Raises `ValueError` for empty agent names

**5. compile_agent_specs(specs: dict[str, AgentSpec], factory) → tuple[EntityId, World]**
- Creates exactly ONE runnable primary entity
- Creates exactly ONE runnable primary entity
- Attaches `LLMComponent`, `SystemPromptConfigSpec`, `PermissionComponent` (when `tools` present), `SubagentRegistryComponent`, `UserPromptConfigComponent` (always), `ToolRegistryComponent` (always). When subagents are declared, also attaches `SubagentSessionTableComponent` and installs `SubagentSystem` with the `subagent` tool. Auto-registers `SystemPromptRenderSystem` (priority `-20`) and `UserPromptNormalizationSystem` (priority `-10`) unconditionally.
- Subagents become `SubagentConfig` in registry
- Raises `ValueError` if zero or multiple primaries

## Conflict Resolution

### Last-One-Wins Policy

When multiple sources define agents with the same name, the **last one in sorted order wins**:

```python
# Directory structure:
# agents/
#   01-base.json       # defines "assistant"
#   02-override.json   # defines "assistant" (wins)

sources = discover_agent_sources("./agents")
# Returns: [Path('agents/01-base.json'), Path('agents/02-override.json')]

specs = []
for source in sources:
    specs.extend(load_json_agents(source))

resolved = resolve_agent_specs(specs)
# resolved['assistant'] uses config from 02-override.json
```

**Determinism Guarantee:**
- `discover_agent_sources()` uses `sorted()` on paths
- Same filesystem state always produces same result
- Lexicographic ordering: `a.json` < `b.json` < `z.md`

## Prompt File References

### Syntax

Use `{file:relative/path}` to reference external prompt files:

```json
{
  "assistant": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "{file:prompts/assistant-system.txt}"
  }
}
```

### Security

Prompt file resolution enforces strict security:

```python
from ecs_agent.dsl.prompt_resolver import resolve_prompt_file

# Allowed: relative paths within source directory
resolved = resolve_prompt_file("{file:prompts/system.txt}", source_dir=Path("./agents"))

# Rejected: absolute paths
resolve_prompt_file("{file:/etc/passwd}", ...)  # ValueError

# Rejected: path traversal
resolve_prompt_file("{file:../../secrets.txt}", ...)  # ValueError

# Rejected: symlink escapes
resolve_prompt_file("{file:link_to_outside}", ...)  # ValueError if target outside source_dir
```

**Security Checks:**
1. Reject absolute paths (`Path.is_absolute()`)
2. Reject path traversal (`..` in `Path.parts`)
3. Validate resolved path stays within `source_dir` (`relative_to()` check)
4. Reject symlinks pointing outside `source_dir`

**Encoding:**
- All files read as UTF-8
- `UnicodeDecodeError` → `ValueError` with context

## Permission Mapping

### Tools Boolean Dict

The `tools` field maps tool names to boolean enabled/disabled flags:

```json
{
  "assistant": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "You are a helpful assistant.",
    "tools": {
      "read_file": true,
      "write_file": true,
      "execute_bash": false,
      "delete_file": false
    }
  }
}
```

### Compilation Behavior

**Enabled tools (true) → PermissionComponent.allowed_tools:**

```python
# Input: {"read_file": true, "write_file": true, "execute_bash": false}
# Output: PermissionComponent(allowed_tools=["read_file", "write_file"])
```

**Rules:**
- Only `true` values included in allowlist
- Order preserved from dict iteration
- Empty allowlist (`[]`) means deny-all (no tools allowed)
- Missing `tools` field → no `PermissionComponent` (default runtime behavior)

**Integration with PermissionSystem:**

```python
# At runtime, PermissionSystem checks:
if permission.allowed_tools:  # Non-empty list
    if tool_name not in permission.allowed_tools:
        raise PermissionError(f"Tool {tool_name} not in allowlist")
else:  # Empty list
    raise PermissionError("All tools denied (empty allowlist)")
```
## Placeholders

Declare `${name}` template variables in your `prompt` field and resolve them via `placeholders`:

### JSON Example

```json
{
  "assistant": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "You are a ${role}. Your tone is ${tone}.",
    "placeholders": [
[object Object]
[object Object]
    ]
  }
}
```

### Validation Rules

- Each entry must have `name` (str) and `value` (str)
- `name` must match `[A-Za-z_][A-Za-z0-9_]*`
- Names starting with `_` are reserved (e.g., `_installed_tools`)

### Compilation

`compile_agent_specs` builds `SystemPromptConfigSpec` with `PlaceholderSpec` objects from these entries.
`SystemPromptRenderSystem` (auto-registered at priority `-20`) resolves `${name}` → value before the LLM call.

## Triggers

Declare trigger rules that inject context into user messages via `UserPromptNormalizationSystem`:

### JSON Example
,
```json
{
  "assistant": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "You are an assistant.",
    "triggers": [
      {
        "pattern": "@help",
        "match_mode": "keyword",
        "action": "inject",
        "content": "The user is requesting help. Show available commands.",
        "priority": 0
      }
    ]
  }
}
```

### Trigger Schema

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `pattern` | str | yes | any string |
| `match_mode` | str | yes | `keyword`, `prefix`, `contains` |
| `action` | str | yes | `replace`, `inject`, `script` |
| `content` | str | yes | any string |
| `priority` | int | no (default 0) | integer |

### Compilation

When `triggers` are present, `compile_agent_specs`:
1. Attaches `UserPromptConfigComponent` with the declared `TriggerSpec` objects to the primary entity

`UserPromptNormalizationSystem` (priority `-10`) is **always** registered regardless of whether triggers are present, because skill slash-command injection also requires it.
> **Note:** The `script` action is not available in the Agent DSL (JSON or Markdown format).
> Script handlers are Python callables and cannot be serialized to text.
> To use script triggers, construct `UserPromptConfigComponent` directly in Python:
>
> ```python
> async def my_handler(world: World, entity_id: EntityId, user_text: str) -> str | None:
>     # rewrite prompt or mutate world
>     return f"[processed] {user_text}"
>
> world.add_component(entity, UserPromptConfigComponent(
>     triggers=[TriggerSpec(pattern="@run", match_mode="keyword", action="script", content="my_handler")],
>     script_handlers={"my_handler": my_handler},
> ))
> ```


## Skills

Declare SKILL.md-based skills to install onto the primary agent at compile time.

### JSON Example

```json
{
  "assistant": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "You are a helpful assistant.",
    "skills": [
      {"path": "skills/ui-ux-reviewer"}
    ]
  }
}
```

### Markdown Example

```yaml
skills:
  - path: skills/ui-ux-reviewer
```

### Skill Path Rules

Each `path` entry is a directory (relative to the DSL source file) that contains a `SKILL.md` file.

**Validation rules:**
- Must be a relative path (absolute paths are rejected)
- Must not contain `..` path traversal
- Must be non-empty string

### Compilation

When `skills` are present, pass `source_dir` to `compile_agent_specs`:

```python
from pathlib import Path

primary_entity, world = compile_agent_specs(
    resolved,
    model_factory,
    source_dir=Path("./examples"),  # skill paths resolved relative to this
)
```

For each skill entry, the compiler:
1. Resolves `(source_dir / path / "SKILL.md").resolve()`
2. Loads the skill with `Skill(skill_path=...)`
3. Installs it via `SkillManager().install(world, primary_entity, skill)`

If `source_dir` is `None` and skills are declared, a warning is logged and skills are skipped (no exception raised).

## Error Handling

### Fail-Fast Policy

All DSL errors raise exceptions immediately with detailed context:

```python
# Missing required field
validate_agent_spec({"mode": "primary", "model": "gpt-4"}, source_name="assistant")
# ValueError: Missing required field(s): prompt in 'assistant'

# Unknown field
validate_agent_spec({
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "...",
    "unknown_field": 123
}, source_name="assistant")
# ValueError: Unknown field(s): unknown_field in 'assistant'

# Invalid mode
validate_agent_spec({
    "mode": "invalid",
    "model": "gpt-4",
    "prompt": "..."
}, source_name="assistant")
# ValueError: Invalid mode 'invalid': must be 'primary' or 'subagent' in 'assistant'

# Multiple primaries
compile_agent_specs({
    "a": AgentSpec(mode="primary", ...),
    "b": AgentSpec(mode="primary", ...)
}, factory)
# ValueError: Expected exactly one primary agent, found 2

# Missing primary
compile_agent_specs({
    "a": AgentSpec(mode="subagent", ...),
    "b": AgentSpec(mode="subagent", ...)
}, factory)
# ValueError: Expected exactly one primary agent, found 0
```

### Error Types

| Stage | Exception | Trigger | Message Format |
|-------|-----------|---------|----------------|
| Discovery | `FileNotFoundError` | Directory missing | `"Directory not found: {path}"` |
| Discovery | `ValueError` | Path is file not dir | `"Path is not a directory: {path}"` |
| JSON Load | `FileNotFoundError` | File missing | `"Agent JSON file not found: {path}"` |
| JSON Load | `ValueError` | Malformed JSON | `"Failed to parse JSON from {path}: {error}"` |
| Markdown Load | `FileNotFoundError` | File missing | `"Markdown agent file not found: {path}"` |
| Markdown Load | `ValueError` | Invalid YAML | `"Failed to parse YAML frontmatter from {path}: {error}"` |
| Validation | `ValueError` | Missing fields | `"Missing required fields: {fields} (source: {name})"` |
| Validation | `ValueError` | Unknown fields | `"Unknown fields: {fields} (source: {name})"` |
| Validation | `TypeError` | Wrong type | `"Field '{field}' must be {type} (source: {name})"` |
| Resolver | `ValueError` | Empty name | `"Agent name cannot be empty (index: {index})"` |
| Compiler | `ValueError` | Wrong primary count | `"Expected exactly one primary agent, found {count}"` |
| Prompt File | `ValueError` | Absolute path | `"Absolute paths not allowed in {file:} reference: {path}"` |
| Prompt File | `ValueError` | Path traversal | `"Path traversal (..) not allowed in {file:} reference: {path}"` |
| Prompt File | `FileNotFoundError` | File missing | `"Prompt file not found: {path}"` |

## Usage Examples

### Basic Single-Agent JSON

```python
import asyncio
import json
from pathlib import Path
from ecs_agent.dsl import discover_agent_sources, load_json_agents, resolve_agent_specs, compile_agent_specs
from ecs_agent.core import Runner
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.components import ConversationComponent
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.types import Message

# Create agent config
config = {
    "assistant": {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are a helpful assistant."
    }
}

# Save to file
Path("agents").mkdir(exist_ok=True)
Path("agents/config.json").write_text(json.dumps(config))

# Load and compile
sources = discover_agent_sources("./agents")
specs = []
for source in sources:
    specs.extend(load_json_agents(source))

resolved = resolve_agent_specs(specs)

def model_factory(model: str, system_prompt: str):
    return Model(
        model,
        base_url="https://api.openai.com/v1",
        api_key="your-key",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )

primary_entity, world = compile_agent_specs(resolved, model_factory)

# Add conversation and systems
world.add_component(
    primary_entity,
    ConversationComponent(messages=[Message(role="user", content="Hello!")])
)
world.register_system(ReasoningSystem(), priority=0)
world.register_system(MemorySystem(), priority=10)

# Run
runner = Runner()
await runner.run(world, max_ticks=3)
```

### Markdown with Subagents

```markdown
<!-- File: agents/main.md -->
---
mode: primary
model: gpt-4
tools:
  subagent: true
  read_file: true
---

# Orchestrator Agent

You coordinate work between specialized subagents.
Use the subagent tool to assign tasks to researchers and writers.
```

```markdown
<!-- File: agents/researcher.md -->
---
mode: subagent
model: gpt-3.5-turbo
tools:
  web_search: true
  read_file: true
---

# Research Specialist

You gather information from web searches and documents.
Provide comprehensive, well-sourced answers.
```

```python
# Load and run
sources = discover_agent_sources("./agents")
specs = [load_markdown_agent(s) for s in sources]
resolved = resolve_agent_specs(specs)

primary_entity, world = compile_agent_specs(resolved, model_factory)

# SubagentRegistryComponent now populated with 'researcher' config
registry = world.get_component(primary_entity, SubagentRegistryComponent)
assert 'researcher' in registry.subagents
```

### Prompt File References

```
agents/
  assistant.json
  prompts/
    system.txt
    guidelines.md
```

**assistant.json:**
```json
{
  "assistant": {
    "mode": "primary",
    "model": "gpt-4",
    "prompt": "{file:prompts/system.txt}"
  }
}
```

**prompts/system.txt:**
```
You are a helpful assistant specialized in code review.

Guidelines:
- Focus on security vulnerabilities
- Check for proper error handling
- Verify test coverage
```

```python
# Load with prompt file resolution
sources = discover_agent_sources("./agents")
specs = []
for source in sources:
    if str(source).endswith('.json'):
        specs_from_json = load_json_agents(source)
        # Prompt file resolution happens automatically during load
        # Each spec's prompt field contains resolved content
        specs.extend(specs_from_json)
```

## API Reference

### ecs_agent.dsl

**AgentSpec**
```python
@dataclass
class AgentSpec:
    mode: Literal["primary", "subagent"]
    model: str
    prompt: str
    tools: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    placeholders: list[dict[str, str]] = field(default_factory=list)
    triggers: list[dict[str, str | int]] = field(default_factory=list)
    skills: list[dict[str, str]] = field(default_factory=list)
```

**validate_agent_spec(data, *, source_name="") → AgentSpec**
- Validates and normalizes agent specification from raw dict
- Raises `ValueError` for schema violations
- Raises `TypeError` for type mismatches

**discover_agent_sources(directory) → list[Path]**
- Discovers `*.json` and `*.md` files in directory (non-recursive)
- Returns sorted paths for deterministic ordering
- Raises `FileNotFoundError` if directory doesn't exist

**load_json_agents(path) → list[AgentSpec]**
- Loads multi-agent dict from JSON file
- Dict keys become agent names
- Returns list of validated `AgentSpec` instances
- Raises `ValueError` for malformed JSON or invalid specs

**load_markdown_agent(path) → AgentSpec**
- Loads single agent from Markdown file with YAML frontmatter
- Filename (sans `.md`) becomes agent name
- Markdown body becomes prompt
- Raises `ValueError` for invalid YAML or missing frontmatter

**resolve_agent_specs(specs) → dict[str, AgentSpec]**
- Resolves conflicts using last-one-wins policy
- Returns dict mapping agent names to specs
- Raises `ValueError` for empty agent names

**compile_agent_specs(specs, model_factory, *, source_dir: Path | None = None) → tuple[EntityId, World]**
- Compiles agent specs into ECS World with components
- Creates exactly one runnable primary entity
- Subagents populate `SubagentRegistryComponent`
- `model_factory: Callable[[str, str], LLMModel]` creates models
- `source_dir`: Optional base directory for resolving skill paths; required when `skills` are declared
- Raises `ValueError` if zero or multiple primaries

### ecs_agent.dsl.prompt_resolver

**resolve_prompt_file(prompt_spec, source_dir) → str**
- Resolves `{file:path}` references in prompt strings
- Returns file content if pattern matches, otherwise returns input unchanged
- Enforces security: rejects absolute paths, path traversal, symlink escapes
- Raises `ValueError` for security violations
- Raises `FileNotFoundError` if referenced file doesn't exist
