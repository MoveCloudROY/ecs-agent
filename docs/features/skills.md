# Skills

The Skills system provides a modular way to group tools and system prompts into composable capabilities.

## Quick Start

```python
from ecs_agent.skills.manager import SkillManager
from my_skills import WeatherSkill

manager = SkillManager()
# Install registers metadata, system prompt, and tools in one call
manager.install(world, agent_entity, WeatherSkill())
```

## Concepts

### What Is a Skill?
A Skill is a package of functionality that includes:
- **Tools**: Async handlers and their JSON schemas.
- **System Prompt**: Contextual instructions for the agent.
- **Lifecycle hooks**: Logic to run during installation and uninstallation.

### Two-Phase Lifecycle (Index → Activate)
To remain token-efficient, the `SkillManager` supports a two-phase loading process:

1.  **Index**: Register skill metadata (name, description, tool names) into the `SkillComponent`. No tools are registered in the `ToolRegistryComponent` yet, and the system prompt is not loaded. `SkillMetadata.activated` is `False`.
2.  **Activate**: Load the skill's system prompt into the `SystemPromptComponent` and register tools into the `ToolRegistryComponent`. `SkillMetadata.activated` is `True`.

### Progressive Disclosure (3 Tiers)
Skills use progressive disclosure to minimize the context window while keeping capabilities discoverable:

1.  **Tier 1: Metadata Summary**: Included in the main system prompt. Lists available skill names and descriptions.
2.  **Tier 2: Detailed Schemas**: The `load_skill_details` meta-tool allows the LLM to fetch full JSON schemas for a specific skill's tools on demand.
3.  **Tier 3: Reference Docs**: Extensive documentation or guides can be requested by the agent (optional/custom implementation).

## Skill Protocol

Any class implementing the `Skill` protocol can be managed by a `SkillManager`.

```python
from typing import Protocol, runtime_checkable
from ecs_agent.core import World
from ecs_agent.types import EntityId, ToolSchema, ToolHandler

@runtime_checkable
class Skill(Protocol):
    name: str
    description: str

    def tools(self) -> dict[str, tuple[ToolSchema, ToolHandler]]:
        """Return tool schemas and their async handlers."""
        ...

    def system_prompt(self) -> str:
        """Return context for the system prompt."""
        ...

    def install(self, world: World, entity_id: EntityId) -> None:
        """Called when the skill is added to an entity."""
        ...

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        """Called when the skill is removed from an entity."""
        ...
```

## SkillManager API

The `SkillManager` handles the installation lifecycle, tool registration, and prompt management.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `index` | `index(world, entity_id, skill)` | Register metadata only. No tools loaded. `activated=False`. |
| `activate` | `activate(world, entity_id, skill_name)` | Load system prompt and register tools. Idempotent. |
| `install` | `install(world, entity_id, skill)` | Convenience: `index()` + `activate()` in one call. |
| `uninstall` | `uninstall(world, entity_id, skill_name)` | Remove metadata, tools, and system prompt fragment. |
| `list_skills` | `list_skills(world, entity_id)` | Return all `SkillMetadata` for installed skills. |
| `get_skill_metadata` | `get_skill_metadata(world, entity_id, skill_name)` | Return metadata for a specific skill, or `None`. |
| `can_invoke_via_slash` | `can_invoke_via_slash(world, entity, slash_cmd)` | Check if a slash command like `/skill-name` is user-invocable. |
| `can_model_auto_invoke_skill` | `can_model_auto_invoke_skill(world, entity, skill_name)` | Check if model can auto-invoke this skill. |
| `format_skill_details` | `format_skill_details(world, entity_id, skill_name)` | Return formatted Tier 2 details string for a skill. |

### Code Examples

#### Two-Phase Loading
```python
manager.index(world, agent_entity, my_skill)
# ... later, when the agent decides to use it ...
manager.activate(world, agent_entity, "my-skill")
```

#### Single-Call Installation
```python
manager.install(world, agent_entity, my_skill)
```

## Markdown Skills (SKILL.md)

Markdown Skills allow defining capabilities using a `.claude/skills/<name>/SKILL.md` file format with YAML frontmatter and an optional `scripts/` directory for tools.

### File Format
A Markdown Skill consists of a YAML frontmatter block followed by the markdown body which becomes the system prompt.

```markdown
---
name: web-scraper
description: Fetch and extract content from web pages
user-invocable: true
---

# Web Scraping Specialist
You can fetch web pages and extract structured content.
...
```

### YAML Frontmatter Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **Required** | Skill identifier (regex: `[a-z0-9-]{1,64}`) |
| `description` | string | **Required** | Brief summary of skill purpose |
| `user-invocable` | boolean | `true` | Whether the user can invoke via slash command. Alias `user-invokable` also accepted. |
| `disable-model-invocation`| boolean | `false` | Whether to exclude from model auto-invocation. |
| `argument-hint` | string | `""` | Hint text shown for arguments (e.g., `<url>`). |
| `allowed-tools` | list[str] | `[]` | List of tool names this skill is allowed to use. |
| `context` | string | `None` | Context routing identifier. |
| `agent` | string | `None` | Agent routing identifier. |
| `model` | string | `None` | Specific model to use for this skill. |
| `hooks` | dict | `{}` | Dictionary of lifecycle hook configurations. |

### String Substitutions
Markdown bodies and tool scripts support Claude-compatible substitutions. They are processed in this specific order:

| Variable | Description | Example |
|----------|-------------|---------|
| `$ARGUMENTS[N]` | Nth word (0-indexed) from input arguments. | `$ARGUMENTS[0]` |
| `$ARGUMENTS` | The entire whitespace-separated arguments string. | `$ARGUMENTS` |
| `$N` | Shorthand for Nth argument (1-indexed). | `$1`, `$2` |
| `${CLAUDE_SESSION_ID}` | Resolves to the current session ID. | `ses_123` |
| `${CLAUDE_SKILL_DIR}` | Absolute path to the skill's directory. | `/path/to/skill` |

### Tool Scripts (scripts/ directory)
Python files in the `scripts/` directory are automatically discovered as tools. Each script must define a `TOOL_SCHEMA` and an async `TOOL_HANDLER`.

```python
# scripts/fetch_url.py
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch HTML content from URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            },
            "required": ["url"]
        }
    }
}

async def TOOL_HANDLER(arguments: dict) -> str:
    url = arguments["url"]
    # ... logic ...
    return "Result content"
```

### Invocation Controls
Use `user-invocable` and `disable-model-invocation` to control how skills are triggered.

```python
# Check if user can run '/web-scraper'
can_run = manager.can_invoke_via_slash(world, agent, "/web-scraper")

# Check if model can use it automatically
can_auto = manager.can_model_auto_invoke_skill(world, agent, "web-scraper")
```

### Path Safety
The `MarkdownSkill` class provides `resolve_supporting_path` to safely resolve paths within the skill directory, blocking any directory traversal attempts.

```python
# Safe
path = skill.resolve_supporting_path("data/config.json")
# Raises ValueError
path = skill.resolve_supporting_path("../../../etc/passwd")
```

### Skip+Warn Policy
If a `SKILL.md` file contains invalid frontmatter or missing required fields, the discovery system logs a warning and skips that specific skill, continuing with the rest of the directory.

## Skill Discovery

### Python Skill Discovery (SkillDiscovery)
Scans directories for `.py` files and instantiates classes implementing the `Skill` protocol.

```python
from ecs_agent.skills.discovery import SkillDiscovery

discovery = SkillDiscovery(skill_paths=["./my_skills"])
skills = discovery.discover() # Returns list[Skill]
```

### Markdown Skill Discovery (discover_markdown_skills)
Recursively scans directories for `SKILL.md` files.

```python
from ecs_agent.skills.markdown_skill import discover_markdown_skills

skills = discover_markdown_skills(directories=[Path(".claude/skills")])
```

### DiscoveryManager
Combines Python and Markdown skill discovery into a single API.

```python
from ecs_agent.skills.discovery import DiscoveryManager

discovery_mgr = DiscoveryManager(
    skill_paths=["./python_skills"],
    mcp_configs=mcp_configs
)

# Returns a DiscoveryReport
report = await discovery_mgr.auto_discover_and_install(
    world, agent_entity, manager, directories=[Path(".claude/skills")]
)
```

**DiscoveryReport Fields:**
- `installed_skills`: List of successfully installed skill names.
- `failed_sources`: List of `(source, error)` tuples.
- `skipped_mcp`: List of skipped MCP servers.

## SkillMetadata Reference

Stored in the `SkillComponent` for each installed skill.

| Field | Description |
|-------|-------------|
| `name` | Unique skill name. |
| `description` | Summary of the skill. |
| `tool_names` | List of tools (populated after activation). |
| `has_system_prompt`| True if a system prompt is loaded. |
| `activated` | True if the skill has been activated. |
| `user_invocable` | If the user can trigger via slash command. |
| `disable_model_invocation` | If the model is blocked from auto-invocation. |
| `argument_hint` | Help text for arguments. |
| `allowed_tools` | Tools the skill is permitted to use. |
| `context` | Routing context. |
| `agent` | Routing agent. |
| `model` | Specific model override. |
| `hooks` | Lifecycle hook configurations. |
| `skill_dir_path` | Path to the `SKILL.md` directory (if applicable). |
| `slash_command` | The command string (e.g., `/my-skill`). |
| `substitution_variables` | Supported variables for this skill. |

## Creating a Custom Skill

```python
from ecs_agent import Skill, ToolSchema
from ecs_agent.core import World
from ecs_agent.types import EntityId

class CalculatorSkill:
    name = "calculator"
    description = "Basic arithmetic operations."

    def tools(self):
        schema = ToolSchema(
            name="add",
            description="Add two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )

        async def add_handler(a: float, b: float) -> str:
            return str(a + b)

        return {"add": (schema, add_handler)}

    def system_prompt(self) -> str:
        return "Use the calculator skill for any math operations."

    def install(self, world: World, entity_id: EntityId) -> None:
        print(f"Installing calculator on {entity_id}")

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        print(f"Uninstalling calculator from {entity_id}")
```

## Built-in Skills
- **BuiltinToolsSkill**: Basic file manipulation (`read_file`, `write_file`, `edit_file`) and shell execution (`bash`). See [Built-in Tools](builtin-tools.md).

## Examples
- [`examples/skill_agent.py`](../../examples/skill_agent.py): Basic skill demo.
- [`examples/skill_discovery_agent.py`](../../examples/skill_discovery_agent.py): File-based auto-discovery.
- [`examples/markdown_skill_agent.py`](../../examples/markdown_skill_agent.py): Markdown skill loading and installation.

## See Also
- [Tool Discovery & Approval](./tool-discovery.md)
- [MCP Integration](./mcp.md)
- [Permissions](./permissions.md)
