# Skills

The Skills system provides a modular way to group tools and system prompts into composable capabilities. It supports both high-level Markdown-based definitions and advanced Python-based script skills.

## Quick Start

The most common way to use skills is via Markdown definitions (`SKILL.md`).

```python
from ecs_agent.skills import discover_skills
from ecs_agent.skills.manager import SkillManager
from pathlib import Path

# discover_skills returns list[SkillDescriptor] — immutable catalog descriptors.
# SkillManager is a facade over the world-local SkillRuntime; pass the same world
# to multiple SkillManager instances and they will share the same runtime state.
descriptors = discover_skills([Path(".claude/skills")])
manager = SkillManager()

for descriptor in descriptors:
    # materialize() returns a Skill instance bound to the descriptor metadata.
    skill = descriptor.materialize()
    # install() indexes metadata and activates (registers tools + system prompt).
    manager.install(world, agent_entity, skill)
```

## Concepts

### What Is a Skill?
A Skill is a package of functionality that includes:
- **System Prompt**: Contextual instructions for the agent.
- **Tools**: Function schemas and their async handlers.
- **Metadata**: Name, description, and invocation controls.
- **Lifecycle hooks**: Logic to run during installation and uninstallation.

### Two-Phase Lifecycle (Index → Activate)
To remain token-efficient, the `SkillManager` supports a two-phase loading process:

1.  **Index**: Register skill metadata (name, description, tool names) into the `SkillComponent`. No tools are registered in the `ToolRegistryComponent` yet, and the system prompt is not loaded. `SkillMetadata.activated` is `False`.
2.  **Activate**: Call `skill.install(world, entity_id)` on the skill, which writes or appends the skill's system prompt to `SystemPromptComponent.content` and registers tools into the `ToolRegistryComponent`. `SkillMetadata.activated` is `True`.

`manager.install()` is a convenience method that performs both `index()` and `activate()` in one call.

### Progressive Disclosure (3 Tiers)
Skills use progressive disclosure to minimize the context window while keeping capabilities discoverable:

1.  **Tier 1: Metadata Summary**: Included in the main system prompt. Lists available skill names and descriptions via the `${_installed_skills}` placeholder.
2.  **Tier 2: Full Context on Demand**: The model calls `load_skill_details('<skill_name>')`. The tool returns the skill's full formatted context block — including the skill name, description, markdown body (system prompt), and all tool schemas — as a string. The string is returned to the LLM via the normal `role=tool` message; the framework does not automatically stage or inject it into any future outbound message.
3.  **Tier 3: Reference Docs**: Extensive documentation or guides can be requested by the agent (optional/custom implementation).

### Full Skill Context Block
The `load_skill_details` tool returns a formatted block containing the skill's complete definition. This block is delivered to the LLM inline in the `role=tool` response message, giving the model the full context it needs to use the skill's tools in the same or subsequent turns.

Format:
```
Skill: <name>
Description: <description>

## Skill Body
<markdown system prompt body>

## Tool Schemas
### Tool: <tool_name>
Description: <description>
parameters:
```json
<JSON parameters>
```
...
```

## Markdown Skills (Primary)

Markdown Skills are the primary way to define capabilities. They use a `SKILL.md` file format with YAML frontmatter and an optional `scripts/` directory for tools.

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
| `user-invocable` | boolean | `true` | Whether the user can invoke via slash command. |
| `disable-model-invocation`| boolean | `false` | Whether to exclude from model auto-invocation. |
| `argument-hint` | string | `""` | Hint text shown for arguments (e.g., `<url>`). |
| `allowed-tools` | list[str] | `[]` | List of tool names this skill is allowed to use. |
| `context` | string | `None` | Context routing identifier. |
| `agent` | string | `None` | Agent routing identifier. |
| `model` | string | `None` | Specific model to use for this skill. |
| `hooks` | dict | `{}` | Dictionary of lifecycle hook configurations. |

### String Substitutions
Markdown bodies and tool scripts support substitutions, processed in this order:

| Variable | Description | Example |
|----------|-------------|---------|
| `$ARGUMENTS[N]` | Nth word (0-indexed) from input arguments. | `$ARGUMENTS[0]` |
| `$ARGUMENTS` | The entire whitespace-separated arguments string. | `$ARGUMENTS` |
| `$N` | Shorthand for Nth argument (1-indexed). | `$1`, `$2` |
| `${CLAUDE_SESSION_ID}` | Resolves to the current session ID. | `ses_123` |
| `${CLAUDE_SKILL_DIR}` | Absolute path to the skill's directory. | `/path/to/skill` |

### Tool Scripts (scripts/ directory)
Python files in the `scripts/` directory are automatically discovered as tools. Each script receives arguments via JSON on stdin and returns results on stdout.

```python
# scripts/fetch_url.py
import sys
import json

async def main():
    args = json.load(sys.stdin)
    url = args["url"]
    # ... logic ...
    print(f"Content from {url}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Note: Markdown skills automatically generate a basic schema for these scripts. For precise control over tool schemas, use [Script Skills](#script-skills-advanced).

### Path Safety
The `Skill` class (markdown-based) provides `resolve_supporting_path` to safely resolve paths within the skill directory, blocking any directory traversal attempts.

```python
# Safe
path = skill.resolve_supporting_path("data/config.json")
# Raises ValueError
path = skill.resolve_supporting_path("../../../etc/passwd")
```

### Trigger Templates
Skills can define trigger templates (keywords or events) that are automatically registered if the agent has a `UserPromptConfigComponent`. These triggers (e.g., `@help` or `event:tool_success`) trigger the injection of specific prompt blocks when detected in user messages or runtime state.
### Slash Command Skill Context Injection

When a user message contains a slash command (e.g., `/skill-name`) and the corresponding skill has `user_invocable=True`, the full skill context is injected transiently into the outbound user message. This injection is performed at call-time by `prepare_outbound_messages()` and does not mutate the stored conversation history or the `PromptContextQueueComponent`.

Key behaviors:
- **Longest Match Wins**: If multiple slash commands overlap in the user text, the longest matching command is selected.
- **Transient Injection**: The context is resolved and injected only for the current LLM provider call. It is not persisted to the entity's components.
- **Priority Placement**: Slash context is injected as a high-priority `ContextEntry` (priority 1,000,000), appearing before any reserved `ContextPool` entries and before the original user text.
- **Preservation**: The original `/skill-name ...` text remains in the user message; it is not replaced by the context block.
- **Distinct from Tool Calls**: This is a prompt-level injection, separate from the `load_skill_details` tool which returns context via a `role=tool` message.
- **Normal Path Only**: Slash injection is skipped when using the `conversation_override` path in `prepare_outbound_messages()`.
## Script Skills (Advanced)

Script Skills are Python classes that implement the `ScriptSkill` protocol. Use these when you need complex tool handlers, precise schema definitions, or custom installation logic.

### ScriptSkill Protocol

```python
from typing import Protocol, runtime_checkable
from ecs_agent.core import World
from ecs_agent.types import EntityId, ToolSchema, ToolHandler

@runtime_checkable
class ScriptSkill(Protocol):
    name: str
    description: str

    # Optional class attribute — default False.
    # Set to True to mark this skill as a pure tool bundle:
    #   - Tools are registered normally on ToolRegistryComponent.
    #   - The skill is NOT listed in SkillComponent / ${_installed_skills}.
    #   - load_skill_details cannot be called for it.
    #   - No system prompt fragment is injected.
    # Use this for infrastructure-level tool sets (e.g. BuiltinToolsSkill)
    # that the LLM should not see or reason about as a "skill".
    is_tool_bundle: bool  # default False

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

### Custom Skill Example

```python
from ecs_agent.skills import ScriptSkill
from ecs_agent.types import ToolSchema

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
        pass

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        pass
```

## SkillManager API

`SkillManager` is a **thin facade** over the world-local `SkillRuntime`. Installed-skill state
lives on the `World` instance, not inside any individual `SkillManager` object.
Creating multiple `SkillManager()` instances that operate on the same `World` is safe — they
all share the same underlying runtime state.

```python
# Two facades — one world. Both see the same installed skills.
m1 = SkillManager()
m2 = SkillManager()
m1.install(world, entity, skill)
assert m2.get_skill_metadata(world, entity, skill.name) is not None  # same runtime
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `index` | `index(world, entity_id, skill)` | Register metadata only. No tools loaded. `activated=False`. |
| `activate` | `activate(world, entity_id, skill_name)` | Call skill's install path, appending system prompt to `SystemPromptComponent.content` and registering tools. Idempotent. |
| `install` | `install(world, entity_id, skill)` | Convenience: `index()` + `activate()` in one call. |
| `uninstall` | `uninstall(world, entity_id, skill_name)` | Remove metadata, tools, and system prompt fragment. |
| `list_skills` | `list_skills(world, entity_id)` | Return all `SkillMetadata` for installed skills. |
| `get_skill_metadata` | `get_skill_metadata(world, entity_id, skill_name)` | Return `SkillMetadata` for a specific skill, or `None`. |
| `can_invoke_via_slash` | `can_invoke_via_slash(world, entity, slash_cmd)` | Check if a slash command like `/skill-name` is user-invocable. |
| `can_model_auto_invoke_skill` | `can_model_auto_invoke_skill(world, entity, skill_name)` | Check if model can auto-invoke this skill. |
| `format_skill_details` | `format_skill_details(world, entity_id, skill_name)` | Return formatted Tier 2 details string for a skill. |

## Skill Discovery

### Discover Markdown Skills
Recursively scans directories for `SKILL.md` files and returns a list of `SkillDescriptor` objects.
Each `SkillDescriptor` is a frozen, immutable catalog record. Call `.materialize()` to obtain a
runnable `Skill` instance when you need to install it at runtime.

```python
from ecs_agent.skills import discover_skills
from pathlib import Path

descriptors = discover_skills([Path(".claude/skills")])
# descriptors is list[SkillDescriptor] — globally shareable and immutable.
# Materialize to a Skill before installing:
skill = descriptors[0].materialize()
```

### Discover Python Skills
Scans directories for `.py` files and instantiates classes implementing the `ScriptSkill` protocol.

```python
from ecs_agent.skills.discovery import SkillDiscovery

discovery = SkillDiscovery(skill_paths=["./my_skills"])
skills = discovery.discover() # Returns list[ScriptSkill]
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

## Built-in Skills
- **BuiltinToolsSkill**: Basic file manipulation (`read_file`, `write_file`, `edit_file`, `glob`) and shell execution (`bash`). See [Built-in Tools](builtin-tools.md).

## Workspace Binding

Skills that access the filesystem (e.g., `BuiltinToolsSkill`) need a **workspace root**.
The workspace belongs to the **agent entity**, not the skill object. Bind it by attaching
`WorkspaceBindingComponent` to the entity before installing the skill:

```python
from ecs_agent.components.definitions import WorkspaceBindingComponent
from ecs_agent.skills.manager import SkillManager
from ecs_agent.tools.builtins import BuiltinToolsSkill
from pathlib import Path

world.add_component(agent, WorkspaceBindingComponent(workspace_root=Path("/workspace")))
SkillManager().install(world, agent, BuiltinToolsSkill())
# The runtime materializes a workspace-bound copy of the skill tools for this entity.
# The shared BuiltinToolsSkill instance is never mutated.
```

When a subagent is spawned, the child entity **inherits** the parent workspace binding by
default (controlled by `InheritancePolicy`). The child can therefore use the same
workspace-scoped file tools without any additional configuration.

## See Also
- [Tool Discovery & Approval](./tool-discovery.md)
- [MCP Integration](./mcp.md)
- [Permissions](./permissions.md)
