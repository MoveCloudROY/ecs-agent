# Skills System

The Skills system provides a modular way to group tools and system prompts into composable capabilities. Skills use a **3-tier progressive disclosure** approach to remain token-efficient while allowing agents to discover detailed tool schemas only when needed.

## Overview

A Skill is a package of functionality that includes:
- **Tools**: Async handlers and their schemas.
- **System Prompt**: Contextual instructions for using the skill.
- **Lifecycle hooks**: Logic to run during installation and uninstallation.

By using Skills, you can avoid overwhelming the LLM with dozens of tool schemas in every request.

## Progressive Disclosure (3 Tiers)

To optimize token usage, Skills are presented to the LLM in three levels of detail:

1.  **Tier 1: Metadata Summary**: Included in the main system prompt. It lists available skill names, descriptions, and tool names.
2.  **Tier 2: Detailed Schemas**: The `load_skill_details` meta-tool allows the LLM to fetch full JSON schemas for a specific skill's tools on demand.
3.  **Tier 3: Reference Docs**: Extensive documentation or guides can be requested by the agent (optional/custom implementation).

## Skill Protocol

Any class implementing the `Skill` protocol can be installed into a `World`.

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

## SkillManager

The `SkillManager` handles the installation lifecycle, tool registration, and prompt management.

```python
from ecs_agent import SkillManager

manager = SkillManager()

# Install a skill
manager.install(world, agent_entity, my_skill)

# List installed skills
skills = manager.list_skills(world, agent_entity)

# Uninstall a skill
manager.uninstall(world, agent_entity, "my-skill")
```
## File-based Skill Discovery

The `SkillDiscovery` class allows you to automatically load and instantiate `Skill` implementations from a directory. This is useful for plugins or modular capability loading.

### SkillDiscovery

`SkillDiscovery` scans a list of filesystem paths for `.py` files, dynamically imports them, and instantiates any classes that implement the `Skill` protocol.

```python
from ecs_agent.skills.discovery import SkillDiscovery

# Initialize discovery with a list of paths
discovery = SkillDiscovery(skill_paths=["./my_skills", "/opt/ecs/plugins"])

# Discover skills (returns list[Skill])
skills = discovery.discover()

# Or discover and install directly
installed_names = discovery.discover_and_install(world, agent_entity, manager)
```

- **Dynamic Loading**: Uses `importlib.util` for dynamic module loading.
- **Protocol Checking**: Only instantiates classes that match the `Skill` protocol (name, description, tools, system_prompt).
- **Graceful Handling**: Skips `__init__.py` and files that don't contain valid skills.

## Discovery Manager

The `DiscoveryManager` provides a high-level API for discovering and installing skills from both the filesystem and external MCP servers.

### Using DiscoveryManager

It combines the capabilities of `SkillDiscovery` and `MCPSkillAdapter` into a single operation.

```python
from ecs_agent.skills.discovery import DiscoveryManager

mcp_configs = [
    {
        "server_name": "sqlite",
        "command": "uvx mcp-server-sqlite --db-path ./test.db"
    }
]

discovery_mgr = DiscoveryManager(
    skill_paths=["./examples/skills"],
    mcp_configs=mcp_configs
)

# Discover and install everything
report = await discovery_mgr.auto_discover_and_install(world, agent_entity, manager)
```

### DiscoveryReport

The `auto_discover_and_install` method returns a `DiscoveryReport` containing the results of the discovery process:

- `installed_skills`: List of successfully installed skill names.
- `failed_sources`: List of `(source, error)` tuples for failed imports or connections.
- `skipped_mcp`: List of MCP servers that were skipped (e.g., due to missing dependencies).

The manager publishes a `SkillDiscoveryEvent` to the `EventBus` for each source scanned, allowing for real-time tracking of the discovery process.


### Installation Side Effects
-   Tools are registered in the `ToolRegistryComponent`.
-   The skill's `system_prompt()` is appended to the `SystemPromptComponent`.
-   `SkillMetadata` is stored in the `SkillComponent`.
-   The `load_skill_details` tool is automatically added to enable Tier 2 discovery.

## Creating a Custom Skill

Here is a simple example of a skill that provides a calculator tool.

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

The framework includes several built-in skills:
-   **BuiltinToolsSkill**: Basic file manipulation (`read_file`, `write_file`, `edit_file`) and shell execution (`bash`). See [Built-in Tools](builtin-tools.md).

## Examples

See [`examples/skill_agent.py`](../../examples/skill_agent.py) for a basic skill demo, and [`examples/skill_discovery_agent.py`](../../examples/skill_discovery_agent.py) for file-based auto-discovery.
