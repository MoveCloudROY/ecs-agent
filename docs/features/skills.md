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

See [`examples/skill_agent.py`](../../examples/skill_agent.py) for a complete demonstration of the skill lifecycle.
