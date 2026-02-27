"""Skill manager for lifecycle handling and tool registry integration."""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Coroutine

from ecs_agent.components import (
    SandboxConfigComponent,
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.core.world import World
from ecs_agent.tools.bwrap_sandbox import wrap_sandbox_handler
from ecs_agent.types import (
    EntityId,
    SkillInstalledEvent,
    SkillUninstalledEvent,
    ToolSchema,
)

from ecs_agent.skills.protocol import Skill


class SkillManager:
    _DETAILS_TOOL_NAME = "load_skill_details"

    def __init__(self) -> None:
        self._installed_skills: dict[tuple[EntityId, str], Skill] = {}

    def install(self, world: World, entity_id: EntityId, skill: Skill) -> None:
        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            registry = ToolRegistryComponent(tools={}, handlers={})
            world.add_component(entity_id, registry)

        self._ensure_skill_details_tool(world, entity_id, registry)

        skill_tools = skill.tools()
        collisions = sorted(set(skill_tools).intersection(registry.tools))
        if collisions:
            collision_list = ", ".join(collisions)
            raise ValueError(
                f"Tool name collision for skill '{skill.name}': {collision_list}"
            )

        sandbox_config = world.get_component(entity_id, SandboxConfigComponent)

        for tool_name, tool_data in skill_tools.items():
            schema, handler = tool_data
            if sandbox_config is not None:
                handler = wrap_sandbox_handler(handler, schema, sandbox_config)
            registry.tools[tool_name] = schema
            registry.handlers[tool_name] = handler

        prompt = skill.system_prompt()
        if prompt:
            prompt_component = world.get_component(entity_id, SystemPromptComponent)
            if prompt_component is None:
                world.add_component(entity_id, SystemPromptComponent(content=prompt))
            elif prompt_component.content:
                prompt_component.content = f"{prompt_component.content}\n\n{prompt}"
            else:
                prompt_component.content = prompt

        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            skill_component = SkillComponent(skills={})
            world.add_component(entity_id, skill_component)

        skill_component.skills[skill.name] = SkillMetadata(
            name=skill.name,
            description=skill.description,
            tool_names=list(skill_tools.keys()),
            has_system_prompt=bool(prompt),
        )
        self._installed_skills[(entity_id, skill.name)] = skill
        skill.install(world, entity_id)

        # Publish SkillInstalledEvent
        self._publish_event(
            world,
            SkillInstalledEvent(
                entity_id=entity_id,
                skill_name=skill.name,
                tool_names=list(skill_tools.keys()),
            ),
        )

    def uninstall(self, world: World, entity_id: EntityId, skill_name: str) -> None:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            return

        metadata = skill_component.skills.pop(skill_name, None)
        if metadata is None:
            return

        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is not None:
            for tool_name in metadata.tool_names:
                registry.tools.pop(tool_name, None)
                registry.handlers.pop(tool_name, None)

        skill = self._installed_skills.pop((entity_id, skill_name), None)
        if skill is not None:
            skill.uninstall(world, entity_id)

        self._cleanup_skill_details_tool(world, entity_id)

        # Publish SkillUninstalledEvent
        self._publish_event(
            world,
            SkillUninstalledEvent(
                entity_id=entity_id,
                skill_name=skill_name,
            ),
        )

    def list_skills(self, world: World, entity_id: EntityId) -> list[SkillMetadata]:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            return []
        return list(skill_component.skills.values())

    def get_skill_metadata(
        self, world: World, entity_id: EntityId, skill_name: str
    ) -> SkillMetadata | None:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            return None
        return skill_component.skills.get(skill_name)

    def format_skill_details(
        self, world: World, entity_id: EntityId, skill_name: str
    ) -> str | None:
        metadata = self.get_skill_metadata(world, entity_id, skill_name)
        if metadata is None:
            return None

        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            return None

        lines = [
            f"Skill: {metadata.name}",
            f"Description: {metadata.description}",
            "Tools:",
        ]

        for tool_name in metadata.tool_names:
            schema = registry.tools.get(tool_name)
            if schema is None:
                continue

            lines.extend(
                [
                    f"- {schema.name}",
                    f"  description: {schema.description}",
                    "  parameters:",
                    json.dumps(schema.parameters, indent=2, sort_keys=True),
                ]
            )

        return "\n".join(lines)

    def _ensure_skill_details_tool(
        self, world: World, entity_id: EntityId, registry: ToolRegistryComponent
    ) -> None:
        if self._DETAILS_TOOL_NAME in registry.tools:
            return

        async def load_skill_details(skill_name: str) -> str:
            details = self.format_skill_details(world, entity_id, skill_name)
            if details is None:
                return f"Skill '{skill_name}' is not installed."
            return details

        registry.tools[self._DETAILS_TOOL_NAME] = ToolSchema(
            name=self._DETAILS_TOOL_NAME,
            description=("Load detailed Tier 2 tool schemas for an installed skill."),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Installed skill name to inspect.",
                    }
                },
                "required": ["skill_name"],
            },
        )
        registry.handlers[self._DETAILS_TOOL_NAME] = load_skill_details

    def _cleanup_skill_details_tool(self, world: World, entity_id: EntityId) -> None:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is not None and skill_component.skills:
            return

        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            return

        registry.tools.pop(self._DETAILS_TOOL_NAME, None)
        registry.handlers.pop(self._DETAILS_TOOL_NAME, None)

    def _publish_event(self, world: World, event: object) -> None:
        """Publish an event synchronously using the sync→async bridge pattern."""
        self._run_sync(world.event_bus.publish(event))

    def _run_sync(self, operation: Coroutine[object, object, object]) -> object:
        """Run an async operation synchronously, handling both sync and async contexts."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Not in an event loop, can use asyncio.run()
            return asyncio.run(operation)

        # In an event loop, need to run in a thread
        def _run_in_thread() -> object:
            return asyncio.run(operation)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            return future.result()
