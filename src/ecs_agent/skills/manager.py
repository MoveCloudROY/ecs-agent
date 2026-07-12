"""Skill manager for lifecycle handling and tool registry integration."""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Coroutine

from ecs_agent.components import (
    SandboxConfigComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import (
    SkillComponent,
    SkillMetadata,
)
from ecs_agent.core.world import World
from ecs_agent.tools.bwrap_sandbox import wrap_sandbox_handler
from ecs_agent.types import (
    EntityId,
    SkillInstalledEvent,
    SkillUninstalledEvent,
    ToolSchema,
)

from ecs_agent.skills.script_skill import ScriptSkill


def render_full_skill_context(
    *,
    skill_name: str,
    description: str,
    body: str,
    tool_schemas: list[ToolSchema],
) -> str:
    lines = [
        f"Skill: {skill_name}",
        f"Description: {description}",
        "",
        "## Skill Body",
    ]

    if body:
        lines.append(body)
    else:
        lines.append("(none)")

    lines.extend(["", "## Tool Schemas"])

    if not tool_schemas:
        lines.append("- none")
        return "\n".join(lines)

    for schema in sorted(tool_schemas, key=lambda item: item.name):
        lines.extend(
            [
                f"### Tool: {schema.name}",
                f"Description: {schema.description}",
                "parameters:",
                "```json",
                json.dumps(schema.parameters, indent=2, sort_keys=True),
                "```",
            ]
        )

    return "\n".join(lines)


class SkillManager:
    _DETAILS_TOOL_NAME = "load_skill_details"

    def index(self, world: World, entity_id: EntityId, skill: ScriptSkill) -> None:
        materialized_skill = world.skill_runtime.materialize_skill_for_entity(
            world,
            entity_id,
            skill,
        )

        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            registry = ToolRegistryComponent(tools={}, handlers={})
            world.add_component(entity_id, registry)

        self._ensure_skill_details_tool(world, entity_id, registry)

        # Tool bundles are pure tool collections — they are not listed as skills.
        # Skip SkillComponent registration entirely.
        if getattr(materialized_skill, "is_tool_bundle", False):
            world.skill_runtime.set_installed_skill(
                entity_id,
                materialized_skill.name,
                materialized_skill,
            )
            return

        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            skill_component = SkillComponent(skills={})
            world.add_component(entity_id, skill_component)

        skill_component.skills[materialized_skill.name] = SkillMetadata(
            name=materialized_skill.name,
            description=materialized_skill.description,
            tool_names=[],
            has_system_prompt=False,
            activated=False,
            user_invocable=getattr(materialized_skill, "user_invocable", True),
            disable_model_invocation=getattr(
                materialized_skill,
                "disable_model_invocation",
                False,
            ),
            argument_hint=getattr(materialized_skill, "argument_hint", ""),
            allowed_tools=getattr(materialized_skill, "allowed_tools", []),
            context=getattr(materialized_skill, "context", None),
            agent=getattr(materialized_skill, "agent", None),
            model=getattr(materialized_skill, "model", None),
            hooks=getattr(materialized_skill, "hooks", {}),
            skill_dir_path=getattr(materialized_skill, "skill_dir_path", None),
            slash_command=getattr(
                materialized_skill,
                "slash_command",
                f"/{materialized_skill.name}",
            ),
            substitution_variables=getattr(
                materialized_skill,
                "substitution_variables",
                [
                    "$ARGUMENTS",
                    "$ARGUMENTS[0]",
                    "$1",
                    "${CLAUDE_SESSION_ID}",
                    "${CLAUDE_SKILL_DIR}",
                ],
            ),
        )
        world.skill_runtime.set_installed_skill(
            entity_id,
            materialized_skill.name,
            materialized_skill,
        )

    def activate(self, world: World, entity_id: EntityId, skill_name: str) -> None:
        skill_component = world.get_component(entity_id, SkillComponent)
        metadata = (
            None if skill_component is None else skill_component.skills.get(skill_name)
        )
        skill = world.skill_runtime.get_installed_skill(entity_id, skill_name)
        if skill is None:
            raise ValueError(
                f"Skill '{skill_name}' is not indexed for entity {entity_id}."
            )

        # Tool bundles have no SkillMetadata — check activated state via a sentinel.
        is_tool_bundle = getattr(skill, "is_tool_bundle", False)
        if not is_tool_bundle and metadata is None:
            raise ValueError(
                f"Skill '{skill_name}' is not indexed for entity {entity_id}."
            )

        if metadata is not None and metadata.activated:
            return

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
                f"Tool name collision for skill '{skill_name}': {collision_list}"
            )

        sandbox_config = world.get_component(entity_id, SandboxConfigComponent)

        for tool_name, tool_data in skill_tools.items():
            schema, handler = tool_data
            if sandbox_config is not None:
                handler = wrap_sandbox_handler(handler, schema, sandbox_config)
            registry.tools[tool_name] = schema
            registry.handlers[tool_name] = handler

        prompt = skill.system_prompt()

        skill.install(world, entity_id)

        # Update SkillMetadata only for real skills (not tool bundles).
        if not is_tool_bundle:
            refreshed_skill_component = world.get_component(entity_id, SkillComponent)
            refreshed_metadata = (
                None
                if refreshed_skill_component is None
                else refreshed_skill_component.skills.get(skill_name)
            )
            target_metadata = (
                refreshed_metadata if refreshed_metadata is not None else metadata
            )
            assert target_metadata is not None
            target_metadata.has_system_prompt = bool(prompt)
            target_metadata.activated = True
            target_metadata.tool_names = list(skill_tools.keys())

        # Publish SkillInstalledEvent
        self._publish_event(
            world,
            SkillInstalledEvent(
                entity_id=entity_id,
                skill_name=skill_name,
                tool_names=list(skill_tools.keys()),
            ),
        )

    def install(self, world: World, entity_id: EntityId, skill: ScriptSkill) -> None:
        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            registry = ToolRegistryComponent(tools={}, handlers={})
            world.add_component(entity_id, registry)

        self._ensure_skill_details_tool(world, entity_id, registry)

        collisions = sorted(set(skill.tools()).intersection(registry.tools))
        if collisions:
            collision_list = ", ".join(collisions)
            raise ValueError(
                f"Tool name collision for skill '{skill.name}': {collision_list}"
            )

        self.index(world, entity_id, skill)
        self.activate(world, entity_id, skill.name)

    def uninstall(self, world: World, entity_id: EntityId, skill_name: str) -> None:
        skill = world.skill_runtime.get_installed_skill(entity_id, skill_name)
        is_tool_bundle = skill is not None and getattr(skill, "is_tool_bundle", False)

        if is_tool_bundle:
            # Tool bundles are not in SkillComponent; remove their tools directly.
            registry = world.get_component(entity_id, ToolRegistryComponent)
            if registry is not None and skill is not None:
                for tool_name in skill.tools():
                    registry.tools.pop(tool_name, None)
                    registry.handlers.pop(tool_name, None)
            world.skill_runtime.pop_installed_skill(entity_id, skill_name)
            if skill is not None:
                skill.uninstall(world, entity_id)
            self._cleanup_skill_details_tool(world, entity_id)
            self._publish_event(
                world,
                SkillUninstalledEvent(entity_id=entity_id, skill_name=skill_name),
            )
            return

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

        world.skill_runtime.pop_installed_skill(entity_id, skill_name)
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
        return world.skill_runtime.list_skills(world, entity_id)

    def get_skill_metadata(
        self, world: World, entity_id: EntityId, skill_name: str
    ) -> SkillMetadata | None:
        return world.skill_runtime.get_skill_metadata(world, entity_id, skill_name)

    def can_invoke_via_slash(
        self, world: World, entity: EntityId, slash_cmd: str
    ) -> bool:
        skill_component = world.get_component(entity, SkillComponent)
        if skill_component is None:
            return False

        skill_name = slash_cmd[1:] if slash_cmd.startswith("/") else slash_cmd
        for metadata in skill_component.skills.values():
            if metadata.slash_command == slash_cmd or metadata.name == skill_name:
                return metadata.user_invocable

        return False

    def can_model_auto_invoke_skill(
        self, world: World, entity: EntityId, skill_name: str
    ) -> bool:
        skill_component = world.get_component(entity, SkillComponent)
        if skill_component is None:
            return False

        metadata = skill_component.skills.get(skill_name)
        if metadata is None:
            return False

        return not metadata.disable_model_invocation

    def format_skill_details(
        self, world: World, entity_id: EntityId, skill_name: str
    ) -> str | None:
        skill = world.skill_runtime.get_installed_skill(entity_id, skill_name)

        if skill is not None and getattr(skill, "is_tool_bundle", False):
            # Tool bundles are not in SkillComponent; build details directly.
            body = skill.system_prompt()
            tool_schemas = [schema for schema, _ in skill.tools().values()]
            return render_full_skill_context(
                skill_name=skill.name,
                description=skill.description,
                body=body,
                tool_schemas=tool_schemas,
            )

        # Regular skill path: read SkillMetadata from SkillComponent.
        # Works for skills installed via SkillManager.install() as well as
        # third-party adapters (e.g. MCPSkillAdapter) that write directly
        # to SkillComponent without going through _installed_skills.
        metadata = self.get_skill_metadata(world, entity_id, skill_name)
        if metadata is None:
            return None

        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            return None

        body = skill.system_prompt() if skill is not None else ""

        schemas: list[ToolSchema] = []
        for tool_name in sorted(metadata.tool_names):
            schema = registry.tools.get(tool_name)
            if schema is not None:
                schemas.append(schema)

        return render_full_skill_context(
            skill_name=metadata.name,
            description=metadata.description,
            body=body,
            tool_schemas=schemas,
        )

    def _ensure_skill_details_tool(
        self, world: World, entity_id: EntityId, registry: ToolRegistryComponent
    ) -> None:
        if self._DETAILS_TOOL_NAME in registry.tools:
            return

        async def load_skill_details(skill_name: str) -> str:
            details = self.format_skill_details(world, entity_id, skill_name)
            if details is None:
                # Corrective payload instead of a dead end: name the skills that
                # *can* be loaded so the model recovers (picks a real one, or
                # stops guessing and proceeds) rather than re-issuing the same
                # failing call. A skill referenced only in a prompt or delegated
                # to a subagent is not loadable here.
                available = world.skill_runtime.loadable_skill_names(
                    world, entity_id
                )
                if available:
                    return (
                        f"Skill '{skill_name}' is not installed. Skills you can "
                        f"load: {', '.join(available)}. Call load_skill_details "
                        "with one of those names, or continue without loading a "
                        "skill."
                    )
                return (
                    f"Skill '{skill_name}' is not installed. No skills are "
                    "available to load here; continue with your current tools "
                    "and do not call load_skill_details again."
                )
            return details

        registry.tools[self._DETAILS_TOOL_NAME] = ToolSchema(
            name=self._DETAILS_TOOL_NAME,
            description=(
                "Load the full instructions and tool schemas for an installed skill.\n"
                "\n"
                "Call this before using a skill for the first time. It returns the skill's\n"
                "system prompt and the complete parameter schemas for every tool the skill\n"
                "provides, so you know exactly what each tool does and how to call it.\n"
                "\n"
                "Usage:\n"
                '  load_skill_details(skill_name="<name>")\n'
                "\n"
                "Where <name> is one of the skill names listed under Available Skills.\n"
                'Example: load_skill_details(skill_name="ui-navigator")'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load, exactly as listed under Available Skills.",
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
