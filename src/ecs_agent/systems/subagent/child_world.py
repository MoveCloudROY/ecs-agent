"""Child-world assembly for subagent delegation.

``ChildWorldBuilder`` (Task 4 of the subagent package refactor) owns the construction
of an isolated child ``World`` for a delegated subagent: LLM/prompt/conversation
components, inheritance-policy application (tools, skills, permissions, workspace,
compaction), and registration of the child system set.

The child system set is still hardcoded here (SystemPromptRenderSystem, ReasoningSystem,
ErrorHandlingSystem, and CompactionSystem when the parent has compaction config).
Task 12 replaces it with configurable runtime profiles; behavior is unchanged for now.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    PermissionComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import (
    SkillComponent,
    SkillMetadata,
    WorkspaceBindingComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.skills import catalog as _skill_catalog
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent._contextvars import _BACKGROUND_RESULT_ENVELOPE_ENABLED
from ecs_agent.systems.subagent.result_envelope import (
    _build_background_child_prompt_template,
    _build_child_prompt_template,
)
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import EntityId, InheritancePolicy, SubagentConfig, ToolSchema

logger = get_logger(__name__)

_SUBAGENT_COMPACTION_PRIORITY = -30


class _InheritedSkill:
    def __init__(
        self, metadata: SkillMetadata, tools: dict[str, tuple[ToolSchema, Any]]
    ) -> None:
        self.name = metadata.name
        self.description = metadata.description
        self._tools = tools
        self._system_prompt = (
            "inherited skill prompt" if metadata.has_system_prompt else ""
        )

    def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
        return self._tools

    def system_prompt(self) -> str:
        return self._system_prompt

    def install(self, world: World, entity_id: EntityId) -> None:
        del world, entity_id

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        del world, entity_id


class ChildWorldBuilder:
    """Assembles an isolated child world + runnable child entity for a subagent."""

    def assemble_child_world(
        self,
        parent_world: World,
        parent_entity: EntityId,
        config: SubagentConfig,
        parent_child_entity: EntityId | None = None,
    ) -> tuple[World, EntityId]:
        """Assemble isolated child world and runnable child entity."""
        policy = config.inheritance_policy
        parent_llm = parent_world.get_component(parent_entity, LLMComponent)
        parent_tools = parent_world.get_component(parent_entity, ToolRegistryComponent)
        parent_permissions = parent_world.get_component(
            parent_entity, PermissionComponent
        )
        parent_skills = parent_world.get_component(parent_entity, SkillComponent)

        effective_system_prompt = config.system_prompt
        if (
            policy.enabled
            and policy.inherit_system_prompt
            and not effective_system_prompt
            and parent_llm is not None
        ):
            effective_system_prompt = parent_llm.system_prompt

        child_world_name = f"{config.name}-{uuid.uuid4().hex[:8]}"
        child_world = World(name=child_world_name)
        child_world_entity_id = child_world.create_entity()
        background_result_envelope = _BACKGROUND_RESULT_ENVELOPE_ENABLED.get()
        child_world.add_component(
            child_world_entity_id,
            LLMComponent(
                model=config.model,
                system_prompt="",  # SystemPromptRenderSystem will populate this
            ),
        )
        child_world.add_component(
            child_world_entity_id,
            SystemPromptConfigSpec(
                template_source=PromptTemplateSource(
                    inline=(
                        _build_background_child_prompt_template(
                            effective_system_prompt or ""
                        )
                        if background_result_envelope
                        else _build_child_prompt_template(effective_system_prompt or "")
                    ),
                ),
            ),
        )
        child_world.add_component(
            child_world_entity_id,
            ConversationComponent(messages=[]),
        )
        child_world.add_component(
            child_world_entity_id,
            OwnerComponent(owner_id=parent_entity),
        )

        target_entities: list[tuple[World, EntityId]] = [
            (child_world, child_world_entity_id)
        ]
        if parent_child_entity is not None:
            target_entities.append((parent_world, parent_child_entity))

        if policy.enabled and policy.inherit_tools:
            for target_world, target_entity in target_entities:
                existing_registry = target_world.get_component(
                    target_entity, ToolRegistryComponent
                )
                if existing_registry is None:
                    target_world.add_component(
                        target_entity,
                        ToolRegistryComponent(tools={}, handlers={}),
                    )

        skill_manager = SkillManager()
        explicit_skills = list(dict.fromkeys(config.skills))
        inherited_tool_names = self._effective_inherited_tool_names(policy)
        inherited_skills = self._skills_for_inherited_tools(
            inherited_tool_names,
            parent_skills,
        )
        required_skills = list(dict.fromkeys(explicit_skills + inherited_skills))

        resolved_skills: list[ScriptSkill] = []
        for skill_name in required_skills:
            skill = self._resolve_parent_skill(
                parent_entity,
                skill_name,
                parent_skills,
                parent_tools,
                policy,
            )
            if skill is not None:
                resolved_skills.append(skill)

        for target_world, target_entity in target_entities:
            for skill in resolved_skills:
                skill_manager.install(target_world, target_entity, skill)

        if policy.enabled and parent_tools is not None:
            for tool_name in inherited_tool_names:
                self._inherit_tool_to_target_entities(
                    tool_name,
                    parent_tools,
                    target_entities,
                    policy,
                )

        if (
            policy.enabled
            and policy.inherit_permissions
            and parent_permissions is not None
        ):
            for target_world, target_entity in target_entities:
                target_world.add_component(
                    target_entity,
                    PermissionComponent(
                        allowed_tools=list(parent_permissions.allowed_tools),
                        denied_tools=list(parent_permissions.denied_tools),
                    ),
                )

        # Inherit parent workspace binding onto child entity when policy allows.
        if policy.enabled:
            parent_binding = parent_world.get_component(
                parent_entity, WorkspaceBindingComponent
            )
            if parent_binding is not None:
                child_world.add_component(
                    child_world_entity_id,
                    WorkspaceBindingComponent(
                        workspace_root=parent_binding.workspace_root
                    ),
                )
                # Also stamp the stub entity in the parent world (for test visibility).
                if parent_child_entity is not None:
                    parent_world.add_component(
                        parent_child_entity,
                        WorkspaceBindingComponent(
                            workspace_root=parent_binding.workspace_root
                        ),
                    )

        parent_compaction = parent_world.get_component(
            parent_entity, CompactionConfigComponent
        )
        if parent_compaction is not None:
            child_world.add_component(
                child_world_entity_id,
                replace(parent_compaction),
            )
            child_world.add_component(
                child_world_entity_id,
                ConversationArchiveComponent(),
            )

        if parent_compaction is not None:
            child_world.register_system(
                CompactionSystem(), priority=_SUBAGENT_COMPACTION_PRIORITY
            )
        child_world.register_system(
            SystemPromptRenderSystem(priority=-20), priority=-20
        )
        child_world.register_system(ReasoningSystem(priority=0), priority=0)
        child_world.register_system(
            ErrorHandlingSystem(priority=99),
            priority=99,
        )
        return child_world, child_world_entity_id

    def _effective_inherited_tool_names(self, policy: InheritancePolicy) -> list[str]:
        if not policy.enabled:
            return []

        return list(policy.inherit_tools)

    def _skills_for_inherited_tools(
        self,
        inherited_tool_names: list[str],
        parent_skills: SkillComponent | None,
    ) -> list[str]:
        if parent_skills is None or not inherited_tool_names:
            return []

        inherited_tool_set = set(inherited_tool_names)
        inherited_skill_names: list[str] = []
        for metadata in parent_skills.skills.values():
            if any(
                tool_name in inherited_tool_set for tool_name in metadata.tool_names
            ):
                inherited_skill_names.append(metadata.name)
        return inherited_skill_names

    def _resolve_parent_skill(
        self,
        parent_entity: EntityId,
        skill_name: str,
        parent_skills: SkillComponent | None,
        parent_tools: ToolRegistryComponent | None,
        policy: InheritancePolicy,
    ) -> ScriptSkill | None:
        if parent_skills is None or parent_tools is None:
            catalog_skill = self._resolve_from_catalog(skill_name)
            if catalog_skill is not None:
                return catalog_skill
            return self._handle_missing_skill(parent_entity, skill_name, policy)

        metadata = parent_skills.skills.get(skill_name)
        if metadata is None:
            catalog_skill = self._resolve_from_catalog(skill_name)
            if catalog_skill is not None:
                return catalog_skill
            return self._handle_missing_skill(parent_entity, skill_name, policy)

        tools: dict[str, tuple[ToolSchema, Any]] = {}
        for tool_name in metadata.tool_names:
            schema = parent_tools.tools.get(tool_name)
            handler = parent_tools.handlers.get(tool_name)
            if schema is None or handler is None:
                return self._handle_missing_skill(parent_entity, skill_name, policy)
            tools[tool_name] = (schema, handler)
        return _InheritedSkill(metadata, tools)

    def _handle_missing_skill(
        self,
        parent_entity: EntityId,
        skill_name: str,
        policy: InheritancePolicy,
    ) -> ScriptSkill | None:
        message = f"Missing skill '{skill_name}' on parent entity {parent_entity} during subagent delegation"
        if policy.missing_skill_policy == "error":
            raise ValueError(message)
        if policy.missing_skill_policy == "warn":
            logger.warning(
                "subagent_missing_skill",
                parent_entity=parent_entity,
                skill_name=skill_name,
                message=message,
            )
            return None
        raise ValueError(
            f"Invalid missing_skill_policy '{policy.missing_skill_policy}' for subagent inheritance"
        )

    def _resolve_from_catalog(self, skill_name: str) -> ScriptSkill | None:
        """Try to materialize a skill by name from the process-level catalog."""
        descriptor = _skill_catalog.lookup(skill_name)
        if descriptor is None:
            return None
        skill: ScriptSkill = descriptor.materialize()
        return skill

    def _inherit_tool_to_target_entities(
        self,
        tool_name: str,
        parent_tools: ToolRegistryComponent,
        target_entities: list[tuple[World, EntityId]],
        policy: InheritancePolicy,
    ) -> None:
        schema = parent_tools.tools.get(tool_name)
        handler = parent_tools.handlers.get(tool_name)
        if schema is None or handler is None:
            return

        for target_world, target_entity in target_entities:
            registry = target_world.get_component(target_entity, ToolRegistryComponent)
            if registry is None:
                registry = ToolRegistryComponent(tools={}, handlers={})
                target_world.add_component(target_entity, registry)

            has_conflict = tool_name in registry.tools or tool_name in registry.handlers
            if has_conflict:
                if policy.tool_conflict_policy == "skip":
                    continue
                if policy.tool_conflict_policy == "error":
                    raise ValueError(f"Tool inheritance conflict for '{tool_name}'")
                if policy.tool_conflict_policy != "override":
                    raise ValueError(
                        f"Invalid tool_conflict_policy '{policy.tool_conflict_policy}'"
                    )
            elif policy.tool_conflict_policy == "error":
                raise ValueError(f"Tool inheritance conflict for '{tool_name}'")

            registry.tools[tool_name] = schema
            registry.handlers[tool_name] = handler


__all__ = ["ChildWorldBuilder", "_InheritedSkill", "_SUBAGENT_COMPACTION_PRIORITY"]
