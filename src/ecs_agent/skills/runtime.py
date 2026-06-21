"""Skill runtime context."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ecs_agent.components.definitions import (
    SkillComponent,
    SkillMetadata,
    WorkspaceBindingComponent,
)
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.types import EntityId

if TYPE_CHECKING:
    from ecs_agent.core.world import World


class SkillRuntime:
    def __init__(self) -> None:
        self._installed_skills: dict[tuple[EntityId, str], ScriptSkill] = {}

    def set_installed_skill(
        self,
        entity_id: EntityId,
        skill_name: str,
        skill: ScriptSkill,
    ) -> None:
        self._installed_skills[(entity_id, skill_name)] = skill

    def get_installed_skill(
        self,
        entity_id: EntityId,
        skill_name: str,
    ) -> ScriptSkill | None:
        return self._installed_skills.get((entity_id, skill_name))

    def pop_installed_skill(
        self,
        entity_id: EntityId,
        skill_name: str,
    ) -> ScriptSkill | None:
        return self._installed_skills.pop((entity_id, skill_name), None)

    def materialize_skill_for_entity(
        self,
        world: World,
        entity_id: EntityId,
        skill: ScriptSkill,
    ) -> ScriptSkill:
        binding = world.get_component(entity_id, WorkspaceBindingComponent)
        if binding is None:
            return skill

        materialized = copy.deepcopy(skill)
        workspace_root = binding.workspace_root

        resolver = getattr(materialized, "resolve_path_references", None)
        if callable(resolver):
            resolved_skill = resolver(workspace_root)
            if resolved_skill is not None:
                materialized = resolved_skill

        binder = getattr(materialized, "bind_workspace", None)
        if callable(binder):
            bound_skill = binder(str(workspace_root))
            if bound_skill is not None:
                materialized = bound_skill

        return materialized

    def inherit_workspace_binding(
        self,
        world: World,
        *,
        parent_entity: EntityId,
        child_entity: EntityId,
    ) -> None:
        parent_binding = world.get_component(parent_entity, WorkspaceBindingComponent)
        if parent_binding is None:
            return

        world.add_component(
            child_entity,
            WorkspaceBindingComponent(workspace_root=parent_binding.workspace_root),
        )

    def list_skills(self, world: World, entity_id: EntityId) -> list[SkillMetadata]:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            return []
        return list(skill_component.skills.values())

    def get_skill_metadata(
        self,
        world: World,
        entity_id: EntityId,
        skill_name: str,
    ) -> SkillMetadata | None:
        skill_component = world.get_component(entity_id, SkillComponent)
        if skill_component is None:
            return None
        return skill_component.skills.get(skill_name)
