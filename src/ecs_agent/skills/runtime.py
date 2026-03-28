from __future__ import annotations

from typing import TYPE_CHECKING

from ecs_agent.components.definitions import SkillComponent, SkillMetadata
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
