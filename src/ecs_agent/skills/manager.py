from ecs_agent.components import SystemPromptComponent, ToolRegistryComponent
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.core.world import World
from ecs_agent.types import EntityId

from ecs_agent.skills.protocol import Skill


class SkillManager:
    def __init__(self) -> None:
        self._installed_skills: dict[tuple[EntityId, str], Skill] = {}

    def install(self, world: World, entity_id: EntityId, skill: Skill) -> None:
        registry = world.get_component(entity_id, ToolRegistryComponent)
        if registry is None:
            registry = ToolRegistryComponent(tools={}, handlers={})
            world.add_component(entity_id, registry)

        skill_tools = skill.tools()
        collisions = sorted(set(skill_tools).intersection(registry.tools))
        if collisions:
            collision_list = ", ".join(collisions)
            raise ValueError(
                f"Tool name collision for skill '{skill.name}': {collision_list}"
            )

        for tool_name, tool_data in skill_tools.items():
            schema, handler = tool_data
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
