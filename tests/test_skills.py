from __future__ import annotations

from dataclasses import asdict

import pytest

from ecs_agent.components import SystemPromptComponent, ToolRegistryComponent
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.core import World
from ecs_agent.skills import Skill, SkillManager
from ecs_agent.types import EntityId, ToolSchema


async def _noop_handler(**_: object) -> str:
    return "ok"


async def _sum_handler(a: int, b: int) -> str:
    return str(a + b)


def _tool(name: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
    )


class DummySkill:
    def __init__(
        self,
        name: str,
        description: str,
        tool_bundle: dict[str, tuple[ToolSchema, object]],
        prompt: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._tool_bundle = tool_bundle
        self._prompt = prompt
        self.install_calls = 0
        self.uninstall_calls = 0

    def tools(self) -> dict[str, tuple[ToolSchema, object]]:
        return self._tool_bundle

    def system_prompt(self) -> str:
        return self._prompt

    def install(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        self.install_calls += 1

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id
        self.uninstall_calls += 1


def test_skill_protocol_duck_typing_compliance() -> None:
    skill = DummySkill("math", "math helpers", {"sum": (_tool("sum"), _sum_handler)})
    assert isinstance(skill, Skill)


def test_skill_component_dataclass_structure() -> None:
    metadata = SkillMetadata(
        name="math",
        description="math helpers",
        tool_names=["sum", "subtract"],
        has_system_prompt=True,
    )
    component = SkillComponent(skills={"math": metadata})

    assert component.skills["math"] == metadata
    assert asdict(metadata) == {
        "name": "math",
        "description": "math helpers",
        "tool_names": ["sum", "subtract"],
        "has_system_prompt": True,
    }


def test_skill_install_merges_tools() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"existing": _tool("existing")},
            handlers={"existing": _noop_handler},
        ),
    )
    world.add_component(entity_id, SystemPromptComponent(content="base prompt"))

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={
            "sum": (_tool("sum"), _sum_handler),
            "multiply": (_tool("multiply"), _noop_handler),
        },
        prompt="Use careful arithmetic.",
    )

    manager.install(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    prompts = world.get_component(entity_id, SystemPromptComponent)
    skills = world.get_component(entity_id, SkillComponent)

    assert registry is not None
    assert set(registry.tools) == {"existing", "sum", "multiply"}
    assert set(registry.handlers) == {"existing", "sum", "multiply"}
    assert callable(registry.handlers["existing"])
    assert callable(registry.handlers["sum"])
    assert prompts is not None
    assert "base prompt" in prompts.content
    assert "Use careful arithmetic." in prompts.content
    assert skills is not None
    assert "math" in skills.skills
    assert skills.skills["math"].tool_names == ["sum", "multiply"]
    assert skill.install_calls == 1


def test_skill_install_get_or_create_components() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
        prompt="Respond with equations.",
    )

    manager.install(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    prompts = world.get_component(entity_id, SystemPromptComponent)
    skills = world.get_component(entity_id, SkillComponent)
    assert registry is not None
    assert list(registry.tools) == ["sum"]
    assert prompts is not None
    assert prompts.content == "Respond with equations."
    assert skills is not None
    assert list(skills.skills) == ["math"]


def test_skill_install_collision_raises() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"read_file": _tool("read_file")},
            handlers={"read_file": _noop_handler},
        ),
    )

    skill = DummySkill(
        name="filesystem",
        description="filesystem tools",
        tool_bundle={"read_file": (_tool("read_file"), _noop_handler)},
    )

    with pytest.raises(ValueError, match="read_file"):
        manager.install(world, entity_id, skill)

    registry = world.get_component(entity_id, ToolRegistryComponent)
    skills = world.get_component(entity_id, SkillComponent)
    assert registry is not None
    assert set(registry.tools) == {"read_file"}
    assert skills is None


def test_skill_uninstall_removes_only_skill_tools() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"existing": _tool("existing")},
            handlers={"existing": _noop_handler},
        ),
    )

    skill = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
    )
    manager.install(world, entity_id, skill)

    manager.uninstall(world, entity_id, "math")

    registry = world.get_component(entity_id, ToolRegistryComponent)
    skills = world.get_component(entity_id, SkillComponent)
    assert registry is not None
    assert set(registry.tools) == {"existing"}
    assert set(registry.handlers) == {"existing"}
    assert skills is not None
    assert skills.skills == {}
    assert skill.uninstall_calls == 1


def test_skill_multiple_skills_can_be_installed_and_listed() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill_one = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
    )
    skill_two = DummySkill(
        name="text",
        description="text helpers",
        tool_bundle={"title": (_tool("title"), _noop_handler)},
    )

    manager.install(world, entity_id, skill_one)
    manager.install(world, entity_id, skill_two)

    listed = manager.list_skills(world, entity_id)
    assert {skill.name for skill in listed} == {"math", "text"}


def test_skill_uninstall_one_does_not_affect_another() -> None:
    world = World()
    manager = SkillManager()
    entity_id = world.create_entity()

    skill_one = DummySkill(
        name="math",
        description="math helpers",
        tool_bundle={"sum": (_tool("sum"), _sum_handler)},
    )
    skill_two = DummySkill(
        name="text",
        description="text helpers",
        tool_bundle={"title": (_tool("title"), _noop_handler)},
    )
    manager.install(world, entity_id, skill_one)
    manager.install(world, entity_id, skill_two)

    manager.uninstall(world, entity_id, "math")

    registry = world.get_component(entity_id, ToolRegistryComponent)
    text_meta = manager.get_skill_metadata(world, entity_id, "text")
    math_meta = manager.get_skill_metadata(world, entity_id, "math")
    assert registry is not None
    assert set(registry.tools) == {"title"}
    assert text_meta is not None
    assert text_meta.tool_names == ["title"]
    assert math_meta is None
