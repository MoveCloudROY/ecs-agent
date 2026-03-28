from __future__ import annotations

from ecs_agent.core import World
from ecs_agent.skills import SkillManager
from ecs_agent.types import EntityId, ToolSchema


async def _noop_handler(**_: object) -> str:
    return "ok"


class DummySkill:
    def __init__(self, *, name: str, description: str, prompt: str) -> None:
        self.name = name
        self.description = description
        self._prompt = prompt

    def tools(self) -> dict[str, tuple[ToolSchema, object]]:
        return {
            "dummy_tool": (
                ToolSchema(
                    name="dummy_tool",
                    description="dummy tool",
                    parameters={"type": "object", "properties": {}},
                ),
                _noop_handler,
            )
        }

    def system_prompt(self) -> str:
        return self._prompt

    def install(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        _ = world
        _ = entity_id


class DummyToolBundleSkill(DummySkill):
    is_tool_bundle = True


def test_skill_runtime_world_local_runtime_shared_across_skill_manager_facades() -> (
    None
):
    world = World()
    entity_id = world.create_entity()

    manager_a = SkillManager()
    manager_b = SkillManager()

    manager_a.install(
        world,
        entity_id,
        DummySkill(
            name="shared-skill",
            description="shared skill",
            prompt="Prompt body owned by runtime state.",
        ),
    )

    details_from_other_facade = manager_b.format_skill_details(
        world,
        entity_id,
        "shared-skill",
    )

    assert details_from_other_facade is not None
    assert "Prompt body owned by runtime state." in details_from_other_facade


def test_skill_runtime_not_process_global_between_worlds() -> None:
    manager = SkillManager()

    world_a = World()
    entity_a = world_a.create_entity()
    manager.install(
        world_a,
        entity_a,
        DummyToolBundleSkill(name="bundle", description="A", prompt="A"),
    )

    world_b = World()
    entity_b = world_b.create_entity()

    leaked = manager.format_skill_details(world_b, entity_b, "bundle")
    assert leaked is None
