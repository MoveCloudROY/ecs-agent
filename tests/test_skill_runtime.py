from __future__ import annotations

from pathlib import Path

from ecs_agent.core import World
from ecs_agent.components import WorkspaceBindingComponent
from ecs_agent.skills import SkillManager
from ecs_agent.skills.skill import Skill
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


class MutableWorkspaceSkill(DummySkill):
    def __init__(self, *, name: str, description: str, prompt: str) -> None:
        super().__init__(name=name, description=description, prompt=prompt)
        self.workspace_seen: str | None = None

    def resolve_path_references(
        self, workspace_root: str | Path
    ) -> MutableWorkspaceSkill:
        self.workspace_seen = str(workspace_root)
        return self


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


def test_skill_runtime_workspace_binding_uses_agent_component(tmp_path: Path) -> None:
    manager = SkillManager()
    world = World()
    entity_id = world.create_entity()

    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        """---
name: writer
description: workspace rewrite
---
Write to @./notes.md
""",
        encoding="utf-8",
    )

    world.add_component(entity_id, WorkspaceBindingComponent(workspace_root=workspace))

    shared_skill = Skill(skill_path=skill_path)
    original_prompt = shared_skill.system_prompt()

    manager.install(world, entity_id, shared_skill)
    installed_skill = world.skill_runtime.get_installed_skill(
        entity_id, shared_skill.name
    )

    assert installed_skill is not None
    assert installed_skill is not shared_skill
    assert shared_skill.system_prompt() == original_prompt
    assert "@./notes.md" in shared_skill.system_prompt()
    assert "@./notes.md" not in installed_skill.system_prompt()
    assert "skills/writer/notes.md" in installed_skill.system_prompt()


def test_skill_runtime_two_agents_different_workspaces_produce_isolated_skills() -> (
    None
):
    manager = SkillManager()
    world = World()
    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, WorkspaceBindingComponent(workspace_root="/tmp/ws-a"))
    world.add_component(entity_b, WorkspaceBindingComponent(workspace_root="/tmp/ws-b"))

    shared_skill = MutableWorkspaceSkill(
        name="mutable",
        description="workspace-sensitive",
        prompt="prompt",
    )

    manager.install(world, entity_a, shared_skill)
    manager.install(world, entity_b, shared_skill)

    installed_a = world.skill_runtime.get_installed_skill(entity_a, "mutable")
    installed_b = world.skill_runtime.get_installed_skill(entity_b, "mutable")

    assert installed_a is not None
    assert installed_b is not None
    assert installed_a is not installed_b
    assert installed_a is not shared_skill
    assert installed_b is not shared_skill
    assert isinstance(installed_a, MutableWorkspaceSkill)
    assert isinstance(installed_b, MutableWorkspaceSkill)
    assert installed_a.workspace_seen == "/tmp/ws-a"
    assert installed_b.workspace_seen == "/tmp/ws-b"
    assert shared_skill.workspace_seen is None


def test_subagent_inherits_parent_workspace_binding() -> None:
    world = World()
    parent_entity = world.create_entity()
    child_entity = world.create_entity()

    world.add_component(
        parent_entity,
        WorkspaceBindingComponent(workspace_root="/tmp/parent-workspace"),
    )

    world.skill_runtime.inherit_workspace_binding(
        world,
        parent_entity=parent_entity,
        child_entity=child_entity,
    )

    child_binding = world.get_component(child_entity, WorkspaceBindingComponent)
    assert child_binding is not None
    assert child_binding.workspace_root == "/tmp/parent-workspace"
