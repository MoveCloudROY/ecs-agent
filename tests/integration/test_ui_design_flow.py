from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.core import World
from ecs_agent.skills.discovery import DiscoveryManager
from ecs_agent.skills.manager import SkillManager


async def _install_markdown_skill(tmp_path: Path, content: str) -> tuple[World, int]:
    skill_dir = tmp_path / ".claude" / "skills" / "ui-design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content)

    world = World()
    entity = world.create_entity()
    await DiscoveryManager().auto_discover_and_install(
        world,
        entity,
        SkillManager(),
        directories=[tmp_path],
    )
    return world, entity


@pytest.mark.parametrize(
    ("user_invocable", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
@pytest.mark.asyncio
async def test_ui_design_flow_contract_slash_invocable_semantics(
    tmp_path: Path,
    user_invocable: bool,
    expected: bool,
) -> None:
    world, entity = await _install_markdown_skill(
        tmp_path,
        "---\n"
        "name: ui-design\n"
        "description: UI design flow\n"
        f"user-invocable: {'true' if user_invocable else 'false'}\n"
        "---\n"
        "Guide the user through visual system choices.",
    )

    manager = SkillManager()
    can_invoke_via_slash = getattr(manager, "can_invoke_via_slash", None)

    assert callable(can_invoke_via_slash)
    assert can_invoke_via_slash(world, entity, "/ui-design") is expected


@pytest.mark.parametrize(
    ("disable_model_invocation", "expected"),
    [
        (True, False),
        (False, True),
    ],
)
@pytest.mark.asyncio
async def test_ui_design_flow_contract_model_auto_activation_semantics(
    tmp_path: Path,
    disable_model_invocation: bool,
    expected: bool,
) -> None:
    world, entity = await _install_markdown_skill(
        tmp_path,
        "---\n"
        "name: ui-design\n"
        "description: UI design flow\n"
        f"disable-model-invocation: {'true' if disable_model_invocation else 'false'}\n"
        "---\n"
        "Guide the user through visual system choices.",
    )

    manager = SkillManager()
    can_auto_invoke = getattr(manager, "can_model_auto_invoke_skill", None)

    assert callable(can_auto_invoke)
    assert can_auto_invoke(world, entity, "ui-design") is expected
