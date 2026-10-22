"""Explicit live integration coverage migrated from mixed/offline directories."""

from __future__ import annotations
import importlib.util
import shutil
from pathlib import Path
import pytest
from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import World
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.skill import Skill
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.core import Runner
from ecs_agent.types import Message

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.workflow
@pytest.mark.asyncio
async def test_ui_design_flow_real_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_model_factory: LiveModelFactory,
) -> None:
    """Full E2E test with the selected OpenAI-compatible profile."""
    from ecs_agent.tools.builtins import BuiltinToolsSkill

    # Setup: Copy example structure to tmp_path
    example_base = tmp_path / "ui-design-flow"
    example_base.mkdir()

    assets_dir = example_base / "assets"
    assets_dir.mkdir()
    (assets_dir / "prompt.txt").write_text(
        "Design a modern UI for a todo app", encoding="utf-8"
    )

    output_dir = example_base / "ui-design"
    output_dir.mkdir()

    source_skill_root = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "e2e"
        / "ui-design-flow"
        / ".claude"
        / "skills"
    )
    target_skill_root = example_base / ".claude" / "skills"
    target_skill_root.mkdir(parents=True)
    shutil.copytree(
        source_skill_root / "ui-navigator",
        target_skill_root / "ui-navigator",
    )
    shutil.copytree(
        source_skill_root / "ui-prompt",
        target_skill_root / "ui-prompt",
    )

    # Create World
    world = World()

    # Create the model from the selected profile
    model = live_model_factory.openai()

    # Create Agent Entity
    agent_id = world.create_entity()

    # Read initial prompt
    initial_prompt = (assets_dir / "prompt.txt").read_text(encoding="utf-8").strip()

    # Add components
    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            system_prompt="You are a UI design expert.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(messages=[Message(role="user", content=initial_prompt)]),
    )

    manager = SkillManager()
    ui_navigator_skill = Skill(
        skill_path=target_skill_root / "ui-navigator" / "SKILL.md"
    )
    assert ui_navigator_skill.valid, (
        "Expected ui-navigator skill to parse in test workspace"
    )
    ui_prompt_skill = Skill(skill_path=target_skill_root / "ui-prompt" / "SKILL.md")
    assert ui_prompt_skill.valid, "Expected ui-prompt skill to parse in test workspace"

    workspace_root = str(tmp_path / "ui-design-flow")
    ui_navigator_skill.resolve_path_references(workspace_root)
    ui_prompt_skill.resolve_path_references(workspace_root)
    manager.install(world, agent_id, ui_navigator_skill)  # type: ignore[arg-type]
    manager.install(world, agent_id, ui_prompt_skill)  # type: ignore[arg-type]

    builtin_skill = BuiltinToolsSkill()
    builtin_skill.bind_workspace(str(tmp_path / "ui-design-flow"))
    manager.install(world, agent_id, builtin_skill)

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    input_responses = iter(
        [
            "请为每个页面生成 Nano Banana prompt，并调用 write_file 将结果写入 ui-design/nano-banana-prompts.md",
            "exit",
        ]
    )

    def fake_input(_prompt: str) -> str:
        return next(input_responses, "exit")

    monkeypatch.setattr("builtins.input", fake_input)

    runtime_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "e2e"
        / "ui-design-flow"
        / "runtime.py"
    )
    spec = importlib.util.spec_from_file_location("ui_design_runtime", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_module)
    setup_interactive_input = getattr(runtime_module, "setup_interactive_input", None)
    assert callable(setup_interactive_input)
    await setup_interactive_input(world, agent_id)

    # Run agent loop
    runner = Runner()
    await runner.run(world, max_ticks=10)

    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assistant_indices = [
        idx
        for idx, msg in enumerate(conv.messages)
        if msg.role == "assistant" and (msg.content.strip() or msg.tool_calls)
    ]
    assert assistant_indices, "Expected at least one assistant completion"

    follow_up_users = [
        msg
        for msg in conv.messages
        if msg.role == "user" and msg.content.strip().lower() not in ("exit", "quit")
    ]
    assert len(assistant_indices) >= 2, (
        "Expected cleanup-enabled runtime to continue into at least a second assistant completion"
    )
    assert len(follow_up_users) >= 2, (
        "Expected at least one non-exit follow-up user turn in addition to the initial prompt"
    )

    tool_messages = [msg for msg in conv.messages if msg.role == "tool"]
    assert tool_messages, (
        "Expected at least one tool message from builtin tool execution"
    )

    artifact_path = tmp_path / "ui-design-flow" / "ui-design" / "nano-banana-prompts.md"
    wrote_artifact_via_tool = any(
        (
            "write_file" in msg.content.lower()
            or "nano-banana-prompts.md" in msg.content.lower()
        )
        for msg in tool_messages
        if isinstance(msg.content, str)
    )
    assert artifact_path.exists() or wrote_artifact_via_tool, (
        "Expected ui-prompt artifact mutation on disk or write_file evidence in tool messages"
    )

    terminal_reasons = [
        terminal.reason
        for _, (terminal,) in world.query(TerminalComponent)
        if isinstance(terminal, TerminalComponent)
    ]
    assert terminal_reasons, "Expected run to terminate with a terminal reason"
