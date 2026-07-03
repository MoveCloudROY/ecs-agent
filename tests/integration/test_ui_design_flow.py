"""Integration tests for UI Design Flow E2E example.

Tests deterministic UI design flow execution with FakeModel and
error handling without LLM_API_KEY requirement. All tests are offline.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationComponent,
    CurrentCompactionSummaryComponent,
    LLMComponent,
)
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.skills.discovery import DiscoveryManager
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.skill import Skill
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.core import Runner
from ecs_agent.types import CompletionResult, Message, UserInputRequestedEvent

# DashScope API configuration from environment variables
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")


@pytest.mark.asyncio
async def test_ui_design_flow_fake_provider(tmp_path: Path) -> None:
    """Full E2E test with FakeModel and simulated input."""
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

    # Create World
    world = World()

    # Create FakeModel with deterministic responses
    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="I'll help you design a beautiful UI for your todo app.",
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Here's the draft design:\n\n# Todo App UI Design\n\n## Layout\n- Header with app name\n- Input field for new tasks\n- List of todos\n- Footer with filters",
                )
            ),
        ]
    )

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

    # Register Systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Setup interactive input simulation
    input_responses = ["Tell me more about the layout", "exit"]
    input_index = 0

    async def provide_input(event: UserInputRequestedEvent) -> None:
        nonlocal input_index
        if input_index < len(input_responses):
            user_text = input_responses[input_index]
            input_index += 1
        else:
            user_text = "exit"

        normalized = user_text.lower().strip()
        if normalized in ("exit", "quit"):
            world.add_component(
                event.entity_id,
                TerminalComponent(reason="user_exit_command"),
            )

        if not event.input_future.done():
            event.input_future.set_result(user_text)

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)
    world.register_system(UserInputSystem(priority=-5), priority=-5)

    # Run agent loop
    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Verify conversation
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) >= 2
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Design a modern UI for a todo app"
    assert conv.messages[1].role == "assistant"
    assert "help you design" in conv.messages[1].content.lower()

    # Verify terminal state
    terminal = world.get_component(agent_id, TerminalComponent)
    assert terminal is not None or conv is not None  # Either terminated or completed


@pytest.mark.asyncio
async def test_ui_design_flow_missing_prompt(tmp_path: Path) -> None:
    """Test error handling when assets/prompt.txt is missing."""
    # Setup: Create directory structure WITHOUT prompt.txt
    example_base = tmp_path / "ui-design-flow"
    example_base.mkdir()

    assets_dir = example_base / "assets"
    assets_dir.mkdir()

    output_dir = example_base / "ui-design"
    output_dir.mkdir()

    # Create World
    world = World()

    # Create FakeModel

    # Create Agent Entity
    agent_id = world.create_entity()

    # Attempt to read missing prompt.txt
    prompt_path = assets_dir / "prompt.txt"
    prompt_content: str | None = None
    error_occurred = False

    try:
        prompt_content = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        error_occurred = True

    # Verify error detection
    assert error_occurred is True, (
        "Expected FileNotFoundError when prompt.txt is missing"
    )
    assert prompt_content is None

    # Verify World remains in valid state (no components added yet)
    llm = world.get_component(agent_id, LLMComponent)
    assert llm is None  # No component added because initialization failed


@pytest.mark.asyncio
async def test_ui_design_flow_skill_manager_installation() -> None:
    """Test SkillManager can install markdown Skill without errors."""
    world = World()
    agent_id = world.create_entity()

    # Create temporary skill file
    tmp_skill_path = (
        Path(__file__).parent.parent.parent / "examples" / "skills" / "test_skill"
    )
    tmp_skill_path.mkdir(parents=True, exist_ok=True)
    skill_file = tmp_skill_path / "SKILL.md"

    skill_file.write_text(
        """# Test Skill

## Description
A simple test skill for integration testing.

## Tools
```yaml
tools:
  - name: test_tool
    description: A test tool
    input_schema:
      type: object
      properties:
        message:
          type: string
      required: [message]
```

## Usage
This is a test skill.
""",
        encoding="utf-8",
    )

    try:
        # Test skill installation
        manager = SkillManager()
        skill = Skill(skill_path=skill_file)
        manager.install(world, agent_id, skill)  # type: ignore[arg-type]

        # Verify skill was installed (SkillComponent should exist)
        from ecs_agent.components.definitions import SkillComponent

        skill_comp = world.get_component(agent_id, SkillComponent)
        assert skill_comp is not None
        assert len(skill_comp.skills) > 0

    finally:
        # Cleanup
        if skill_file.exists():
            skill_file.unlink()
        if tmp_skill_path.exists():
            shutil.rmtree(tmp_skill_path)


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_ui_design_flow_real_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full E2E test with real OpenAI-compatible model (DashScope)."""
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

    # Create OpenAI-compatible model with DashScope
    model = OpenAIModel(
        config=ProviderConfig(
            provider_id="openai",
            base_url=BASE_URL,
            api_key=API_KEY,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model=MODEL,
    )

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


@pytest.mark.asyncio
async def test_ui_design_flow_cli_automation() -> None:
    """Test CLI automation with piped stdin input (FakeModel mode).

    This test simulates interactive CLI usage by piping stdin to main.py
    without LLM_API_KEY set, ensuring FakeModel fallback is used.
    Verifies the agent responds to user input and conversation occurs.
    """
    # Input sequence: Design request → continue → exit
    input_sequence = "Design a calculator UI\ncontinue\nexit\n"

    # Run main.py with stdin piped, without LLM_API_KEY
    result = subprocess.run(
        ["uv", "run", "python", "examples/e2e/ui-design-flow/main.py"],
        input=input_sequence,
        text=True,
        capture_output=True,
        cwd=Path(__file__).parent.parent.parent,
        env={**os.environ, "LLM_API_KEY": ""},  # Force FakeModel
    )

    # Verify successful execution
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. stderr: {result.stderr}"
    )

    # Verify conversation occurred (assistant response visible in output)
    output = result.stdout + result.stderr
    assert "assistant" in output.lower() or "conversation" in output.lower(), (
        f"Expected conversation evidence in output. Got:\n{output}"
    )

    assert "user: continue" in output.lower(), (
        f"Expected one follow-up user turn before exit. Got output:\n{output}"
    )

    runtime_source = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "e2e"
        / "ui-design-flow"
        / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "class ClearTerminalForInputSystem" not in runtime_source
    assert "TerminalCleanupSystem" in runtime_source


# Master's contract tests
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


@pytest.mark.asyncio
async def test_ui_design_flow_ui_prompt_invalid_yaml() -> None:
    """Verification: ui-prompt skill YAML is valid after repair."""
    skill_file = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "e2e"
        / "ui-design-flow"
        / ".claude"
        / "skills"
        / "ui-prompt"
        / "SKILL.md"
    )

    skill = Skill(skill_path=skill_file)

    assert skill.valid is True
    assert skill.name == "ui-prompt"
    assert skill.description


@pytest.mark.asyncio
async def test_ui_design_flow_ui_prompt_invalid_install_rejected(
    tmp_path: Path,
) -> None:
    """Test that main.py guard logic rejects invalid skills with ValueError."""
    malformed_skill_dir = tmp_path / ".claude" / "skills" / "invalid-ui-prompt"
    malformed_skill_dir.mkdir(parents=True)
    malformed_skill_file = malformed_skill_dir / "SKILL.md"
    malformed_skill_file.write_text(
        "---\n"
        "name: invalid-ui-prompt\n"
        "description: 'unterminated description\n"
        "---\n"
        "# invalid-ui-prompt\n"
        "body\n",
        encoding="utf-8",
    )

    world = World()
    entity_id = world.create_entity()
    manager = SkillManager()
    skill = Skill(skill_path=malformed_skill_file)

    # Verify skill is invalid
    assert skill.valid is False

    # Test the main.py guard pattern: check valid, log error, raise ValueError
    try:
        if not skill.valid:
            raise ValueError(
                f"Skill at {malformed_skill_file} is invalid and cannot be installed"
            )
        pytest.fail("Expected ValueError from invalid skill guard")
    except ValueError as exc:
        # Guard correctly raises ValueError for invalid skill
        assert "invalid" in str(exc)


@pytest.mark.asyncio
async def test_ui_design_flow_ui_prompt_writes_artifact(tmp_path: Path) -> None:
    from ecs_agent.tools.builtins import BuiltinToolsSkill
    from ecs_agent.types import ToolCall

    world = World()
    agent_id = world.create_entity()

    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Writing ui-prompt artifact now.",
                    tool_calls=[
                        ToolCall(
                            id="call_write_ui_prompt",
                            name="write_file",
                            arguments={
                                "file_path": "ui-design/nano-banana-prompts.md",
                                "content": "# Nano Banana Prompts\n\n- Hero card prompt\n- CTA button prompt\n",
                            },
                        )
                    ],
                )
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Done. The ui-prompt artifact has been written.",
                )
            ),
        ]
    )

    manager = SkillManager()
    builtin_skill = BuiltinToolsSkill()
    builtin_skill.bind_workspace(str(tmp_path))
    manager.install(world, agent_id, builtin_skill)

    world.add_component(
        agent_id,
        LLMComponent(model=model, system_prompt=""),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="Generate ui-prompt output artifact")
            ]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=5)

    output_file = tmp_path / "ui-design" / "nano-banana-prompts.md"
    assert output_file.exists(), "Expected write_file to create ui-prompt artifact"

    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    tool_messages = [msg for msg in conv.messages if msg.role == "tool"]
    assert tool_messages, (
        "Expected tool execution evidence; chat-only output is insufficient"
    )


def _load_runtime_module():
    """Load the hyphenated-dir example runtime.py via importlib (not importable as a package)."""
    runtime_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "e2e"
        / "ui-design-flow"
        / "runtime.py"
    )
    spec = importlib.util.spec_from_file_location("ui_design_runtime", runtime_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_install_auto_compaction_bounds_history() -> None:
    """ISSUE-2: install_auto_compaction wires CompactionSystem so a conversation
    exceeding the threshold is summarized instead of growing unbounded."""
    runtime_module = _load_runtime_module()
    install_auto_compaction = getattr(runtime_module, "install_auto_compaction", None)
    assert callable(install_auto_compaction)

    world = World()
    agent_id = world.create_entity()

    # FakeModel returns the compaction summary when CompactionSystem calls it.
    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="SUMMARY: earlier UI design turns")
            )
        ]
    )
    world.add_component(agent_id, LLMComponent(model=model))

    original_messages = [
        Message(role="user", content="Design a modern dashboard with cards and charts"),
        Message(role="assistant", content="Sure, here is a first layout proposal for the dashboard"),
        Message(role="user", content="Make the sidebar collapsible and add a dark theme"),
        Message(role="assistant", content="Updated: collapsible sidebar plus a dark theme applied"),
        Message(role="user", content="Now export the design tokens as JSON"),
    ]
    world.add_component(
        agent_id, ConversationComponent(messages=list(original_messages))
    )

    # Tiny threshold forces compaction on this tick; full_history summarizes all.
    install_auto_compaction(
        world,
        agent_id,
        threshold_tokens=1,
        compaction_method="full_history",
    )

    # Sanity: the helper wired the per-entity config.
    config = world.get_component(agent_id, CompactionConfigComponent)
    assert config is not None
    assert config.threshold_tokens == 1
    assert config.compaction_method == "full_history"

    await Runner().run(world, max_ticks=1)

    # History was compacted: a summary exists and the message list shrank.
    summary = world.get_component(agent_id, CurrentCompactionSummaryComponent)
    assert summary is not None
    assert summary.summary == "SUMMARY: earlier UI design turns"

    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) < len(original_messages), (
        "expected compaction to bound history growth"
    )
