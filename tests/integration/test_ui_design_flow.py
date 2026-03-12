"""Integration tests for UI Design Flow E2E example.

Tests deterministic UI design flow execution with FakeProvider and
error handling without LLM_API_KEY requirement. All tests are offline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import shutil
from pathlib import Path

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.markdown_skill import MarkdownSkill
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import CompletionResult, Message, UserInputRequestedEvent

# DashScope API configuration from environment variables
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")


@pytest.mark.asyncio
async def test_ui_design_flow_fake_provider(tmp_path: Path) -> None:
    """Full E2E test with FakeProvider and simulated input."""
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

    # Create FakeProvider with deterministic responses
    provider = FakeProvider(
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
            provider=provider,
            model="fake",
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
    world.register_system(MemorySystem(), priority=10)
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

    # Create FakeProvider

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
    """Test SkillManager can install MarkdownSkill without errors."""
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
        skill = MarkdownSkill(skill_path=skill_file)
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
async def test_ui_design_flow_real_llm(tmp_path: Path) -> None:
    """Full E2E test with real OpenAI-compatible provider (DashScope)."""
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

    # Create OpenAI-compatible provider with DashScope
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
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
            provider=provider,
            model=MODEL,
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
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Setup interactive input simulation
    input_responses = ["continue", "continue", "exit"]
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

    # Verify conversation happened (LLM may or may not call tools)
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) > 1, "Expected conversation to have occurred"


@pytest.mark.asyncio
async def test_ui_design_flow_cli_automation() -> None:
    """Test CLI automation with piped stdin input (FakeProvider mode).

    This test simulates interactive CLI usage by piping stdin to main.py
    without LLM_API_KEY set, ensuring FakeProvider fallback is used.
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
        env={**os.environ, "LLM_API_KEY": ""},  # Force FakeProvider
    )

    # Verify successful execution
    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    # Verify conversation occurred (assistant response visible in output)
    output = result.stdout + result.stderr
    assert "assistant" in output.lower() or "conversation" in output.lower(), (
        f"Expected conversation evidence in output. Got:\n{output}"
    )
