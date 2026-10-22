"""Live LLM integration tests for new built-in tools.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.skills import SkillManager
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import Message

from tests.support.models import LiveModelFactory


def _build_world(
    tmp_path: Path,
    user_message: str,
    system_prompt: str = "",
    *,
    live_model_factory: LiveModelFactory,
) -> tuple[World, int]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    world = World()
    model = live_model_factory.create()
    agent = world.create_entity()

    skill = BuiltinToolsSkill().bind_workspace(str(workspace))
    manager = SkillManager()
    manager.install(world, agent, skill)

    world.add_component(
        agent,
        LLMComponent(
            model=model,
            system_prompt=system_prompt
            or (
                "You are a helpful assistant with file, shell, and web tools. "
                "Use the provided tools precisely as instructed."
            ),
        ),
    )
    world.add_component(
        agent,
        ConversationComponent(messages=[Message(role="user", content=user_message)]),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    return world, agent


def _last_assistant_message(world: World, agent: int) -> str:
    conv = world.get_component(agent, ConversationComponent)
    assert conv is not None
    msg = next((m for m in reversed(conv.messages) if m.role == "assistant"), None)
    assert msg is not None, "No assistant message found"
    return msg.content or ""


# ---------------------------------------------------------------------------
# grep live test
# ---------------------------------------------------------------------------


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_grep_tool(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    """LLM should use grep to find function definitions in a Python file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    world, agent = _build_world(
        tmp_path,
        "Use the grep tool to search for lines containing 'def ' in module.py. "
        "Tell me how many function definitions you found.",
        live_model_factory=live_model_factory,
    )
    runner = Runner()
    await runner.run(world, max_ticks=8)

    reply = _last_assistant_message(world, agent)
    assert (
        "2" in reply
        or "two" in reply.lower()
        or "add" in reply.lower()
        or "subtract" in reply.lower()
    )


# ---------------------------------------------------------------------------
# read_file with range live test
# ---------------------------------------------------------------------------


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_read_file_with_range(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    """LLM should use read_file with offset and limit to read a specific range."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lines = [f"line {i}" for i in range(1, 21)]
    (workspace / "big.txt").write_text("\n".join(lines), encoding="utf-8")

    world, agent = _build_world(
        tmp_path,
        "Use read_file to read only lines 5 to 7 of big.txt (offset=5, limit=3). "
        "Tell me the content of those three lines.",
        live_model_factory=live_model_factory,
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "line 5" in reply.lower() or "5" in reply


# ---------------------------------------------------------------------------
# explore live test
# ---------------------------------------------------------------------------


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_explore_tool(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    """LLM should use explore to describe workspace directory structure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("x = 1")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_main.py").write_text("pass")

    world, agent = _build_world(
        tmp_path,
        "Use the explore tool to explore '.' with max_depth=2 and describe the directory structure.",
        live_model_factory=live_model_factory,
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "src" in reply.lower() or "tests" in reply.lower() or "main" in reply.lower()


# ---------------------------------------------------------------------------
# code_execution live test
# ---------------------------------------------------------------------------


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_code_execution_tool(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    """LLM should use code_execution to run Python code and report the result."""
    world, agent = _build_world(
        tmp_path,
        "Use the code_execution tool to run this Python code: "
        "print(sum(range(1, 11))). Tell me what it prints.",
        live_model_factory=live_model_factory,
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "55" in reply


# ---------------------------------------------------------------------------
# webfetch live test (uses a reliable public endpoint)
# ---------------------------------------------------------------------------


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses", "anthropic_messages"],
    capabilities=["text", "tools"],
)
@pytest.mark.tools
@pytest.mark.asyncio
async def test_live_webfetch_tool(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    """LLM should use webfetch to retrieve a URL and report content."""
    world, agent = _build_world(
        tmp_path,
        "Use the webfetch tool to fetch https://httpbin.org/get and tell me the "
        "value of the 'url' field in the JSON response.",
        live_model_factory=live_model_factory,
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "httpbin.org" in reply.lower() or "get" in reply.lower()
