"""Live LLM integration tests for new built-in tools.

Run with real credentials:
    Anthropic-compatible (kimi):
        LLM_BASE_URL=https://api.anthropic.com
        LLM_API_KEY=<key>
        LLM_MODEL=kimi-for-coding
        LLM_API_FORMAT=anthropic  (optional, defaults to openai)

    OpenAI-compatible (qwen):
        LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        LLM_API_KEY=<key>
        LLM_MODEL=qwen3.5-flash
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent, ToolRegistryComponent
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.skills import SkillManager
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import Message


def _make_provider(live_api_key: str) -> OpenAIModel | ClaudeModel:
    model = os.getenv("LLM_MODEL") or "qwen3.5-flash"
    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    api_format_str = os.getenv("LLM_API_FORMAT", "openai").lower()

    config = ProviderConfig(
        provider_id="live",
        base_url=base_url,
        api_key=live_api_key,
        api_format=ApiFormat.ANTHROPIC_MESSAGES
        if api_format_str == "anthropic"
        else ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )

    if api_format_str == "anthropic":
        return ClaudeModel(config=config, model=model)
    return OpenAIModel(config=config, model=model)


def _build_world(
    live_api_key: str,
    tmp_path: Path,
    user_message: str,
    system_prompt: str = "",
    model: str = "",
) -> tuple[World, int]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    world = World()
    model = _make_provider(live_api_key)
    agent = world.create_entity()

    skill = BuiltinToolsSkill().bind_workspace(str(workspace))
    manager = SkillManager()
    manager.install(world, agent, skill)

    resolved_model = model or os.getenv("LLM_MODEL") or "qwen3.5-flash"
    model_model = getattr(model, "model", None) or getattr(model, "_model", resolved_model)
    world.add_component(
        agent,
        LLMComponent(
            model=model,
            
            system_prompt=system_prompt or (
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


@pytest.mark.asyncio
async def test_live_grep_tool(live_api_key: str, tmp_path: Path) -> None:
    """LLM should use grep to find function definitions in a Python file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    world, agent = _build_world(
        live_api_key,
        tmp_path,
        "Use the grep tool to search for lines containing 'def ' in module.py. "
        "Tell me how many function definitions you found.",
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


@pytest.mark.asyncio
async def test_live_read_file_with_range(live_api_key: str, tmp_path: Path) -> None:
    """LLM should use read_file with offset and limit to read a specific range."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lines = [f"line {i}" for i in range(1, 21)]
    (workspace / "big.txt").write_text("\n".join(lines), encoding="utf-8")

    world, agent = _build_world(
        live_api_key,
        tmp_path,
        "Use read_file to read only lines 5 to 7 of big.txt (offset=5, limit=3). "
        "Tell me the content of those three lines.",
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "line 5" in reply.lower() or "5" in reply


# ---------------------------------------------------------------------------
# explore live test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_explore_tool(live_api_key: str, tmp_path: Path) -> None:
    """LLM should use explore to describe workspace directory structure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("x = 1")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_main.py").write_text("pass")

    world, agent = _build_world(
        live_api_key,
        tmp_path,
        "Use the explore tool to explore '.' with max_depth=2 and describe the directory structure.",
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "src" in reply.lower() or "tests" in reply.lower() or "main" in reply.lower()


# ---------------------------------------------------------------------------
# code_execution live test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_code_execution_tool(live_api_key: str, tmp_path: Path) -> None:
    """LLM should use code_execution to run Python code and report the result."""
    world, agent = _build_world(
        live_api_key,
        tmp_path,
        "Use the code_execution tool to run this Python code: "
        "print(sum(range(1, 11))). Tell me what it prints.",
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "55" in reply


# ---------------------------------------------------------------------------
# webfetch live test (uses a reliable public endpoint)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_webfetch_tool(live_api_key: str, tmp_path: Path) -> None:
    """LLM should use webfetch to retrieve a URL and report content."""
    world, agent = _build_world(
        live_api_key,
        tmp_path,
        "Use the webfetch tool to fetch https://httpbin.org/get and tell me the "
        "value of the 'url' field in the JSON response.",
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    reply = _last_assistant_message(world, agent)
    assert "httpbin.org" in reply.lower() or "get" in reply.lower()
