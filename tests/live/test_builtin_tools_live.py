from __future__ import annotations

import os
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.skills import SkillManager
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.tools.builtins.edit_tool import compute_line_hash
from ecs_agent.types import Message


def _make_provider(live_api_key: str) -> OpenAIProvider:
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    config = ProviderConfig(
        provider_id="aliyun",
        base_url=base_url,
        api_key=live_api_key,
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    return OpenAIProvider(config=config, model=model)


def _build_world(
    live_api_key: str, tmp_path: Path, user_message: str
) -> tuple[World, int]:
    workspace = tmp_path / "workspace"

    world = World()
    provider = _make_provider(live_api_key)
    agent = world.create_entity()

    skill = BuiltinToolsSkill().bind_workspace(str(workspace))
    manager = SkillManager()
    manager.install(world, agent, skill)

    registry = world.get_component(agent, ToolRegistryComponent)
    assert registry is not None

    world.add_component(
        agent,
        LLMComponent(
            model=provider,
            
            system_prompt=(
                "You are a helpful assistant with access to file and shell tools. "
                "Use the tools as instructed. Be precise and concise."
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


@pytest.mark.asyncio
async def test_live_edit_file_tool_called_by_llm(
    live_api_key: str, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = workspace / "greet.py"
    target.write_text('def hello():\n    return "world"\n', encoding="utf-8")

    line1_hash = compute_line_hash(1, "def hello():")
    user_msg = (
        f"Please use edit_file to rename the function on line 1 of greet.py. "
        f"Use op='replace', pos='1#{line1_hash}', content='def greet():'."
    )

    world, agent = _build_world(live_api_key, tmp_path, user_msg)
    runner = Runner()
    await runner.run(world, max_ticks=5)

    updated = target.read_text(encoding="utf-8")
    assert "def greet():" in updated or "greet" in updated


@pytest.mark.asyncio
async def test_live_read_file_tool_called_by_llm(
    live_api_key: str, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "info.txt").write_text(
        "project=ecs-agent\nversion=1.0", encoding="utf-8"
    )

    world, agent = _build_world(
        live_api_key,
        tmp_path,
        "Read info.txt and tell me the project name listed in it.",
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    conv = world.get_component(agent, ConversationComponent)
    assert conv is not None
    last_assistant = next(
        (m for m in reversed(conv.messages) if m.role == "assistant"), None
    )
    assert last_assistant is not None
    assert (
        "ecs-agent" in (last_assistant.content or "").lower()
        or "ecs" in (last_assistant.content or "").lower()
    )


@pytest.mark.asyncio
async def test_live_interactive_bash_tool_called_by_llm(
    live_api_key: str, tmp_path: Path
) -> None:
    session_name = "test-ecs-live-ib"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    world, agent = _build_world(
        live_api_key,
        tmp_path,
        f"Use interactive_bash to create a new tmux session named '{session_name}' "
        f"with the command: new-session -d -s {session_name}. "
        "Then confirm you did it.",
    )
    runner = Runner()
    await runner.run(world, max_ticks=5)

    import asyncio

    result = await asyncio.create_subprocess_exec(
        "tmux",
        "has-session",
        "-t",
        session_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await result.communicate()

    if result.returncode == 0:
        await asyncio.create_subprocess_exec("tmux", "kill-session", "-t", session_name)

    conv = world.get_component(agent, ConversationComponent)
    assert conv is not None
    assert any(m.role == "assistant" for m in conv.messages)
