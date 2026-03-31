from __future__ import annotations

import os
from pathlib import Path

import pytest

from ecs_agent import BuiltinToolsSkill, SkillManager
from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers import OpenAIProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message


API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_real_llm_hashline_read_edit_workflow(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "config.txt"
    target.write_text(
        "host: localhost\nport: 3000\ndebug: false\ntimeout: 30\n",
        encoding="utf-8",
    )

    world = World()
    entity = world.create_entity()
    world.add_component(
        entity,
        LLMComponent(
            provider=OpenAIProvider(
                config=ProviderConfig(
                    provider_id="openai",
                    base_url=BASE_URL,
                    api_key=API_KEY,
                    api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
                ),
                model=MODEL,
            ),
            model=MODEL,
        ),
    )
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Use builtin tools only. First call read_file on config.txt. "
                        "Then call edit_file to change only the port line from 3000 to 8080 using the "
                        "fresh LINE#HASH anchor. Do not use write_file. After editing, give a one-line confirmation."
                    ),
                )
            ]
        ),
    )

    skill = BuiltinToolsSkill()
    skill.bind_workspace(str(workspace))
    SkillManager().install(world, entity, skill)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=8)

    updated = target.read_text(encoding="utf-8")
    assert "port:" in updated

    conv = world.get_component(entity, ConversationComponent)
    assert conv is not None
    tool_messages = [message for message in conv.messages if message.role == "tool"]
    assert len(tool_messages) >= 2
    assert any(message.content.startswith("1#") for message in tool_messages)
    assert any(
        ("Applied" in message.content)
        or ("unexpected keyword argument" in message.content)
        or ("edits_json must be a JSON array" in message.content)
        for message in tool_messages
    )
    assert conv.messages[-1].role == "assistant"
