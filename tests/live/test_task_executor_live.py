"""Live TaskExecutor subagent dispatch tests."""

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import ApiFormat, OpenAIProvider, ProviderConfig
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.task import DispatchRequest, TaskExecutor
from ecs_agent.types import SubagentConfig


def _build_provider(
    api_key: str, *, base_url: str, api_format: ApiFormat
) -> OpenAIProvider:
    model = "qwen3.5-flash"
    return OpenAIProvider(
        config=ProviderConfig(
            provider_id="aliyun",
            base_url=base_url,
            api_key=api_key,
            api_format=api_format,
        ),
        model=model,
    )


async def _execute_live_dispatch(
    live_api_key: str,
    *,
    base_url: str,
    api_format: ApiFormat,
) -> None:
    world = World(name="task-executor-live")
    entity_id = world.create_entity()
    provider = _build_provider(
        live_api_key,
        base_url=base_url,
        api_format=api_format,
    )
    child_provider = _build_provider(
        live_api_key,
        base_url=base_url,
        api_format=api_format,
    )

    world.add_component(
        entity_id,
        LLMComponent(
            provider=provider,
            model=provider.model,
            system_prompt="You are a helpful parent agent.",
        ),
    )
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    provider=child_provider,
                    model=child_provider.model,
                    system_prompt="Answer the user's request directly and briefly.",
                    max_ticks=3,
                )
            }
        ),
    )

    subagent_system = SubagentSystem()
    await subagent_system.process(world)

    executor = TaskExecutor()
    result = await executor.execute_dispatch_request(
        world,
        entity_id,
        DispatchRequest(
            task_id="live-task-1",
            wave_number=0,
            sequence_number=0,
            description="Say 'hello' in one word.",
            expected_output="A one-word greeting.",
            assigned_agent="worker",
            tools=(),
            context_dependencies=(),
            priority=0,
        ),
    )

    assert result.success is True
    assert len(result.result_content) > 0
    assert result.backend_type == "subagent"


@pytest.mark.live
@pytest.mark.asyncio
async def test_task_executor_dispatch_compatible_mode(live_api_key: str) -> None:
    await _execute_live_dispatch(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_task_executor_dispatch_responses_mode(live_api_key: str) -> None:
    await _execute_live_dispatch(
        live_api_key,
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        api_format=ApiFormat.OPENAI_RESPONSES,
    )
