import asyncio
import os
from typing import Any
from collections.abc import AsyncIterator

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    UserPromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    ToolRegistryComponent,
    TurnStateComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.prompts.contracts import SystemPromptConfigSpec, PromptTemplateSource
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import CompletionResult, Message, StreamDelta, ToolSchema


class RecordingProvider:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.last_messages: list[Message] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        self.last_messages = list(messages)
        return await self._provider.complete(
            messages,
            tools=tools,
            stream=stream,
            response_format=response_format,
        )


def _build_provider_from_env() -> tuple[LLMProvider, str, str]:
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    if api_key:
        real_provider: LLMProvider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
        return real_provider, model, "real"

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Fake mode response: prompt normalization wiring is active.",
                )
            )
        ]
    )
    return provider, "fake", "fake"


def _extract_outbound_user_message(messages: list[Message]) -> Message:
    if len(messages) < 2:
        raise RuntimeError(
            "Provider did not receive expected outbound message sequence"
        )
    return messages[-1]


async def main() -> None:
    world = World()
    base_provider, model, mode = _build_provider_from_env()
    provider = RecordingProvider(base_provider)

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(role="user", content="Please @code summarize latest findings")
            ]
        ),
    )
    world.add_component(
        entity,
        UserPromptConfigComponent(
            triggers={
                "@code": "Prioritize deterministic code-first reasoning.",
                "event:tool_success": "Prefer successful tool outputs as evidence.",
            },
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity,
        OneShotContextPoolComponent(
            items=[
                (
                    30,
                    0,
                    "tool:search",
                    "source: tool:search\nstatus: success\nresult: citation-A",
                ),
                (
                    20,
                    1,
                    "subagent:researcher",
                    "source: subagent:researcher\nstatus: success\nresult: synthesis-B",
                ),
            ],
            _counter=2,
        ),
    )
    world.add_component(entity, TurnStateComponent(current_turn_id="demo-turn-1"))
    world.add_component(
        entity,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "You are a helpful coding assistant.\n\n"
                    "Available tools:\n"
                    "${_installed_tools}\n\n"
                    "Keep responses evidence-based and concise."
                )
            ),
        ),
    )
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={
                "demo_tool": ToolSchema(
                    name="demo_tool",
                    description="Demo capability wiring check",
                    parameters={"type": "object", "properties": {}},
                )
            },
            handlers={},
        ),
    )
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    outbound_user = _extract_outbound_user_message(provider.last_messages)
    rendered_system = world.get_component(entity, RenderedSystemPromptComponent)
    rendered_user = world.get_component(entity, RenderedUserPromptComponent)
    user_tail = "Please @code summarize latest findings"

    print(f"[mode] {mode}")
    print("\n=== Rendered System Prompt ===")
    print(rendered_system.text if rendered_system is not None else "<missing>")
    print("\n=== Outbound User Message (Injected) ===")
    print(outbound_user.content)
    print("\n=== Rendered User Prompt Component ===")
    print(rendered_user.text if rendered_user is not None else "<missing>")
    print("\n=== Rendered Component Presence ===")
    print(f"rendered system present: {rendered_system is not None}")
    print(f"rendered user present: {rendered_user is not None}")
    print("\n=== Marker Checks ===")
    print(f"[PROMPT_INJECT:@code]: {'[PROMPT_INJECT:@code]' in outbound_user.content}")
    print(f"[PROMPT_CONTEXT_POOL]: {'[PROMPT_CONTEXT_POOL]' in outbound_user.content}")
    print(f"user tail preserved: {outbound_user.content.endswith(user_tail)}")


if __name__ == "__main__":
    asyncio.run(main())
