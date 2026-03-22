import asyncio
import os
from typing import Any
from collections.abc import AsyncIterator

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PromptConfigComponent,
    SystemPromptComponent,
    TurnStateComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
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
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
        return provider, model, "real"

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


def _extract_outbound_messages(messages: list[Message]) -> tuple[Message, Message]:
    if len(messages) < 2:
        raise RuntimeError(
            "Provider did not receive expected outbound message sequence"
        )
    return messages[0], messages[-1]


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
        PromptConfigComponent(
            trigger_templates={
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
        SystemPromptComponent(
            content=(
                "# Markdown Linked Prompt\n\n"
                "## toolSelection\n\n"
                "Prefer deterministic tools and concise synthesis.\n\n"
                "## exploreSection\n\n"
                "Surface concrete evidence from context entries first.\n\n"
                "## librarianSection\n\n"
                "Preserve exact references in final responses."
            ),
        ),
    )
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    outbound_system, outbound_user = _extract_outbound_messages(provider.last_messages)
    user_tail = "Please @code summarize latest findings"

    print(f"[mode] {mode}")
    print("\n=== Assembled System Prompt ===")
    print(outbound_system.content)
    print("\n=== Outbound User Message (Injected) ===")
    print(outbound_user.content)
    print("\n=== Marker Checks ===")
    print(f"[PROMPT_INJECT:@code]: {'[PROMPT_INJECT:@code]' in outbound_user.content}")
    print(f"[PROMPT_CONTEXT_POOL]: {'[PROMPT_CONTEXT_POOL]' in outbound_user.content}")
    print(f"user tail preserved: {outbound_user.content.endswith(user_tail)}")


if __name__ == "__main__":
    asyncio.run(main())
