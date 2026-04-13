"""Live tests for conversation compaction behavior.

Run:
    LLM_API_KEY=your-api-key \
        uv run pytest tests/live/test_compaction_live.py -v
"""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.components.definitions import EntityRegistryComponent
from ecs_agent.core import World
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.types import CompletionResult, Message

_registry_module = pytest.importorskip("ecs_agent.providers.registry")
ProviderRegistry = _registry_module.ProviderRegistry
get_llm_provider = _registry_module.get_llm_provider

COMPLETIONS_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
RESPONSES_URL = (
    "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
)
MODEL = "qwen3.5-flash"


def _live_registry() -> ProviderRegistry:
    return ProviderRegistry.from_dict(
        {
            "aliyun": {
                "base_url": COMPLETIONS_URL,
                "api_format": "openai_chat_completions",
            },
            "aliyun-responses": {
                "base_url": RESPONSES_URL,
                "api_format": "openai_responses",
            },
        }
    )


def _make_manual_chat_provider(api_key: str) -> OpenAIProvider:
    return OpenAIProvider(
        config=ProviderConfig(
            provider_id="aliyun",
            base_url=COMPLETIONS_URL,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ),
        model=MODEL,
    )


def _build_python_conversation() -> list[Message]:
    messages = [
        Message(
            role="system",
            content=(
                "You are a helpful Python tutor who explains syntax choices, debugging steps, "
                "and code organization clearly."
            ),
        )
    ]
    topics = [
        "list comprehensions for filtering and mapping cleanly in data processing tasks",
        "generator expressions for memory efficient iteration over large Python datasets",
        "dictionary merging strategies when combining configuration values from several sources",
        "exception handling patterns that keep tracebacks useful during debugging sessions",
        "writing small pure functions that are easy to test and reuse safely",
        "using pathlib for readable filesystem logic instead of fragile string concatenation",
        "type hints that clarify function contracts and improve editor assistance significantly",
        "asyncio coroutines for coordinating network calls without blocking program progress",
        "dataclasses for lightweight structured state in application components and services",
        "pytest assertions that describe behavior clearly and fail with actionable details",
        "refactoring nested conditionals into helper functions with explicit intent boundaries",
        "tool result summaries that preserve context without repeating every raw response line",
        "conversation state management across multiple turns while keeping relevant details concise",
        "breaking algorithms into steps so future maintenance remains understandable and safe",
        "using enums and literals when state transitions need stricter validation rules",
        "balancing readability and performance when processing moderate sized Python collections",
        "preserving user intent when summarizing prior reasoning or implementation decisions",
        "careful prompt design so summaries keep pending tasks and important outcomes intact",
        "troubleshooting API integration issues by checking payload shape and message ordering",
    ]
    for index, topic in enumerate(topics):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            Message(
                role=role,
                content=(
                    f"We are discussing Python programming details about {topic}, and we want "
                    "practical examples, tradeoffs, and next steps for implementation."
                ),
            )
        )
    return messages


def _build_compaction_world(provider: object) -> tuple[World, object]:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model=MODEL))
    world.add_component(
        entity_id,
        ConversationComponent(messages=_build_python_conversation()),
    )
    world.add_component(entity_id, ConversationArchiveComponent())
    return world, entity_id


def _get_compaction_message(messages: list[Message]) -> Message:
    return next(message for message in messages if message.role == "compaction")


@pytest.mark.asyncio
async def test_live_compaction_role_chat_completions(live_api_key: str) -> None:
    provider = get_llm_provider(
        "aliyun/qwen3.5-flash",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    result = await provider.complete(
        [
            Message(
                role="compaction",
                content=(
                    "Previous conversation summary: The user asked about Python. "
                    "We discussed list comprehensions."
                ),
            ),
            Message(role="user", content="Continue our discussion."),
        ]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.asyncio
async def test_live_compaction_role_responses_api(live_api_key: str) -> None:
    provider = get_llm_provider(
        "aliyun-responses/qwen3.5-flash",
        registry=_live_registry(),
        api_key=live_api_key,
    )

    result = await provider.complete(
        [
            Message(
                role="compaction",
                content=(
                    "Previous conversation summary: The user asked about Python. "
                    "We discussed list comprehensions."
                ),
            ),
            Message(role="user", content="Continue our discussion."),
        ]
    )

    assert isinstance(result, CompletionResult)
    assert len(result.message.content.strip()) > 0


@pytest.mark.asyncio
async def test_live_compaction_system_triggers_and_summarizes_chat(
    live_api_key: str,
) -> None:
    provider = get_llm_provider(
        "aliyun/qwen3.5-flash",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(provider)
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=30, compaction_method="bisect"),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    archive = world.get_component(entity_id, ConversationArchiveComponent)

    assert conv is not None
    assert archive is not None
    assert sum(message.role == "compaction" for message in conv.messages) == 1
    compaction_message = _get_compaction_message(conv.messages)
    assert compaction_message.content.startswith("Previous conversation summary:")
    assert len(archive.archived_summaries) == 1
    assert len(conv.messages) < 20


@pytest.mark.asyncio
async def test_live_compaction_system_triggers_and_summarizes_responses(
    live_api_key: str,
) -> None:
    provider = get_llm_provider(
        "aliyun-responses/qwen3.5-flash",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(provider)
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=30, compaction_method="bisect"),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    archive = world.get_component(entity_id, ConversationArchiveComponent)

    assert conv is not None
    assert archive is not None
    assert sum(message.role == "compaction" for message in conv.messages) == 1
    compaction_message = _get_compaction_message(conv.messages)
    assert compaction_message.content.startswith("Previous conversation summary:")
    assert len(archive.archived_summaries) == 1
    assert len(conv.messages) < 20


@pytest.mark.asyncio
async def test_live_compaction_full_history_method(live_api_key: str) -> None:
    provider = get_llm_provider(
        "aliyun/qwen3.5-flash",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(provider)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30,
            compaction_method="full_history",
        ),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)

    assert conv is not None
    assert len(conv.messages) <= 2
    assert any(message.role == "compaction" for message in conv.messages)
    assert _get_compaction_message(conv.messages).content.strip()


@pytest.mark.asyncio
async def test_live_compaction_custom_prompt_template(live_api_key: str) -> None:
    provider = get_llm_provider(
        "aliyun/qwen3.5-flash",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(provider)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30,
            compaction_prompt_template="Summarize the key points briefly.",
        ),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    archive = world.get_component(entity_id, ConversationArchiveComponent)

    assert conv is not None
    assert archive is not None
    assert any(message.role == "compaction" for message in conv.messages)
    assert len(archive.archived_summaries) == 1
    assert len(archive.archived_summaries[0]) > 0


@pytest.mark.asyncio
async def test_live_compaction_summary_model_id_routing(live_api_key: str) -> None:
    provider = _make_manual_chat_provider(live_api_key)
    world, entity_id = _build_compaction_world(provider)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30,
            summary_model_id="aliyun/qwen3.5-flash",
        ),
    )
    world.add_component(
        entity_id,
        EntityRegistryComponent(
            entity_id=entity_id,
            name="test-agent",
            metadata={"provider_registry": _live_registry()},
        ),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    archive = world.get_component(entity_id, ConversationArchiveComponent)

    assert conv is not None
    assert archive is not None
    assert any(message.role == "compaction" for message in conv.messages)
    assert len(archive.archived_summaries) == 1


@pytest.mark.asyncio
async def test_live_memory_system_preserves_compaction_boundary(
    live_api_key: str,
) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            max_messages=5,
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="old1"),
                Message(role="assistant", content="old2"),
                Message(
                    role="compaction",
                    content="Previous conversation summary: We discussed A.",
                ),
                Message(role="user", content="post-compact1"),
                Message(role="assistant", content="post-compact2"),
                Message(role="user", content="post-compact3"),
                Message(role="assistant", content="post-compact4"),
            ],
        ),
    )

    memory_system = MemorySystem()
    await memory_system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)

    assert conv is not None
    assert conv.messages[0].role == "system"
    assert any(message.role == "compaction" for message in conv.messages)
    assert conv.messages[1:] == [
        Message(
            role="compaction",
            content="Previous conversation summary: We discussed A.",
        ),
        Message(role="user", content="post-compact1"),
        Message(role="assistant", content="post-compact2"),
        Message(role="user", content="post-compact3"),
        Message(role="assistant", content="post-compact4"),
    ]
    assert all(message.content != "old1" for message in conv.messages)
    assert all(message.content != "old2" for message in conv.messages)
