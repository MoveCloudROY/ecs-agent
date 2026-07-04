"""Live tests for conversation compaction behavior.

Run:
    LLM_API_KEY=your-api-key \\
        uv run pytest tests/live/test_compaction_live.py -v

Optional overrides (default: Aliyun DashScope + qwen3.5-flash):
    LLM_BASE_URL            chat-completions base URL
    LLM_RESPONSES_BASE_URL  Responses API base URL (falls back to LLM_BASE_URL)
    LLM_MODEL               model identifier
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    CurrentCompactionSummaryComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
)
from ecs_agent.components.definitions import EntityRegistryComponent
from ecs_agent.core import World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import CompletionResult, Message

_registry_module = pytest.importorskip("ecs_agent.providers.registry")
ProviderRegistry = _registry_module.ProviderRegistry
get_model = _registry_module.get_model

COMPLETIONS_URL = os.getenv(
    "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
RESPONSES_URL = os.getenv(
    "LLM_RESPONSES_BASE_URL",
    os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    ),
)
MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")


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


def _make_manual_chat_provider(api_key: str) -> OpenAIModel:
    return OpenAIModel(
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


def _build_compaction_world(model: object) -> tuple[World, EntityId]:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=_build_python_conversation()),
    )
    world.add_component(entity_id, ConversationArchiveComponent())
    return world, entity_id


@pytest.mark.asyncio
async def test_live_compaction_xml_summary_visible_chat_completions(
    live_api_key: str,
) -> None:
    model = get_model(
        f"aliyun/{MODEL}",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
            
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=4000),
    )
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(
            summary="The secret codename for this session is: orchid-47"
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="You are a helpful assistant.")
        ),
    )

    render_system = SystemPromptRenderSystem()
    await render_system.process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "<chat_history_summary>" in rendered.text
    assert "orchid-47" in rendered.text

    result = await model.complete(
        [
            Message(role="system", content=rendered.text),
            Message(role="user", content="What is the secret codename?"),
        ]
    )

    assert isinstance(result, CompletionResult)
    assert "orchid-47" in result.message.content


@pytest.mark.asyncio
async def test_live_compaction_xml_summary_visible_responses_api(
    live_api_key: str,
) -> None:
    model = get_model(
        f"aliyun-responses/{MODEL}",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
            
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=4000),
    )
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(
            summary="The secret codename for this session is: orchid-47"
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="You are a helpful assistant.")
        ),
    )

    render_system = SystemPromptRenderSystem()
    await render_system.process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None

    result = await model.complete(
        [
            Message(role="system", content=rendered.text),
            Message(role="user", content="What is the secret codename?"),
        ]
    )

    assert isinstance(result, CompletionResult)
    assert "orchid-47" in result.message.content


@pytest.mark.asyncio
async def test_live_compaction_system_triggers_and_summarizes_chat(
    live_api_key: str,
) -> None:
    model = get_model(
        f"aliyun/{MODEL}",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(model)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30, compaction_method="full_history"
        ),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    archive = world.get_component(entity_id, ConversationArchiveComponent)
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)

    assert conv is not None
    assert archive is not None
    assert current_summary is not None
    assert len(current_summary.summary) > 0
    assert len(archive.archived_summaries) == 1
    assert archive.archived_summaries[0] == current_summary.summary
    assert len(conv.messages) < 20
    assert all(message.role != "compaction" for message in conv.messages)


@pytest.mark.asyncio
async def test_live_compaction_system_triggers_and_summarizes_responses(
    live_api_key: str,
) -> None:
    model = get_model(
        f"aliyun-responses/{MODEL}",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(model)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30, compaction_method="full_history"
        ),
    )

    system = CompactionSystem()
    await system.process(world)

    conv = world.get_component(entity_id, ConversationComponent)
    archive = world.get_component(entity_id, ConversationArchiveComponent)
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)

    assert conv is not None
    assert archive is not None
    assert current_summary is not None
    assert len(current_summary.summary) > 0
    assert len(archive.archived_summaries) == 1
    assert archive.archived_summaries[0] == current_summary.summary
    assert len(conv.messages) < 20
    assert all(message.role != "compaction" for message in conv.messages)


@pytest.mark.asyncio
async def test_live_compaction_full_history_method(live_api_key: str) -> None:
    model = get_model(
        f"aliyun/{MODEL}",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(model)
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
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)

    assert conv is not None
    assert current_summary is not None
    assert len(current_summary.summary) > 0
    # full_history retains the system message plus the last-user continuation
    # anchor (869269b) — nothing else survives.
    assert [message.role for message in conv.messages] == ["system", "user"]
    assert all(message.role != "compaction" for message in conv.messages)


@pytest.mark.asyncio
async def test_live_compaction_custom_prompt_template(live_api_key: str) -> None:
    model = get_model(
        f"aliyun/{MODEL}",
        registry=_live_registry(),
        api_key=live_api_key,
    )
    world, entity_id = _build_compaction_world(model)
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
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)

    assert conv is not None
    assert archive is not None
    assert current_summary is not None
    assert len(archive.archived_summaries) == 1
    assert len(archive.archived_summaries[0]) > 0
    assert archive.archived_summaries[0] == current_summary.summary


@pytest.mark.asyncio
async def test_live_compaction_summary_model_id_routing(live_api_key: str) -> None:
    model = _make_manual_chat_provider(live_api_key)
    world, entity_id = _build_compaction_world(model)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30,
            summary_model_id=f"aliyun/{MODEL}",
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
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)

    assert conv is not None
    assert archive is not None
    assert current_summary is not None
    assert len(archive.archived_summaries) == 1
    assert archive.archived_summaries[0] == current_summary.summary


@pytest.mark.asyncio
async def test_live_memory_system_preserves_post_compaction_turns(
    live_api_key: str,
) -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        CurrentCompactionSummaryComponent(
            summary="Previous conversation summary: We discussed A."
        ),
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            max_messages=5,
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="post-compact1"),
                Message(role="assistant", content="post-compact2"),
                Message(role="user", content="post-compact3"),
                Message(role="assistant", content="post-compact4"),
                Message(role="user", content="post-compact5"),
                Message(role="assistant", content="post-compact6"),
            ],
        ),
    )

    # Post-compaction truncation no longer applies

    conv = world.get_component(entity_id, ConversationComponent)

    assert conv is not None
    assert len(conv.messages) == 7
    assert conv.messages[0].content == "You are helpful."
    assert conv.messages[-1].content == "post-compact6"
