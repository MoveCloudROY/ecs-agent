"""Live tests for conversation compaction behavior.

Select a profile with --live; see docs/testing.md.
"""

from __future__ import annotations


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
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import CompletionResult, EntityId, Message

from ecs_agent.providers.registry import ProviderRegistry, get_model

from tests.support.models import LiveModelFactory


def _live_registry(*, live_model_factory: LiveModelFactory) -> ProviderRegistry:
    return ProviderRegistry.from_dict(
        {
            "live": live_model_factory.registry_entry(),
        }
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


@pytest.mark.live(formats=["openai_chat_completions"], capabilities=["text"])
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_xml_summary_visible_chat_completions(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.track(
        get_model(
            f"live/{live_model_factory.profile.model}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
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


@pytest.mark.live(formats=["openai_responses"], capabilities=["text"])
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_xml_summary_visible_responses_api(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.track(
        get_model(
            f"live/{live_model_factory.profile.model}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
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


@pytest.mark.live(formats=["openai_chat_completions"], capabilities=["text"])
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_system_triggers_and_summarizes_chat(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.track(
        get_model(
            f"live/{live_model_factory.profile.model}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
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


@pytest.mark.live(formats=["openai_responses"], capabilities=["text"])
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_system_triggers_and_summarizes_responses(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.track(
        get_model(
            f"live/{live_model_factory.profile.model}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
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


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"], capabilities=["text"]
)
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_full_history_method(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.track(
        get_model(
            f"live/{live_model_factory.profile.model}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
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


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"], capabilities=["text"]
)
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_custom_prompt_template(
    live_api_key: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.track(
        get_model(
            f"live/{live_model_factory.profile.model}",
            registry=_live_registry(live_model_factory=live_model_factory),
            api_key=live_api_key,
        )
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


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"], capabilities=["text"]
)
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_compaction_summary_model_id_routing(
    *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.openai()
    world, entity_id = _build_compaction_world(model)
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=30,
            summary_model_id=f"live/{live_model_factory.profile.model}",
        ),
    )
    world.add_component(
        entity_id,
        EntityRegistryComponent(
            entity_id=entity_id,
            name="test-agent",
            metadata={
                "provider_registry": _live_registry(
                    live_model_factory=live_model_factory
                )
            },
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


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"], capabilities=["text"]
)
@pytest.mark.compaction
@pytest.mark.asyncio
async def test_live_memory_system_preserves_post_compaction_turns() -> None:
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
