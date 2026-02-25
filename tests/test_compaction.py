from __future__ import annotations

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.types import (
    CompactionCompleteEvent,
    CompletionResult,
    Message,
    ToolSchema,
)


class RecordingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None, bool]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult:
        _ = response_format
        self.calls.append((list(messages), tools, stream))
        result = await super().complete(messages, tools=tools, stream=stream)
        assert isinstance(result, CompletionResult)
        return result


def _message(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


@pytest.mark.asyncio
async def test_compaction_triggers_when_threshold_exceeded() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="brief"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[_message("alpha beta gamma delta") for _ in range(6)]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=20, summary_model="summary-model"),
    )
    world.add_component(entity_id, ConversationArchiveComponent())

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert len(provider.calls) == 1
    assert len(conversation.messages) == 4
    assert conversation.messages[0].content.startswith("Previous conversation summary:")


@pytest.mark.asyncio
async def test_compaction_bisects_messages_and_keeps_recent_half() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="older summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    original = [
        _message("old-0"),
        _message("old-1"),
        _message("new-2"),
        _message("new-3"),
        _message("new-4"),
    ]
    world.add_component(entity_id, ConversationComponent(messages=list(original)))
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [message.content for message in conversation.messages[1:]] == [
        "new-2",
        "new-3",
        "new-4",
    ]


@pytest.mark.asyncio
async def test_compaction_calls_llm_with_expected_summarization_prompt() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="s"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="base-model"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("context-one"),
                _message("context-two"),
                _message("context-three"),
                _message("context-four"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )

    await CompactionSystem().process(world)

    assert len(provider.calls) == 1
    sent_messages, sent_tools, sent_stream = provider.calls[0]
    assert sent_tools is None
    assert sent_stream is False
    assert len(sent_messages) == 1
    assert sent_messages[0].role == "user"
    assert sent_messages[0].content.startswith(
        "Summarize the following conversation, preserving key decisions, facts, and context:"
    )
    assert "context-one" in sent_messages[0].content
    assert "context-two" in sent_messages[0].content


@pytest.mark.asyncio
async def test_summary_is_stored_in_archive_component() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="saved summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[_message("one"), _message("two"), _message("three")]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )
    world.add_component(entity_id, ConversationArchiveComponent())

    await CompactionSystem().process(world)

    archive = world.get_component(entity_id, ConversationArchiveComponent)
    assert archive is not None
    assert archive.archived_summaries == ["saved summary"]


@pytest.mark.asyncio
async def test_compaction_publishes_event_with_original_and_compacted_token_counts() -> (
    None
):
    world = World()
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="s"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[_message("a b c d"), _message("e f g h")]),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )

    seen: list[CompactionCompleteEvent] = []

    async def on_compaction(event: CompactionCompleteEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(CompactionCompleteEvent, on_compaction)

    await CompactionSystem().process(world)

    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].original_tokens >= seen[0].compacted_tokens


@pytest.mark.asyncio
async def test_no_compaction_when_threshold_not_exceeded() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    original_messages = [_message("hello"), _message("world")]
    world.add_component(
        entity_id, ConversationComponent(messages=list(original_messages))
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1000, summary_model="summary-model"),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == original_messages
    assert provider.calls == []


@pytest.mark.asyncio
async def test_system_message_is_preserved_during_compaction() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("You are a strict assistant", role="system"),
                _message("old-a"),
                _message("old-b"),
                _message("new-c"),
                _message("new-d"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == "You are a strict assistant"
    assert conversation.messages[1].content.startswith("Previous conversation summary:")


@pytest.mark.asyncio
async def test_bisect_ratio_is_configurable() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    original_messages = [
        _message("m0"),
        _message("m1"),
        _message("m2"),
        _message("m3"),
        _message("m4"),
        _message("m5"),
    ]
    world.add_component(
        entity_id, ConversationComponent(messages=list(original_messages))
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )

    await CompactionSystem(bisect_ratio=0.25).process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [message.content for message in conversation.messages[1:]] == [
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
    ]
