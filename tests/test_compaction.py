from __future__ import annotations

import hashlib

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ContextTrimConfig,
    CurrentCompactionSummaryComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
    RenderedUserPromptComponent,
    RenderedSystemPromptComponent,
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
    SystemPromptComponent,
    TerminalComponent,
    TokenUsageComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.observability import RecordingTelemetrySink, install_observability
from ecs_agent.serialization import WorldSerializer
from ecs_agent.providers import FakeModel
from ecs_agent.providers.registry import ProviderRegistry
from ecs_agent.systems.compaction import DEFAULT_COMPACTION_PROMPT, CompactionSystem
import ecs_agent.systems.compaction as compaction_module
from ecs_agent.systems.system_prompt_render_system import render_compaction_prompt
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
)
from ecs_agent.prompts.message_assembly import trim_context_to_fit
from ecs_agent.types import (
    CompactionCompleteEvent,
    CompletionResult,
    EntityId,
    Message,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    ToolCall,
    ToolSchema,
)
from ecs_agent.components.definitions import ContextEntry


class RecordingFakeModel(FakeModel):
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


class RegistryBackedRecordingFakeModel(RecordingFakeModel):
    def __init__(
        self,
        responses: list[CompletionResult],
        registry: ProviderRegistry,
    ) -> None:
        super().__init__(responses=responses)
        self.registry = registry


class TerminatingSystem:
    """Stops runner-driven compaction observability tests after one tick."""

    async def process(self, world: World) -> None:
        """Attach a terminal component."""
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


def _message(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


def _fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _subagent_session(
    session_id: str,
    *,
    status: str,
    parent_entity_id: EntityId,
) -> SubagentSessionRecord:
    return SubagentSessionRecord(
        session_id=session_id,
        category="research",
        prompt="Gather context",
        parent_entity_id=parent_entity_id,
        created_at="2026-04-01T10:00:00Z",
        updated_at="2026-04-01T10:05:00Z",
        status=status,
    )


def test_legacy_compaction_message_restores_current_summary_state() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="hello"),
                Message(
                    role="compaction",
                    content="Previous conversation summary: compacted state",
                ),
                Message(
                    role="compaction",
                    content="Previous conversation summary: newer compacted state",
                ),
                Message(role="assistant", content="latest reply"),
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})
    restored_conversation = restored.get_component(entity_id, ConversationComponent)
    current_summary = restored.get_component(
        entity_id, CurrentCompactionSummaryComponent
    )

    assert restored_conversation is not None
    assert current_summary == CurrentCompactionSummaryComponent(
        summary="newer compacted state"
    )
    assert restored_conversation.messages == [
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="latest reply",
        ),
    ]


def test_reserialized_world_does_not_emit_compaction_role_messages() -> None:
    legacy_data = {
        "next_entity_id": 2,
        "entities": {
            "1": {
                "ConversationComponent": {
                    "messages": [
                        {
                            "role": "compaction",
                            "content": "Previous conversation summary: compacted state",
                            "tool_calls": None,
                            "tool_call_id": None,
                        },
                        {
                            "role": "user",
                            "content": "hello",
                            "tool_calls": None,
                            "tool_call_id": None,
                        },
                    ]
                }
            }
        },
        "_entity_registry": {},
        "_entity_tags": {},
        "world_name": None,
    }

    restored = WorldSerializer.from_dict(legacy_data, providers={}, tool_handlers={})

    reserialized = WorldSerializer.to_dict(restored)
    messages = reserialized["entities"]["1"]["ConversationComponent"]["messages"]

    assert all(message["role"] != "compaction" for message in messages)
    assert reserialized["entities"]["1"]["CurrentCompactionSummaryComponent"] == {
        "summary": "compacted state",
        "metadata": None,
    }


def test_current_compaction_summary_component_can_be_stored_independently() -> None:
    world = World()
    entity_id = world.create_entity()
    component = CurrentCompactionSummaryComponent(summary="plain-text summary")

    world.add_component(entity_id, component)

    assert (
        world.get_component(entity_id, CurrentCompactionSummaryComponent) == component
    )


@pytest.mark.asyncio
async def test_compaction_triggers_when_threshold_exceeded() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="brief"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert conversation is not None
    assert len(model.calls) == 1
    assert conversation.messages == [Message(role="user", content="alpha beta gamma delta")]
    assert current_summary == CurrentCompactionSummaryComponent(summary="brief")


@pytest.mark.asyncio
async def test_compaction_full_history_method_preserves_system_and_user_anchor() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="full summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("You are a strict assistant", role="system"),
                _message("old-0"),
                _message("old-1"),
                _message("new-2"),
                _message("new-3"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            compaction_method="full_history",
        ),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert conversation is not None
    assert [(message.role, message.content) for message in conversation.messages] == [
        ("system", "You are a strict assistant"),
        ("user", "new-3"),
    ]
    assert current_summary == CurrentCompactionSummaryComponent(summary="full summary")

    sent_messages, _, _ = model.calls[0]
    assert sent_messages[1].role == "user"
    assert "user: old-0" in sent_messages[1].content
    assert "user: old-1" in sent_messages[1].content
    assert "user: new-2" in sent_messages[1].content
    assert "user: new-3" in sent_messages[1].content


@pytest.mark.asyncio
async def test_compaction_predrop_then_compact_uses_budgeted_view_without_mutating_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="budgeted summary")
            )
        ]
    )
    entity_id = world.create_entity()
    original_messages = [
        _message("You are a strict assistant", role="system"),
        Message(
            role="assistant",
            content="tool call",
            tool_calls=[ToolSchema(name="calc", description="", parameters={})],
        ),
        Message(role="tool", content="tool result", tool_call_id="call-1"),
        _message("keep me"),
    ]
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id, ConversationComponent(messages=list(original_messages))
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            compaction_method="predrop_then_compact",
        ),
    )

    seen: dict[str, object] = {}

    def fake_trim_context_to_fit(
        messages: list[Message],
        system_prompt: str,
        context_entries: list[object],
        config: ContextTrimConfig,
    ) -> list[Message]:
        seen["messages"] = list(messages)
        seen["system_prompt"] = system_prompt
        seen["context_entries"] = list(context_entries)
        seen["config"] = config
        return [messages[-1]]

    monkeypatch.setattr(
        compaction_module,
        "trim_context_to_fit",
        fake_trim_context_to_fit,
    )

    await CompactionSystem().process(world)

    assert seen["system_prompt"] == ""
    assert seen["context_entries"] == []
    budget_config = seen["config"]
    assert isinstance(budget_config, ContextTrimConfig)
    assert budget_config.trim_tool_results is True

    summarized_input = model.calls[0][0][1].content
    assert "assistant: tool call" not in summarized_input
    assert "tool: tool result" not in summarized_input
    assert "user: keep me" in summarized_input

    conversation = world.get_component(entity_id, ConversationComponent)
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="system", content="You are a strict assistant"),
        Message(role="user", content="keep me"),
    ]
    assert current_summary == CurrentCompactionSummaryComponent(
        summary="budgeted summary"
    )
    assert original_messages[1].tool_calls is not None


@pytest.mark.asyncio
async def test_compaction_preserves_last_user_anchor_without_system_message() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("old request"),
                _message("old answer", role="assistant"),
                _message("continue the active task"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            compaction_method="predrop_then_compact",
        ),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="user", content="continue the active task")
    ]


@pytest.mark.asyncio
async def test_compaction_prefers_rendered_user_prompt_for_retained_anchor() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity_id = world.create_entity()
    raw_resume_command = "/task:resume web-novel-writing-assistant"
    rendered_resume_prompt = "Task resumed: continue T03-work-management"
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("older context"),
                _message("older response", role="assistant"),
                _message(raw_resume_command),
            ]
        ),
    )
    world.add_component(
        entity_id,
        RenderedUserPromptComponent(
            text=rendered_resume_prompt,
            source_fingerprint=_fingerprint(raw_resume_command),
            source_message_index=2,
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            compaction_method="predrop_then_compact",
        ),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="user", content=rendered_resume_prompt)
    ]


@pytest.mark.asyncio
async def test_compaction_ignores_stale_rendered_user_prompt_for_anchor() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("older command"),
                _message("older response", role="assistant"),
                _message("continue with current task"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        RenderedUserPromptComponent(
            text="stale rendered command result",
            source_fingerprint=_fingerprint("older command"),
            source_message_index=0,
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            compaction_method="predrop_then_compact",
        ),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="user", content="continue with current task")
    ]


@pytest.mark.asyncio
async def test_compaction_summary_input_includes_canonical_subagent_session_states() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="subagent-aware summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("conversation text claims sess-pending succeeded"),
                _message("recent user turn"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, compaction_method="full_history"),
    )
    world.add_component(
        entity_id,
        SubagentSessionTableComponent(
            sessions={
                "sess-pending": _subagent_session(
                    "sess-pending",
                    status="running",
                    parent_entity_id=entity_id,
                ),
                "sess-failed": _subagent_session(
                    "sess-failed",
                    status="failed",
                    parent_entity_id=entity_id,
                ),
                "sess-cancelled": _subagent_session(
                    "sess-cancelled",
                    status="cancelled",
                    parent_entity_id=entity_id,
                ),
            }
        ),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="sess-failed:failed",
                    session_id="sess-failed",
                    parent_entity_id=int(entity_id),
                    terminal_status="failed",
                    summary=None,
                    error="boom",
                    created_at="2026-04-01T10:06:00Z",
                    delivered_at=None,
                )
            ]
        ),
    )

    await CompactionSystem().process(world)

    summary_input = model.calls[0][0][1].content
    assert "conversation text claims sess-pending succeeded" in summary_input
    assert "Subagent session state:" in summary_input
    assert "Pending: sess-pending" in summary_input
    assert "Completed (failed): sess-failed" in summary_input
    assert "Completed (cancelled): sess-cancelled" in summary_input
    assert "notification status=failed" in summary_input


@pytest.mark.asyncio
async def test_compaction_calls_llm_with_expected_summarization_prompt() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="s"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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

    assert len(model.calls) == 1
    sent_messages, sent_tools, sent_stream = model.calls[0]
    assert sent_tools is None
    assert sent_stream is False
    assert len(sent_messages) == 2
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == DEFAULT_COMPACTION_PROMPT
    assert sent_messages[1].role == "user"
    assert "user: context-one" in sent_messages[1].content
    assert "user: context-two" in sent_messages[1].content


@pytest.mark.asyncio
async def test_summary_is_stored_in_archive_component() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="saved summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert archive is not None
    assert archive.archived_summaries == ["saved summary"]
    assert current_summary == CurrentCompactionSummaryComponent(summary="saved summary")


@pytest.mark.asyncio
async def test_compaction_publishes_event_with_original_and_compacted_token_counts() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="s"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    assert model.calls == []


@pytest.mark.asyncio
async def test_noop_compaction_run_emits_no_compaction_tracing() -> None:
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[_message("hello"), _message("world")]),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1000, summary_model="summary-model"),
    )
    world.register_system(CompactionSystem(), priority=-30)
    world.register_system(TerminatingSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    assert model.calls == []
    assert not any(record.name == "compaction.complete" for record in sink.records)
    assert not any(record.name.endswith("CompactionSystem") for record in sink.records)


@pytest.mark.asyncio
async def test_compaction_run_emits_only_compaction_complete_tracing() -> None:
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="summary"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[_message("a b c d"), _message("e f g h")]),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )
    world.register_system(CompactionSystem(), priority=-30)
    world.register_system(TerminatingSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    compaction_records = [record for record in sink.records if record.name == "compaction.complete"]
    assert len(compaction_records) == 1
    assert compaction_records[0].kind == "event"
    assert compaction_records[0].entity_id == int(entity_id)
    assert compaction_records[0].metadata is not None
    assert compaction_records[0].metadata["original_tokens"] >= compaction_records[0].metadata["compacted_tokens"]
    assert not any(record.name.endswith("CompactionSystem") for record in sink.records)


@pytest.mark.asyncio
async def test_system_message_is_preserved_during_compaction() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
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
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert conversation is not None
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == "You are a strict assistant"
    assert all(message.role != "compaction" for message in conversation.messages)
    assert current_summary == CurrentCompactionSummaryComponent(summary="summary")


@pytest.mark.asyncio
async def test_compaction_updates_current_summary_and_clears_rendered_prompt_cache() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="state-backed summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[_message("first"), _message("second"), _message("third")]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )
    world.add_component(
        entity_id,
        RenderedSystemPromptComponent(
            text="cached runtime prompt",
            placeholder_snapshot={"_cache_key": "runtime-cache"},
        ),
    )

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)

    assert conversation is not None
    assert all(message.role != "compaction" for message in conversation.messages)
    assert current_summary == CurrentCompactionSummaryComponent(
        summary="state-backed summary"
    )
    assert world.get_component(entity_id, RenderedSystemPromptComponent) is None


@pytest.mark.asyncio
async def test_repeated_compaction_folds_previous_current_summary_into_next_summary_input() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="first summary")
            ),
            CompletionResult(
                message=Message(role="assistant", content="second summary")
            ),
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("old-0"),
                _message("old-1"),
                _message("new-2"),
                _message("new-3"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, summary_model="summary-model"),
    )
    world.add_component(entity_id, ConversationArchiveComponent())

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.extend([_message("latest-4"), _message("latest-5")])

    await CompactionSystem().process(world)

    archive = world.get_component(entity_id, ConversationArchiveComponent)
    current_summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    second_summary_input = model.calls[1][0][1].content
    previous_summary_index = second_summary_input.index(
        "user: Previous summary:\n\nfirst summary"
    )
    retained_message_index = second_summary_input.index("user: latest-4")

    assert previous_summary_index < retained_message_index
    assert "Conversation to summarize:" in second_summary_input
    assert archive is not None
    assert archive.archived_summaries == ["first summary", "second summary"]
    assert current_summary == CurrentCompactionSummaryComponent(
        summary="second summary"
    )


def test_compaction_prompt_render_does_not_mutate_runtime_state() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Summary for ${topic}"),
            placeholders=[PlaceholderSpec(name="topic", value="compaction")],
        ),
    )
    world.add_component(
        entity_id,
        RenderedSystemPromptComponent(
            text="cached runtime prompt",
            placeholder_snapshot={"_cache_key": "runtime-cache"},
        ),
    )
    world.add_component(
        entity_id,
        LLMComponent(
            model=FakeModel(responses=[]),
            system_prompt="runtime llm prompt",
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptComponent(content="runtime legacy prompt"),
    )

    rendered = render_compaction_prompt(
        template="Summary for ${topic}",
        world=world,
        entity=entity_id,
    )

    runtime_cache = world.get_component(entity_id, RenderedSystemPromptComponent)
    llm = world.get_component(entity_id, LLMComponent)
    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)
    config = world.get_component(entity_id, SystemPromptConfigSpec)
    assert rendered == "Summary for compaction"
    assert runtime_cache is not None
    assert runtime_cache.text == "cached runtime prompt"
    assert runtime_cache.placeholder_snapshot == {"_cache_key": "runtime-cache"}
    assert llm is not None
    assert llm.system_prompt == "runtime llm prompt"
    assert legacy_prompt is not None
    assert legacy_prompt.content == "runtime legacy prompt"
    assert config is not None
    assert config.template_source.inline == "Summary for ${topic}"


def test_compaction_prompt_render_does_not_pollute_runtime_state() -> None:
    test_compaction_prompt_render_does_not_mutate_runtime_state()


@pytest.mark.asyncio
async def test_compaction_uses_summary_model_id_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    registry = ProviderRegistry.from_dict(
        {
            "fake": {
                "base_url": "https://example.invalid/v1",
                "api_format": "openai_chat_completions",
                "api_key": "test-key",
            }
        }
    )
    primary_provider = RegistryBackedRecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ],
        registry=registry,
    )
    summary_provider = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="model-id summary")
            )
        ]
    )

    def fake_get_model(
        model_id: str,
        *,
        registry: ProviderRegistry,
        api_key: str | None = None,
    ) -> RecordingFakeModel:
        assert model_id == "fake/test-model"
        assert registry is primary_provider.registry
        assert api_key is None
        return summary_provider

    monkeypatch.setattr(compaction_module, "get_model", fake_get_model)

    entity_id = world.create_entity()
    world.add_component(
        entity_id, LLMComponent(model=primary_provider)
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("first message content"),
                _message("second message content"),
                _message("third message content"),
                _message("fourth message content"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            summary_model_id="fake/test-model",
            summary_model="legacy-model",
        ),
    )

    await CompactionSystem().process(world)

    assert primary_provider.calls == []
    assert len(summary_provider.calls) == 1


@pytest.mark.asyncio
async def test_compaction_falls_back_to_legacy_summary_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="legacy summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("first message content"),
                _message("second message content"),
                _message("third message content"),
                _message("fourth message content"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            summary_model="legacy-model",
        ),
    )

    warnings: list[tuple[str, dict[str, object]]] = []

    def record_warning(event: str, **kwargs: object) -> None:
        warnings.append((event, kwargs))

    monkeypatch.setattr(compaction_module.logger, "warning", record_warning)

    await CompactionSystem().process(world)

    assert len(model.calls) == 1
    assert model.calls[0][0][0].content == DEFAULT_COMPACTION_PROMPT


def test_overflow_pruning_logs_observable_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, dict[str, object]]] = []

    def record_info(event_name: str, **kwargs: object) -> None:
        observed.append((event_name, kwargs))

    monkeypatch.setattr("ecs_agent.prompts.message_assembly.logger.info", record_info)

    messages = [
        Message(
            role="user",
            content=(
                "Question\n\n[PROMPT_CONTEXT_POOL]\n"
                "tool output that should be pruned\n\n---\n\nkeep this context"
            ),
        )
    ]
    context_entries = [
        ContextEntry(
            entry_id="tool-entry",
            priority=10,
            source_label="tool:search",
            content="tool output that should be pruned",
            registration_order=0,
            droppable_kind="tool_result",
        ),
        ContextEntry(
            entry_id="keep-entry",
            priority=5,
            source_label="memory",
            content="keep this context",
            registration_order=1,
        ),
    ]

    reduced = trim_context_to_fit(
        messages,
        system_prompt="",
        context_entries=context_entries,
        config=ContextTrimConfig(
            max_tokens=40,
            token_estimation_chars_per_token=1.0,
            overflow_behavior="warn",
        ),
    )

    assert "tool output that should be pruned" not in reduced[0].content
    assert observed == [
        (
            "context_entry_pruned",
            {
                "reason": "tool_result",
                "entry_id": "tool-entry",
                "source_label": "tool:search",
            },
        )
    ]


@pytest.mark.asyncio
async def test_compaction_uses_default_prompt_when_no_template() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="default prompt summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("first message content"),
                _message("second message content"),
                _message("third message content"),
                _message("fourth message content"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1, compaction_prompt_template=None),
    )

    await CompactionSystem().process(world)

    assert len(model.calls) == 1
    sent_messages, _, _ = model.calls[0]
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == DEFAULT_COMPACTION_PROMPT


@pytest.mark.asyncio
async def test_compaction_renders_custom_prompt_template() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="custom prompt summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="Agent topic: ${topic}"),
            placeholders=[PlaceholderSpec(name="topic", value="compaction")],
        ),
    )
    world.add_component(
        entity_id,
        RenderedSystemPromptComponent(
            text="cached runtime prompt",
            placeholder_snapshot={"_cache_key": "runtime-cache"},
        ),
    )
    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
            system_prompt="runtime llm prompt",
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptComponent(content="runtime legacy prompt"),
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("first message content"),
                _message("second message content"),
                _message("third message content"),
                _message("fourth message content"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(
            threshold_tokens=1,
            compaction_prompt_template="Summary for ${topic}",
        ),
    )

    await CompactionSystem().process(world)

    sent_messages, _, _ = model.calls[0]
    llm = world.get_component(entity_id, LLMComponent)
    runtime_cache = world.get_component(entity_id, RenderedSystemPromptComponent)
    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)

    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == "Summary for compaction"
    assert llm is not None
    assert llm.system_prompt == "runtime llm prompt"
    assert runtime_cache is None
    assert legacy_prompt is not None
    assert legacy_prompt.content == "runtime legacy prompt"


def test_compaction_config_component_accepts_legacy_summary_model_string() -> None:
    component = CompactionConfigComponent(
        threshold_tokens=1000,
        summary_model="gpt-4o-mini",
    )

    assert component.summary_model == "gpt-4o-mini"


def test_compaction_config_component_new_fields_have_expected_defaults() -> None:
    component = CompactionConfigComponent(
        threshold_tokens=1000,
        summary_model="gpt-4o-mini",
    )

    assert component.compaction_method == "full_history"
    assert component.summary_model_id is None
    assert component.compaction_prompt_template is None


@pytest.mark.asyncio
async def test_compaction_calibrates_on_real_prompt_tokens() -> None:
    """Real last_prompt_tokens triggers compaction even when the local estimate
    of the (small) conversation content is well below the threshold."""
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="brief"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    # Small conversation: a pure local estimate is far below threshold=1000.
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[_message("a"), _message("b", role="assistant"), _message("c")]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1000, summary_model="summary-model"),
    )
    world.add_component(entity_id, ConversationArchiveComponent())
    # Ground truth: the last call actually used 5000 input tokens (huge system +
    # tools), with a single message as its basis.
    world.add_component(
        entity_id,
        TokenUsageComponent(
            last_prompt_tokens=5000,
            call_count=1,
            last_prompt_message_count=1,
        ),
    )

    # Sanity: without calibration the local estimate would NOT trigger.
    assert CompactionSystem()._estimate_tokens(
        world.get_component(entity_id, ConversationComponent).messages
    ) <= 1000

    await CompactionSystem().process(world)

    # Calibration (5000 + delta) crossed the threshold -> compaction ran.
    summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert summary == CurrentCompactionSummaryComponent(summary="brief")
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_compaction_falls_back_when_anchor_exceeds_conversation() -> None:
    """A stale anchor (e.g. after the conversation shrank) is ignored and the
    pure local estimate is used, so a small conversation does not compact."""
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="brief"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[_message("a"), _message("b", role="assistant"), _message("c")]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1000, summary_model="summary-model"),
    )
    world.add_component(entity_id, ConversationArchiveComponent())
    # Anchor points past the current conversation length (stale) -> ignored.
    world.add_component(
        entity_id,
        TokenUsageComponent(
            last_prompt_tokens=5000,
            call_count=1,
            last_prompt_message_count=99,
        ),
    )

    await CompactionSystem().process(world)

    # Fell back to the local estimate (small) -> no compaction.
    assert world.get_component(entity_id, CurrentCompactionSummaryComponent) is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_trim_frees_space_and_skips_summary() -> None:
    """ISSUE-5: when trimming old tool spans gets under budget, history is
    permanently reduced and no LLM summary is produced."""
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="brief"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("u" * 40),
                Message(
                    role="assistant",
                    content="a" * 40,
                    tool_calls=[ToolCall(id="c1", name="search", arguments={})],
                ),
                Message(role="tool", content="t" * 400, tool_call_id="c1"),
                _message("b" * 40, role="assistant"),
                _message("c" * 40),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1000, summary_model="summary-model"),
    )
    world.add_component(entity_id, ContextTrimConfig(max_tokens=60))

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    # Tool span dropped; no summary.
    assert [m.role for m in conversation.messages] == ["user", "assistant", "user"]
    assert all("t" * 400 not in (m.content or "") for m in conversation.messages)
    assert world.get_component(entity_id, CurrentCompactionSummaryComponent) is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_trim_insufficient_falls_back_to_summary() -> None:
    """ISSUE-5: when trimming cannot get under budget (essential content too
    large), it falls back to compaction summarization."""
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="brief"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                _message("x" * 400),
                Message(
                    role="assistant",
                    content="a" * 20,
                    tool_calls=[ToolCall(id="c1", name="search", arguments={})],
                ),
                Message(role="tool", content="t" * 20, tool_call_id="c1"),
                _message("y" * 400, role="assistant"),
                _message("z" * 400),
            ]
        ),
    )
    world.add_component(
        entity_id,
        CompactionConfigComponent(threshold_tokens=1000, summary_model="summary-model"),
    )
    world.add_component(entity_id, ContextTrimConfig(max_tokens=60))

    await CompactionSystem().process(world)

    # Trim ran (tool span gone) but was not enough -> summary produced.
    summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert summary == CurrentCompactionSummaryComponent(summary="brief")
    assert len(model.calls) == 1


def test_resolve_trim_budget_from_model_window() -> None:
    from ecs_agent.providers import FakeModel

    system = CompactionSystem()
    claude_llm = LLMComponent(model=FakeModel(responses=[], model_id="claude-opus-4-8"))
    # Explicit max_tokens wins.
    assert (
        system._resolve_trim_budget(ContextTrimConfig(max_tokens=500), claude_llm) == 500
    )
    # None -> derived from model window (200000 - 8192 reserve).
    assert (
        system._resolve_trim_budget(ContextTrimConfig(max_tokens=None), claude_llm)
        == 200_000 - 8_192
    )
    # Unknown model + None -> no budget.
    unknown_llm = LLMComponent(model=FakeModel(responses=[], model_id="fake"))
    assert system._resolve_trim_budget(ContextTrimConfig(max_tokens=None), unknown_llm) is None
    # No trim config -> no budget.
    assert system._resolve_trim_budget(None, claude_llm) is None
