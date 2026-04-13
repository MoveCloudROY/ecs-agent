from __future__ import annotations

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ContextBudgetConfig,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
    SystemPromptComponent,
)
from ecs_agent.core import World
from ecs_agent.serialization import WorldSerializer
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.registry import ProviderRegistry
from ecs_agent.systems.compaction import DEFAULT_COMPACTION_PROMPT, CompactionSystem
import ecs_agent.systems.compaction as compaction_module
from ecs_agent.systems.system_prompt_render_system import render_compaction_prompt
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
)
from ecs_agent.prompts.message_assembly import apply_outbound_budget
from ecs_agent.types import (
    CompactionCompleteEvent,
    CompletionResult,
    EntityId,
    Message,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    ToolSchema,
)
from ecs_agent.components.definitions import ContextEntry


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


class RegistryBackedRecordingFakeProvider(RecordingFakeProvider):
    def __init__(
        self,
        responses: list[CompletionResult],
        registry: ProviderRegistry,
    ) -> None:
        super().__init__(responses=responses)
        self.registry = registry


def _message(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


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


def test_compaction_role_survives_serialization() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(
                    role="compaction",
                    content="Previous conversation summary: compacted state",
                )
            ]
        ),
    )

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})
    restored_conversation = restored.get_component(entity_id, ConversationComponent)

    assert restored_conversation is not None
    assert restored_conversation.messages == [
        Message(
            role="compaction",
            content="Previous conversation summary: compacted state",
        )
    ]


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
async def test_compaction_full_history_method_replaces_all_non_system_history() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="full summary"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    assert conversation is not None
    assert [(message.role, message.content) for message in conversation.messages] == [
        ("system", "You are a strict assistant"),
        ("compaction", "Previous conversation summary: full summary"),
    ]

    sent_messages, _, _ = provider.calls[0]
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
    provider = RecordingFakeProvider(
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
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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

    def fake_apply_outbound_budget(
        messages: list[Message],
        system_prompt: str,
        context_entries: list[object],
        config: ContextBudgetConfig,
    ) -> list[Message]:
        seen["messages"] = list(messages)
        seen["system_prompt"] = system_prompt
        seen["context_entries"] = list(context_entries)
        seen["config"] = config
        return [messages[-1]]

    monkeypatch.setattr(
        compaction_module,
        "apply_outbound_budget",
        fake_apply_outbound_budget,
    )

    await CompactionSystem().process(world)

    assert seen["system_prompt"] == ""
    assert seen["context_entries"] == []
    budget_config = seen["config"]
    assert isinstance(budget_config, ContextBudgetConfig)
    assert budget_config.prune_tool_results is True

    summarized_input = provider.calls[0][0][1].content
    assert "assistant: tool call" not in summarized_input
    assert "tool: tool result" not in summarized_input
    assert "user: keep me" in summarized_input

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].role == "system"
    assert conversation.messages[1].role == "compaction"
    assert conversation.messages == [
        Message(role="system", content="You are a strict assistant"),
        Message(
            role="compaction", content="Previous conversation summary: budgeted summary"
        ),
    ]
    assert original_messages[1].tool_calls is not None


@pytest.mark.asyncio
async def test_compaction_summary_input_includes_canonical_subagent_session_states() -> (
    None
):
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="subagent-aware summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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

    summary_input = provider.calls[0][0][1].content
    assert "conversation text claims sess-pending succeeded" in summary_input
    assert "Subagent session state:" in summary_input
    assert "Pending: sess-pending" in summary_input
    assert "Completed (failed): sess-failed" in summary_input
    assert "Completed (cancelled): sess-cancelled" in summary_input
    assert "notification status=failed" in summary_input


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
    assert len(sent_messages) == 2
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == DEFAULT_COMPACTION_PROMPT
    assert sent_messages[1].role == "user"
    assert "user: context-one" in sent_messages[1].content
    assert "user: context-two" in sent_messages[1].content


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
            provider=FakeProvider(responses=[]),
            model="fake",
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
    primary_provider = RegistryBackedRecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused"))
        ],
        registry=registry,
    )
    summary_provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="model-id summary")
            )
        ]
    )

    def fake_get_llm_provider(
        model_id: str,
        *,
        registry: ProviderRegistry,
        api_key: str | None = None,
    ) -> RecordingFakeProvider:
        assert model_id == "fake/test-model"
        assert registry is primary_provider.registry
        assert api_key is None
        return summary_provider

    monkeypatch.setattr(compaction_module, "get_llm_provider", fake_get_llm_provider)

    entity_id = world.create_entity()
    world.add_component(
        entity_id, LLMComponent(provider=primary_provider, model="base-model")
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
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="legacy summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="base-model"))
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

    assert len(provider.calls) == 1
    assert provider.calls[0][0][0].content == DEFAULT_COMPACTION_PROMPT


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

    reduced = apply_outbound_budget(
        messages,
        system_prompt="",
        context_entries=context_entries,
        config=ContextBudgetConfig(
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
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="default prompt summary")
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="base-model"))
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

    assert len(provider.calls) == 1
    sent_messages, _, _ = provider.calls[0]
    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == DEFAULT_COMPACTION_PROMPT


@pytest.mark.asyncio
async def test_compaction_renders_custom_prompt_template() -> None:
    world = World()
    provider = RecordingFakeProvider(
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
            provider=provider,
            model="base-model",
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

    sent_messages, _, _ = provider.calls[0]
    llm = world.get_component(entity_id, LLMComponent)
    runtime_cache = world.get_component(entity_id, RenderedSystemPromptComponent)
    legacy_prompt = world.get_component(entity_id, SystemPromptComponent)

    assert sent_messages[0].role == "system"
    assert sent_messages[0].content == "Summary for compaction"
    assert llm is not None
    assert llm.system_prompt == "runtime llm prompt"
    assert runtime_cache is not None
    assert runtime_cache.text == "cached runtime prompt"
    assert legacy_prompt is not None
    assert legacy_prompt.content == "runtime legacy prompt"


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

    assert component.compaction_method == "bisect"
    assert component.summary_model_id is None
    assert component.compaction_prompt_template is None
