from collections.abc import AsyncIterator

import pytest

from ecs_agent.components import (
    ContextTrimConfig,
    ContextEntry,
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PromptContextQueueComponent,
    StreamingComponent,
    TokenUsageComponent,
    UserPromptConfigComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema
from ecs_agent.types import StreamDelta


class RecordingFakeModel(FakeModel):
    def __init__(self, responses: list[CompletionResult], model_id: str = "fake") -> None:
        super().__init__(responses=responses, model_id=model_id)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: object = None,
    ) -> CompletionResult:
        self.calls.append((list(messages), tools))
        return await super().complete(messages, tools, stream=stream, response_format=response_format)  # type: ignore[return-value]


class ErrorFakeModel(FakeModel):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        raise RuntimeError("provider exploded")


class ReasoningContentStreamingFakeModel(FakeModel):
    async def _stream_complete(
        self, result: CompletionResult
    ) -> AsyncIterator[StreamDelta]:
        _ = result
        yield StreamDelta(reasoning_content="first thought ")
        yield StreamDelta(reasoning_content="second thought")
        yield StreamDelta(content="done")
        yield StreamDelta(finish_reason="stop")


@pytest.mark.asyncio
async def test_basic_conversation_appends_assistant_response() -> None:
    world = World()
    model = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Hi there!"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert len(conversation.messages) == 2
    assert conversation.messages[1] == Message(role="assistant", content="Hi there!")


@pytest.mark.asyncio
async def test_tool_calls_attach_pending_tool_calls_component() -> None:
    world = World()
    tool_call = ToolCall(id="call-1", name="get_weather", arguments={"city": "Paris"})
    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[tool_call],
                )
            )
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need weather")]),
    )

    await ReasoningSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert pending.tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_pending_tool_calls_block_follow_up_reasoning() -> None:
    world = World()
    tool_call = ToolCall(id="call-1", name="get_weather", arguments={"city": "Paris"})
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[tool_call],
                )
            ),
            CompletionResult(message=Message(role="assistant", content="should not run")),
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need weather")]),
    )

    system = ReasoningSystem()
    await system.process(world)
    await system.process(world)

    assert len(model.calls) == 1
    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert pending.tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_system_prompt_component_is_prepended_to_messages() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, SystemPromptComponent(content="You are concise"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    await ReasoningSystem().process(world)

    assert len(model.calls) == 1
    sent_messages, _ = model.calls[0]
    assert sent_messages[0] == Message(
        role="system", content="You are concise", cache_control=True
    )
    assert sent_messages[1] == Message(role="user", content="Hello")


@pytest.mark.asyncio
async def test_empty_conversation_does_not_call_provider() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(entity_id, SystemPromptComponent(content="You are concise"))
    world.add_component(entity_id, ConversationComponent(messages=[]))

    await ReasoningSystem().process(world)

    assert model.calls == []
    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == []
    assert world.get_component(entity_id, ErrorComponent) is None


@pytest.mark.asyncio
async def test_tool_registry_tools_are_passed_to_provider() -> None:
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    tool_schema = ToolSchema(
        name="get_weather",
        description="Get weather by city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Weather")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"get_weather": tool_schema},
            handlers={},
        ),
    )

    await ReasoningSystem().process(world)

    assert len(model.calls) == 1
    _, sent_tools = model.calls[0]
    assert sent_tools == [tool_schema]


@pytest.mark.asyncio
async def test_provider_exhaustion_adds_terminal_component_not_error() -> None:
    world = World()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="one"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    system = ReasoningSystem()
    await system.process(world)
    await system.process(world)

    terminal = world.get_component(entity_id, TerminalComponent)
    error = world.get_component(entity_id, ErrorComponent)
    assert terminal is not None
    assert terminal.reason == "provider_exhausted"
    assert error is None


@pytest.mark.asyncio
async def test_stop_iteration_also_adds_terminal_component() -> None:
    world = World()
    model = FakeModel(responses=[])

    def raise_stop_iteration(
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        raise StopIteration("done")

    model.complete = raise_stop_iteration  # type: ignore[method-assign]

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    await ReasoningSystem().process(world)

    terminal = world.get_component(entity_id, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "provider_exhausted"


@pytest.mark.asyncio
async def test_error_handling_adds_error_component() -> None:
    world = World()
    model = ErrorFakeModel(responses=[])
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    await ReasoningSystem().process(world)

    error = world.get_component(entity_id, ErrorComponent)
    terminal = world.get_component(entity_id, TerminalComponent)
    assert error is not None
    assert error.system_name == "ReasoningSystem"
    assert "provider exploded" in error.error
    assert terminal is None


@pytest.mark.asyncio
async def test_multiple_entities_are_processed() -> None:
    world = World()
    model_one = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="A1"))]
    )
    model_two = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="B1"))]
    )
    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(model=model_one))
    world.add_component(entity_b, LLMComponent(model=model_two))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="A")]),
    )
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="B")]),
    )

    await ReasoningSystem().process(world)

    conv_a = world.get_component(entity_a, ConversationComponent)
    conv_b = world.get_component(entity_b, ConversationComponent)
    assert conv_a is not None
    assert conv_b is not None
    assert conv_a.messages[-1].content == "A1"
    assert conv_b.messages[-1].content == "B1"


@pytest.mark.asyncio
async def test_entities_missing_required_components_are_skipped() -> None:
    world = World()
    incomplete = world.create_entity()
    world.add_component(
        incomplete,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    valid = world.create_entity()
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    world.add_component(valid, LLMComponent(model=model))
    world.add_component(
        valid,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    await ReasoningSystem().process(world)

    incomplete_error = world.get_component(incomplete, ErrorComponent)
    incomplete_terminal = world.get_component(incomplete, TerminalComponent)
    valid_conversation = world.get_component(valid, ConversationComponent)

    assert incomplete_error is None
    assert incomplete_terminal is None
    assert valid_conversation is not None
    assert valid_conversation.messages[-1].content == "ok"


@pytest.mark.asyncio
async def test_entity_scoped_model_switching() -> None:
    """Two entities with different models should be isolated."""
    world = World()
    model_alpha = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="alpha"))]
    )
    model_beta = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="beta"))]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(
        entity_a, LLMComponent(model=model_alpha)
    )
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    world.add_component(entity_b, LLMComponent(model=model_beta))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    # Switch entity_b's model via pending_model
    model_beta_override = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="beta-override"))],
        model_id="model-b-override",
    )
    llm_b = world.get_component(entity_b, LLMComponent)
    assert llm_b is not None
    llm_b.pending_model = model_beta_override

    await ReasoningSystem().process(world)

    # Verify entity_a used its model
    assert len(model_alpha.calls) == 1
    # Verify entity_b used the pending_model override (not original model_beta)
    assert len(model_beta_override.calls) == 1
    assert len(model_beta.calls) == 0


@pytest.mark.asyncio
async def test_entity_scoped_provider_switch() -> None:
    """Switching model should not leak to other entities."""
    world = World()
    model_main = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="main"))]
    )
    model_override = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="override"))
        ]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(model=model_main))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="a")]),
    )

    world.add_component(entity_b, LLMComponent(model=model_main))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="b")]),
    )

    # Switch entity_b's model via pending_model
    llm_b = world.get_component(entity_b, LLMComponent)
    assert llm_b is not None
    llm_b.pending_model = model_override

    await ReasoningSystem().process(world)

    # entity_a should use model_main
    assert len(model_main.calls) == 1
    # entity_b should use model_override
    assert len(model_override.calls) == 1


@pytest.mark.asyncio
async def test_multi_entity_model_switch_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_models: dict[int, str] = {}

    class _RecordingLogger:
        def info(self, event_name: str, **kwargs: object) -> None:
            if event_name != "reasoning_start":
                return
            entity_id = kwargs.get("entity_id")
            model = kwargs.get("model")
            if isinstance(entity_id, int) and isinstance(model, str):
                recorded_models[entity_id] = model

        def error(self, event_name: str, **kwargs: object) -> None:
            _ = event_name
            _ = kwargs

    monkeypatch.setattr("ecs_agent.systems.reasoning.logger", _RecordingLogger())

    world = World()
    model_a = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="a"))],
        model_id="model-a",
    )
    model_b = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="b"))],
        model_id="model-b",
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(model=model_a))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="hello a")]),
    )

    world.add_component(entity_b, LLMComponent(model=model_b))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="hello b")]),
    )

    llm_a = world.get_component(entity_a, LLMComponent)
    assert llm_a is not None
    model_a_override = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="a-override"))],
        model_id="model-a-override",
    )
    llm_a.pending_model = model_a_override

    await ReasoningSystem().process(world)

    assert recorded_models[int(entity_a)] == "model-a-override"
    assert recorded_models[int(entity_b)] == "model-b"


@pytest.mark.asyncio
async def test_model_switching_in_flight_stability() -> None:
    """Model should remain stable during request (sample at start)."""
    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="stable"))
        ]
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="test")]),
    )

    # Set pending_model before processing
    llm = world.get_component(entity, LLMComponent)
    assert llm is not None
    override_provider = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="override"))],
        model_id="override-model",
    )
    llm.pending_model = override_provider

    await ReasoningSystem().process(world)

    # Verify override_provider was called (pending_model took effect)
    assert len(override_provider.calls) == 1
    # Original model was not called since pending_model replaced it
    assert len(model.calls) == 0


@pytest.mark.asyncio
async def test_per_entity_model_override() -> None:
    """pending_model and pending_provider override defaults."""
    world = World()
    model_default = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="default"))
        ]
    )
    model_override = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="override"))
        ]
    )

    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(model=model_default)
    )
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    # Set pending_model to override the default
    llm = world.get_component(entity, LLMComponent)
    assert llm is not None
    llm.pending_model = model_override

    await ReasoningSystem().process(world)

    # model_override should be called, not model_default
    assert len(model_default.calls) == 0
    assert len(model_override.calls) == 1


@pytest.mark.asyncio
async def test_prompt_context_injection_is_transient_for_reasoning_provider_call() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="subagent-writer-1",
                    priority=20,
                    registration_order=1,
                    source_label="subagent:writer",
                    content="source: subagent\nresult: drafted",
                ),
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool\nresult: found facts",
                ),
            ]
        ),
    )

    await ReasoningSystem().process(world)

    sent_messages, _ = model.calls[0]
    transient_user = sent_messages[-1]
    assert transient_user.role == "user"
    assert "[PROMPT_CONTEXT_POOL]" in transient_user.content
    assert transient_user.content.index("source: tool") < transient_user.content.index(
        "source: subagent"
    )
    assert transient_user.content.endswith("Need summary")

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Need summary"


@pytest.mark.asyncio
async def test_event_trigger_injection_is_transient_for_reasoning_provider_call() -> (
    None
):
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(
                    pattern="summary",
                    match_mode="keyword",
                    action="inject",
                    content="Prefer using successful tool evidence",
                    priority=0,
                )
            ],
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool:search\nstatus: success\nresult: found facts\nerror: ",
                )
            ]
        ),
    )

    await ReasoningSystem().process(world)

    sent_messages, _ = model.calls[0]
    transient_user = sent_messages[-1]
    assert transient_user.role == "user"
    assert transient_user.content.startswith(
        "[PROMPT_INJECT:summary]\nPrefer using successful tool evidence"
    )
    assert transient_user.content.endswith("Need summary")

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Need summary"


@pytest.mark.asyncio
async def test_reasoning_content_becomes_droppable_context_entry_when_enabled() -> None:
    world = World()
    model = ReasoningContentStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    world.add_component(
        entity_id,
        ContextTrimConfig(max_tokens=1024, trim_reasoning=True),
    )

    await ReasoningSystem().process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert len(queue.entries) == 1
    assert queue.entries[0].droppable_kind == "reasoning"
    assert queue.entries[0].source_label == "reasoning"
    assert queue.entries[0].content == "first thought second thought"


@pytest.mark.asyncio
async def test_reasoning_context_is_noop_when_disabled() -> None:
    world = World()
    model = ReasoningContentStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    world.add_component(
        entity_id,
        ContextTrimConfig(max_tokens=1024, trim_reasoning=False),
    )

    await ReasoningSystem().process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert queue.entries == []

    no_reasoning_world = World()
    no_reasoning_provider = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="plain reply"))
        ]
    )
    no_reasoning_entity = no_reasoning_world.create_entity()
    no_reasoning_world.add_component(
        no_reasoning_entity,
        LLMComponent(model=no_reasoning_provider),
    )
    no_reasoning_world.add_component(
        no_reasoning_entity,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    no_reasoning_world.add_component(
        no_reasoning_entity,
        PromptContextQueueComponent(),
    )
    no_reasoning_world.add_component(
        no_reasoning_entity,
        ContextTrimConfig(max_tokens=1024, trim_reasoning=True),
    )

    await ReasoningSystem().process(no_reasoning_world)

    no_reasoning_queue = no_reasoning_world.get_component(
        no_reasoning_entity, PromptContextQueueComponent
    )
    assert no_reasoning_queue is not None
    assert no_reasoning_queue.entries == []


@pytest.mark.asyncio
async def test_reasoning_context_capture_does_not_mutate_conversation() -> None:
    world = World()
    model = ReasoningContentStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    world.add_component(
        entity_id,
        ContextTrimConfig(max_tokens=1024, trim_reasoning=True),
    )

    await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [(message.role, message.content) for message in conversation.messages] == [
        ("user", "Hi"),
        ("assistant", "done"),
    ]
    assert all("thought" not in message.content for message in conversation.messages)


@pytest.mark.asyncio
async def test_reasoning_records_api_token_usage_on_entity() -> None:
    """The API-reported usage is persisted on the entity (real token info)."""
    from ecs_agent.types import UsageRecord

    world = World()
    model = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="one"),
                usage=UsageRecord(
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                    cache_read_tokens=40,
                    cache_creation_tokens=10,
                ),
            ),
            CompletionResult(
                message=Message(role="assistant", content="two"),
                usage=UsageRecord(
                    prompt_tokens=200,
                    completion_tokens=30,
                    total_tokens=230,
                    cache_read_tokens=150,
                    cache_creation_tokens=0,
                ),
            ),
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    await ReasoningSystem().process(world)

    usage = world.get_component(entity_id, TokenUsageComponent)
    assert usage is not None
    # First call is reflected in both last_* and total_*.
    assert usage.last_prompt_tokens == 100
    assert usage.last_completion_tokens == 20
    assert usage.last_total_tokens == 120
    assert usage.last_cache_read_tokens == 40
    assert usage.last_cache_creation_tokens == 10
    assert usage.total_prompt_tokens == 100
    assert usage.call_count == 1

    # Second call: last_* replaced, total_* accumulated.
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="one"),
                Message(role="user", content="again"),
            ]
        ),
    )
    await ReasoningSystem().process(world)

    usage = world.get_component(entity_id, TokenUsageComponent)
    assert usage is not None
    assert usage.last_prompt_tokens == 200
    assert usage.last_cache_read_tokens == 150
    assert usage.total_prompt_tokens == 300
    assert usage.total_completion_tokens == 50
    assert usage.total_tokens == 350
    assert usage.total_cache_read_tokens == 190
    assert usage.total_cache_creation_tokens == 10
    assert usage.call_count == 2


@pytest.mark.asyncio
async def test_reasoning_without_usage_does_not_create_usage_component() -> None:
    """A provider that reports no usage leaves TokenUsageComponent absent."""
    world = World()
    model = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    await ReasoningSystem().process(world)

    assert world.get_component(entity_id, TokenUsageComponent) is None
