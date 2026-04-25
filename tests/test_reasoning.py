from collections.abc import AsyncIterator

import pytest

from ecs_agent.components import (
    ContextBudgetConfig,
    ContextEntry,
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PromptContextQueueComponent,
    StreamingComponent,
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
    provider = FakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Hi there!"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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
    provider = FakeModel(
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
    world.add_component(entity_id, LLMComponent(model=provider))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need weather")]),
    )

    await ReasoningSystem().process(world)

    pending = world.get_component(entity_id, PendingToolCallsComponent)
    assert pending is not None
    assert pending.tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_system_prompt_component_is_prepended_to_messages() -> None:
    world = World()
    provider = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
    world.add_component(entity_id, SystemPromptComponent(content="You are concise"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello")]),
    )

    await ReasoningSystem().process(world)

    assert len(provider.calls) == 1
    sent_messages, _ = provider.calls[0]
    assert sent_messages[0] == Message(role="system", content="You are concise")
    assert sent_messages[1] == Message(role="user", content="Hello")


@pytest.mark.asyncio
async def test_tool_registry_tools_are_passed_to_provider() -> None:
    world = World()
    provider = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    tool_schema = ToolSchema(
        name="get_weather",
        description="Get weather by city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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

    assert len(provider.calls) == 1
    _, sent_tools = provider.calls[0]
    assert sent_tools == [tool_schema]


@pytest.mark.asyncio
async def test_provider_exhaustion_adds_terminal_component_not_error() -> None:
    world = World()
    provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="one"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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
    provider = FakeModel(responses=[])

    def raise_stop_iteration(
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        raise StopIteration("done")

    provider.complete = raise_stop_iteration  # type: ignore[method-assign]

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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
    provider = ErrorFakeModel(responses=[])
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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
    provider_one = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="A1"))]
    )
    provider_two = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="B1"))]
    )
    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(model=provider_one))
    world.add_component(entity_b, LLMComponent(model=provider_two))
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
    provider = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    world.add_component(valid, LLMComponent(model=provider))
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
    provider_alpha = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="alpha"))]
    )
    provider_beta = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="beta"))]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(
        entity_a, LLMComponent(model=provider_alpha)
    )
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    world.add_component(entity_b, LLMComponent(model=provider_beta))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    # Switch entity_b's model via pending_model
    provider_beta_override = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="beta-override"))],
        model_id="model-b-override",
    )
    llm_b = world.get_component(entity_b, LLMComponent)
    assert llm_b is not None
    llm_b.pending_model = provider_beta_override

    await ReasoningSystem().process(world)

    # Verify entity_a used its provider
    assert len(provider_alpha.calls) == 1
    # Verify entity_b used the pending_model override (not original provider_beta)
    assert len(provider_beta_override.calls) == 1
    assert len(provider_beta.calls) == 0


@pytest.mark.asyncio
async def test_entity_scoped_provider_switch() -> None:
    """Switching provider should not leak to other entities."""
    world = World()
    provider_main = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="main"))]
    )
    provider_override = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="override"))
        ]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(model=provider_main))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="a")]),
    )

    world.add_component(entity_b, LLMComponent(model=provider_main))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="b")]),
    )

    # Switch entity_b's model via pending_model
    llm_b = world.get_component(entity_b, LLMComponent)
    assert llm_b is not None
    llm_b.pending_model = provider_override

    await ReasoningSystem().process(world)

    # entity_a should use provider_main
    assert len(provider_main.calls) == 1
    # entity_b should use provider_override
    assert len(provider_override.calls) == 1


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
    provider_a = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="a"))],
        model_id="model-a",
    )
    provider_b = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="b"))],
        model_id="model-b",
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(model=provider_a))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="hello a")]),
    )

    world.add_component(entity_b, LLMComponent(model=provider_b))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="hello b")]),
    )

    llm_a = world.get_component(entity_a, LLMComponent)
    assert llm_a is not None
    provider_a_override = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="a-override"))],
        model_id="model-a-override",
    )
    llm_a.pending_model = provider_a_override

    await ReasoningSystem().process(world)

    assert recorded_models[int(entity_a)] == "model-a-override"
    assert recorded_models[int(entity_b)] == "model-b"


@pytest.mark.asyncio
async def test_model_switching_in_flight_stability() -> None:
    """Model should remain stable during request (sample at start)."""
    world = World()
    provider = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="stable"))
        ]
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=provider))
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
    # Original provider was not called since pending_model replaced it
    assert len(provider.calls) == 0


@pytest.mark.asyncio
async def test_per_entity_model_override() -> None:
    """pending_model and pending_provider override defaults."""
    world = World()
    provider_default = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="default"))
        ]
    )
    provider_override = RecordingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="override"))
        ]
    )

    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(model=provider_default)
    )
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    # Set pending_model to override the default
    llm = world.get_component(entity, LLMComponent)
    assert llm is not None
    llm.pending_model = provider_override

    await ReasoningSystem().process(world)

    # provider_override should be called, not provider_default
    assert len(provider_default.calls) == 0
    assert len(provider_override.calls) == 1


@pytest.mark.asyncio
async def test_prompt_context_injection_is_transient_for_reasoning_provider_call() -> (
    None
):
    world = World()
    provider = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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

    sent_messages, _ = provider.calls[0]
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
    provider = RecordingFakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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

    sent_messages, _ = provider.calls[0]
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
    provider = ReasoningContentStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    world.add_component(
        entity_id,
        ContextBudgetConfig(max_tokens=1024, prune_reasoning=True),
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
    provider = ReasoningContentStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    world.add_component(
        entity_id,
        ContextBudgetConfig(max_tokens=1024, prune_reasoning=False),
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
        ContextBudgetConfig(max_tokens=1024, prune_reasoning=True),
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
    provider = ReasoningContentStreamingFakeModel(
        responses=[
            CompletionResult(message=Message(role="assistant", content="ignored"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.add_component(entity_id, PromptContextQueueComponent())
    world.add_component(
        entity_id,
        ContextBudgetConfig(max_tokens=1024, prune_reasoning=True),
    )

    await ReasoningSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [(message.role, message.content) for message in conversation.messages] == [
        ("user", "Hi"),
        ("assistant", "done"),
    ]
    assert all("thought" not in message.content for message in conversation.messages)
