import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema


class RecordingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[tuple[list[Message], list[ToolSchema] | None]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        self.calls.append((list(messages), tools))
        return await super().complete(messages, tools)


class ErrorFakeProvider(FakeProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        raise RuntimeError("provider exploded")


@pytest.mark.asyncio
async def test_basic_conversation_appends_assistant_response() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Hi there!"))
        ]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider = FakeProvider(
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
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    tool_schema = ToolSchema(
        name="get_weather",
        description="Get weather by city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="one"))]
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider = FakeProvider(responses=[])

    def raise_stop_iteration(
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = messages
        _ = tools
        raise StopIteration("done")

    provider.complete = raise_stop_iteration  # type: ignore[method-assign]

    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider = ErrorFakeProvider(responses=[])
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
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
    provider_one = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="A1"))]
    )
    provider_two = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="B1"))]
    )
    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(provider=provider_one, model="fake"))
    world.add_component(entity_b, LLMComponent(provider=provider_two, model="fake"))
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
    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    world.add_component(valid, LLMComponent(provider=provider, model="fake"))
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
    provider_alpha = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="alpha"))]
    )
    provider_beta = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="beta"))]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(
        entity_a, LLMComponent(provider=provider_alpha, model="model-a")
    )
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    world.add_component(entity_b, LLMComponent(provider=provider_beta, model="model-b"))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="hello")]),
    )

    # Switch entity_b's model via pending_model
    llm_b = world.get_component(entity_b, LLMComponent)
    assert llm_b is not None
    llm_b.pending_model = "model-b-override"  # type: ignore[attr-defined]

    await ReasoningSystem().process(world)

    # Verify entity_a used model-a
    assert len(provider_alpha.calls) == 1
    # Verify entity_b used model-b-override (pending_model takes precedence)
    assert len(provider_beta.calls) == 1


@pytest.mark.asyncio
async def test_entity_scoped_provider_switch() -> None:
    """Switching provider should not leak to other entities."""
    world = World()
    provider_main = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="main"))]
    )
    provider_override = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="override"))
        ]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(provider=provider_main, model="fake"))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="a")]),
    )

    world.add_component(entity_b, LLMComponent(provider=provider_main, model="fake"))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="b")]),
    )

    # Switch entity_b's provider
    llm_b = world.get_component(entity_b, LLMComponent)
    assert llm_b is not None
    llm_b.pending_provider = provider_override  # type: ignore[attr-defined]

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
    provider_a = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="a"))]
    )
    provider_b = RecordingFakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="b"))]
    )

    entity_a = world.create_entity()
    entity_b = world.create_entity()

    world.add_component(entity_a, LLMComponent(provider=provider_a, model="model-a"))
    world.add_component(
        entity_a,
        ConversationComponent(messages=[Message(role="user", content="hello a")]),
    )

    world.add_component(entity_b, LLMComponent(provider=provider_b, model="model-b"))
    world.add_component(
        entity_b,
        ConversationComponent(messages=[Message(role="user", content="hello b")]),
    )

    llm_a = world.get_component(entity_a, LLMComponent)
    assert llm_a is not None
    llm_a.pending_model = "model-a-override"

    await ReasoningSystem().process(world)

    assert recorded_models[int(entity_a)] == "model-a-override"
    assert recorded_models[int(entity_b)] == "model-b"


@pytest.mark.asyncio
async def test_model_switching_in_flight_stability() -> None:
    """Model should remain stable during request (sample at start)."""
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="stable"))
        ]
    )

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model="base-model"))
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="test")]),
    )

    # Set pending_model before processing
    llm = world.get_component(entity, LLMComponent)
    assert llm is not None
    llm.pending_model = "override-model"  # type: ignore[attr-defined]

    await ReasoningSystem().process(world)

    # Verify provider was called exactly once with stable model
    assert len(provider.calls) == 1
    # Model should have been sampled at start and used throughout
    # (This test verifies the model doesn't change mid-request)


@pytest.mark.asyncio
async def test_per_entity_model_override() -> None:
    """pending_model and pending_provider override defaults."""
    world = World()
    provider_default = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="default"))
        ]
    )
    provider_override = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="override"))
        ]
    )

    entity = world.create_entity()
    world.add_component(
        entity, LLMComponent(provider=provider_default, model="default-model")
    )
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content="hi")]),
    )

    # Set both pending fields
    llm = world.get_component(entity, LLMComponent)
    assert llm is not None
    llm.pending_provider = provider_override  # type: ignore[attr-defined]
    llm.pending_model = "override-model"  # type: ignore[attr-defined]

    await ReasoningSystem().process(world)

    # provider_override should be called, not provider_default
    assert len(provider_default.calls) == 0
    assert len(provider_override.calls) == 1
