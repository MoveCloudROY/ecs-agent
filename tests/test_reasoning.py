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
    tool_call = ToolCall(id="call-1", name="get_weather", arguments='{"city":"Paris"}')
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
