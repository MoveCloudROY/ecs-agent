import pytest

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message, ToolCall, ToolSchema


@pytest.mark.asyncio
async def test_process_executes_pending_tool_calls_and_appends_tool_messages() -> None:
    world = World()
    entity_id = world.create_entity()
    seen: list[tuple[str, str]] = []

    async def get_weather(city: str) -> str:
        seen.append(("city", city))
        return f"sunny in {city}"

    async def get_time(zone: str) -> str:
        seen.append(("zone", zone))
        return f"10:00 in {zone}"

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="tools please")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "get_weather": ToolSchema(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"},
                ),
                "get_time": ToolSchema(
                    name="get_time",
                    description="Get time",
                    parameters={"type": "object"},
                ),
            },
            handlers={"get_weather": get_weather, "get_time": get_time},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="call-1", name="get_weather", arguments={"city": "Paris"}),
                ToolCall(id="call-2", name="get_time", arguments={"zone": "UTC"}),
            ]
        ),
    )

    await ToolExecutionSystem().process(world)

    assert seen == [("city", "Paris"), ("zone", "UTC")]

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-2] == Message(
        role="tool",
        content="sunny in Paris",
        tool_call_id="call-1",
    )
    assert conversation.messages[-1] == Message(
        role="tool",
        content="10:00 in UTC",
        tool_call_id="call-2",
    )

    assert world.get_component(entity_id, PendingToolCallsComponent) is None
    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert results.results == {
        "call-1": "sunny in Paris",
        "call-2": "10:00 in UTC",
    }


@pytest.mark.asyncio
async def test_unknown_tool_is_converted_to_error_result_string() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="missing-1", name="does_not_exist", arguments={}),
            ]
        ),
    )

    await ToolExecutionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(
            role="tool",
            content="Error: unknown tool 'does_not_exist'",
            tool_call_id="missing-1",
        )
    ]
    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert results.results == {"missing-1": "Error: unknown tool 'does_not_exist'"}
    assert world.get_component(entity_id, PendingToolCallsComponent) is None


@pytest.mark.asyncio
async def test_handler_exception_is_converted_to_error_result_string() -> None:
    world = World()
    entity_id = world.create_entity()

    async def exploding_tool(city: str) -> str:
        _ = city
        raise RuntimeError("boom")

    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "get_weather": ToolSchema(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"},
                )
            },
            handlers={"get_weather": exploding_tool},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="boom-1", name="get_weather", arguments={"city": "Paris"}),
            ]
        ),
    )

    await ToolExecutionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "tool"
    assert conversation.messages[0].tool_call_id == "boom-1"
    assert (
        conversation.messages[0].content == "Error executing tool 'get_weather': boom"
    )

    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert results.results == {
        "boom-1": "Error executing tool 'get_weather': boom",
    }
    assert world.get_component(entity_id, PendingToolCallsComponent) is None


@pytest.mark.asyncio
async def test_empty_pending_calls_removes_pending_without_results_component() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(entity_id, PendingToolCallsComponent(tool_calls=[]))

    await ToolExecutionSystem().process(world)

    assert world.get_component(entity_id, PendingToolCallsComponent) is None
    assert world.get_component(entity_id, ToolResultsComponent) is None


@pytest.mark.asyncio
async def test_entities_missing_required_components_are_skipped() -> None:
    world = World()

    incomplete = world.create_entity()
    world.add_component(incomplete, PendingToolCallsComponent(tool_calls=[]))

    valid = world.create_entity()

    async def ping() -> str:
        return "pong"

    world.add_component(valid, ConversationComponent(messages=[]))
    world.add_component(
        valid,
        ToolRegistryComponent(
            tools={
                "ping": ToolSchema(
                    name="ping",
                    description="Ping",
                    parameters={"type": "object"},
                )
            },
            handlers={"ping": ping},
        ),
    )
    world.add_component(
        valid,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="ok-1", name="ping", arguments={})]
        ),
    )

    await ToolExecutionSystem().process(world)

    assert world.get_component(incomplete, PendingToolCallsComponent) is not None

    conversation = world.get_component(valid, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="tool", content="pong", tool_call_id="ok-1")
    ]
    assert world.get_component(valid, PendingToolCallsComponent) is None
