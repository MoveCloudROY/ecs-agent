import pytest

from ecs_agent.components import (
    ConversationComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.context import current_tool_context
from ecs_agent.types import Message, ToolCall, ToolExecutionCompletedEvent, ToolSchema


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
async def test_process_exposes_internal_tool_execution_context() -> None:
    world = World()
    entity_id = world.create_entity()

    async def inspect_context() -> str:
        context = current_tool_context()
        assert context.world is world
        assert context.entity_id == entity_id
        assert context.tool_name == "inspect_context"
        assert context.tool_call_id == "ctx-1"
        return "context-ok"

    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "inspect_context": ToolSchema(
                    name="inspect_context",
                    description="Inspect context",
                    parameters={"type": "object"},
                )
            },
            handlers={"inspect_context": inspect_context},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="ctx-1", name="inspect_context", arguments={})]
        ),
    )

    await ToolExecutionSystem().process(world)

    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert results.results == {"ctx-1": "context-ok"}


@pytest.mark.asyncio
async def test_process_publishes_tool_completion_duration_and_status() -> None:
    world = World()
    entity_id = world.create_entity()
    completed: list[ToolExecutionCompletedEvent] = []

    async def ping() -> str:
        return "pong"

    async def on_completed(event: ToolExecutionCompletedEvent) -> None:
        completed.append(event)

    world.event_bus.subscribe(ToolExecutionCompletedEvent, on_completed)
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
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
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-1", name="ping", arguments={})]
        ),
    )

    await ToolExecutionSystem().process(world)

    assert len(completed) == 1
    assert completed[0].tool_name == "ping"
    assert completed[0].status == "success"
    assert completed[0].duration_seconds is not None
    assert completed[0].duration_seconds >= 0


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


@pytest.mark.asyncio
async def test_tool_execution_caches_result_on_overflow(tmp_path) -> None:
    scratchbook_root = tmp_path / ".scratchbook"
    registry = ArtifactRegistry(root=scratchbook_root)

    world = World()
    entity_id = world.create_entity()

    async def verbose_tool() -> str:
        return "cached payload " * 80

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="run the tool")]),
    )
    world.add_component(
        entity_id,
        ContextTrimConfig(
            max_tokens=5,
            token_estimation_chars_per_token=1.0,
            overflow_behavior="warn",
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "verbose_tool": ToolSchema(
                    name="verbose_tool",
                    description="Return large payload",
                    parameters={"type": "object"},
                )
            },
            handlers={"verbose_tool": verbose_tool},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="overflow-1", name="verbose_tool", arguments={})]
        ),
    )

    await ToolExecutionSystem(registry=registry).process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1].role == "tool"
    assert "cached" in conversation.messages[-1].content.lower()
    assert "scratchbook/records/tool/tool_" in conversation.messages[-1].content

    cache = world.get_component(entity_id, ContextCacheComponent)
    assert cache is not None
    assert len(cache.cached_tool_results) == 1
    assert cache.cached_tool_results[0].tool_call_id == "overflow-1"
    assert cache.cached_tool_results[0].original_content == "cached payload " * 80

    artifact_path = scratchbook_root / cache.cached_tool_results[0].artifact_path
    assert artifact_path.exists()
    artifact_payload = artifact_path.read_text(encoding="utf-8")
    assert "cached payload " in artifact_payload
