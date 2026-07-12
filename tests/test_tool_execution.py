import asyncio
from typing import Awaitable, Callable

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    PendingToolCallsComponent,
    ToolExecutionConfigComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.context import current_tool_context
from ecs_agent.types import (
    EntityId,
    Message,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolSchema,
)


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


def _safe_schema(name: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"{name} (concurrency-safe)",
        parameters={"type": "object"},
        concurrency_safe=True,
    )


def _unsafe_schema(name: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"{name} (serial-only)",
        parameters={"type": "object"},
    )


def _install_tool_entity(
    world: World,
    tools: dict[str, ToolSchema],
    handlers: dict[str, Callable[..., Awaitable[str]]],
    tool_calls: list[ToolCall],
) -> EntityId:
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools=tools, handlers=handlers))
    world.add_component(entity_id, PendingToolCallsComponent(tool_calls=tool_calls))
    return entity_id


@pytest.mark.asyncio
async def test_concurrency_safe_tools_run_concurrently() -> None:
    """Two safe tools rendezvous: each returns only after the other started."""
    world = World()
    started_a = asyncio.Event()
    started_b = asyncio.Event()

    async def tool_a() -> str:
        started_a.set()
        await asyncio.wait_for(started_b.wait(), timeout=2.0)
        return "a-done"

    async def tool_b() -> str:
        started_b.set()
        await asyncio.wait_for(started_a.wait(), timeout=2.0)
        return "b-done"

    entity_id = _install_tool_entity(
        world,
        tools={"tool_a": _safe_schema("tool_a"), "tool_b": _safe_schema("tool_b")},
        handlers={"tool_a": tool_a, "tool_b": tool_b},
        tool_calls=[
            ToolCall(id="par-1", name="tool_a", arguments={}),
            ToolCall(id="par-2", name="tool_b", arguments={}),
        ],
    )

    await ToolExecutionSystem().process(world)

    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert results.results == {"par-1": "a-done", "par-2": "b-done"}


@pytest.mark.asyncio
async def test_concurrent_results_land_in_original_tool_call_order() -> None:
    """Completion order (fast first) must not affect conversation order."""
    world = World()
    completion_order: list[str] = []

    async def slow_tool() -> str:
        await asyncio.sleep(0.05)
        completion_order.append("slow")
        return "slow-result"

    async def fast_tool() -> str:
        completion_order.append("fast")
        return "fast-result"

    entity_id = _install_tool_entity(
        world,
        tools={"slow": _safe_schema("slow"), "fast": _safe_schema("fast")},
        handlers={"slow": slow_tool, "fast": fast_tool},
        tool_calls=[
            ToolCall(id="ord-1", name="slow", arguments={}),
            ToolCall(id="ord-2", name="fast", arguments={}),
        ],
    )

    await ToolExecutionSystem().process(world)

    assert completion_order == ["fast", "slow"]
    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="tool", content="slow-result", tool_call_id="ord-1"),
        Message(role="tool", content="fast-result", tool_call_id="ord-2"),
    ]


@pytest.mark.asyncio
async def test_unsafe_tool_is_a_barrier_between_safe_groups() -> None:
    """A serial-only tool never overlaps surrounding concurrency-safe groups."""
    world = World()
    events: list[tuple[str, str]] = []

    def make_tool(name: str, delay: float) -> Callable[[], Awaitable[str]]:
        async def handler() -> str:
            events.append(("start", name))
            await asyncio.sleep(delay)
            events.append(("end", name))
            return name

        return handler

    entity_id = _install_tool_entity(
        world,
        tools={
            "safe1": _safe_schema("safe1"),
            "safe2": _safe_schema("safe2"),
            "unsafe": _unsafe_schema("unsafe"),
            "safe3": _safe_schema("safe3"),
        },
        handlers={
            "safe1": make_tool("safe1", 0.03),
            "safe2": make_tool("safe2", 0.01),
            "unsafe": make_tool("unsafe", 0.01),
            "safe3": make_tool("safe3", 0.0),
        },
        tool_calls=[
            ToolCall(id="b-1", name="safe1", arguments={}),
            ToolCall(id="b-2", name="safe2", arguments={}),
            ToolCall(id="b-3", name="unsafe", arguments={}),
            ToolCall(id="b-4", name="safe3", arguments={}),
        ],
    )

    await ToolExecutionSystem().process(world)

    unsafe_start = events.index(("start", "unsafe"))
    assert events.index(("end", "safe1")) < unsafe_start
    assert events.index(("end", "safe2")) < unsafe_start
    assert events.index(("end", "unsafe")) < events.index(("start", "safe3"))

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert [message.tool_call_id for message in conversation.messages] == [
        "b-1",
        "b-2",
        "b-3",
        "b-4",
    ]


@pytest.mark.asyncio
async def test_max_concurrency_bounds_in_flight_tools() -> None:
    world = World()
    in_flight = 0
    peak = 0

    def make_tool() -> Callable[[], Awaitable[str]]:
        async def handler() -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return "ok"

        return handler

    tool_names = [f"tool{i}" for i in range(4)]
    entity_id = _install_tool_entity(
        world,
        tools={name: _safe_schema(name) for name in tool_names},
        handlers={name: make_tool() for name in tool_names},
        tool_calls=[
            ToolCall(id=f"cap-{i}", name=name, arguments={})
            for i, name in enumerate(tool_names)
        ],
    )
    world.add_component(entity_id, ToolExecutionConfigComponent(max_concurrency=2))

    await ToolExecutionSystem().process(world)

    assert peak == 2
    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert len(results.results) == 4


@pytest.mark.asyncio
async def test_max_concurrency_one_serializes_safe_tools() -> None:
    world = World()
    in_flight = 0
    peak = 0
    completion_order: list[str] = []

    def make_tool(name: str) -> Callable[[], Awaitable[str]]:
        async def handler() -> str:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            completion_order.append(name)
            return name

        return handler

    entity_id = _install_tool_entity(
        world,
        tools={"first": _safe_schema("first"), "second": _safe_schema("second")},
        handlers={"first": make_tool("first"), "second": make_tool("second")},
        tool_calls=[
            ToolCall(id="ser-1", name="first", arguments={}),
            ToolCall(id="ser-2", name="second", arguments={}),
        ],
    )
    world.add_component(entity_id, ToolExecutionConfigComponent(max_concurrency=1))

    await ToolExecutionSystem().process(world)

    assert peak == 1
    assert completion_order == ["first", "second"]


@pytest.mark.asyncio
async def test_concurrent_batch_publishes_paired_events_per_tool_call() -> None:
    world = World()
    started: list[ToolExecutionStartedEvent] = []
    completed: list[ToolExecutionCompletedEvent] = []

    async def on_started(event: ToolExecutionStartedEvent) -> None:
        started.append(event)

    async def on_completed(event: ToolExecutionCompletedEvent) -> None:
        completed.append(event)

    world.event_bus.subscribe(ToolExecutionStartedEvent, on_started)
    world.event_bus.subscribe(ToolExecutionCompletedEvent, on_completed)

    async def ping() -> str:
        await asyncio.sleep(0.01)
        return "pong"

    tool_names = [f"ping{i}" for i in range(3)]
    _install_tool_entity(
        world,
        tools={name: _safe_schema(name) for name in tool_names},
        handlers={name: ping for name in tool_names},
        tool_calls=[
            ToolCall(id=f"evt-{i}", name=name, arguments={})
            for i, name in enumerate(tool_names)
        ],
    )

    await ToolExecutionSystem().process(world)

    started_ids = sorted(event.tool_call.id for event in started)
    completed_ids = sorted(event.tool_call_id for event in completed)
    assert started_ids == ["evt-0", "evt-1", "evt-2"]
    assert completed_ids == ["evt-0", "evt-1", "evt-2"]
    assert all(event.success for event in completed)
    assert all(
        event.duration_seconds is not None and event.duration_seconds >= 0
        for event in completed
    )


@pytest.mark.asyncio
async def test_unknown_tool_between_safe_tools_lands_error_in_order() -> None:
    world = World()

    async def ok_tool() -> str:
        return "ok"

    entity_id = _install_tool_entity(
        world,
        tools={"ok1": _safe_schema("ok1"), "ok2": _safe_schema("ok2")},
        handlers={"ok1": ok_tool, "ok2": ok_tool},
        tool_calls=[
            ToolCall(id="mix-1", name="ok1", arguments={}),
            ToolCall(id="mix-2", name="missing", arguments={}),
            ToolCall(id="mix-3", name="ok2", arguments={}),
        ],
    )

    await ToolExecutionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [
        Message(role="tool", content="ok", tool_call_id="mix-1"),
        Message(
            role="tool",
            content="Error: unknown tool 'missing'",
            tool_call_id="mix-2",
        ),
        Message(role="tool", content="ok", tool_call_id="mix-3"),
    ]


def test_concurrency_metadata_survives_serialization_roundtrip() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"grep": _safe_schema("grep"), "bash": _unsafe_schema("bash")},
            handlers={},
        ),
    )
    world.add_component(entity_id, ToolExecutionConfigComponent(max_concurrency=3))

    serialized = WorldSerializer.to_dict(world)
    restored = WorldSerializer.from_dict(serialized, providers={}, tool_handlers={})

    entries = restored.query(ToolRegistryComponent)
    assert len(entries) == 1
    _, (registry,) = entries[0]
    assert isinstance(registry, ToolRegistryComponent)
    assert registry.tools["grep"].concurrency_safe is True
    assert registry.tools["bash"].concurrency_safe is False

    config_entries = restored.query(ToolExecutionConfigComponent)
    assert len(config_entries) == 1
    _, (config,) = config_entries[0]
    assert isinstance(config, ToolExecutionConfigComponent)
    assert config.max_concurrency == 3
