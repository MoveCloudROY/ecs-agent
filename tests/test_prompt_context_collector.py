from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ecs_agent.components import (
    OneShotContextPoolComponent,
    PromptConfigComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.systems.prompt_context_collector import (
    CONTEXT_ENTRY_DELIMITER,
    CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX,
    CONTEXT_POOL_OVERFLOW_SOURCE,
    PromptContextCollectorSystem,
    STRUCTURED_OUTPUT_CONTEXT_PRIORITY,
)
from ecs_agent.types import DelegationCompletedEvent, ToolExecutionCompletedEvent


@pytest.mark.asyncio
async def test_collector_orders_entries_by_priority_desc_then_registration_order() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        PromptConfigComponent(enable_context_pool=True, context_pool_max_chars=10000),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(
            items=[(5, 0, "existing", "existing-content")], _counter=1
        ),
    )
    world.add_component(
        entity_id,
        ToolResultsComponent(results={"result-1": '{"ok": true}'}),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )

    await system.process(world)
    await world.event_bus.publish(
        ToolExecutionCompletedEvent(
            entity_id=entity_id,
            tool_call_id="tool-1",
            result="tool-result",
            success=True,
        )
    )
    await world.event_bus.publish(
        DelegationCompletedEvent(
            entity_id=entity_id,
            subagent_name="researcher",
            result="subagent-result",
            success=False,
            error="failed",
        )
    )

    await system.process(world)

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert len(pool.items) == 4
    assert pool.items == sorted(pool.items, key=lambda item: (-item[0], item[1]))
    for _, _, source, content in pool.items:
        if source == "existing":
            continue
        assert "source:" in content
        assert "status:" in content
        assert "result:" in content
        assert "error:" in content
        assert "timestamp:" in content


@pytest.mark.asyncio
async def test_collector_truncation_appends_footer_when_entries_are_dropped() -> None:
    world = World()
    entity_id = world.create_entity()
    existing_items = [
        (STRUCTURED_OUTPUT_CONTEXT_PRIORITY, 0, "structured-output:0", "first"),
        (STRUCTURED_OUTPUT_CONTEXT_PRIORITY, 1, "structured-output:1", "second"),
    ]
    footer = f"{CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX} dropped_entries=1"
    max_chars = (
        len(existing_items[0][3])
        + len(CONTEXT_ENTRY_DELIMITER)
        + len(existing_items[1][3])
        + len(CONTEXT_ENTRY_DELIMITER)
        + len(footer)
    )

    world.add_component(
        entity_id,
        PromptConfigComponent(
            enable_context_pool=True, context_pool_max_chars=max_chars
        ),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(items=list(existing_items), _counter=2),
    )
    world.add_component(
        entity_id,
        ToolResultsComponent(results={"result-1": "new-structured-result"}),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )
    await system.process(world)

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert len(pool.items) == 3
    assert pool.items[0] == existing_items[0]
    assert pool.items[1] == existing_items[1]
    assert pool.items[2][2] == CONTEXT_POOL_OVERFLOW_SOURCE
    assert pool.items[2][3] == footer


@pytest.mark.asyncio
async def test_collector_truncation_drops_more_entries_to_fit_footer() -> None:
    world = World()
    entity_id = world.create_entity()
    existing_items = [
        (STRUCTURED_OUTPUT_CONTEXT_PRIORITY, 0, "structured-output:0", "alpha"),
        (STRUCTURED_OUTPUT_CONTEXT_PRIORITY, 1, "structured-output:1", "beta"),
    ]
    footer = f"{CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX} dropped_entries=2"
    max_chars = len(existing_items[0][3]) + len(CONTEXT_ENTRY_DELIMITER) + len(footer)

    world.add_component(
        entity_id,
        PromptConfigComponent(
            enable_context_pool=True, context_pool_max_chars=max_chars
        ),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(items=list(existing_items), _counter=2),
    )
    world.add_component(
        entity_id,
        ToolResultsComponent(results={"result-1": "new-structured-result"}),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )
    await system.process(world)

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert len(pool.items) == 2
    assert pool.items[0] == existing_items[0]
    assert pool.items[1][2] == CONTEXT_POOL_OVERFLOW_SOURCE
    assert pool.items[1][3] == footer


@pytest.mark.asyncio
async def test_collector_does_not_clear_pool_state_or_items() -> None:
    world = World()
    entity_id = world.create_entity()
    original_item = (1, 0, "baseline", "keep-me")
    world.add_component(
        entity_id,
        PromptConfigComponent(enable_context_pool=True, context_pool_max_chars=10000),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(
            items=[original_item],
            state="reserved",
            reserved_turn_id="turn-123",
            _counter=1,
        ),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )
    await system.process(world)
    await world.event_bus.publish(
        ToolExecutionCompletedEvent(
            entity_id=entity_id,
            tool_call_id="tool-1",
            result="ok",
            success=True,
        )
    )
    await system.process(world)

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert pool.state == "reserved"
    assert pool.reserved_turn_id == "turn-123"
    assert original_item in pool.items


@pytest.mark.asyncio
async def test_collector_non_opt_in_entity_is_unchanged() -> None:
    world = World()
    entity_id = world.create_entity()
    original_item = (1, 0, "baseline", "keep-me")
    world.add_component(
        entity_id,
        PromptConfigComponent(enable_context_pool=False, context_pool_max_chars=10000),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(items=[original_item], _counter=1),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )
    await system.process(world)
    await world.event_bus.publish(
        ToolExecutionCompletedEvent(
            entity_id=entity_id,
            tool_call_id="tool-1",
            result="ignored",
            success=True,
        )
    )
    await system.process(world)

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert pool.items == [original_item]
    assert pool._counter == 1
