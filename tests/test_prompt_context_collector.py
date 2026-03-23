from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ecs_agent.components import (
    ContextEntry,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    UserPromptConfigComponent,
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
        UserPromptConfigComponent(
            enable_context_pool=True, context_pool_max_chars=10000
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="existing-0",
                    priority=5,
                    registration_order=0,
                    source_label="existing",
                    content="existing-content",
                )
            ]
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

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert len(queue.entries) == 4
    assert queue.entries == sorted(
        queue.entries, key=lambda entry: (-entry.priority, entry.registration_order)
    )
    for entry in queue.entries:
        source = entry.source_label
        content = entry.content
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
    existing_entries = [
        ContextEntry(
            entry_id="structured-output-0",
            priority=STRUCTURED_OUTPUT_CONTEXT_PRIORITY,
            registration_order=0,
            source_label="structured-output:0",
            content="first",
        ),
        ContextEntry(
            entry_id="structured-output-1",
            priority=STRUCTURED_OUTPUT_CONTEXT_PRIORITY,
            registration_order=1,
            source_label="structured-output:1",
            content="second",
        ),
    ]
    footer = f"{CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX} dropped_entries=1"
    max_chars = (
        len(existing_entries[0].content)
        + len(CONTEXT_ENTRY_DELIMITER)
        + len(existing_entries[1].content)
        + len(CONTEXT_ENTRY_DELIMITER)
        + len(footer)
    )

    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            enable_context_pool=True, context_pool_max_chars=max_chars
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(entries=list(existing_entries)),
    )
    world.add_component(
        entity_id,
        ToolResultsComponent(results={"result-1": "new-structured-result"}),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )
    await system.process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert len(queue.entries) == 3
    assert queue.entries[0] == existing_entries[0]
    assert queue.entries[1] == existing_entries[1]
    assert queue.entries[2].source_label == CONTEXT_POOL_OVERFLOW_SOURCE
    assert queue.entries[2].content == footer


@pytest.mark.asyncio
async def test_collector_truncation_drops_more_entries_to_fit_footer() -> None:
    world = World()
    entity_id = world.create_entity()
    existing_entries = [
        ContextEntry(
            entry_id="structured-output-0",
            priority=STRUCTURED_OUTPUT_CONTEXT_PRIORITY,
            registration_order=0,
            source_label="structured-output:0",
            content="alpha",
        ),
        ContextEntry(
            entry_id="structured-output-1",
            priority=STRUCTURED_OUTPUT_CONTEXT_PRIORITY,
            registration_order=1,
            source_label="structured-output:1",
            content="beta",
        ),
    ]
    footer = f"{CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX} dropped_entries=2"
    max_chars = (
        len(existing_entries[0].content) + len(CONTEXT_ENTRY_DELIMITER) + len(footer)
    )

    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            enable_context_pool=True, context_pool_max_chars=max_chars
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(entries=list(existing_entries)),
    )
    world.add_component(
        entity_id,
        ToolResultsComponent(results={"result-1": "new-structured-result"}),
    )

    system = PromptContextCollectorSystem(
        now_provider=lambda: datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    )
    await system.process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert len(queue.entries) == 2
    assert queue.entries[0] == existing_entries[0]
    assert queue.entries[1].source_label == CONTEXT_POOL_OVERFLOW_SOURCE
    assert queue.entries[1].content == footer


@pytest.mark.asyncio
async def test_collector_does_not_clear_pool_state_or_items() -> None:
    world = World()
    entity_id = world.create_entity()
    original_entry = ContextEntry(
        entry_id="baseline-0",
        priority=1,
        registration_order=0,
        source_label="baseline",
        content="keep-me",
    )
    original_reservation = PromptContextReservationComponent(
        reservation_id="reservation-123",
        created_at_tick=7,
        reserved_entries=[original_entry],
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            enable_context_pool=True, context_pool_max_chars=10000
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(entries=[original_entry]),
    )
    world.add_component(entity_id, original_reservation)

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

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    reservation = world.get_component(entity_id, PromptContextReservationComponent)
    assert queue is not None
    assert reservation is not None
    assert original_entry in queue.entries
    assert reservation == original_reservation


@pytest.mark.asyncio
async def test_collector_non_opt_in_entity_is_unchanged() -> None:
    world = World()
    entity_id = world.create_entity()
    original_entry = ContextEntry(
        entry_id="baseline-0",
        priority=1,
        registration_order=0,
        source_label="baseline",
        content="keep-me",
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            enable_context_pool=False, context_pool_max_chars=10000
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(entries=[original_entry]),
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

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert queue.entries == [original_entry]
