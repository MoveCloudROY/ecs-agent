from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest

from ecs_agent.core import EventBus


@dataclass(slots=True)
class SampleEvent:
    value: int


@dataclass(slots=True)
class OtherEvent:
    name: str


def test_event_bus_instantiation() -> None:
    bus = EventBus()
    assert bus is not None
    assert bus._handlers == {}


def test_subscribe_registers_handler() -> None:
    bus = EventBus()

    async def handler(event: SampleEvent) -> None:
        _ = event

    bus.subscribe(SampleEvent, handler)
    assert SampleEvent in bus._handlers
    assert handler in bus._handlers[SampleEvent]


@pytest.mark.asyncio
async def test_publish_triggers_matching_handler() -> None:
    bus = EventBus()
    seen: list[int] = []

    async def handler(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, handler)
    await bus.publish(SampleEvent(value=7))
    assert seen == [7]


@pytest.mark.asyncio
async def test_publish_triggers_all_handlers_for_same_event_type() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler_one(event: SampleEvent) -> None:
        seen.append(f"one:{event.value}")

    async def handler_two(event: SampleEvent) -> None:
        seen.append(f"two:{event.value}")

    bus.subscribe(SampleEvent, handler_one)
    bus.subscribe(SampleEvent, handler_two)

    await bus.publish(SampleEvent(value=3))
    assert sorted(seen) == ["one:3", "two:3"]


@pytest.mark.asyncio
async def test_publish_executes_handlers_in_parallel() -> None:
    bus = EventBus()

    async def handler_one(event: SampleEvent) -> None:
        _ = event
        await asyncio.sleep(0.05)

    async def handler_two(event: SampleEvent) -> None:
        _ = event
        await asyncio.sleep(0.05)

    bus.subscribe(SampleEvent, handler_one)
    bus.subscribe(SampleEvent, handler_two)

    started = time.perf_counter()
    await bus.publish(SampleEvent(value=1))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.09


@pytest.mark.asyncio
async def test_unsubscribe_removes_specific_handler() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler_keep(event: SampleEvent) -> None:
        seen.append(f"keep:{event.value}")

    async def handler_remove(event: SampleEvent) -> None:
        seen.append(f"remove:{event.value}")

    bus.subscribe(SampleEvent, handler_keep)
    bus.subscribe(SampleEvent, handler_remove)
    bus.unsubscribe(SampleEvent, handler_remove)

    await bus.publish(SampleEvent(value=9))
    assert seen == ["keep:9"]


@pytest.mark.asyncio
async def test_handler_exception_isolated_from_other_handlers() -> None:
    bus = EventBus()
    seen: list[int] = []

    async def bad_handler(event: SampleEvent) -> None:
        _ = event
        raise RuntimeError("boom")

    async def good_handler(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, bad_handler)
    bus.subscribe(SampleEvent, good_handler)

    await bus.publish(SampleEvent(value=42))
    assert seen == [42]


@pytest.mark.asyncio
async def test_publish_unsubscribed_event_type_is_silent_noop() -> None:
    bus = EventBus()
    await bus.publish(OtherEvent(name="unused"))


@pytest.mark.asyncio
async def test_clear_removes_all_subscriptions() -> None:
    bus = EventBus()
    seen: list[int] = []

    async def handler(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, handler)
    bus.clear()

    await bus.publish(SampleEvent(value=5))
    assert seen == []
    assert bus._handlers == {}
