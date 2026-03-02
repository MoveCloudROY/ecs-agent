from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest


def _create_message_bus_system(
    *,
    buffer_size: int = 100,
    publish_timeout: float = 2.0,
    request_timeout: float = 30.0,
) -> Any:
    try:
        module = importlib.import_module("ecs_agent.systems.message_bus")
    except ModuleNotFoundError as exc:
        pytest.fail(f"MessageBusSystem not implemented: {exc}")

    message_bus_system = getattr(module, "MessageBusSystem", None)
    if message_bus_system is None:
        pytest.fail("MessageBusSystem not implemented: missing MessageBusSystem class")

    return message_bus_system(
        buffer_size=buffer_size,
        publish_timeout=publish_timeout,
        request_timeout=request_timeout,
    )


def _single_pending_request_id(message_bus: Any) -> str:
    pending_requests = getattr(message_bus, "_pending_requests", None)
    if not isinstance(pending_requests, dict):
        pytest.fail("MessageBusSystem contract requires _pending_requests dict")
    if len(pending_requests) != 1:
        pytest.fail("Expected exactly one pending request for contract assertion")
    return next(iter(pending_requests.keys()))


class TestMessageBusContract:
    @pytest.mark.asyncio
    async def test_fifo_per_subscriber_ordering(self) -> None:
        """FIFO is per-subscriber: each subscriber receives its own ordered stream."""
        bus = _create_message_bus_system(buffer_size=10)

        subscriber_a = bus.subscribe(topic="agent.updates", subscriber_id="alpha")
        subscriber_b = bus.subscribe(topic="agent.updates", subscriber_id="beta")

        await bus.publish(topic="agent.updates", message={"seq": 1})
        await bus.publish(topic="agent.updates", message={"seq": 2})
        await bus.publish(topic="agent.updates", message={"seq": 3})

        seen_a = [
            await asyncio.wait_for(subscriber_a.get(), timeout=0.1) for _ in range(3)
        ]
        seen_b = [
            await asyncio.wait_for(subscriber_b.get(), timeout=0.1) for _ in range(3)
        ]

        assert [msg["seq"] for msg in seen_a] == [1, 2, 3]
        assert [msg["seq"] for msg in seen_b] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_publish_timeout_when_buffer_full(self) -> None:
        """Backpressure: publish times out when a subscriber buffer is full."""
        bus = _create_message_bus_system(buffer_size=1, publish_timeout=0.01)
        _ = bus.subscribe(topic="agent.updates", subscriber_id="alpha")

        await bus.publish(topic="agent.updates", message={"seq": 1})

        with pytest.raises(TimeoutError):
            await bus.publish(topic="agent.updates", message={"seq": 2})

    @pytest.mark.asyncio
    async def test_request_timeout_cleans_pending(self) -> None:
        """Request timeout must remove pending futures to prevent leaks."""
        bus = _create_message_bus_system(request_timeout=0.02)

        with pytest.raises(TimeoutError):
            await bus.request(topic="rpc.compute", message={"value": 5})

        pending_requests = getattr(bus, "_pending_requests", None)
        assert isinstance(pending_requests, dict)
        assert pending_requests == {}

    @pytest.mark.asyncio
    async def test_late_response_after_timeout_ignored(self) -> None:
        """Late response after timeout is rejected and does not recreate pending state."""
        bus = _create_message_bus_system(request_timeout=0.01)

        with pytest.raises(TimeoutError):
            await bus.request(topic="rpc.compute", message={"value": 9})

        accepted = await bus.respond(
            correlation_id="expired-request-id",
            message={"result": 99},
        )

        assert accepted is False
        pending_requests = getattr(bus, "_pending_requests", None)
        assert isinstance(pending_requests, dict)
        assert pending_requests == {}

    @pytest.mark.asyncio
    async def test_single_response_contract_first_response_wins(self) -> None:
        """Single-response contract: first response resolves request, duplicates rejected."""
        bus = _create_message_bus_system(request_timeout=0.2)

        request_task = asyncio.create_task(
            bus.request(topic="rpc.compute", message={"value": 11})
        )
        await asyncio.sleep(0)

        request_id = _single_pending_request_id(bus)

        first_accepted = await bus.respond(
            correlation_id=request_id,
            message={"result": 12},
        )
        second_accepted = await bus.respond(
            correlation_id=request_id,
            message={"result": 13},
        )

        result = await asyncio.wait_for(request_task, timeout=0.1)

        assert first_accepted is True
        assert second_accepted is False
        assert result == {"result": 12}
