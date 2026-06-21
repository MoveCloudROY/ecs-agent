"""Publish-subscribe event bus implementation."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, TypeVar, cast

from ecs_agent.logging import get_logger, log_bus_deliver, log_bus_publish

T = TypeVar("T")
Handler = Callable[[Any], Awaitable[None]]

logger = get_logger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = {}

    def subscribe(
        self, event_type: type[T], handler: Callable[[T], Awaitable[None]]
    ) -> None:
        handlers = self._handlers.setdefault(event_type, [])
        handlers.append(cast(Handler, handler))

    def unsubscribe(
        self, event_type: type[T], handler: Callable[[T], Awaitable[None]]
    ) -> None:
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return

        try:
            handlers.remove(cast(Handler, handler))
        except ValueError:
            return

        if not handlers:
            del self._handlers[event_type]

    async def publish(self, event: T) -> None:
        handlers = list(self._handlers.get(type(event), []))
        if not handlers:
            return

        # Generate trace context for this publish operation
        trace_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        topic = type(event).__name__

        # Log publish event
        log_bus_publish(
            logger=logger,
            topic=topic,
            trace_id=trace_id,
            correlation_id=correlation_id,
            payload_type=type(event).__name__,
        )

        # Deliver to all handlers
        for idx, handler in enumerate(handlers):
            subscriber_id = getattr(handler, "__name__", f"handler_{idx}")
            log_bus_deliver(
                logger=logger,
                topic=topic,
                subscriber_id=subscriber_id,
                trace_id=trace_id,
                correlation_id=correlation_id,
            )

        results = await asyncio.gather(
            *(handler(event) for handler in handlers), return_exceptions=True
        )
        for idx, result in enumerate(results):
            if not isinstance(result, Exception):
                continue
            subscriber_id = getattr(handlers[idx], "__name__", f"handler_{idx}")
            logger.error(
                "event_bus_subscriber_error",
                topic=topic,
                subscriber_id=subscriber_id,
                exception=str(result),
            )

    def clear(self) -> None:
        self._handlers.clear()
