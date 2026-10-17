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

    def has_subscribers(self, event_type: type) -> bool:
        """Return True when at least one handler is subscribed to ``event_type``.

        Lets a publisher that blocks on a subscriber-resolved future (e.g. an
        interactive tool awaiting user input) fail fast instead of hanging when
        no listener is attached.
        """
        return bool(self._handlers.get(event_type))

    def clear(self) -> None:
        self._handlers.clear()


_PUBLISHER_CLOSE = object()

# Strong references to in-flight publisher drain tasks: asyncio only keeps
# weak references to tasks, and these intentionally outlive the scope that
# spawned them (producers finish while subscribers catch up asynchronously).
_ACTIVE_PUBLISHER_TASKS: set[asyncio.Task[None]] = set()


class SequentialEventPublisher:
    """Order-preserving, non-blocking publisher for a burst of related events.

    Events are enqueued without blocking the producer; a single lazily
    started consumer task publishes them strictly in FIFO order, so a
    subscriber can never observe a later event (e.g. a stream end marker)
    before an earlier one. Publish errors are logged and never reach the
    producer — matching EventBus.publish's own subscriber-error isolation.
    """

    def __init__(self, event_bus: "EventBus", label: str) -> None:
        self._event_bus = event_bus
        self._label = label
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def publish_nowait(self, event: Any) -> None:
        """Enqueue an event; the consumer task is started lazily."""
        if self._task is None:
            self._task = asyncio.create_task(self._drain_loop())
            _ACTIVE_PUBLISHER_TASKS.add(self._task)
            self._task.add_done_callback(_ACTIVE_PUBLISHER_TASKS.discard)
        self._queue.put_nowait(event)

    async def _drain_loop(self) -> None:
        while True:
            event = await self._queue.get()
            if event is _PUBLISHER_CLOSE:
                return
            try:
                await self._event_bus.publish(event)
            except Exception as exc:
                logger.error(
                    "sequential_publisher_error",
                    label=self._label,
                    error=str(exc),
                )

    def close(self) -> None:
        """Stop the consumer after the queued tail has been delivered.

        Deliberately not awaited: producers finish immediately while the
        flush continues in the background. FIFO ordering still guarantees the
        tail is delivered in order; the task stays alive via
        _ACTIVE_PUBLISHER_TASKS until the flush completes.
        """
        if self._task is None:
            return
        self._queue.put_nowait(_PUBLISHER_CLOSE)
        self._task = None

    def cancel(self) -> None:
        """Drop queued events (teardown path: the burst is being abandoned)."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
