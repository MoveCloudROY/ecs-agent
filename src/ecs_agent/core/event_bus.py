from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar, cast

T = TypeVar("T")
Handler = Callable[[Any], Awaitable[None]]

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type[Any], list[Handler]] = {}

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

        results = await asyncio.gather(
            *(handler(event) for handler in handlers), return_exceptions=True
        )

        for result in results:
            if isinstance(result, BaseException):
                logger.error(
                    "Event handler failed for %s",
                    type(event).__name__,
                    exc_info=(type(result), result, result.__traceback__),
                )

    def clear(self) -> None:
        self._handlers.clear()
