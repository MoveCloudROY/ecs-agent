"""Public tool definitions shared by discovery and entity registration."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ecs_agent.types import ToolSchema


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """A tool's JSON parameter contract and its asynchronous handler.

    Explicit handlers receive JSON values as keyword arguments. Adapt custom
    Python objects inside the handler; a schema does not perform conversion.
    """

    schema: ToolSchema
    handler: Callable[..., Awaitable[str]]
