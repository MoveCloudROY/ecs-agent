"""Skill protocol definitions."""

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from ecs_agent.core.world import World
from ecs_agent.types import EntityId, ToolSchema

ToolHandler = Callable[..., Awaitable[str]]


@runtime_checkable
class Skill(Protocol):
    name: str
    description: str

    def tools(self) -> dict[str, tuple[ToolSchema, ToolHandler]]: ...

    def system_prompt(self) -> str: ...

    def install(self, world: World, entity_id: EntityId) -> None: ...

    def uninstall(self, world: World, entity_id: EntityId) -> None: ...
