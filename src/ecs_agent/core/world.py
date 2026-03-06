from __future__ import annotations

import uuid
from typing import Any, TypeVar

from ecs_agent.components.definitions import EntityRegistryComponent
from ecs_agent.core.component import ComponentStore
from ecs_agent.core.entity import EntityIdGenerator
from ecs_agent.core.event_bus import EventBus
from ecs_agent.core.query import Query
from ecs_agent.core.system import System, SystemExecutor
from ecs_agent.types import EntityId
from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger

logger = get_logger(__name__)
T = TypeVar("T")


class World:
    def __init__(self) -> None:
        self._entity_gen = EntityIdGenerator()
        self._components = ComponentStore()
        self._systems = SystemExecutor()
        self._event_bus = EventBus()
        self._query = Query(self._components)
        self._registry_entity = EntityId(1)
        self.add_component(self._registry_entity, EntityRegistryComponent())

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    def create_entity(self) -> EntityId:
        entity_id = self._entity_gen.next()
        logger.debug(STANDARD_EVENT_NAMES["ENTITY_CREATED"], entity_id=int(entity_id))
        return entity_id

    def add_component(self, entity_id: EntityId, component: Any) -> None:
        self._components.add(entity_id, component)
        logger.info(
            STANDARD_EVENT_NAMES["COMPONENT_ADDED"],
            entity_id=int(entity_id),
            component_type=type(component).__name__,
        )

    def get_component(self, entity_id: EntityId, component_type: type[T]) -> T | None:
        return self._components.get(entity_id, component_type)

    def remove_component(self, entity_id: EntityId, component_type: type[Any]) -> None:
        self._components.remove(entity_id, component_type)

    def has_component(self, entity_id: EntityId, component_type: type[Any]) -> bool:
        return self._components.has(entity_id, component_type)

    def delete_entity(self, entity_id: EntityId) -> None:
        self._components.delete_entity(entity_id)

    def register_entity(
        self, entity_id: EntityId, name: str, tags: set[str] | None = None
    ) -> None:
        registry = self.get_component(self._registry_entity, EntityRegistryComponent)
        if registry:
            registry.names[name] = entity_id
            if tags:
                for tag in tags:
                    registry.tags.setdefault(tag, set()).add(entity_id)
            logger.info(
                "entity_registered",
                entity_id=int(entity_id),
                name=name,
                tags=list(tags or []),
            )

    def resolve_entity(self, name: str) -> EntityId:
        registry = self.get_component(self._registry_entity, EntityRegistryComponent)
        if registry and name in registry.names:
            return registry.names[name]
        raise KeyError(f"Entity name not found: {name}")

    def list_entities_by_tag(self, tag: str) -> list[EntityId]:
        registry = self.get_component(self._registry_entity, EntityRegistryComponent)
        if registry and tag in registry.tags:
            return list(registry.tags[tag])
        return []

    def unregister_entity(self, entity_id: EntityId) -> None:
        registry = self.get_component(self._registry_entity, EntityRegistryComponent)
        if registry:
            # Remove from names
            names_to_remove = [
                n for n, eid in registry.names.items() if eid == entity_id
            ]
            for name in names_to_remove:
                del registry.names[name]
            # Remove from tags
            for tag_entities in registry.tags.values():
                if entity_id in tag_entities:
                    tag_entities.remove(entity_id)
            logger.info("entity_unregistered", entity_id=int(entity_id))

    def register_system(self, system: System, priority: int) -> str:
        handle = str(uuid.uuid4())
        self._systems.register(system, priority, handle)
        return handle

    def remove_system(self, handle: str) -> None:
        self._systems.remove(handle)

    def replace_system(self, handle: str, new_system: System) -> None:
        self._systems.replace(handle, new_system)

    def apply_pending_system_operations(self) -> None:
        self._systems.apply_pending()

    async def process(self) -> None:
        await self._systems.execute(self)

    def query(
        self, *component_types: type[Any]
    ) -> list[tuple[EntityId, tuple[Any, ...]]]:
        return self._query.get(*component_types)
