from __future__ import annotations

from typing import Any, TypeVar, cast

from ecs_agent.types import EntityId

T = TypeVar("T")


class ComponentStore:
    def __init__(self) -> None:
        self._components: dict[type[Any], dict[EntityId, Any]] = {}

    def add(self, entity_id: EntityId, component: Any) -> None:
        component_type = type(component)
        entities = self._components.setdefault(component_type, {})
        entities[entity_id] = component

    def get(self, entity_id: EntityId, component_type: type[T]) -> T | None:
        entities = self._components.get(component_type)
        if entities is None:
            return None
        return cast(T | None, entities.get(entity_id))

    def remove(self, entity_id: EntityId, component_type: type[Any]) -> None:
        entities = self._components.get(component_type)
        if entities is None:
            return

        entities.pop(entity_id, None)
        if not entities:
            del self._components[component_type]

    def has(self, entity_id: EntityId, component_type: type[Any]) -> bool:
        entities = self._components.get(component_type)
        if entities is None:
            return False
        return entity_id in entities

    def delete_entity(self, entity_id: EntityId) -> None:
        empty_component_types: list[type[Any]] = []
        for component_type, entities in self._components.items():
            entities.pop(entity_id, None)
            if not entities:
                empty_component_types.append(component_type)

        for component_type in empty_component_types:
            del self._components[component_type]

    def get_all(self, component_type: type[T]) -> dict[EntityId, T]:
        entities = self._components.get(component_type)
        if entities is None:
            return {}
        return cast(dict[EntityId, T], entities)
