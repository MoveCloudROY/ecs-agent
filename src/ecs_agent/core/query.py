from __future__ import annotations

from typing import Any

from ecs_agent.core.component import ComponentStore
from ecs_agent.types import EntityId


class Query:
    def __init__(self, component_store: ComponentStore) -> None:
        self._component_store = component_store

    def get(
        self, *component_types: type[Any]
    ) -> list[tuple[EntityId, tuple[Any, ...]]]:
        if not component_types:
            return []

        first_component_type = component_types[0]
        first_components = self._component_store.get_all(first_component_type)
        results: list[tuple[EntityId, tuple[Any, ...]]] = []

        for entity_id in first_components:
            if not all(
                self._component_store.has(entity_id, component_type)
                for component_type in component_types
            ):
                continue

            components = tuple(
                self._component_store.get(entity_id, component_type)
                for component_type in component_types
            )
            results.append((entity_id, components))

        return results
