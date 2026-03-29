from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from ecs_agent.core.component import ComponentStore
from ecs_agent.core.entity import EntityIdGenerator
from ecs_agent.core.event_bus import EventBus
from ecs_agent.core.query import Query
from ecs_agent.core.system import System, SystemExecutor
from ecs_agent.types import EntityId, SystemHandle

from ecs_agent.logging import STANDARD_EVENT_NAMES, get_logger

logger = get_logger(__name__)
T = TypeVar("T")

if TYPE_CHECKING:
    from ecs_agent.skills.runtime import SkillRuntime


class World:
    def __init__(self, name: str | None = None) -> None:
        self._name: str | None = name
        self._entity_gen = EntityIdGenerator()
        self._components = ComponentStore()
        self._systems = SystemExecutor()
        self._event_bus = EventBus()
        self._query = Query(self._components)
        self._skill_runtime: SkillRuntime | None = None
        self._entity_registry: dict[str, EntityId] = {}
        self._entity_tags: dict[str, set[EntityId]] = {}
        self._entity_ids: set[EntityId] = set()

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def name(self) -> str | None:
        """Optional human-readable name for this World instance."""
        return self._name

    @property
    def skill_runtime(self) -> SkillRuntime:
        if self._skill_runtime is None:
            from ecs_agent.skills.runtime import SkillRuntime

            self._skill_runtime = SkillRuntime()
        return self._skill_runtime

    def create_entity(self) -> EntityId:
        entity_id = self._entity_gen.next()
        self._entity_ids.add(entity_id)
        logger.debug(
            STANDARD_EVENT_NAMES["ENTITY_CREATED"],
            entity_id=int(entity_id),
            world_name=self._name,
        )
        return entity_id

    def add_component(self, entity_id: EntityId, component: Any) -> None:
        self._components.add(entity_id, component)
        logger.info(
            STANDARD_EVENT_NAMES["COMPONENT_ADDED"],
            entity_id=int(entity_id),
            component_type=type(component).__name__,
            world_name=self._name,
        )

    def get_component(self, entity_id: EntityId, component_type: type[T]) -> T | None:
        return self._components.get(entity_id, component_type)

    def remove_component(self, entity_id: EntityId, component_type: type[Any]) -> None:
        self._components.remove(entity_id, component_type)

    def has_component(self, entity_id: EntityId, component_type: type[Any]) -> bool:
        return self._components.has(entity_id, component_type)

    def has_entity(self, entity_id: EntityId) -> bool:
        """Return True if entity_id was created in this world."""
        return entity_id in self._entity_ids

    def delete_entity(self, entity_id: EntityId) -> None:
        self.unregister_entity(entity_id)
        self._components.delete_entity(entity_id)
        self._entity_ids.discard(entity_id)

    def register_entity(
        self, entity_id: EntityId, name: str, tags: set[str] | None = None
    ) -> None:
        """Register entity with unique name and optional tags.

        Args:
            entity_id: Entity to register
            name: Unique name for entity lookup
            tags: Optional set of tags for entity grouping

        Raises:
            ValueError: If name already registered
        """
        if name in self._entity_registry:
            raise ValueError(f"Entity name '{name}' already registered")

        self._entity_registry[name] = entity_id

        if tags:
            for tag in tags:
                if tag not in self._entity_tags:
                    self._entity_tags[tag] = set()
                self._entity_tags[tag].add(entity_id)

    def resolve_entity(self, name: str) -> EntityId | None:
        """Lookup entity by registered name.

        Args:
            name: Registered entity name

        Returns:
            EntityId if found, None otherwise
        """
        return self._entity_registry.get(name)

    def list_entities_by_tag(self, tag: str) -> list[EntityId]:
        """Find all entities with given tag.

        Args:
            tag: Tag to search for

        Returns:
            List of entity IDs with the tag (empty if tag not found)
        """
        return list(self._entity_tags.get(tag, set()))

    def unregister_entity(self, entity_id: EntityId) -> None:
        """Remove entity from registry and tag indexes.

        Args:
            entity_id: Entity to unregister
        """
        # Find and remove from name registry
        name_to_remove = None
        for name, eid in self._entity_registry.items():
            if eid == entity_id:
                name_to_remove = name
                break

        if name_to_remove:
            del self._entity_registry[name_to_remove]

        # Remove from all tag indexes
        for tag_set in self._entity_tags.values():
            tag_set.discard(entity_id)

    def register_system(self, system: System, priority: int) -> SystemHandle:
        return self._systems.register(system, priority)

    def remove_system(self, handle: SystemHandle) -> None:
        self._systems.remove(handle)

    def replace_system(
        self, handle: SystemHandle, system: System, priority: int | None = None
    ) -> None:
        self._systems.replace(handle, system, priority)

    def apply_pending_system_operations(self) -> None:
        self._systems.apply_queued_operations()

    async def process(self) -> None:
        self.apply_pending_system_operations()
        await self._systems.execute(self)

    def query(
        self, *component_types: type[Any]
    ) -> list[tuple[EntityId, tuple[Any, ...]]]:
        return self._query.get(*component_types)
