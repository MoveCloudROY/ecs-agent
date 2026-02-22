from ecs_agent.types import EntityId


class EntityIdGenerator:
    def __init__(self) -> None:
        self._counter = 0

    def next(self) -> EntityId:
        self._counter += 1
        return EntityId(self._counter)
