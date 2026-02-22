from ecs_agent.core.entity import EntityIdGenerator
from ecs_agent.types import EntityId


def test_entity_id_generator_starts_at_one() -> None:
    generator = EntityIdGenerator()
    assert generator.next() == EntityId(1)


def test_entity_id_generator_increments_sequentially() -> None:
    generator = EntityIdGenerator()
    ids = [generator.next(), generator.next(), generator.next()]
    assert ids == [EntityId(1), EntityId(2), EntityId(3)]


def test_entity_id_generator_produces_unique_ids() -> None:
    generator = EntityIdGenerator()
    ids = {generator.next() for _ in range(100)}
    assert len(ids) == 100
