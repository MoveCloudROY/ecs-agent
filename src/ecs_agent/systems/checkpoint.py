"""World state checkpoint system."""

from __future__ import annotations

import time
from typing import Any

from ecs_agent.components import CheckpointComponent
from ecs_agent.core.query import Query
from ecs_agent.core.world import World
from ecs_agent.serialization import WorldSerializer
from ecs_agent.types import CheckpointCreatedEvent, CheckpointRestoredEvent, EntityId
from ecs_agent.logging import get_logger

logger = get_logger(__name__)

class CheckpointSystem:
    """Creates and restores tick-level World snapshots for undo functionality."""
    async def process(self, world: World) -> None:
        start_time = time.monotonic()
        snapshot = WorldSerializer.to_dict(world)
        timestamp = time.time()

        for entity_id, components in world.query(CheckpointComponent):
            checkpoint = components[0]
            checkpoint.snapshots.append(snapshot)

            if len(checkpoint.snapshots) > checkpoint.max_snapshots:
                overflow = len(checkpoint.snapshots) - checkpoint.max_snapshots
                del checkpoint.snapshots[:overflow]

            await world.event_bus.publish(
                CheckpointCreatedEvent(
                    entity_id=entity_id,
                    checkpoint_id=len(checkpoint.snapshots) - 1,
                    timestamp=timestamp,
                )
            )
            
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "checkpoint_saved",
                success=True,
                duration_ms=duration_ms,
                checkpoint_id=len(checkpoint.snapshots) - 1,
            )

    @staticmethod
    async def undo(
        world: World,
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> None:
        start_time = time.monotonic()
        try:
            checkpoints = world.query(CheckpointComponent)
            if not checkpoints:
                logger.error("checkpoint_restore_failed", reason="No checkpoint component found")
                raise ValueError("No checkpoint snapshots available to restore")

            entity_id, components = checkpoints[0]
            checkpoint = components[0]
            if not checkpoint.snapshots:
                logger.error("checkpoint_restore_failed", reason="No snapshots available")
                raise ValueError("No checkpoint snapshots available to restore")

            popped_snapshot = checkpoint.snapshots.pop()
            snapshot = checkpoint.snapshots[-1] if checkpoint.snapshots else popped_snapshot

            restored_world = WorldSerializer.from_dict(
                snapshot,
                providers=providers,
                tool_handlers=tool_handlers,
            )

            world._entity_gen = restored_world._entity_gen
            world._components = restored_world._components
            world._entity_registry = restored_world._entity_registry
            world._entity_tags = restored_world._entity_tags
            world._entity_ids = restored_world._entity_ids
            world._query = Query(world._components)

            restored_checkpoint = world.get_component(entity_id, CheckpointComponent)
            if restored_checkpoint is None:
                world.add_component(
                    entity_id,
                    CheckpointComponent(
                        snapshots=checkpoint.snapshots,
                        max_snapshots=checkpoint.max_snapshots,
                    ),
                )
            else:
                restored_checkpoint.snapshots = checkpoint.snapshots
                restored_checkpoint.max_snapshots = checkpoint.max_snapshots

            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "checkpoint_restored",
                success=True,
                checkpoint_id=len(checkpoint.snapshots),
                duration_ms=duration_ms,
            )

            await world.event_bus.publish(
                CheckpointRestoredEvent(
                    entity_id=EntityId(entity_id),
                    checkpoint_id=len(checkpoint.snapshots),
                    timestamp=time.time(),
                )
            )
        except ValueError:
            # Already logged above, just re-raise
            raise
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "checkpoint_restore_failed",
                reason=str(exc),
                duration_ms=duration_ms,
            )
            raise


__all__ = ["CheckpointSystem"]
