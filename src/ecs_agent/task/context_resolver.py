"""Context resolver for task dependencies and execution inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecs_agent.components import TaskComponent
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook import ScratchbookService
from ecs_agent.types import TaskStatus

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ResolvedContext:
    """Resolved context data for a task."""

    task_id: str
    resolved_data: dict[str, Any]
    missing_refs: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ContextResolutionError:
    """Error during context resolution."""

    task_id: str
    missing_refs: tuple[str, ...]
    reason: str


class ContextResolver:
    """Resolves task context dependencies from scratchbook artifacts."""

    def __init__(self, service: ScratchbookService) -> None:
        """Initialize resolver with scratchbook service.

        Args:
            service: ScratchbookService for artifact retrieval
        """
        self.service = service

    def resolve_context(
        self, task: TaskComponent, running_task_ids: set[str] | None = None
    ) -> ResolvedContext | ContextResolutionError:
        """Resolve context dependencies for a task.

        Args:
            task: TaskComponent to resolve context for
            running_task_ids: Set of task IDs currently running (grandfathered)

        Returns:
            ResolvedContext with resolved data or ContextResolutionError
        """
        # Grandfathering: Running tasks keep existing context
        if running_task_ids and task.task_id in running_task_ids:
            logger.info(
                "context_grandfathered",
                task_id=task.task_id,
                status=task.status.value,
            )
            return ResolvedContext(
                task_id=task.task_id, resolved_data={}, missing_refs=()
            )

        resolved_data: dict[str, Any] = {}
        missing_refs: list[str] = []

        for ref in task.context_dependencies:
            # Parse ref format: "category/artifact_id" or just "artifact_id"
            if "/" in ref:
                parts = ref.split("/", 1)
                category = parts[0]
                artifact_id = parts[1]
            else:
                # Default to common categories if no category specified
                # Try tool_results first, then planning, then replanning
                artifact_id = ref
                artifact = self._try_categories(artifact_id)
                if artifact is not None:
                    resolved_data[ref] = artifact
                    continue
                else:
                    missing_refs.append(ref)
                    continue

            # Fetch artifact from scratchbook
            artifact = self.service.read_artifact(
                artifact_id=artifact_id, category=category
            )

            if artifact is None:
                missing_refs.append(ref)
                logger.warning(
                    "context_missing_artifact",
                    task_id=task.task_id,
                    ref=ref,
                    artifact_id=artifact_id,
                    category=category,
                )
            else:
                resolved_data[ref] = artifact
                logger.debug(
                    "context_resolved_artifact",
                    task_id=task.task_id,
                    ref=ref,
                    artifact_id=artifact_id,
                    category=category,
                )

        if missing_refs:
            return ContextResolutionError(
                task_id=task.task_id,
                missing_refs=tuple(missing_refs),
                reason=f"missing dependencies: {', '.join(missing_refs)}",
            )

        logger.info(
            "context_resolved",
            task_id=task.task_id,
            ref_count=len(task.context_dependencies),
            resolved_count=len(resolved_data),
        )

        return ResolvedContext(
            task_id=task.task_id, resolved_data=resolved_data, missing_refs=()
        )

    def _try_categories(self, artifact_id: str) -> dict[str, Any] | None:
        """Try multiple default categories for artifact_id.

        Args:
            artifact_id: Artifact ID to search for

        Returns:
            Artifact data or None if not found in any category
        """
        categories = ["tool_results", "planning", "replanning"]
        for category in categories:
            artifact = self.service.read_artifact(
                artifact_id=artifact_id, category=category
            )
            if artifact is not None:
                return artifact
        return None

    def inject_context_into_snapshot(
        self, snapshot: dict[str, Any], resolved: ResolvedContext
    ) -> dict[str, Any]:
        """Inject resolved context into execution snapshot.

        Args:
            snapshot: Base execution snapshot
            resolved: Resolved context data

        Returns:
            Enhanced snapshot with context data injected
        """
        enhanced = dict(snapshot)
        enhanced["context"] = resolved.resolved_data
        logger.debug(
            "context_injected",
            task_id=resolved.task_id,
            context_keys=list(resolved.resolved_data.keys()),
        )
        return enhanced


__all__ = ["ContextResolver", "ResolvedContext", "ContextResolutionError"]
