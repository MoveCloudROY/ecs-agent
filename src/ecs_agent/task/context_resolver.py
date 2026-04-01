"""Context resolver for task dependencies and execution inputs."""

from __future__ import annotations

import json
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
            if ref.startswith("scratchbook/"):
                artifact = self._read_artifact_by_record_path(record_path=ref)
                if artifact is None:
                    missing_refs.append(ref)
                    logger.warning(
                        "context_missing_artifact",
                        task_id=task.task_id,
                        ref=ref,
                        record_path=ref,
                    )
                else:
                    resolved_data[ref] = artifact
                    logger.debug(
                        "context_resolved_artifact",
                        task_id=task.task_id,
                        ref=ref,
                        record_path=ref,
                    )
                continue

            elif "/" not in ref:
                missing_refs.append(ref)
                logger.warning(
                    "context_missing_artifact",
                    task_id=task.task_id,
                    ref=ref,
                    reason="unsupported_ref_format",
                )

            elif ref.startswith("tasks/"):
                category, artifact_id = ref.split("/", 1)
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

            else:
                missing_refs.append(ref)
                logger.warning(
                    "context_missing_artifact",
                    task_id=task.task_id,
                    ref=ref,
                    reason="unknown_ref_format",
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

    def _read_artifact_by_record_path(self, record_path: str) -> Any | None:
        artifact_path = self.service.root / record_path
        if not artifact_path.exists():
            return None

        text = artifact_path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

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
