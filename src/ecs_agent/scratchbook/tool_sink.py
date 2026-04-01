"""Tool result append-only persistence sink."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.scratchbook.artifact_registry import (
    ArtifactKind,
    ArtifactPersistResult,
    ArtifactRegistry,
)

logger = get_logger(__name__)


class ToolResultsSink:
    """Append-only persistence sink for tool execution results."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        """Initialize sink with artifact registry.

        Args:
            registry: ArtifactRegistry instance for persisting artifacts
        """
        self.registry = registry
        self._persisted_call_ids: set[str] = set()
        # NOTE: _persisted_call_ids is a session-scoped in-memory guard only.
        # It prevents duplicate persistence within a single process lifetime,
        # but does NOT provide durable cross-restart idempotency.
        # If the process restarts, the same tool_call_id can be persisted again.

    def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        arguments: dict[str, Any] | None = None,
    ) -> ArtifactPersistResult:
        """Persist tool execution result as immutable artifact.

        Args:
            tool_call_id: Unique identifier for tool call
            tool_name: Name of the tool
            result: Result string from tool execution
            arguments: Optional tool arguments dict

        Returns:
            Artifact persistence envelope with canonical record path

        Raises:
            ValueError: If attempting to persist result for already-persisted call_id
        """
        if tool_call_id in self._persisted_call_ids:
            msg = (
                f"Tool call {tool_call_id} already persisted; overwrites are immutable"
            )
            logger.error("tool_result_overwrite_attempted", tool_call_id=tool_call_id)
            raise ValueError(msg)

        artifact_data: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arguments": arguments,
        }

        persist_result = self.registry.persist(
            kind=ArtifactKind.TOOL,
            content=json.dumps(artifact_data),
        )

        # Track persisted call
        self._persisted_call_ids.add(tool_call_id)

        logger.info(
            "tool_result_persisted",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            record_path=persist_result.record_path,
        )

        return persist_result

    def read_tool_result(self, stable_id: str) -> dict[str, Any] | None:
        """Read persisted tool result artifact.

        Args:
            stable_id: Relative record path from persist_tool_result

        Returns:
            Artifact data dict or None if not found
        """
        artifact_path = self.registry.root / stable_id
        if not artifact_path.exists():
            return None
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload
