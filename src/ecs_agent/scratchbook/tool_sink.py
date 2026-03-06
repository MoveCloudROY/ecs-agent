"""Tool result append-only persistence sink."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ecs_agent.logging import get_logger
from ecs_agent.scratchbook.service import ScratchbookService

logger = get_logger(__name__)


class ToolResultsSink:
    """Append-only persistence sink for tool execution results."""

    CATEGORY = "tool_results"
    STABLE_ID_PREFIX = "tool-result-"

    def __init__(self, service: ScratchbookService) -> None:
        """Initialize sink with scratchbook service.

        Args:
            service: ScratchbookService instance for persisting artifacts
        """
        self.service = service
        self._persisted_call_ids: set[str] = set()

    def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """Persist tool execution result as immutable artifact.

        Args:
            tool_call_id: Unique identifier for tool call
            tool_name: Name of the tool
            result: Result string from tool execution
            arguments: Optional tool arguments dict

        Returns:
            Stable artifact ID (refs only, not full payload)

        Raises:
            ValueError: If attempting to persist result for already-persisted call_id
        """
        if tool_call_id in self._persisted_call_ids:
            msg = (
                f"Tool call {tool_call_id} already persisted; overwrites are immutable"
            )
            logger.error("tool_result_overwrite_attempted", tool_call_id=tool_call_id)
            raise ValueError(msg)

        # Create immutable artifact
        stable_id = f"{self.STABLE_ID_PREFIX}{tool_call_id}"
        artifact_data: dict[str, Any] = {
            "stable_id": stable_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if arguments:
            artifact_data["arguments"] = arguments

        # Write to scratchbook (permanent)
        self.service.write_artifact(
            artifact_id=stable_id, category=self.CATEGORY, data=artifact_data
        )

        # Track persisted call
        self._persisted_call_ids.add(tool_call_id)

        logger.info(
            "tool_result_persisted",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            stable_id=stable_id,
        )

        return stable_id

    def read_tool_result(self, stable_id: str) -> dict[str, Any] | None:
        """Read persisted tool result artifact.

        Args:
            stable_id: Stable artifact ID (from persist_tool_result)

        Returns:
            Artifact data dict or None if not found
        """
        return self.service.read_artifact(artifact_id=stable_id, category=self.CATEGORY)
