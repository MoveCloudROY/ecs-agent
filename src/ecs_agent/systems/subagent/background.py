"""Background subagent session lifecycle (Task 7).

``BackgroundSessionRunner`` replaces the ``_build_background_coroutine`` closure that
used to live on ``SubagentSystem``. It builds the ``run_in_background`` coroutine that
drives one background session through its phases: mark running, run (with optional
timeout) via the delegation core, then finalize success / failure / timeout /
cancellation — updating the runtime manager, persisting the artifact, enqueuing the
parent notification, and publishing delegation events.

The delegation-core / publish / persist operations are passed in per build (they are
monkeypatchable seams on ``SubagentSystem`` — capturing them at build time, which
happens at tool-invocation time after any test patching, keeps the patches effective).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.systems.subagent._contextvars import (
    _BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT,
    _PUBLISH_COMPLETION_EVENT,
)
from ecs_agent.systems.subagent.notifications import NotificationCoordinator
from ecs_agent.systems.subagent.result_envelope import parse_background_result_envelope
from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager
from ecs_agent.types import EntityId, SubagentConfig, SubagentSessionRecord

logger = get_logger(__name__)

# Seam signatures captured from SubagentSystem at build time.
ExecuteCore = Callable[..., Awaitable[tuple[str, bool, str | None]]]
PersistResult = Callable[[str], tuple[str, str, str | None] | None]
PublishEvents = Callable[..., Awaitable[None]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class BackgroundSessionRunner:
    """Builds the run-in-background coroutine for a background subagent session."""

    def __init__(
        self,
        runtime_manager: SubagentRuntimeManager,
        notification_coordinator: NotificationCoordinator,
    ) -> None:
        self._runtime_manager = runtime_manager
        self._notification_coordinator = notification_coordinator

    def build_coroutine(
        self,
        world: World,
        parent_entity_id: EntityId,
        category: str,
        prompt: str,
        session_id: str,
        metadata: SubagentSessionRecord,
        config: SubagentConfig,
        resolved_timeout: float | None,
        *,
        execute_core: ExecuteCore,
        persist_result: PersistResult,
        publish_events: PublishEvents,
    ) -> Callable[[], Awaitable[None]]:
        runtime_manager = self._runtime_manager
        notification_coordinator = self._notification_coordinator

        async def execute_with_config() -> tuple[str, bool, str | None]:
            token = _PUBLISH_COMPLETION_EVENT.set(False)
            launch_context_token = _BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT.set(
                (
                    metadata.launch_trace_id,
                    metadata.launch_run_id,
                    metadata.launch_parent_observation_id,
                )
            )
            try:
                if metadata.stream:
                    return await execute_core(
                        world,
                        parent_entity_id,
                        category,
                        prompt,
                        metadata.correlation_id,
                        metadata.traceparent,
                        config,
                        session_id=session_id,
                        stream=True,
                    )

                return await execute_core(
                    world,
                    parent_entity_id,
                    category,
                    prompt,
                    metadata.correlation_id,
                    metadata.traceparent,
                    config,
                )
            finally:
                _BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT.reset(launch_context_token)
                _PUBLISH_COMPLETION_EVENT.reset(token)

        async def run_in_background() -> None:
            metadata.status = "running"
            metadata.started_at = _utc_now_iso()
            metadata.updated_at = metadata.started_at
            await runtime_manager.update_status(session_id, metadata.status)
            await runtime_manager.sync_to_component(world, parent_entity_id)
            try:
                if resolved_timeout is not None:
                    result, success, error = await asyncio.wait_for(
                        execute_with_config(), timeout=resolved_timeout
                    )
                else:
                    result, success, error = await execute_with_config()
            except asyncio.CancelledError:
                metadata.finished_at = _utc_now_iso()
                metadata.updated_at = metadata.finished_at
                await runtime_manager.sync_to_component(world, parent_entity_id)
                raise
            except asyncio.TimeoutError:
                error_msg = f"Error: Subagent timeout after {resolved_timeout}s"
                metadata.finished_at = _utc_now_iso()
                metadata.updated_at = metadata.finished_at
                logger.error(
                    "subagent_background_timeout",
                    timeout=resolved_timeout,
                    category=category,
                )
                await runtime_manager.update_timeout(session_id, error_msg)
                await runtime_manager.sync_to_component(world, parent_entity_id)
                notification_coordinator.enqueue_parent_notification(world, metadata)
                await publish_events(
                    world,
                    parent_entity_id,
                    category,
                    correlation_id=metadata.correlation_id,
                    traceparent=metadata.traceparent,
                    task=metadata.prompt,
                    result=error_msg,
                    success=False,
                    error=error_msg,
                    trace_id=metadata.launch_trace_id,
                    run_id=metadata.launch_run_id,
                    parent_observation_id=metadata.launch_parent_observation_id,
                    observation_id=metadata.correlation_id,
                    publish_started=False,
                )
                return
            except Exception as exc:
                error_msg = str(exc)
                metadata.status = "failed"
                metadata.error = error_msg
                metadata.finished_at = _utc_now_iso()
                metadata.updated_at = metadata.finished_at
                await runtime_manager.update_status(session_id, metadata.status)
                await runtime_manager.sync_to_component(world, parent_entity_id)
                notification_coordinator.enqueue_parent_notification(world, metadata)
                await publish_events(
                    world,
                    parent_entity_id,
                    category,
                    correlation_id=metadata.correlation_id,
                    traceparent=metadata.traceparent,
                    task=metadata.prompt,
                    result=error_msg,
                    success=False,
                    error=error_msg,
                    trace_id=metadata.launch_trace_id,
                    run_id=metadata.launch_run_id,
                    parent_observation_id=metadata.launch_parent_observation_id,
                    observation_id=metadata.correlation_id,
                    publish_started=False,
                )
                return

            metadata.updated_at = _utc_now_iso()
            if success:
                parsed_result = parse_background_result_envelope(result)
                if parsed_result is None:
                    metadata.result_summary = None
                    full_result = result
                else:
                    metadata.result_summary, full_result = parsed_result
                metadata.status = "succeeded"
                metadata.finished_at = metadata.updated_at
                metadata.result_excerpt = full_result[:200]
                if len(full_result.encode("utf-8")) <= 8192:
                    metadata.artifact_inline_content = full_result
                persisted = persist_result(full_result)
                if persisted is not None:
                    artifact_id, record_path, inline_content = persisted
                    metadata.artifact_id = artifact_id
                    metadata.artifact_record_path = record_path
                    metadata.artifact_inline_content = inline_content
            else:
                metadata.status = "failed"
                metadata.finished_at = metadata.updated_at
                metadata.error = error
            await runtime_manager.update_status(session_id, metadata.status)
            logger.info(
                "subagent_background_finished",
                session_id=session_id,
                category=category,
                status=metadata.status,
                result_length=len(result) if success else 0,
                error=error,
            )
            await runtime_manager.sync_to_component(world, parent_entity_id)
            notification_coordinator.enqueue_parent_notification(world, metadata)
            await publish_events(
                world,
                parent_entity_id,
                category,
                correlation_id=metadata.correlation_id,
                traceparent=metadata.traceparent,
                task=metadata.prompt,
                result=result,
                success=success,
                error=error,
                trace_id=metadata.launch_trace_id,
                run_id=metadata.launch_run_id,
                parent_observation_id=metadata.launch_parent_observation_id,
                observation_id=metadata.correlation_id,
                publish_started=False,
            )

        return run_in_background


__all__ = ["BackgroundSessionRunner"]
