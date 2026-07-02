"""Subagent delegation system."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Literal

from ecs_agent.components import (
    ChildStubComponent,
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    StreamingComponent,
    SubagentNotificationQueueComponent,
    SubagentRegistryComponent,
    SubagentWaitComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.systems.subagent_wait import (
    ResumeCallback,
    notification_matches_wait,
    wait_scope_is_terminal,
)
from ecs_agent.scratchbook.artifact_registry import ArtifactKind, ArtifactRegistry
from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager
from ecs_agent.observability import generate_traceparent
from ecs_agent.systems.subagent._contextvars import (
    _BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT,
    _BACKGROUND_RESULT_ENVELOPE_ENABLED,
    _PUBLISH_COMPLETION_EVENT,
)
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.subagent.result_envelope import (
    _build_background_child_prompt_template,
    _build_child_prompt_template,
    parse_background_result_envelope,
)
from ecs_agent.systems.subagent.tool_schemas import (
    build_cancel_schema,
    build_result_schema,
    build_resume_schema,
    build_status_schema,
    build_subagent_schema,
    build_wait_schema,
)
from ecs_agent.observability.context import current_run_id, current_trace_id
from ecs_agent.observability.install import install_observability
from ecs_agent.types import (
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    FreeSubagentConfig,
    InheritancePolicy,
    Message,
    render_subagent_session_reminder_table,
    RetryConfig,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamStartEvent,
    SubagentConfig,
    SubagentNotificationRecord,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
    SubagentSessionRecord,
    ToolSchema,
    is_wake_worthy,
)

from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.retry_model import RetryModel

logger = get_logger(__name__)

# ContextVars moved to _contextvars.py (cycle-safe home for submodules); imported below.

# Background-result envelope + child prompt templates live in result_envelope.py,
# imported at the top of this module and re-exported to preserve historical paths:
#   from ecs_agent.systems.subagent import _build_child_prompt_template, ...


class SubagentSystem:
    """System that manages subagent delegation lifecycle.

    This system automatically registers a 'subagent' tool for entities that have
    a SubagentRegistryComponent. When the subagent tool is called, it:
    1. Creates a child entity with the specified subagent configuration
    2. Runs the child entity to completion
    3. Returns the child's final assistant message
    4. Publishes delegation events to the event bus
    """

    def __init__(
        self,
        priority: int = -1,
        default_timeout: float | None = None,
        registry: ArtifactRegistry | None = None,
        max_background_concurrency: int = 5,
        allow_unregistered_subagents: bool = False,
        free_subagent_config: FreeSubagentConfig | None = None,
    ) -> None:
        self.priority = priority
        self._runtime_manager = SubagentRuntimeManager(
            max_background_concurrency=max_background_concurrency
        )
        self._default_timeout = default_timeout
        self._registry = registry
        self._free_subagent_config = free_subagent_config or FreeSubagentConfig(
            enabled=allow_unregistered_subagents
        )
        self._child_world_builder = ChildWorldBuilder()
        self._reconciled_session_ids: set[str] = set()

    def _persist_subagent_result(
        self,
        result: str,
    ) -> tuple[str, str, str | None] | None:
        if self._registry is None:
            return None

        persist_result = self._registry.persist(
            kind=ArtifactKind.SUBAGENT,
            content=result,
        )
        return (
            persist_result.descriptor.artifact_id,
            persist_result.record_path,
            persist_result.inline_content,
        )

    def _session_inline_content(
        self,
        session: SubagentSessionRecord,
    ) -> str | None:
        if session.artifact_inline_content is not None:
            return session.artifact_inline_content

        if session.artifact_record_path is not None:
            return (
                f"Result persisted to {session.artifact_record_path}. "
                "Read that file to access the full content."
            )

        return None

    def _session_payload(
        self,
        session: SubagentSessionRecord,
        *,
        status: str,
        queue_position: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "session_id": session.session_id,
            "category": session.category,
            "lifecycle_status": session.status,
            "artifact_id": session.artifact_id,
            "record_path": session.artifact_record_path,
            "inline_content": self._session_inline_content(session),
            "error": session.error,
        }
        if queue_position is not None:
            payload["queue_position"] = queue_position

        return payload

    def _summary_payload(
        self,
        session: SubagentSessionRecord,
        *,
        status: str,
        queue_position: int | None = None,
    ) -> dict[str, Any]:
        payload = self._session_payload(
            session,
            status=status,
            queue_position=queue_position,
        )
        payload["read_method"] = "summary"
        payload["inline_content"] = session.result_summary
        return payload

    def _resolve_timeout(self, per_call_timeout: float | None) -> float | None:
        """Resolve timeout with precedence: per-call > global > None."""
        return (
            per_call_timeout if per_call_timeout is not None else self._default_timeout
        )

    def _active_observability_context(
        self,
        world: World,
    ) -> tuple[str | None, str | None, str | None]:
        """Return active trace, run, and root observation IDs for this world."""
        active_run_id = current_run_id()
        active_trace_id = current_trace_id()
        active_parent_observation_id: str | None = None
        parent_subscriber = getattr(
            world,
            "_ecs_agent_observability_subscriber",
            None,
        )
        trace_states = getattr(parent_subscriber, "trace_states", None)
        if isinstance(active_run_id, str) and isinstance(trace_states, dict):
            trace_state = trace_states.get(active_run_id)
            trace_state_id = getattr(trace_state, "trace_id", None)
            trace_state_observation_id = getattr(trace_state, "observation_id", None)
            if isinstance(trace_state_id, str):
                active_trace_id = trace_state_id
            if isinstance(trace_state_observation_id, str):
                active_parent_observation_id = trace_state_observation_id
        return (active_trace_id, active_run_id, active_parent_observation_id)

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _notification_summary(
        self,
        metadata: SubagentSessionRecord,
    ) -> str | None:
        return metadata.result_summary

    def _get_or_create_notification_queue(
        self,
        world: World,
        parent_entity_id: EntityId,
    ) -> SubagentNotificationQueueComponent:
        queue = world.get_component(
            parent_entity_id, SubagentNotificationQueueComponent
        )
        if queue is not None:
            return queue

        queue = SubagentNotificationQueueComponent()
        world.add_component(parent_entity_id, queue)
        return queue

    def _all_waited_sessions_terminal(
        self,
        world: World,
        parent_entity_id: EntityId,
        wait_component: SubagentWaitComponent,
    ) -> bool:
        """Return True when every session in the wait scope is terminal."""
        from ecs_agent.components.definitions import SubagentSessionTableComponent

        table = world.get_component(parent_entity_id, SubagentSessionTableComponent)
        effective_ids = (
            wait_component.resolved_session_ids or wait_component.session_ids
        )
        return wait_scope_is_terminal(table, effective_ids).all_terminal

    def _enqueue_parent_notification(
        self,
        world: World,
        metadata: SubagentSessionRecord,
    ) -> None:
        if not metadata.background or not is_wake_worthy(metadata.status):
            return

        terminal_status: Literal["succeeded", "failed", "timed_out"]
        if metadata.status == "succeeded":
            terminal_status = "succeeded"
        elif metadata.status == "failed":
            terminal_status = "failed"
        else:
            terminal_status = "timed_out"

        notification_id = f"{metadata.session_id}:{terminal_status}"
        queue = self._get_or_create_notification_queue(world, metadata.parent_entity_id)
        if any(
            notification.notification_id == notification_id
            for notification in queue.notifications
        ):
            return

        notification = SubagentNotificationRecord(
            notification_id=notification_id,
            session_id=metadata.session_id,
            parent_entity_id=metadata.parent_entity_id,
            terminal_status=terminal_status,
            summary=self._notification_summary(metadata),
            error=metadata.error,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            delivered_at=None,
        )
        queue.notifications.append(notification)

        wait_component = world.get_component(
            metadata.parent_entity_id, SubagentWaitComponent
        )
        if wait_component is None:
            return

        future = wait_component.future
        if not notification_matches_wait(notification, wait_component):
            return
        if isinstance(future, asyncio.Future) and not future.done():
            if self._all_waited_sessions_terminal(
                world, metadata.parent_entity_id, wait_component
            ):
                future.set_result(None)

    def _config_for_session(
        self,
        registry: SubagentRegistryComponent,
        session: SubagentSessionRecord,
    ) -> SubagentConfig:
        base_config = self._resolve_subagent_config(registry, session.category)
        if session.load_skills == base_config.skills:
            return base_config

        return replace(base_config, skills=list(session.load_skills))

    def _build_background_coroutine(
        self,
        world: World,
        parent_entity_id: EntityId,
        category: str,
        prompt: str,
        session_id: str,
        metadata: SubagentSessionRecord,
        config: SubagentConfig,
        resolved_timeout: float | None,
    ) -> Any:
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
                    return await self._execute_subagent_core(
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

                return await self._execute_subagent_core(
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
            metadata.started_at = self._utc_now_iso()
            metadata.updated_at = metadata.started_at
            await self._runtime_manager.update_status(session_id, metadata.status)
            await self._runtime_manager.sync_to_component(world, parent_entity_id)
            try:
                if resolved_timeout is not None:
                    result, success, error = await asyncio.wait_for(
                        execute_with_config(), timeout=resolved_timeout
                    )
                else:
                    result, success, error = await execute_with_config()
            except asyncio.CancelledError:
                metadata.finished_at = self._utc_now_iso()
                metadata.updated_at = metadata.finished_at
                await self._runtime_manager.sync_to_component(world, parent_entity_id)
                raise
            except asyncio.TimeoutError:
                error_msg = f"Error: Subagent timeout after {resolved_timeout}s"
                metadata.finished_at = self._utc_now_iso()
                metadata.updated_at = metadata.finished_at
                logger.error(
                    "subagent_background_timeout",
                    timeout=resolved_timeout,
                    category=category,
                )
                await self._runtime_manager.update_timeout(session_id, error_msg)
                await self._runtime_manager.sync_to_component(world, parent_entity_id)
                self._enqueue_parent_notification(world, metadata)
                await self._publish_delegation_events(
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
                metadata.finished_at = self._utc_now_iso()
                metadata.updated_at = metadata.finished_at
                await self._runtime_manager.update_status(session_id, metadata.status)
                await self._runtime_manager.sync_to_component(world, parent_entity_id)
                self._enqueue_parent_notification(world, metadata)
                await self._publish_delegation_events(
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

            metadata.updated_at = self._utc_now_iso()
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
                persisted = self._persist_subagent_result(full_result)
                if persisted is not None:
                    artifact_id, record_path, inline_content = persisted
                    metadata.artifact_id = artifact_id
                    metadata.artifact_record_path = record_path
                    metadata.artifact_inline_content = inline_content
            else:
                metadata.status = "failed"
                metadata.finished_at = metadata.updated_at
                metadata.error = error
            await self._runtime_manager.update_status(session_id, metadata.status)
            logger.info(
                "subagent_background_finished",
                session_id=session_id,
                category=category,
                status=metadata.status,
                result_length=len(result) if success else 0,
                error=error,
            )
            await self._runtime_manager.sync_to_component(world, parent_entity_id)
            self._enqueue_parent_notification(world, metadata)
            await self._publish_delegation_events(
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

    async def _launch_background_session(
        self,
        world: World,
        parent_entity_id: EntityId,
        *,
        category: str,
        prompt: str,
        effective_config: SubagentConfig,
        normalized_skills: list[str],
        stream: bool,
        timeout_seconds: float | None,
    ) -> tuple[str, SubagentSessionRecord]:
        """Create, enqueue, and sync a background subagent session.

        Shared by ``subagent_handler(background=True)`` and ``_resume_session``.
        Returns ``(session_id, metadata)``.
        """
        correlation_id = str(uuid.uuid4())
        traceparent = generate_traceparent()
        resolved_timeout = self._resolve_timeout(timeout_seconds)
        (
            launch_trace_id,
            launch_run_id,
            launch_parent_observation_id,
        ) = self._active_observability_context(world)

        session_id = self._runtime_manager.create_session()
        now_iso = self._utc_now_iso()
        metadata = SubagentSessionRecord(
            session_id=session_id,
            category=category,
            prompt=prompt,
            parent_entity_id=parent_entity_id,
            created_at=now_iso,
            updated_at=now_iso,
            load_skills=normalized_skills,
            stream=stream,
            background=True,
            correlation_id=correlation_id,
            traceparent=traceparent,
            launch_trace_id=launch_trace_id,
            launch_run_id=launch_run_id,
            launch_parent_observation_id=launch_parent_observation_id,
            timeout_seconds=timeout_seconds,
        )
        run_in_background = self._build_background_coroutine(
            world,
            parent_entity_id,
            category,
            prompt,
            session_id,
            metadata,
            effective_config,
            resolved_timeout,
        )

        await self._runtime_manager.enqueue_session(
            session_id,
            metadata,
            run_in_background,
        )
        await self._runtime_manager.sync_to_component(world, parent_entity_id)
        return session_id, metadata

    async def _reconcile_restored_sessions(
        self,
        world: World,
        entity_id: EntityId,
        registry: SubagentRegistryComponent,
    ) -> None:
        from ecs_agent.components.definitions import SubagentSessionTableComponent

        table = world.get_component(entity_id, SubagentSessionTableComponent)
        if table is None or not table.sessions:
            return

        unreconciled_sessions = [
            session
            for session in table.sessions.values()
            if session.session_id not in self._reconciled_session_ids
        ]
        if not unreconciled_sessions:
            return

        now_iso = self._utc_now_iso()
        queued_sessions = sorted(
            (
                session
                for session in unreconciled_sessions
                if session.status == "queued"
            ),
            key=lambda session: (session.created_at, session.session_id),
        )

        for session in unreconciled_sessions:
            if session.status == "queued":
                continue

            if session.status == "running":
                session.status = "failed"
                session.error = "restored_without_live_task_handle"
                session.updated_at = now_iso
                session.finished_at = now_iso

            await self._runtime_manager.restore_session_metadata(session)
            self._reconciled_session_ids.add(session.session_id)

        for session in queued_sessions:
            config = self._config_for_session(registry, session)
            resolved_timeout = self._resolve_timeout(session.timeout_seconds)
            coroutine_factory = self._build_background_coroutine(
                world,
                entity_id,
                session.category,
                session.prompt,
                session.session_id,
                session,
                config,
                resolved_timeout,
            )
            await self._runtime_manager.enqueue_session(
                session.session_id,
                session,
                coroutine_factory,
            )
            self._reconciled_session_ids.add(session.session_id)

        await self._runtime_manager.sync_to_component(world, entity_id)

    def _wrap_retry_model_if_needed(self, model: LLMModel) -> LLMModel:
        """Wrap model with RetryModel if not already wrapped.

        Args:
            model: LLM model to wrap

        Returns:
            RetryModel-wrapped model, or original if already wrapped or FakeModel
        """
        if isinstance(model, RetryModel):
            return model

        # Skip FakeModel (deterministic tests)
        if isinstance(model, FakeModel):
            return model

        return RetryModel(model=model, retry_config=RetryConfig())

    def install_subagent_tool(
        self,
        world: World,
        entity_id: EntityId,
        tool_name: str = "subagent",
        override: bool = False,
    ) -> None:
        registry = world.get_component(entity_id, SubagentRegistryComponent)
        if registry is None:
            raise ValueError(f"Entity {entity_id} missing SubagentRegistryComponent")

        tool_registry = world.get_component(entity_id, ToolRegistryComponent)
        if tool_registry is None:
            raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")

        free_mode_enabled = registry.free_subagent_config.enabled
        tool_registry.tools[tool_name] = build_subagent_schema(
            tool_name, free_mode_enabled=free_mode_enabled
        )

        if tool_name in tool_registry.handlers and not override:
            return

        tool_registry.handlers[tool_name] = self._make_subagent_handler(
            world, entity_id
        )

        logger.info(
            "subagent_tool_installed",
            entity_id=entity_id,
            tool_name=tool_name,
            available_subagents=list(registry.subagents.keys()),
        )

    async def process(self, world: World) -> None:
        """Register subagent tool for entities with SubagentRegistryComponent.

        System registers the unified subagent tool on entities with both
        SubagentRegistryComponent and ToolRegistryComponent.
        """
        if self._free_subagent_config.enabled:
            for entity_id, (tool_registry,) in world.query(ToolRegistryComponent):
                registry_comp = world.get_component(entity_id, SubagentRegistryComponent)
                if registry_comp is None:
                    registry_comp = SubagentRegistryComponent(
                        free_subagent_config=replace(self._free_subagent_config)
                    )
                    world.add_component(entity_id, registry_comp)
                elif not registry_comp.free_subagent_config.enabled:
                    registry_comp.free_subagent_config = replace(
                        self._free_subagent_config
                    )

                await self._reconcile_restored_sessions(world, entity_id, registry_comp)
                if "subagent" in tool_registry.tools:
                    continue
                self.install_subagent_tool(
                    world, entity_id, tool_name="subagent", override=False
                )

                logger.info(
                    "subagent_tool_registered",
                    entity_id=entity_id,
                    available_subagents=list(registry_comp.subagents.keys()),
                    free_subagents_enabled=True,
                )
            return

        for entity_id, components in world.query(
            SubagentRegistryComponent, ToolRegistryComponent
        ):
            registry_comp, tool_registry = components
            assert isinstance(registry_comp, SubagentRegistryComponent)
            assert isinstance(tool_registry, ToolRegistryComponent)

            await self._reconcile_restored_sessions(world, entity_id, registry_comp)

            # Skip if subagent tool already registered
            if "subagent" in tool_registry.tools:
                continue

            # Use public installer API for unified subagent tool
            self.install_subagent_tool(
                world, entity_id, tool_name="subagent", override=False
            )

            logger.info(
                "subagent_tool_registered",
                entity_id=entity_id,
                available_subagents=list(registry_comp.subagents.keys()),
            )

    def install_subagent_control_tools(
        self,
        world: World,
        entity_id: EntityId,
    ) -> None:
        from ecs_agent.components.definitions import SubagentSessionTableComponent

        table = world.get_component(entity_id, SubagentSessionTableComponent)
        if table is None:
            raise ValueError(
                f"Entity {entity_id} missing SubagentSessionTableComponent"
            )

        tool_registry = world.get_component(entity_id, ToolRegistryComponent)
        if tool_registry is None:
            raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")

        # Install subagent_status tool
        tool_registry.tools["subagent_status"] = build_status_schema()
        tool_registry.handlers["subagent_status"] = self._make_status_handler(
            world, entity_id
        )

        tool_registry.tools["subagent_wait"] = build_wait_schema()
        tool_registry.handlers["subagent_wait"] = self._make_wait_handler(
            world, entity_id
        )

        # Install subagent_result tool
        tool_registry.tools["subagent_result"] = build_result_schema()
        tool_registry.handlers["subagent_result"] = self._make_result_handler(
            world, entity_id
        )

        # Install subagent_cancel tool
        tool_registry.tools["subagent_cancel"] = build_cancel_schema()
        tool_registry.handlers["subagent_cancel"] = self._make_cancel_handler(
            world, entity_id
        )

        # Install subagent_resume tool
        tool_registry.tools["subagent_resume"] = build_resume_schema()
        tool_registry.handlers["subagent_resume"] = self._make_resume_handler(
            world, entity_id
        )

        logger.info("subagent_control_tools_installed", entity_id=entity_id)

    def _make_status_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        """Create handler for subagent_status tool."""

        async def status_handler(session_id: str | None = None) -> str:
            from ecs_agent.components.definitions import SubagentSessionTableComponent

            logger.debug(
                "subagent_status_queried",
                parent_entity=parent_entity_id,
                session_id=session_id,
            )

            table = world.get_component(parent_entity_id, SubagentSessionTableComponent)
            if table is None:
                return json.dumps({"error": "SubagentSessionTableComponent not found"})

            if session_id is None:
                table_text = render_subagent_session_reminder_table(table.sessions)
                return json.dumps(
                    {
                        "status": "ok",
                        "session_count": len(table.sessions),
                        "summary_table": table_text,
                    }
                )

            session = table.sessions.get(session_id)
            if session is None:
                return json.dumps(
                    {
                        "error": f"Session not found: {session_id}",
                        "session_id": session_id,
                    }
                )

            logger.debug(
                "subagent_status_found",
                parent_entity=parent_entity_id,
                session_id=session_id,
                lifecycle_status=session.status,
            )
            queue_position: int | None = None
            if session.status == "queued":
                queue_position = await self._runtime_manager.get_queue_position(
                    session.session_id
                )

            return json.dumps(
                self._session_payload(
                    session,
                    status="ok",
                    queue_position=queue_position,
                )
            )

        return status_handler

    def _make_wait_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        async def wait_handler(
            session_ids: list[str] | None = None,
            timeout: float | str | None = None,
            auto_restart_budget: int = 0,
        ) -> str:
            if isinstance(timeout, str):
                timeout = float(timeout)

            wait_component = SubagentWaitComponent(
                session_ids=session_ids,
                timeout=timeout,
                future=None,
                started_at=self._utc_now_iso(),
                auto_restart_budget=auto_restart_budget,
            )
            world.add_component(parent_entity_id, wait_component)
            logger.info(
                "subagent_wait_requested",
                parent_entity=parent_entity_id,
                session_ids=session_ids,
                timeout=timeout,
                auto_restart_budget=auto_restart_budget,
            )
            return (
                "Waiting for background subagents. "
                "Will be notified when all sessions complete."
            )

        return wait_handler

    def _make_result_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        """Create handler for subagent_result tool."""

        async def result_handler(
            session_id: str,
            read_method: str = "full",
            timeout: float | str | None = None,
        ) -> str:
            if read_method not in {"full", "summary"}:
                return json.dumps(
                    {
                        "error": (
                            f"Invalid read_method '{read_method}'. "
                            "Expected one of: full, summary"
                        ),
                        "read_method": read_method,
                        "session_id": session_id,
                    }
                )

            if isinstance(timeout, str):
                timeout = float(timeout)

            logger.info(
                "subagent_result_requested",
                parent_entity=parent_entity_id,
                session_id=session_id,
                read_method=read_method,
                timeout=timeout,
            )
            session = await self._runtime_manager.get_session(session_id)
            if session is None:
                return json.dumps(
                    {
                        "error": f"Session not found: {session_id}",
                        "session_id": session_id,
                    }
                )

            if session.status in ("failed", "timed_out", "cancelled"):
                if read_method == "summary":
                    if session.result_summary is None:
                        return json.dumps(
                            {
                                "error": 'Summary not available for this session. Retry with read_method="full".',
                                "read_method": "summary",
                                "session_id": session_id,
                            }
                        )
                    return json.dumps(self._summary_payload(session, status="terminal"))
                return json.dumps(self._session_payload(session, status="terminal"))

            if session.status == "succeeded":
                logger.info(
                    "subagent_result_ready",
                    parent_entity=parent_entity_id,
                    session_id=session_id,
                    result_length=len(session.result_excerpt or ""),
                )
                if read_method == "summary":
                    if session.result_summary is None:
                        return json.dumps(
                            {
                                "error": 'Summary not available for this session. Retry with read_method="full".',
                                "read_method": "summary",
                                "session_id": session_id,
                            }
                        )
                    return json.dumps(self._summary_payload(session, status="success"))
                return json.dumps(self._session_payload(session, status="success"))

            logger.info(
                "subagent_result_awaiting",
                parent_entity=parent_entity_id,
                session_id=session_id,
                timeout=timeout,
            )
            try:
                loop = asyncio.get_running_loop()
                deadline = None if timeout is None else loop.time() + timeout
                # Backward-compat direct-result poll path:
                # This loop polls the session table every 0.1s until the session
                # reaches a terminal state. This is NOT the recommended path for
                # background sessions — callers should instead use subagent_wait()
                # (future-based, zero-poll) followed by subagent_result(). This
                # polling path exists for callers that invoke subagent_result()
                # directly without a preceding subagent_wait(), preserving
                # backward compatibility.
                while True:
                    session = await self._runtime_manager.get_session(session_id)
                    if session is None:
                        return json.dumps(
                            {
                                "error": f"Session disappeared while waiting: {session_id}",
                                "session_id": session_id,
                            }
                        )

                    if session.status == "succeeded":
                        logger.info(
                            "subagent_result_collected",
                            parent_entity=parent_entity_id,
                            session_id=session_id,
                            lifecycle_status=session.status,
                        )
                        if read_method == "summary":
                            if session.result_summary is None:
                                return json.dumps(
                                    {
                                        "error": 'Summary not available for this session. Retry with read_method="full".',
                                        "read_method": "summary",
                                        "session_id": session_id,
                                    }
                                )
                            return json.dumps(
                                self._summary_payload(session, status="success")
                            )
                        return json.dumps(
                            self._session_payload(session, status="success")
                        )

                    if session.status in ("failed", "timed_out", "cancelled"):
                        logger.info(
                            "subagent_result_collected",
                            parent_entity=parent_entity_id,
                            session_id=session_id,
                            lifecycle_status=session.status,
                        )
                        if read_method == "summary":
                            if session.result_summary is None:
                                return json.dumps(
                                    {
                                        "error": 'Summary not available for this session. Retry with read_method="full".',
                                        "read_method": "summary",
                                        "session_id": session_id,
                                    }
                                )
                            return json.dumps(
                                self._summary_payload(session, status="terminal")
                            )
                        return json.dumps(
                            self._session_payload(session, status="terminal")
                        )

                    remaining = None if deadline is None else deadline - loop.time()
                    if remaining is not None and remaining <= 0:
                        raise asyncio.TimeoutError

                    sleep_for = 0.1 if remaining is None else min(0.1, remaining)
                    await asyncio.sleep(sleep_for)

            except asyncio.TimeoutError:
                logger.warning(
                    "subagent_result_timeout",
                    parent_entity=parent_entity_id,
                    session_id=session_id,
                    timeout=timeout,
                )
                return json.dumps(
                    {
                        "error": f"Timeout waiting for session result after {timeout}s",
                        "session_id": session_id,
                        "timeout": timeout,
                    }
                )
            except asyncio.CancelledError:
                logger.warning(
                    "subagent_result_cancelled",
                    parent_entity=parent_entity_id,
                    session_id=session_id,
                )
                return json.dumps(
                    {
                        "error": f"Session was cancelled: {session_id}",
                        "session_id": session_id,
                    }
                )

        return result_handler

    def _make_cancel_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        """Create handler for subagent_cancel tool."""

        async def cancel_handler(session_id: str) -> str:
            logger.info(
                "subagent_cancel_requested",
                parent_entity=parent_entity_id,
                session_id=session_id,
            )
            session = await self._runtime_manager.get_session(session_id)
            if session is None:
                return json.dumps(
                    {
                        "error": f"Session not found: {session_id}",
                        "session_id": session_id,
                    }
                )

            if session.status in ("failed", "timed_out", "cancelled", "succeeded"):
                return json.dumps(
                    {
                        "error": f"Session already terminal: {session.status}",
                        "session_id": session_id,
                        "lifecycle_status": session.status,
                    }
                )

            await self._runtime_manager.cancel_session(session_id)
            await self._runtime_manager.sync_to_component(world, parent_entity_id)

            session = await self._runtime_manager.get_session(session_id)

            logger.info(
                "subagent_cancel_completed",
                parent_entity=parent_entity_id,
                session_id=session_id,
                final_status=session.status if session else "cancelled",
            )
            return json.dumps(
                {
                    "status": "cancelled",
                    "session_id": session_id,
                    "lifecycle_status": session.status if session else "cancelled",
                }
            )

        return cancel_handler

    def _make_resume_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        async def resume_handler(session_id: str) -> str:
            logger.info(
                "subagent_resume_requested",
                parent_entity=parent_entity_id,
                session_id=session_id,
            )
            try:
                new_session_id = await self._resume_session(
                    world, parent_entity_id, session_id
                )
            except ValueError as exc:
                return json.dumps({"error": str(exc), "session_id": session_id})

            new_session = await self._runtime_manager.get_session(new_session_id)
            return json.dumps(
                {
                    "status": "resumed",
                    "original_session_id": session_id,
                    "new_session_id": new_session_id,
                    "session_id": new_session_id,
                    "category": new_session.category if new_session else "",
                    "lifecycle_status": new_session.status if new_session else "queued",
                }
            )

        return resume_handler

    async def _resume_session(
        self,
        world: World,
        parent_entity_id: EntityId,
        original_session_id: str,
    ) -> str:
        """Restart a failed/timed_out/cancelled session with the same config.

        Returns the new session_id. The original session record is preserved
        as history; the new session starts fresh in ``queued`` status.
        """
        original = await self._runtime_manager.get_session(original_session_id)
        if original is None:
            raise ValueError(f"Session not found: {original_session_id}")

        if original.status not in ("failed", "timed_out", "cancelled"):
            raise ValueError(
                f"Session {original_session_id} is not in a resumable state: "
                f"{original.status}. Only failed, timed_out, or cancelled "
                "sessions can be resumed."
            )

        registry_comp = world.get_component(
            parent_entity_id, SubagentRegistryComponent
        )
        if registry_comp is None:
            raise ValueError(
                f"SubagentRegistryComponent not found on entity {parent_entity_id}"
            )

        parent_llm = world.get_component(parent_entity_id, LLMComponent)
        config = self._resolve_subagent_config(
            registry_comp,
            original.category,
            parent_model=parent_llm.model if parent_llm is not None else None,
        )
        normalized_skills = self._normalize_load_skills(
            config, original.load_skills
        )
        effective_config = (
            config
            if normalized_skills == config.skills
            else replace(config, skills=normalized_skills)
        )

        session_id, metadata = await self._launch_background_session(
            world,
            parent_entity_id,
            category=original.category,
            prompt=original.prompt,
            effective_config=effective_config,
            normalized_skills=normalized_skills,
            stream=original.stream,
            timeout_seconds=original.timeout_seconds,
        )

        logger.info(
            "subagent_resumed",
            parent_entity=parent_entity_id,
            original_session_id=original_session_id,
            new_session_id=session_id,
            category=original.category,
        )
        return session_id

    def make_resume_callback(self) -> ResumeCallback:
        """Return a callback suitable for SubagentWaitSystem auto-restart."""

        async def callback(
            original_session_id: str,
            parent_entity_id: EntityId,
            world: World,
        ) -> str:
            return await self._resume_session(
                world, parent_entity_id, original_session_id
            )

        return callback

    def _make_subagent_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        async def subagent_handler(
            category: str,
            prompt: str,
            load_skills: list[str] | None = None,
            background: bool = False,
            stream: bool = False,
            timeout: float | None = None,
        ) -> str:
            effective_load_skills = [] if load_skills is None else load_skills
            # Coerce timeout to float: LLM may pass it as a JSON string (e.g. '180')
            resolved_timeout_raw: float | None = (
                float(timeout) if timeout is not None else None
            )
            logger.info(
                "subagent_handler_called",
                parent_entity=parent_entity_id,
                category=category,
                background=background,
                load_skills=effective_load_skills,
                timeout=timeout,
                prompt_length=len(prompt),
            )

            registry_comp = world.get_component(
                parent_entity_id, SubagentRegistryComponent
            )
            if registry_comp is None:
                return f"Error: SubagentRegistryComponent not found on entity {parent_entity_id}"

            # Validate parameters BEFORE the try/except — raise so callers can catch
            self._validate_subagent_params(category, prompt, effective_load_skills)

            parent_llm = world.get_component(parent_entity_id, LLMComponent)
            try:
                config = self._resolve_subagent_config(
                    registry_comp,
                    category,
                    parent_model=parent_llm.model if parent_llm is not None else None,
                )
            except ValueError as exc:
                return str(exc)

            normalized_skills = self._normalize_load_skills(
                config, effective_load_skills
            )
            effective_config = (
                config
                if normalized_skills == config.skills
                else replace(config, skills=normalized_skills)
            )

            correlation_id = str(uuid.uuid4())
            traceparent = generate_traceparent()

            # Resolve timeout for this subagent execution
            resolved_timeout = self._resolve_timeout(resolved_timeout_raw)
            (
                launch_trace_id,
                launch_run_id,
                launch_parent_observation_id,
            ) = self._active_observability_context(world)

            metadata: SubagentSessionRecord | None = None

            async def execute_with_effective_config() -> tuple[str, bool, str | None]:
                return await self._execute_subagent_core(
                    world,
                    parent_entity_id,
                    category,
                    prompt,
                    correlation_id,
                    traceparent,
                    effective_config,
                )

            if not background:
                # Sync mode: wrap with timeout
                try:
                    if resolved_timeout is not None:
                        result, success, _ = await asyncio.wait_for(
                            execute_with_effective_config(), timeout=resolved_timeout
                        )
                    else:
                        result, success, _ = await execute_with_effective_config()
                except asyncio.TimeoutError:
                    error_msg = f"Error: Subagent timeout after {resolved_timeout}s"
                    logger.error(
                        "subagent_timeout", timeout=resolved_timeout, category=category
                    )
                    result = error_msg
                    success = False

                if success:
                    self._persist_subagent_result(result)

                logger.info(
                    "subagent_sync_completed",
                    parent_entity=parent_entity_id,
                    category=category,
                    result_length=len(result),
                )
                return result

            session_id, metadata = await self._launch_background_session(
                world,
                parent_entity_id,
                category=category,
                prompt=prompt,
                effective_config=effective_config,
                normalized_skills=normalized_skills,
                stream=stream,
                timeout_seconds=timeout,
            )

            logger.info(
                "subagent_background_launched",
                parent_entity=parent_entity_id,
                session_id=session_id,
                category=category,
                timeout=resolved_timeout,
            )

            return json.dumps(
                {
                    "session_id": session_id,
                    "status": "queued",
                    "lifecycle_status": metadata.status,
                    "category": category,
                    "created_at": metadata.created_at,
                    "timeout": timeout,
                    "stream": stream,
                }
            )

        return subagent_handler

    async def _execute_subagent_core(
        self,
        world: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        task: str,
        correlation_id: str,
        traceparent: str,
        config: SubagentConfig,
        *,
        session_id: str | None = None,
        stream: bool = False,
    ) -> tuple[str, bool, str | None]:
        """Shared subagent execution core for the subagent API.

        Args:
            world: Parent world instance
            parent_entity_id: Parent entity delegating the task
            subagent_name: Name of subagent to execute
            task: Task description
            correlation_id: CloudEvents correlation ID
            traceparent: W3C trace context

        Returns:
            Tuple of (result, success, error):
            - result: Result string from delegation
            - success: True if successful, False otherwise
            - error: Error message if failed, None otherwise
        """
        publish_completion_event = _PUBLISH_COMPLETION_EVENT.get()
        # Execute delegation
        try:
            child_entity_id = world.create_entity()
            logger.info(
                "child_entity_created",
                parent_entity=parent_entity_id,
                child_entity=child_entity_id,
                subagent_name=subagent_name,
            )

            world.add_component(
                child_entity_id,
                LLMComponent(
                    model=config.model,
                    system_prompt=config.system_prompt,
                ),
            )
            world.add_component(
                child_entity_id,
                ConversationComponent(messages=[Message(role="user", content=task)]),
            )
            world.add_component(
                child_entity_id, OwnerComponent(owner_id=parent_entity_id)
            )
            world.add_component(child_entity_id, ChildStubComponent())

            background_result_token = _BACKGROUND_RESULT_ENVELOPE_ENABLED.set(
                session_id is not None
            )
            try:
                child_world, child_world_entity_id = self._assemble_child_world(
                    world,
                    parent_entity_id,
                    config,
                    parent_child_entity=child_entity_id,
                )
            finally:
                _BACKGROUND_RESULT_ENVELOPE_ENABLED.reset(background_result_token)
            bridge_cleanup: Any = None
            if stream and session_id is not None:
                child_world_name = child_world.name or subagent_name
                child_world.add_component(
                    child_world_entity_id,
                    StreamingComponent(enabled=True),
                )
                bridge_cleanup = self._bridge_subagent_stream_events(
                    parent_world=world,
                    child_world=child_world,
                    parent_entity_id=parent_entity_id,
                    session_id=session_id,
                    category=subagent_name,
                    child_world_name=child_world_name,
                )

            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                task=task,
                child_world_name=child_world.name,
                observation_id=correlation_id,
                start_time=datetime.now(timezone.utc),
            )
            delegation_started_monotonic = time.monotonic()
            launch_context = _BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT.get()
            launch_trace_id: str | None = None
            launch_run_id: str | None = None
            if launch_context is not None:
                launch_trace_id, launch_run_id, _ = launch_context
            self._install_child_observability(
                parent_world=world,
                child_world=child_world,
                trace_id=launch_trace_id,
                run_id=launch_run_id,
                parent_observation_id=correlation_id,
            )

            try:
                result = await self._execute_delegation(
                    child_world,
                    child_world_entity_id,
                    task,
                    config,
                )
            finally:
                if bridge_cleanup is not None:
                    bridge_cleanup()
            # Sync rendered system prompt (populated by SystemPromptRenderSystem) to parent-world stub
            child_llm = child_world.get_component(child_world_entity_id, LLMComponent)
            stub_llm = world.get_component(child_entity_id, LLMComponent)
            if child_llm is not None and stub_llm is not None:
                stub_llm.system_prompt = child_llm.system_prompt

            child_conv = child_world.get_component(
                child_world_entity_id,
                ConversationComponent,
            )
            parent_child_conv = world.get_component(
                child_entity_id,
                ConversationComponent,
            )
            if child_conv is not None and parent_child_conv is not None:
                parent_child_conv.messages = list(child_conv.messages)

            logger.info(
                "delegation_completed",
                parent_entity=parent_entity_id,
                child_entity=child_entity_id,
                subagent_name=subagent_name,
                result_length=len(result),
            )

            if publish_completion_event:
                await self._publish_delegation_events(
                    world,
                    parent_entity_id,
                    subagent_name,
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                    result=result,
                    success=True,
                    error=None,
                    child_world_name=child_world.name,
                    observation_id=correlation_id,
                    end_time=datetime.now(timezone.utc),
                    duration_seconds=time.monotonic() - delegation_started_monotonic,
                )

            return (result, True, None)

        except TimeoutError as exc:
            error_msg = "Error: Subagent timeout"
            logger.error(
                "delegation_timeout",
                parent_entity=parent_entity_id,
                subagent_name=subagent_name,
                correlation_id=correlation_id,
                exception=str(exc),
            )
            if publish_completion_event:
                await self._publish_delegation_events(
                    world,
                    parent_entity_id,
                    subagent_name,
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                    result=error_msg,
                    success=False,
                    error=error_msg,
                    observation_id=correlation_id,
                )
            return (error_msg, False, error_msg)

        except ValueError as exc:
            error_msg = str(exc)
            logger.error(
                "delegation_exception",
                parent_entity=parent_entity_id,
                subagent_name=subagent_name,
                exception=error_msg,
            )
            if publish_completion_event:
                await self._publish_delegation_events(
                    world,
                    parent_entity_id,
                    subagent_name,
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                    result=error_msg,
                    success=False,
                    error=error_msg,
                    observation_id=correlation_id,
                )
            raise

        except Exception as exc:
            error_msg = f"Error during delegation: {exc}"
            logger.error(
                "delegation_exception",
                parent_entity=parent_entity_id,
                subagent_name=subagent_name,
                exception=str(exc),
            )
            if publish_completion_event:
                await self._publish_delegation_events(
                    world,
                    parent_entity_id,
                    subagent_name,
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                    result=error_msg,
                    success=False,
                    error=error_msg,
                    observation_id=correlation_id,
                )
            return (error_msg, False, error_msg)

    def _bridge_subagent_stream_events(
        self,
        *,
        parent_world: World,
        child_world: World,
        parent_entity_id: EntityId,
        session_id: str,
        category: str,
        child_world_name: str,
    ) -> Any:
        seq = 0

        def next_seq() -> int:
            nonlocal seq
            current = seq
            seq += 1
            return current

        def publish_translated_event(event: object) -> None:
            asyncio.create_task(parent_world.event_bus.publish(event))

        async def on_start(event: StreamStartEvent) -> None:
            publish_translated_event(
                SubagentStreamStartEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._iso_timestamp(event.timestamp),
                )
            )

        async def on_reasoning_delta(event: StreamReasoningDeltaEvent) -> None:
            publish_translated_event(
                SubagentStreamDeltaEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._utc_now_iso(),
                    delta="",
                    reasoning_delta=event.reasoning_delta,
                )
            )

        async def on_content_delta(event: StreamContentDeltaEvent) -> None:
            publish_translated_event(
                SubagentStreamDeltaEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._utc_now_iso(),
                    delta=event.delta,
                )
            )

        async def on_end(event: StreamEndEvent) -> None:
            publish_translated_event(
                SubagentStreamEndEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._iso_timestamp(event.timestamp),
                )
            )

        child_world.event_bus.subscribe(StreamStartEvent, on_start)
        child_world.event_bus.subscribe(StreamReasoningDeltaEvent, on_reasoning_delta)
        child_world.event_bus.subscribe(StreamContentDeltaEvent, on_content_delta)
        child_world.event_bus.subscribe(StreamEndEvent, on_end)

        def cleanup() -> None:
            child_world.event_bus.unsubscribe(StreamStartEvent, on_start)
            child_world.event_bus.unsubscribe(
                StreamReasoningDeltaEvent,
                on_reasoning_delta,
            )
            child_world.event_bus.unsubscribe(StreamContentDeltaEvent, on_content_delta)
            child_world.event_bus.unsubscribe(StreamEndEvent, on_end)

        return cleanup

    def _iso_timestamp(self, timestamp: float) -> str:
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _resolve_subagent_config(
        self,
        registry: SubagentRegistryComponent,
        subagent_name: str,
        *,
        parent_model: LLMModel | None = None,
    ) -> SubagentConfig:
        """Resolve and validate subagent configuration from registry."""
        config = registry.subagents.get(subagent_name)
        if config is None:
            config = self._resolve_free_subagent_config(
                registry,
                subagent_name,
                parent_model,
            )
            if config is None:
                raise ValueError(
                    f"Error: Unknown subagent '{subagent_name}'. Available subagents: {list(registry.subagents.keys())}"
                )

        # Wrap model with RetryModel by default
        wrapped_model = self._wrap_retry_model_if_needed(config.model)

        # Return config with wrapped model (use replace to preserve other fields)
        return replace(config, model=wrapped_model)

    def _resolve_free_subagent_config(
        self,
        registry: SubagentRegistryComponent,
        subagent_name: str,
        parent_model: LLMModel | None,
    ) -> SubagentConfig | None:
        free_config = registry.free_subagent_config
        if not free_config.enabled:
            return None
        if parent_model is None:
            raise ValueError(
                f"Error: Unknown subagent '{subagent_name}' cannot be created without a parent LLMComponent model"
            )
        return SubagentConfig(
            name=subagent_name,
            model=parent_model,
            description="Dynamically created free-form subagent.",
            system_prompt=free_config.system_prompt_template.replace(
                "{name}", subagent_name
            ),
            skills=list(free_config.skills),
            max_ticks=free_config.max_ticks,
            inheritance_policy=replace(free_config.inheritance_policy),
        )

    def _validate_subagent_params(
        self, category: str, prompt: str, load_skills: list[str]
    ) -> None:
        """Validate subagent invocation parameters.

        Args:
            category: Subagent category/name
            prompt: Task description
            load_skills: List of skill names to load

        Raises:
            ValueError: If parameters are invalid
        """
        if not category or not category.strip():
            raise ValueError("Error: category cannot be empty")
        if not prompt or not prompt.strip():
            raise ValueError("Error: prompt cannot be empty")
        if not isinstance(load_skills, list):
            raise ValueError(
                f"Error: load_skills must be a list, got {type(load_skills).__name__}"
            )

    def _normalize_load_skills(
        self, config: SubagentConfig, load_skills: list[str]
    ) -> list[str]:
        """Normalize load_skills as ordered unique merge of config.skills + load_skills.

        Args:
            config: Resolved subagent configuration
            load_skills: Additional skills requested by caller

        Returns:
            List of skill names (ordered, deduplicated)
        """
        # Preserve order: config.skills first, then load_skills
        # Remove duplicates while maintaining order
        seen: set[str] = set()
        result: list[str] = []
        for skill in config.skills + load_skills:
            if skill not in seen:
                seen.add(skill)
                result.append(skill)
        return result

    def _assemble_child_world(
        self,
        parent_world: World,
        parent_entity: EntityId,
        config: SubagentConfig,
        parent_child_entity: EntityId | None = None,
    ) -> tuple[World, EntityId]:
        """Delegate to ChildWorldBuilder.

        Retained as a thin instance method so white-box tests can monkeypatch or
        call this seam while the assembly logic lives in child_world.ChildWorldBuilder.
        """
        return self._child_world_builder.assemble_child_world(
            parent_world,
            parent_entity,
            config,
            parent_child_entity,
        )

    async def _execute_delegation(
        self,
        child_world: World,
        child_entity: EntityId,
        task: str,
        config: SubagentConfig,
    ) -> str:
        """Execute child world delegation run and return extracted result."""
        child_world.add_component(
            child_entity,
            ConversationComponent(messages=[Message(role="user", content=task)]),
        )
        runner = Runner()
        await runner.run(
            child_world,
            max_ticks=config.max_ticks,
            trace_id=getattr(
                child_world,
                "_ecs_agent_trace_id",
                current_trace_id(),
            ),
            run_id=getattr(
                child_world,
                "_ecs_agent_run_id",
                current_run_id(),
            ),
            parent_observation_id=getattr(
                child_world,
                "_ecs_agent_parent_observation_id",
                None,
            ),
            emit_root_trace=False,
        )
        return self._extract_delegation_result(child_world, child_entity)

    def _install_child_observability(
        self,
        *,
        parent_world: World,
        child_world: World,
        trace_id: str | None = None,
        run_id: str | None = None,
        parent_observation_id: str,
    ) -> None:
        """Install parent observability sink on a child world when available."""
        parent_sink = getattr(parent_world, "_ecs_agent_observability_sink", None)
        if parent_sink is None:
            return
        parent_config = getattr(
            parent_world,
            "_ecs_agent_observability_config",
            None,
        )
        install_observability(child_world, parent_sink, config=parent_config)
        active_trace_id, active_run_id, _ = self._active_observability_context(parent_world)
        if trace_id is not None:
            active_trace_id = trace_id
        if run_id is not None:
            active_run_id = run_id
        if active_trace_id is not None:
            setattr(child_world, "_ecs_agent_trace_id", active_trace_id)
        if active_run_id is not None:
            setattr(child_world, "_ecs_agent_run_id", active_run_id)
        setattr(
            child_world,
            "_ecs_agent_parent_observation_id",
            parent_observation_id,
        )

    def _extract_delegation_result(
        self, child_world: World, child_entity: EntityId
    ) -> str:
        """Extract terminal delegation result from child conversation."""
        child_conv = child_world.get_component(child_entity, ConversationComponent)
        if child_conv is None:
            return "Error: No conversation found"

        for message in reversed(child_conv.messages):
            if message.role == "assistant":
                return message.content
        return "Error: No assistant message found in subagent conversation"

    async def _publish_delegation_events(
        self,
        world: World,
        parent_entity_id: EntityId,
        subagent_name: str,
        *,
        correlation_id: str,
        traceparent: str,
        task: str | None = None,
        result: str | None = None,
        success: bool | None = None,
        error: str | None = None,
        child_world_name: str | None = None,
        observation_id: str = "",
        trace_id: str | None = None,
        run_id: str | None = None,
        parent_observation_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        duration_seconds: float | None = None,
        publish_started: bool = True,
    ) -> None:
        """Publish start/completion delegation events via one wrapper API."""
        if task is not None and publish_started:
            await world.event_bus.publish(
                DelegationStartedEvent(
                    entity_id=parent_entity_id,
                    subagent_name=subagent_name,
                    task=task,
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                    child_world_name=child_world_name,
                    observation_id=observation_id,
                    start_time=start_time,
                )
            )

        if result is not None and success is not None:
            await world.event_bus.publish(
                DelegationCompletedEvent(
                    entity_id=parent_entity_id,
                    subagent_name=subagent_name,
                    result=result,
                    success=success,
                    error=error,
                    task=task or "",
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                    child_world_name=child_world_name,
                    observation_id=observation_id,
                    trace_id=trace_id,
                    run_id=run_id,
                    parent_observation_id=parent_observation_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration_seconds,
                )
            )
