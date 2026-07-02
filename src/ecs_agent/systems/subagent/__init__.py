"""Subagent delegation system."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ecs_agent.components import (
    ChildStubComponent,
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    StreamingComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.systems.subagent_wait import ResumeCallback
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager
from ecs_agent.observability import generate_traceparent
from ecs_agent.systems.subagent._contextvars import (
    _BACKGROUND_LAUNCH_OBSERVABILITY_CONTEXT,
    _BACKGROUND_RESULT_ENVELOPE_ENABLED,
    _PUBLISH_COMPLETION_EVENT,
)
from ecs_agent.systems.subagent.background import BackgroundSessionRunner
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.subagent.delegation import (
    DelegationExecutor,
    active_observability_context,
)
from ecs_agent.systems.subagent.notifications import NotificationCoordinator
from ecs_agent.systems.subagent.service import SubagentService
from ecs_agent.systems.subagent.result_envelope import (
    _build_background_child_prompt_template,
    _build_child_prompt_template,
    parse_background_result_envelope,  # re-exported for backward-compat import path
)
from ecs_agent.systems.subagent import tools as subagent_tools
from ecs_agent.types import (
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    FreeSubagentConfig,
    Message,
    SubagentConfig,
    SubagentSessionRecord,
    ToolSchema,
)

from ecs_agent.providers.protocol import LLMModel

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
        self._service = SubagentService(
            default_timeout=default_timeout, registry=registry
        )
        self._child_world_builder = ChildWorldBuilder()
        self._delegation_executor = DelegationExecutor()
        self._notification_coordinator = NotificationCoordinator()
        self._background_runner = BackgroundSessionRunner(
            self._runtime_manager, self._notification_coordinator
        )
        self._reconciled_session_ids: set[str] = set()

    def _persist_subagent_result(
        self,
        result: str,
    ) -> tuple[str, str, str | None] | None:
        return self._service.persist_result(result)

    def _session_payload(
        self,
        session: SubagentSessionRecord,
        *,
        status: str,
        queue_position: int | None = None,
    ) -> dict[str, Any]:
        return self._service.session_payload(
            session, status=status, queue_position=queue_position
        )

    def _summary_payload(
        self,
        session: SubagentSessionRecord,
        *,
        status: str,
        queue_position: int | None = None,
    ) -> dict[str, Any]:
        return self._service.summary_payload(
            session, status=status, queue_position=queue_position
        )

    def _resolve_timeout(self, per_call_timeout: float | None) -> float | None:
        """Resolve timeout with precedence: per-call > global > None."""
        return self._service.resolve_timeout(per_call_timeout)

    def _active_observability_context(
        self,
        world: World,
    ) -> tuple[str | None, str | None, str | None]:
        """Return active trace, run, and root observation IDs for this world.

        Thin seam delegating to delegation.active_observability_context.
        """
        return active_observability_context(world)

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _config_for_session(
        self,
        registry: SubagentRegistryComponent,
        session: SubagentSessionRecord,
    ) -> SubagentConfig:
        return self._service.config_for_session(registry, session)

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
        run_in_background = self._background_runner.build_coroutine(
            world,
            parent_entity_id,
            category,
            prompt,
            session_id,
            metadata,
            effective_config,
            resolved_timeout,
            execute_core=self._execute_subagent_core,
            persist_result=self._persist_subagent_result,
            publish_events=self._publish_delegation_events,
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
            coroutine_factory = self._background_runner.build_coroutine(
                world,
                entity_id,
                session.category,
                session.prompt,
                session.session_id,
                session,
                config,
                resolved_timeout,
                execute_core=self._execute_subagent_core,
                persist_result=self._persist_subagent_result,
                publish_events=self._publish_delegation_events,
            )
            await self._runtime_manager.enqueue_session(
                session.session_id,
                session,
                coroutine_factory,
            )
            self._reconciled_session_ids.add(session.session_id)

        await self._runtime_manager.sync_to_component(world, entity_id)

    def install_subagent_tool(
        self,
        world: World,
        entity_id: EntityId,
        tool_name: str = "subagent",
        override: bool = False,
    ) -> None:
        subagent_tools.install_subagent_tool(
            self, world, entity_id, tool_name, override
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
        subagent_tools.install_subagent_control_tools(self, world, entity_id)

    def _terminal_result_payload(
        self,
        session: SubagentSessionRecord,
        read_method: str,
        session_id: str,
    ) -> str:
        return self._service.terminal_result_payload(session, read_method, session_id)

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
                bridge_cleanup = self._delegation_executor.bridge_subagent_stream_events(
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
            self._delegation_executor.install_child_observability(
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

    def _resolve_subagent_config(
        self,
        registry: SubagentRegistryComponent,
        subagent_name: str,
        *,
        parent_model: LLMModel | None = None,
    ) -> SubagentConfig:
        """Resolve and validate subagent configuration from registry."""
        return self._service.resolve_subagent_config(
            registry, subagent_name, parent_model=parent_model
        )

    def _validate_subagent_params(
        self, category: str, prompt: str, load_skills: list[str]
    ) -> None:
        """Validate subagent invocation parameters. Raises ValueError if invalid."""
        self._service.validate_subagent_params(category, prompt, load_skills)

    def _normalize_load_skills(
        self, config: SubagentConfig, load_skills: list[str]
    ) -> list[str]:
        """Ordered, de-duplicated merge of config.skills followed by load_skills."""
        return self._service.normalize_load_skills(config, load_skills)

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
        """Delegate to DelegationExecutor.run_delegation.

        Retained as a thin instance method so white-box tests can monkeypatch this
        seam while the run/extract logic lives in delegation.DelegationExecutor.
        """
        return await self._delegation_executor.run_delegation(
            child_world,
            child_entity,
            task,
            config,
        )

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
