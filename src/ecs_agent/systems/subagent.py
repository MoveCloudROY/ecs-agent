"""Subagent delegation system."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    PermissionComponent,
    SubagentRegistryComponent,
    SystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent_runtime import SubagentRuntimeManager
from ecs_agent.observability import generate_traceparent
from ecs_agent.types import (
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    InheritancePolicy,
    Message,
    RetryConfig,
    SubagentConfig,
    SubagentSessionRecord,
    ToolSchema,
)

# Task 9: Import providers for retry wrapping
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.providers.retry_provider import RetryProvider

logger = get_logger(__name__)


class _InheritedSkill:
    def __init__(
        self, metadata: SkillMetadata, tools: dict[str, tuple[ToolSchema, Any]]
    ) -> None:
        self.name = metadata.name
        self.description = metadata.description
        self._tools = tools
        self._system_prompt = (
            "inherited skill prompt" if metadata.has_system_prompt else ""
        )

    def tools(self) -> dict[str, tuple[ToolSchema, Any]]:
        return self._tools

    def system_prompt(self) -> str:
        return self._system_prompt

    def install(self, world: World, entity_id: EntityId) -> None:
        del world, entity_id

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        del world, entity_id


class SubagentSystem:
    """System that manages subagent delegation lifecycle.

    This system automatically registers a 'delegate' tool for entities that have
    a SubagentRegistryComponent. When the delegate tool is called, it:
    1. Creates a child entity with the specified subagent configuration
    2. Runs the child entity to completion
    3. Returns the child's final assistant message
    4. Publishes delegation events to the event bus
    """

    def __init__(
        self, priority: int = -1, default_timeout: float | None = None
    ) -> None:
        self.priority = priority
        self._runtime_manager = SubagentRuntimeManager()
        self._default_timeout = default_timeout

    def _resolve_timeout(self, per_call_timeout: float | None) -> float | None:
        """Resolve timeout with precedence: per-call > global > None."""
        return (
            per_call_timeout if per_call_timeout is not None else self._default_timeout
        )

    def _wrap_retry_provider_if_needed(self, provider: LLMProvider) -> LLMProvider:
        """Wrap provider with RetryProvider if not already wrapped.

        Args:
            provider: LLM provider to wrap

        Returns:
            RetryProvider-wrapped provider, or original if already wrapped or FakeProvider
        """
        # Skip if already wrapped (idempotent)
        if isinstance(provider, RetryProvider):
            return provider

        # Skip FakeProvider (deterministic tests)
        if isinstance(provider, FakeProvider):
            return provider

        # Wrap with default config
        return RetryProvider(provider=provider, retry_config=RetryConfig())

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

        tool_registry.tools[tool_name] = ToolSchema(
            name=tool_name,
            description=(
                "Spawn a subagent to handle a self-contained task. The subagent runs in its own "
                "isolated World with inherited tools and skills, then returns its final answer.\n\n"
                "WHEN TO CALL:\n"
                "  - Use when a subtask is independent and can be fully delegated (research, "
                "    code generation, analysis, tool-heavy work).\n"
                "  - Use background=True when you want to launch multiple subagents in parallel "
                "    and collect results later via subagent_result.\n"
                "  - Use background=False (default) when you need the result before continuing.\n\n"
                "INTERFACE:\n"
                "  category (required) — subagent type/name registered in SubagentRegistryComponent.\n"
                "  prompt   (required) — full task instruction for the subagent; be specific.\n"
                "  load_skills        — extra skill names to inject on top of category defaults.\n"
                "  background         — if True, returns a JSON payload with session_id immediately; "
                "                       use subagent_result(session_id) to retrieve the answer later.\n"
                "  timeout            — max seconds to wait before aborting (null = no limit).\n\n"
                "RETURNS (sync): final answer string from the subagent.\n"
                "RETURNS (background): JSON {session_id, status, category, created_at}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Registered subagent type/name (e.g. \"researcher\", \"coder\"). Must match a key in SubagentRegistryComponent.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Full task instruction for the subagent. Be explicit: include goal, context, expected output format, and any constraints.",
                    },
                    "load_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra skill names to load on top of this category's defaults. Use when the task requires capabilities not in the base skill set.",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "If true, launch the subagent asynchronously and return immediately with a session_id. Use subagent_result(session_id) to collect the answer later. Default false (synchronous — blocks until done).",
                    },
                    "timeout": {
                        "type": ["number", "null"],
                        "description": "Maximum seconds to wait before aborting. null means no timeout. Only respected in sync mode and for background collection in subagent_result.",
                    },
                },
                "required": ["category", "prompt"],
            },
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

    def install_delegate_tool(
        self,
        world: World,
        entity_id: EntityId,
        tool_name: str = "delegate",
        override: bool = False,
    ) -> None:
        """Install delegate tool with explicit control over name and overwrite behavior.

        Args:
            world: World instance containing the entity
            entity_id: Entity with SubagentRegistryComponent and ToolRegistryComponent
            tool_name: Name for the delegate tool (default: "delegate")
            override: If True, replaces existing handler; if False, skips if exists

        Raises:
            ValueError: If entity missing required components
        """
        # Validate entity has SubagentRegistryComponent and ToolRegistryComponent
        registry = world.get_component(entity_id, SubagentRegistryComponent)
        if registry is None:
            raise ValueError(f"Entity {entity_id} missing SubagentRegistryComponent")

        tool_registry = world.get_component(entity_id, ToolRegistryComponent)
        if tool_registry is None:
            raise ValueError(f"Entity {entity_id} missing ToolRegistryComponent")

        # Build schema
        subagent_names = list(registry.subagents.keys())
        schema_dict = self._build_delegate_tool_schema(subagent_names)
        function_schema = schema_dict["function"]

        # Install tool schema (always update schema to match tool_name)
        tool_registry.tools[tool_name] = ToolSchema(
            name=tool_name,
            description=function_schema["description"],
            parameters=function_schema["parameters"],
        )

        # Install handler (helper respects override parameter)
        self._install_delegate_handler(world, entity_id, tool_name, override)

        logger.info(
            "delegate_tool_installed",
            entity_id=entity_id,
            tool_name=tool_name,
            available_subagents=list(registry.subagents.keys()),
        )
    async def process(self, world: World) -> None:
        """Register delegate tool for entities with SubagentRegistryComponent.

        Backward compatible: uses public installer API with default parameters.
        """
        for entity_id, components in world.query(
            SubagentRegistryComponent, ToolRegistryComponent
        ):
            registry_comp, tool_registry = components
            assert isinstance(registry_comp, SubagentRegistryComponent)
            assert isinstance(tool_registry, ToolRegistryComponent)

            # Skip if delegate tool already registered
            if "delegate" in tool_registry.tools:
                continue

            # Use public installer API
            self.install_delegate_tool(
                world, entity_id, tool_name="delegate", override=False
            )

            logger.info(
                "delegate_tool_registered",
                entity_id=entity_id,
                available_subagents=list(registry_comp.subagents.keys()),
            )

    def install_subagent_control_tools(
        self,
        world: World,
        entity_id: EntityId,
    ) -> None:
        """Install all three subagent control tools: status, result, cancel."""
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
        tool_registry.tools["subagent_status"] = ToolSchema(
            name="subagent_status",
            description=(
                "Check the status of background subagent sessions. Use this to decide when "
                "to call subagent_result.\n\n"
                "WHEN TO CALL:\n"
                "  - After launching one or more background subagents (background=True) to see "
                "    which have finished (Idle) and which are still running (Working).\n"
                "  - Without arguments to get a summary table of all active sessions.\n"
                "  - With a specific session_id to get detailed info on one session.\n\n"
                "INTERFACE:\n"
                "  session_id (optional) — omit to list all sessions; provide to inspect one.\n\n"
                "RETURNS (no session_id): JSON {status, session_count, summary_table}.\n"
                "RETURNS (with session_id): JSON {session_id, status, category, created_at, ...}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": ["string", "null"],
                        "description": "Session ID returned by a background subagent call. Omit to list all active sessions.",
                    }
                },
                "required": [],
            },
        )
        tool_registry.handlers["subagent_status"] = self._make_status_handler(
            world, entity_id
        )

        # Install subagent_result tool
        tool_registry.tools["subagent_result"] = ToolSchema(
            name="subagent_result",
            description=(
                "Block until a background subagent session finishes, then return its result.\n\n"
                "WHEN TO CALL:\n"
                "  - After launching a subagent with background=True and you are ready to use "
                "    its output.\n"
                "  - You may call subagent_status first to check if the session is already Idle "
                "    (avoiding unnecessary blocking).\n\n"
                "INTERFACE:\n"
                "  session_id (required) — the session_id from the background subagent response.\n"
                "  timeout    (optional) — max seconds to wait; null = wait indefinitely.\n\n"
                "RETURNS: final answer string from the subagent, or an error/timeout message."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID of the background subagent to wait for. Obtained from the subagent(background=True) response.",
                    },
                    "timeout": {
                        "type": ["number", "null"],
                        "description": "Max seconds to wait before returning a timeout error. null = wait indefinitely until the session finishes.",
                    },
                },
                "required": ["session_id"],
            },
        )
        tool_registry.handlers["subagent_result"] = self._make_result_handler(
            world, entity_id
        )

        # Install subagent_cancel tool
        tool_registry.tools["subagent_cancel"] = ToolSchema(
            name="subagent_cancel",
            description=(
                "Abort a running background subagent session and free its resources.\n\n"
                "WHEN TO CALL:\n"
                "  - When a background subagent is no longer needed (e.g. another session "
                "    already produced the answer, or the task was superseded).\n"
                "  - After a timeout or error, to clean up a stuck session.\n\n"
                "INTERFACE:\n"
                "  session_id (required) — the session_id to cancel.\n\n"
                "RETURNS: JSON {status, session_id, lifecycle_status}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID of the background subagent to abort. Obtained from the subagent(background=True) response.",
                    }
                },
                "required": ["session_id"],
            },
        )
        tool_registry.handlers["subagent_cancel"] = self._make_cancel_handler(
            world, entity_id
        )

        logger.info("subagent_control_tools_installed", entity_id=entity_id)

    def _make_status_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        """Create handler for subagent_status tool."""

        async def status_handler(session_id: str | None = None) -> str:
            from ecs_agent.components.definitions import SubagentSessionTableComponent
            from ecs_agent.systems.subagent_runtime import (
                render_subagent_session_reminder_table,
            )

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
            return json.dumps(
                {
                    "status": "ok",
                    "session_id": session.session_id,
                    "category": session.category,
                    "lifecycle_status": session.status,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "result_excerpt": session.result_excerpt,
                    "error": session.error,
                }
            )

        return status_handler

    def _make_result_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        """Create handler for subagent_result tool."""

        async def result_handler(session_id: str, timeout: float | None = None) -> str:
            logger.info(
                "subagent_result_requested",
                parent_entity=parent_entity_id,
                session_id=session_id,
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

            if session.status in ("Dead", "Timeout", "Cancelled"):
                return json.dumps(
                    {
                        "status": "terminal",
                        "session_id": session_id,
                        "lifecycle_status": session.status,
                        "result_excerpt": session.result_excerpt,
                        "error": session.error,
                    }
                )

            if session.status == "Idle":
                logger.info(
                    "subagent_result_ready",
                    parent_entity=parent_entity_id,
                    session_id=session_id,
                    result_length=len(session.result_excerpt or ""),
                )
                return json.dumps(
                    {
                        "status": "success",
                        "session_id": session_id,
                        "lifecycle_status": session.status,
                        "result_excerpt": session.result_excerpt,
                    }
                )

            task = await self._runtime_manager.get_task(session_id)
            if task is None:
                return json.dumps(
                    {
                        "error": f"Task handle not found for session: {session_id}",
                        "session_id": session_id,
                    }
                )

            logger.info(
                "subagent_result_awaiting",
                parent_entity=parent_entity_id,
                session_id=session_id,
                timeout=timeout,
            )
            try:
                if timeout is not None:
                    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                else:
                    await asyncio.shield(task)

                session = await self._runtime_manager.get_session(session_id)
                if session is None:
                    return json.dumps(
                        {
                            "error": f"Session disappeared after await: {session_id}",
                            "session_id": session_id,
                        }
                    )

                logger.info(
                    "subagent_result_collected",
                    parent_entity=parent_entity_id,
                    session_id=session_id,
                    lifecycle_status=session.status,
                )
                return json.dumps(
                    {
                        "status": "completed",
                        "session_id": session_id,
                        "lifecycle_status": session.status,
                        "result_excerpt": session.result_excerpt,
                        "error": session.error,
                    }
                )

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

            if session.status in ("Dead", "Timeout", "Cancelled"):
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
                final_status=session.status if session else "Cancelled",
            )
            return json.dumps(
                {
                    "status": "cancelled",
                    "session_id": session_id,
                    "lifecycle_status": session.status if session else "Cancelled",
                }
            )

        return cancel_handler

    def _build_delegate_tool_schema(self, subagent_names: list[str]) -> dict[str, Any]:
        """Build OpenAI-style function schema for the delegate tool."""
        del subagent_names
        return {
            "type": "function",
            "function": {
                "name": "delegate",
                "description": (
                    "Delegate a self-contained task to a named subagent and return its result synchronously.\n\n"
                    "WHEN TO CALL:\n"
                    "  - Use when a subtask can be fully handled by a specific registered subagent.\n"
                    "  - This is a synchronous, fire-and-forget call: it blocks until the subagent completes.\n"
                    "  - For parallel execution or finer control, prefer the 'subagent' tool with background=True.\n\n"
                    "INTERFACE:\n"
                    "  subagent_name (required) — name of the registered subagent to invoke.\n"
                    "  task          (required) — full task description; include goal, context, and expected output.\n\n"
                    "RETURNS: final answer string from the subagent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_name": {
                            "type": "string",
                            "description": "Name of the registered subagent to invoke. Must match a key in SubagentRegistryComponent.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Full task description for the subagent. Be explicit: include goal, context, expected output format, and any constraints.",
                        },
                    },
                    "required": ["subagent_name", "task"],
                },
            },
        }

    def _install_delegate_handler(
        self,
        world: World,
        entity_id: EntityId,
        tool_name: str,
        override: bool,
    ) -> None:
        """Install delegate tool handler on ToolRegistryComponent."""
        tool_registry = world.get_component(entity_id, ToolRegistryComponent)
        if tool_registry is None:
            raise ValueError(
                f"Error: ToolRegistryComponent not found on entity {entity_id}"
            )

        if tool_name in tool_registry.handlers and not override:
            return

        tool_registry.handlers[tool_name] = self._make_delegate_handler(
            world, entity_id
        )

    def _make_delegate_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        """Create a delegate handler closure that captures world and parent entity."""

        async def delegate_handler(subagent_name: str, task: str) -> str:
            """Execute a subagent delegation.

            Args:
                subagent_name: Name of the subagent to delegate to
                task: Task description for the subagent

            Returns:
                Result string from the subagent's final assistant message
            """
            correlation_id = str(uuid.uuid4())
            traceparent = generate_traceparent()

            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                task=task,
            )

            logger.info(
                "delegation_started",
                parent_entity=parent_entity_id,
                subagent_name=subagent_name,
                task=task,
            )

            # Resolve timeout and call shared execution core with timeout wrapping
            resolved_timeout = self._resolve_timeout(
                None
            )  # delegate uses global default only

            try:
                if resolved_timeout is not None:
                    result, success, error = await asyncio.wait_for(
                        self._execute_subagent_core(
                            world,
                            parent_entity_id,
                            subagent_name,
                            task,
                            correlation_id,
                            traceparent,
                        ),
                        timeout=resolved_timeout,
                    )
                else:
                    result, success, error = await self._execute_subagent_core(
                        world,
                        parent_entity_id,
                        subagent_name,
                        task,
                        correlation_id,
                        traceparent,
                    )
            except asyncio.TimeoutError:
                error_msg = f"Error: Subagent timeout after {resolved_timeout}s"
                logger.error(
                    "delegation_timeout",
                    timeout=resolved_timeout,
                    subagent=subagent_name,
                )
                # Emit completion event with error
                await world.event_bus.publish(
                    DelegationCompletedEvent(
                        entity_id=parent_entity_id,
                        subagent_name=subagent_name,
                        success=False,
                        result=error_msg,
                        correlation_id=correlation_id,
                        traceparent=traceparent,
                    )
                )
                result = error_msg

            return result

        return delegate_handler

    def _make_subagent_handler(self, world: World, parent_entity_id: EntityId) -> Any:
        async def subagent_handler(
            category: str,
            prompt: str,
            load_skills: list[str] | None = None,
            background: bool = False,
            timeout: float | None = None,
        ) -> str:
            effective_load_skills = [] if load_skills is None else load_skills
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

            try:
                config = self._resolve_subagent_config(registry_comp, category)
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
            resolved_timeout = self._resolve_timeout(timeout)

            async def execute_with_effective_skills() -> tuple[str, bool, str | None]:
                original_config = registry_comp.subagents[category]
                registry_comp.subagents[category] = effective_config
                try:
                    return await self._execute_subagent_core(
                        world,
                        parent_entity_id,
                        category,
                        prompt,
                        correlation_id,
                        traceparent,
                    )
                finally:
                    registry_comp.subagents[category] = original_config

            if not background:
                # Sync mode: wrap with timeout
                try:
                    if resolved_timeout is not None:
                        result, _, _ = await asyncio.wait_for(
                            execute_with_effective_skills(), timeout=resolved_timeout
                        )
                    else:
                        result, _, _ = await execute_with_effective_skills()
                except asyncio.TimeoutError:
                    error_msg = f"Error: Subagent timeout after {resolved_timeout}s"
                    logger.error(
                        "subagent_timeout", timeout=resolved_timeout, category=category
                    )
                    result = error_msg
                logger.info(
                    "subagent_sync_completed",
                    parent_entity=parent_entity_id,
                    category=category,
                    result_length=len(result),
                )
                return result

            session_id = self._runtime_manager.create_session()
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            metadata = SubagentSessionRecord(
                session_id=session_id,
                category=category,
                prompt=prompt,
                parent_entity_id=parent_entity_id,
                created_at=now_iso,
                updated_at=now_iso,
                load_skills=normalized_skills,
                background=True,
                status="Working",
                correlation_id=correlation_id,
                traceparent=traceparent,
                timeout_seconds=timeout,
            )

            async def run_in_background() -> None:
                try:
                    if resolved_timeout is not None:
                        result, success, error = await asyncio.wait_for(
                            execute_with_effective_skills(), timeout=resolved_timeout
                        )
                    else:
                        result, success, error = await execute_with_effective_skills()
                except asyncio.TimeoutError:
                    error_msg = f"Error: Subagent timeout after {resolved_timeout}s"
                    logger.error(
                        "subagent_background_timeout",
                        timeout=resolved_timeout,
                        category=category,
                    )
                    # Update session to Timeout status
                    await self._runtime_manager.update_timeout(session_id, error_msg)
                    # Hook: Sync to component after timeout
                    await self._runtime_manager.sync_to_component(
                        world, parent_entity_id
                    )
                    return

                metadata.updated_at = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )
                if success:
                    metadata.status = "Idle"
                    metadata.result_excerpt = result[:200]
                else:
                    metadata.status = "Dead"
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
                # Hook: Sync to component after completion/error
                await self._runtime_manager.sync_to_component(world, parent_entity_id)

            logger.info(
                "subagent_background_launched",
                parent_entity=parent_entity_id,
                session_id=session_id,
                category=category,
                timeout=resolved_timeout,
            )
            background_task = asyncio.create_task(run_in_background())
            await self._runtime_manager.register_task(
                session_id, background_task, metadata
            )
            # Hook: Sync to component after launch
            await self._runtime_manager.sync_to_component(world, parent_entity_id)

            return json.dumps(
                {
                    "session_id": session_id,
                    "status": "Working",
                    "category": category,
                    "timeout": timeout,
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
    ) -> tuple[str, bool, str | None]:
        """Shared subagent execution core for both delegate and subagent APIs.

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
        # Get registry component
        registry_comp = world.get_component(parent_entity_id, SubagentRegistryComponent)
        if registry_comp is None:
            error_msg = f"Error: SubagentRegistryComponent not found on entity {parent_entity_id}"
            logger.error("delegation_failed", reason=error_msg)
            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                result=error_msg,
                success=False,
                error=error_msg,
            )
            return (error_msg, False, error_msg)

        # Resolve config
        try:
            config = self._resolve_subagent_config(registry_comp, subagent_name)
        except ValueError as exc:
            error_msg = str(exc)
            logger.error(
                "delegation_failed",
                reason="unknown_subagent",
                subagent_name=subagent_name,
                available=list(registry_comp.subagents.keys()),
            )
            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                result=error_msg,
                success=False,
                error=error_msg,
            )
            return (error_msg, False, error_msg)

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
                    provider=config.provider,
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

            child_world, child_world_entity_id = self._assemble_child_world(
                world,
                parent_entity_id,
                config,
                parent_child_entity=child_entity_id,
            )
            result = await self._execute_delegation(
                child_world,
                child_world_entity_id,
                task,
                config,
            )

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

            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                result=result,
                success=True,
                error=None,
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
            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                result=error_msg,
                success=False,
                error=error_msg,
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
            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                result=error_msg,
                success=False,
                error=error_msg,
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
            await self._publish_delegation_events(
                world,
                parent_entity_id,
                subagent_name,
                correlation_id=correlation_id,
                traceparent=traceparent,
                result=error_msg,
                success=False,
                error=error_msg,
            )
            return (error_msg, False, error_msg)

    def _resolve_subagent_config(
        self,
        registry: SubagentRegistryComponent,
        subagent_name: str,
    ) -> SubagentConfig:
        """Resolve and validate subagent configuration from registry."""
        config = registry.subagents.get(subagent_name)
        if config is None:
            raise ValueError(
                f"Error: Unknown subagent '{subagent_name}'. Available subagents: {list(registry.subagents.keys())}"
            )

        # Task 9: Wrap provider with RetryProvider by default
        wrapped_provider = self._wrap_retry_provider_if_needed(config.provider)

        # Return config with wrapped provider (use replace to preserve other fields)
        return replace(config, provider=wrapped_provider)

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
        """Assemble isolated child world and runnable child entity."""
        policy = config.inheritance_policy
        parent_llm = parent_world.get_component(parent_entity, LLMComponent)
        parent_tools = parent_world.get_component(parent_entity, ToolRegistryComponent)
        parent_permissions = parent_world.get_component(
            parent_entity, PermissionComponent
        )
        parent_skills = parent_world.get_component(parent_entity, SkillComponent)

        effective_system_prompt = config.system_prompt
        if (
            policy.enabled
            and policy.inherit_system_prompt
            and not effective_system_prompt
            and parent_llm is not None
        ):
            effective_system_prompt = parent_llm.system_prompt

        child_world = World()
        child_world_entity_id = child_world.create_entity()
        child_world.add_component(
            child_world_entity_id,
            LLMComponent(
                provider=config.provider,
                model=config.model,
                system_prompt=effective_system_prompt,
            ),
        )
        child_world.add_component(
            child_world_entity_id,
            SystemPromptComponent(
                template=effective_system_prompt,
                content=effective_system_prompt,
            ),
        )
        child_world.add_component(
            child_world_entity_id,
            ConversationComponent(messages=[]),
        )
        child_world.add_component(
            child_world_entity_id,
            OwnerComponent(owner_id=parent_entity),
        )

        if parent_child_entity is not None:
            parent_world.add_component(
                parent_child_entity,
                LLMComponent(
                    provider=config.provider,
                    model=config.model,
                    system_prompt=effective_system_prompt,
                ),
            )
            parent_world.add_component(
                parent_child_entity,
                SystemPromptComponent(
                    template=effective_system_prompt,
                    content=effective_system_prompt,
                ),
            )

        target_entities: list[tuple[World, EntityId]] = [
            (child_world, child_world_entity_id)
        ]
        if parent_child_entity is not None:
            target_entities.append((parent_world, parent_child_entity))

        if policy.enabled and policy.inherit_tools:
            for target_world, target_entity in target_entities:
                existing_registry = target_world.get_component(
                    target_entity, ToolRegistryComponent
                )
                if existing_registry is None:
                    target_world.add_component(
                        target_entity,
                        ToolRegistryComponent(tools={}, handlers={}),
                    )

        skill_manager = SkillManager()
        explicit_skills = list(dict.fromkeys(config.skills))
        inherited_tool_names = self._effective_inherited_tool_names(policy)
        inherited_skills = self._skills_for_inherited_tools(
            inherited_tool_names,
            parent_skills,
        )
        required_skills = list(dict.fromkeys(explicit_skills + inherited_skills))

        resolved_skills: list[ScriptSkill] = []
        for skill_name in required_skills:
            skill = self._resolve_parent_skill(
                parent_entity,
                skill_name,
                parent_skills,
                parent_tools,
                policy,
            )
            if skill is not None:
                resolved_skills.append(skill)

        for target_world, target_entity in target_entities:
            for skill in resolved_skills:
                skill_manager.install(target_world, target_entity, skill)

        if policy.enabled and parent_tools is not None:
            for tool_name in inherited_tool_names:
                self._inherit_tool_to_target_entities(
                    tool_name,
                    parent_tools,
                    target_entities,
                    policy,
                )

        if (
            policy.enabled
            and policy.inherit_permissions
            and parent_permissions is not None
        ):
            for target_world, target_entity in target_entities:
                target_world.add_component(
                    target_entity,
                    PermissionComponent(
                        allowed_tools=list(parent_permissions.allowed_tools),
                        denied_tools=list(parent_permissions.denied_tools),
                    ),
                )

        child_world.register_system(ReasoningSystem(priority=0), priority=0)
        child_world.register_system(MemorySystem(), priority=10)
        child_world.register_system(
            ErrorHandlingSystem(priority=99),
            priority=99,
        )
        return child_world, child_world_entity_id

    def _effective_inherited_tool_names(self, policy: InheritancePolicy) -> list[str]:
        if not policy.enabled:
            return []

        if policy.allow_delegate_tool:
            return list(policy.inherit_tools)

        return [
            tool_name for tool_name in policy.inherit_tools if tool_name != "delegate"
        ]

    def _skills_for_inherited_tools(
        self,
        inherited_tool_names: list[str],
        parent_skills: SkillComponent | None,
    ) -> list[str]:
        if parent_skills is None or not inherited_tool_names:
            return []

        inherited_tool_set = set(inherited_tool_names)
        inherited_skill_names: list[str] = []
        for metadata in parent_skills.skills.values():
            if any(
                tool_name in inherited_tool_set for tool_name in metadata.tool_names
            ):
                inherited_skill_names.append(metadata.name)
        return inherited_skill_names

    def _resolve_parent_skill(
        self,
        parent_entity: EntityId,
        skill_name: str,
        parent_skills: SkillComponent | None,
        parent_tools: ToolRegistryComponent | None,
        policy: InheritancePolicy,
    ) -> ScriptSkill | None:
        if parent_skills is None or parent_tools is None:
            return self._handle_missing_skill(parent_entity, skill_name, policy)

        metadata = parent_skills.skills.get(skill_name)
        if metadata is None:
            return self._handle_missing_skill(parent_entity, skill_name, policy)

        tools: dict[str, tuple[ToolSchema, Any]] = {}
        for tool_name in metadata.tool_names:
            schema = parent_tools.tools.get(tool_name)
            handler = parent_tools.handlers.get(tool_name)
            if schema is None or handler is None:
                return self._handle_missing_skill(parent_entity, skill_name, policy)
            tools[tool_name] = (schema, handler)
        return _InheritedSkill(metadata, tools)

    def _handle_missing_skill(
        self,
        parent_entity: EntityId,
        skill_name: str,
        policy: InheritancePolicy,
    ) -> ScriptSkill | None:
        message = f"Missing skill '{skill_name}' on parent entity {parent_entity} during subagent delegation"
        if policy.missing_skill_policy == "error":
            raise ValueError(message)
        if policy.missing_skill_policy == "warn":
            logger.warning(
                "subagent_missing_skill",
                parent_entity=parent_entity,
                skill_name=skill_name,
                message=message,
            )
            return None
        raise ValueError(
            f"Invalid missing_skill_policy '{policy.missing_skill_policy}' for subagent inheritance"
        )

    def _inherit_tool_to_target_entities(
        self,
        tool_name: str,
        parent_tools: ToolRegistryComponent,
        target_entities: list[tuple[World, EntityId]],
        policy: InheritancePolicy,
    ) -> None:
        schema = parent_tools.tools.get(tool_name)
        handler = parent_tools.handlers.get(tool_name)
        if schema is None or handler is None:
            return

        for target_world, target_entity in target_entities:
            registry = target_world.get_component(target_entity, ToolRegistryComponent)
            if registry is None:
                registry = ToolRegistryComponent(tools={}, handlers={})
                target_world.add_component(target_entity, registry)

            has_conflict = tool_name in registry.tools or tool_name in registry.handlers
            if has_conflict:
                if policy.tool_conflict_policy == "skip":
                    continue
                if policy.tool_conflict_policy == "error":
                    raise ValueError(f"Tool inheritance conflict for '{tool_name}'")
                if policy.tool_conflict_policy != "override":
                    raise ValueError(
                        f"Invalid tool_conflict_policy '{policy.tool_conflict_policy}'"
                    )
            elif policy.tool_conflict_policy == "error":
                raise ValueError(f"Tool inheritance conflict for '{tool_name}'")

            registry.tools[tool_name] = schema
            registry.handlers[tool_name] = handler

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
        await runner.run(child_world, max_ticks=config.max_ticks)
        return self._extract_delegation_result(child_world, child_entity)

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
    ) -> None:
        """Publish start/completion delegation events via one wrapper API."""
        if task is not None:
            await world.event_bus.publish(
                DelegationStartedEvent(
                    entity_id=parent_entity_id,
                    subagent_name=subagent_name,
                    task=task,
                    correlation_id=correlation_id,
                    traceparent=traceparent,
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
                    correlation_id=correlation_id,
                    traceparent=traceparent,
                )
            )
