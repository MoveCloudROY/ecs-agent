"""Subagent delegation system."""

from __future__ import annotations

import uuid
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
)
from ecs_agent.components.definitions import SkillComponent
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.observability import generate_traceparent
from ecs_agent.types import (
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    Message,
    SubagentConfig,
    ToolSchema,
)

logger = get_logger(__name__)


class SubagentSystem:
    """System that manages subagent delegation lifecycle.

    This system automatically registers a 'delegate' tool for entities that have
    a SubagentRegistryComponent. When the delegate tool is called, it:
    1. Creates a child entity with the specified subagent configuration
    2. Runs the child entity to completion
    3. Returns the child's final assistant message
    4. Publishes delegation events to the event bus
    """

    def __init__(self, priority: int = -1) -> None:
        self.priority = priority

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
            raise ValueError(
                f"Entity {entity_id} missing SubagentRegistryComponent"
            )

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
            self.install_delegate_tool(world, entity_id, tool_name="delegate", override=False)

            logger.info(
                "delegate_tool_registered",
                entity_id=entity_id,
                available_subagents=list(registry_comp.subagents.keys()),
            )
    def _build_delegate_tool_schema(self, subagent_names: list[str]) -> dict[str, Any]:
        """Build OpenAI-style function schema for the delegate tool."""
        del subagent_names
        return {
            "type": "function",
            "function": {
                "name": "delegate",
                "description": (
                    "Delegate a task to a named subagent. The subagent will execute "
                    "the task independently and return its result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_name": {
                            "type": "string",
                            "description": "Name of the subagent to delegate to",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task description for the subagent",
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

            registry_comp = world.get_component(
                parent_entity_id, SubagentRegistryComponent
            )
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
                return error_msg

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
                return error_msg

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
                    ConversationComponent(
                        messages=[Message(role="user", content=task)]
                    ),
                )

                world.add_component(
                    child_entity_id, OwnerComponent(owner_id=parent_entity_id)
                )

                if config.skills:
                    parent_skill_comp = world.get_component(
                        parent_entity_id, SkillComponent
                    )
                    parent_tool_reg = world.get_component(
                        parent_entity_id, ToolRegistryComponent
                    )

                    if parent_skill_comp is not None and parent_tool_reg is not None:
                        child_skill_comp = world.get_component(
                            child_entity_id, SkillComponent
                        )
                        child_tool_reg = world.get_component(
                            child_entity_id, ToolRegistryComponent
                        )

                        if child_skill_comp is None:
                            child_skill_comp = SkillComponent(skills={})
                            world.add_component(child_entity_id, child_skill_comp)

                        if child_tool_reg is None:
                            child_tool_reg = ToolRegistryComponent(
                                tools={}, handlers={}
                            )
                            world.add_component(child_entity_id, child_tool_reg)

                        for skill_name in config.skills:
                            if skill_name in parent_skill_comp.skills:
                                child_skill_comp.skills[skill_name] = (
                                    parent_skill_comp.skills[skill_name]
                                )

                                metadata = parent_skill_comp.skills[skill_name]
                                for tool_name in metadata.tool_names:
                                    if tool_name in parent_tool_reg.tools:
                                        child_tool_reg.tools[tool_name] = (
                                            parent_tool_reg.tools[tool_name]
                                        )
                                    if tool_name in parent_tool_reg.handlers:
                                        child_tool_reg.handlers[tool_name] = (
                                            parent_tool_reg.handlers[tool_name]
                                        )

                                logger.info(
                                    "skill_copied_to_child",
                                    parent_entity=parent_entity_id,
                                    child_entity=child_entity_id,
                                    skill_name=skill_name,
                                )
                            else:
                                logger.warning(
                                    "skill_not_found_on_parent",
                                    parent_entity=parent_entity_id,
                                    child_entity=child_entity_id,
                                    skill_name=skill_name,
                                )
                    else:
                        logger.warning(
                            "parent_missing_skill_components",
                            parent_entity=parent_entity_id,
                            has_skill_comp=parent_skill_comp is not None,
                            has_tool_reg=parent_tool_reg is not None,
                        )
                child_world, child_world_entity_id = self._assemble_child_world(
                    world, parent_entity_id, config
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

                return result

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
                return error_msg

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
                return error_msg

        return delegate_handler

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
        return config

    def _assemble_child_world(
        self,
        parent_world: World,
        parent_entity: EntityId,
        config: SubagentConfig,
    ) -> tuple[World, EntityId]:
        """Assemble isolated child world and runnable child entity."""
        _ = parent_world

        child_world = World()
        child_world_entity_id = child_world.create_entity()
        child_world.add_component(
            child_world_entity_id,
            LLMComponent(
                provider=config.provider,
                model=config.model,
                system_prompt=config.system_prompt,
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
        child_world.register_system(ReasoningSystem(priority=0), priority=0)
        child_world.register_system(MemorySystem(), priority=10)
        child_world.register_system(
            ErrorHandlingSystem(priority=99),
            priority=99,
        )
        return child_world, child_world_entity_id

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
