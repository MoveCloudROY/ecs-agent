"""Subagent delegation system."""

from __future__ import annotations

from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    SubagentRegistryComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
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

    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        """Register delegate tool for entities with SubagentRegistryComponent."""
        for entity_id, components in world.query(
            SubagentRegistryComponent, ToolRegistryComponent
        ):
            registry_comp, tool_registry = components
            assert isinstance(registry_comp, SubagentRegistryComponent)
            assert isinstance(tool_registry, ToolRegistryComponent)

            # Skip if delegate tool already registered
            if "delegate" in tool_registry.tools:
                continue

            # Register the delegate tool
            delegate_schema = ToolSchema(
                name="delegate",
                description=(
                    "Delegate a task to a named subagent. The subagent will execute "
                    "the task independently and return its result."
                ),
                parameters={
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
            )

            # Create the handler closure that captures world and entity_id
            handler = self._make_delegate_handler(world, entity_id)

            tool_registry.tools["delegate"] = delegate_schema
            tool_registry.handlers["delegate"] = handler

            logger.info(
                "delegate_tool_registered",
                entity_id=entity_id,
                available_subagents=list(registry_comp.subagents.keys()),
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
            # Publish DelegationStartedEvent
            await world.event_bus.publish(
                DelegationStartedEvent(
                    entity_id=parent_entity_id,
                    subagent_name=subagent_name,
                    task=task,
                )
            )

            logger.info(
                "delegation_started",
                parent_entity=parent_entity_id,
                subagent_name=subagent_name,
                task=task,
            )

            # Look up subagent config
            registry_comp = world.get_component(
                parent_entity_id, SubagentRegistryComponent
            )
            if registry_comp is None:
                error_msg = f"Error: SubagentRegistryComponent not found on entity {parent_entity_id}"
                logger.error("delegation_failed", reason=error_msg)
                await world.event_bus.publish(
                    DelegationCompletedEvent(
                        entity_id=parent_entity_id,
                        subagent_name=subagent_name,
                        result=error_msg,
                    )
                )
                return error_msg

            config = registry_comp.subagents.get(subagent_name)
            if config is None:
                error_msg = f"Error: Unknown subagent '{subagent_name}'. Available subagents: {list(registry_comp.subagents.keys())}"
                logger.error(
                    "delegation_failed",
                    reason="unknown_subagent",
                    subagent_name=subagent_name,
                    available=list(registry_comp.subagents.keys()),
                )
                await world.event_bus.publish(
                    DelegationCompletedEvent(
                        entity_id=parent_entity_id,
                        subagent_name=subagent_name,
                        result=error_msg,
                    )
                )
                return error_msg

            try:
                # Create child entity
                child_entity_id = world.create_entity()
                logger.info(
                    "child_entity_created",
                    parent_entity=parent_entity_id,
                    child_entity=child_entity_id,
                    subagent_name=subagent_name,
                )

                # Add components to child entity
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

                # TODO: Install skills if config.skills is not empty
                # This requires SkillManager integration - defer to Task 15

                # Ensure minimal systems are registered in world for child execution
                # Check if ReasoningSystem is already registered
                has_reasoning = False
                for sys, _ in world._systems._systems:
                    if isinstance(sys, ReasoningSystem):
                        has_reasoning = True
                        break

                if not has_reasoning:
                    # Register minimal systems needed for subagent execution
                    world.register_system(ReasoningSystem(priority=0), priority=0)
                    world.register_system(MemorySystem(), priority=10)
                    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

                # Run child entity to completion
                runner = Runner()
                await runner.run(world, max_ticks=config.max_ticks)

                # Extract result from child's conversation
                child_conv = world.get_component(child_entity_id, ConversationComponent)
                result = "Error: No conversation found"
                if child_conv is not None:
                    # Find the last assistant message
                    for message in reversed(child_conv.messages):
                        if message.role == "assistant":
                            result = message.content
                            break
                    else:
                        result = (
                            "Error: No assistant message found in subagent conversation"
                        )

                logger.info(
                    "delegation_completed",
                    parent_entity=parent_entity_id,
                    child_entity=child_entity_id,
                    subagent_name=subagent_name,
                    result_length=len(result),
                )

                # Publish DelegationCompletedEvent
                await world.event_bus.publish(
                    DelegationCompletedEvent(
                        entity_id=parent_entity_id,
                        subagent_name=subagent_name,
                        result=result,
                    )
                )

                return result

            except Exception as exc:
                error_msg = f"Error during delegation: {exc}"
                logger.error(
                    "delegation_exception",
                    parent_entity=parent_entity_id,
                    subagent_name=subagent_name,
                    exception=str(exc),
                )
                await world.event_bus.publish(
                    DelegationCompletedEvent(
                        entity_id=parent_entity_id,
                        subagent_name=subagent_name,
                        result=error_msg,
                    )
                )
                return error_msg

        return delegate_handler
