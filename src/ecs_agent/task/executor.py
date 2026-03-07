"""Task executor adapter for routing to local tool/skill or subagent backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PendingToolCallsComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.task.fetching_unit import DispatchRequest
from ecs_agent.types import CompletionResult, EntityId, Message, ToolCall

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    """Normalized result from backend execution."""

    task_id: str
    success: bool
    result_content: str
    backend_type: str  # "local" or "subagent"


class TaskExecutor:
    """Routes dispatch requests to local tool/skill or subagent backend.

    Backend selection logic:
    - assigned_agent: EntityId → local tool/skill execution
    - assigned_agent: str → subagent delegation (str is subagent name)
    - assigned_agent: None → default to local execution

    Both backends return normalized ExecutionResult.
    """

    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        """System processing hook (not currently used - adapter pattern)."""
        # This is an adapter, not a standalone system
        # External systems call execute_dispatch_request directly
        del world

    async def execute_dispatch_request(
        self,
        world: World,
        entity_id: EntityId,
        request: DispatchRequest,
    ) -> ExecutionResult:
        """Execute a dispatch request via appropriate backend.

        Args:
            world: World instance
            entity_id: Entity performing the task
            request: Dispatch request with task metadata

        Returns:
            ExecutionResult with normalized status/result

        Raises:
            ValueError: If backend routing fails or backend validation fails
        """
        backend_type = self._determine_backend(request)

        logger.info(
            "task_executor_routing",
            task_id=request.task_id,
            backend_type=backend_type,
            assigned_agent=request.assigned_agent,
        )

        if backend_type == "subagent":
            return await self._execute_via_subagent(world, entity_id, request)
        else:
            return await self._execute_via_local_tools(world, entity_id, request)

    def _determine_backend(self, request: DispatchRequest) -> str:
        """Determine backend type from assigned_agent field.

        Returns:
            "local" or "subagent"

        Raises:
            ValueError: If assigned_agent has invalid type
        """
        if request.assigned_agent is None:
            # Default policy: local execution
            return "local"
        elif isinstance(request.assigned_agent, str):
            # String → subagent name
            return "subagent"
        elif isinstance(request.assigned_agent, int):
            # EntityId (NewType of int) → local execution
            return "local"
        else:
            raise ValueError(
                f"Invalid assigned_agent type: {type(request.assigned_agent).__name__}. "
                f"Expected EntityId, str, or None."
            )

    async def _execute_via_subagent(
        self,
        world: World,
        entity_id: EntityId,
        request: DispatchRequest,
    ) -> ExecutionResult:
        """Execute task via subagent delegation.

        Reuses SubagentSystem contract via delegate tool handler.
        """
        if not isinstance(request.assigned_agent, str):
            raise ValueError(
                f"Subagent backend requires str subagent name, got {type(request.assigned_agent).__name__}"
            )

        subagent_name = request.assigned_agent

        # Validate subagent registry
        registry = world.get_component(entity_id, SubagentRegistryComponent)
        if registry is None:
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Entity {entity_id} missing SubagentRegistryComponent for subagent delegation",
                backend_type="subagent",
            )

        if subagent_name not in registry.subagents:
            available = list(registry.subagents.keys())
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Unknown subagent '{subagent_name}'. Available: {available}",
                backend_type="subagent",
            )

        # Validate ToolRegistryComponent has delegate handler
        tool_registry = world.get_component(entity_id, ToolRegistryComponent)
        if tool_registry is None or "delegate" not in tool_registry.handlers:
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Entity {entity_id} missing delegate tool handler",
                backend_type="subagent",
            )

        # Execute via delegate handler
        delegate_handler = tool_registry.handlers["delegate"]
        try:
            result = await delegate_handler(
                subagent_name=subagent_name, task=request.description
            )
            success = not result.startswith("Error")
            return ExecutionResult(
                task_id=request.task_id,
                success=success,
                result_content=result,
                backend_type="subagent",
            )
        except Exception as exc:
            logger.error(
                "subagent_execution_failed",
                task_id=request.task_id,
                subagent_name=subagent_name,
                exception=str(exc),
            )
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Subagent execution failed: {exc}",
                backend_type="subagent",
            )

    async def _execute_via_local_tools(
        self,
        world: World,
        entity_id: EntityId,
        request: DispatchRequest,
    ) -> ExecutionResult:
        """Execute task via local tool/skill execution.

        Reuses ToolExecutionSystem contract by setting up PendingToolCallsComponent
        and waiting for ToolResultsComponent.
        """
        if not isinstance(request.assigned_agent, (int, type(None))):
            raise ValueError(
                f"Local backend requires EntityId or None, got {type(request.assigned_agent).__name__}"
            )

        # Validate ToolRegistryComponent
        tool_registry = world.get_component(entity_id, ToolRegistryComponent)
        if tool_registry is None:
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Entity {entity_id} missing ToolRegistryComponent for local execution",
                backend_type="local",
            )

        # Validate LLMComponent for reasoning
        llm_component = world.get_component(entity_id, LLMComponent)
        if llm_component is None:
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Entity {entity_id} missing LLMComponent for local execution",
                backend_type="local",
            )

        # Create user message for task
        conv = world.get_component(entity_id, ConversationComponent)
        if conv is None:
            conv = ConversationComponent(messages=[])
            world.add_component(entity_id, conv)

        task_message = Message(role="user", content=request.description)
        conv.messages.append(task_message)

        # Execute LLM reasoning step to get tool calls
        try:
            completion_result = await llm_component.provider.complete(
                messages=conv.messages,
                tools=[schema for schema in tool_registry.tools.values()],
            )

            # Type guard: ensure we have CompletionResult, not AsyncIterator
            if not isinstance(completion_result, CompletionResult):
                return ExecutionResult(
                    task_id=request.task_id,
                    success=False,
                    result_content="Error: Unexpected streaming result from local execution",
                    backend_type="local",
                )

            # Add assistant response to conversation
            conv.messages.append(completion_result.message)

            # Check for tool calls
            if completion_result.message.tool_calls:
                # Set up pending tool calls for ToolExecutionSystem
                world.add_component(
                    entity_id,
                    PendingToolCallsComponent(tool_calls=completion_result.message.tool_calls),
                )

                # Execute tools via handlers (inline execution, not via ToolExecutionSystem tick)
                tool_results: dict[str, str] = {}
                for tool_call in completion_result.message.tool_calls:
                    tool_result = await self._execute_single_tool(
                        world, entity_id, tool_call, tool_registry
                    )
                    tool_results[tool_call.id] = tool_result
                    conv.messages.append(
                        Message(
                            role="tool",
                            content=tool_result,
                            tool_call_id=tool_call.id,
                        )
                    )

                # Clean up pending component
                world.remove_component(entity_id, PendingToolCallsComponent)
                world.add_component(
                    entity_id, ToolResultsComponent(results=tool_results)
                )

                # Return aggregated tool results
                success = all(
                    not result.startswith("Error") for result in tool_results.values()
                )
                combined_result = "\n".join(
                    f"{tc.name}: {tool_results[tc.id]}"
                    for tc in completion_result.message.tool_calls
                )
                return ExecutionResult(
                    task_id=request.task_id,
                    success=success,
                    result_content=combined_result,
                    backend_type="local",
                )
            else:
                # No tool calls, return assistant message content
                return ExecutionResult(
                    task_id=request.task_id,
                    success=True,
                    result_content=completion_result.message.content,
                    backend_type="local",
                )

        except Exception as exc:
            logger.error(
                "local_execution_failed",
                task_id=request.task_id,
                exception=str(exc),
            )
            return ExecutionResult(
                task_id=request.task_id,
                success=False,
                result_content=f"Error: Local execution failed: {exc}",
                backend_type="local",
            )

    async def _execute_single_tool(
        self,
        world: World,
        entity_id: EntityId,
        tool_call: ToolCall,
        tool_registry: ToolRegistryComponent,
    ) -> str:
        """Execute a single tool call and return result string.

        Reuses ToolExecutionSystem's handler execution pattern.
        """
        del world, entity_id  # Not used in this simplified version

        handler = tool_registry.handlers.get(tool_call.name)
        if handler is None:
            return f"Error: unknown tool '{tool_call.name}'"

        try:
            result = await handler(**tool_call.arguments)
            logger.info(
                "tool_executed",
                tool_name=tool_call.name,
                success=True,
            )
            return str(result)
        except Exception as exc:
            logger.error(
                "tool_execution_failed",
                tool_name=tool_call.name,
                exception=str(exc),
            )
            return f"Error executing tool '{tool_call.name}': {exc}"


__all__ = ["TaskExecutor", "ExecutionResult"]
