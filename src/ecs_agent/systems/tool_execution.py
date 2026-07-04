"""Tool call dispatch system."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from ecs_agent.components import (
    ConversationComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    PendingToolCallsComponent,
    PlanComponent,
    SandboxConfigComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook import ArtifactRegistry, ScratchbookService, ToolResultsSink
from ecs_agent.token_counting import count_tokens
from ecs_agent.systems._plan_utils import derive_plan_name
from ecs_agent.tools.context import ToolExecutionContext, use_tool_context
from ecs_agent.tools.sandbox import sandboxed_execute
from ecs_agent.types import (
    CachedToolResultRef,
    EntityId,
    Message,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolResultCachedEvent,
)

logger = get_logger(__name__)


class ToolExecutionSystem:
    def __init__(
        self,
        priority: int = 0,
        scratchbook_service: ScratchbookService | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        self.priority = priority
        self.scratchbook_service = scratchbook_service
        effective_registry = registry
        if effective_registry is None and scratchbook_service is not None:
            effective_registry = ArtifactRegistry(root=scratchbook_service.root)
        self._registry = effective_registry
        self.tool_sink = (
            ToolResultsSink(effective_registry) if effective_registry else None
        )

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        ):
            pending, registry, conversation = components
            assert isinstance(pending, PendingToolCallsComponent)
            assert isinstance(registry, ToolRegistryComponent)
            assert isinstance(conversation, ConversationComponent)

            results: dict[str, str] = {}
            plan = world.get_component(entity_id, PlanComponent)
            plan_name: str | None = None
            if self._registry is not None and plan is not None:
                plan_name = derive_plan_name(
                    plan=plan,
                    conversation=conversation,
                    entity_id=entity_id,
                )

            for tool_call in pending.tool_calls:
                start_time = time.monotonic()
                tool_start_time = datetime.now(timezone.utc)
                await world.event_bus.publish(
                    ToolExecutionStartedEvent(
                        entity_id=entity_id,
                        tool_call=tool_call,
                        start_time=tool_start_time,
                    )
                )

                result = await self._execute_tool_call(
                    entity_id,
                    world,
                    tool_call,
                    registry.handlers,
                )

                success = not result.startswith("Error")
                tool_end_time = datetime.now(timezone.utc)
                await world.event_bus.publish(
                    ToolExecutionCompletedEvent(
                        entity_id=entity_id,
                        tool_call_id=tool_call.id,
                        result=result,
                        success=success,
                        tool_name=tool_call.name,
                        duration_seconds=time.monotonic() - start_time,
                        start_time=tool_start_time,
                        end_time=tool_end_time,
                    )
                )

                persisted_record_path: str | None = None
                if self.tool_sink is not None:
                    persist_result = self.tool_sink.persist_tool_result(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=result,
                        arguments=tool_call.arguments,
                    )
                    persisted_record_path = persist_result.record_path
                    results[tool_call.id] = persist_result.record_path
                    conversation.messages.append(
                        Message(
                            role="tool",
                            content=persist_result.record_path,
                            tool_call_id=tool_call.id,
                        )
                    )
                    await self._cache_overflowed_tool_result(
                        world=world,
                        entity_id=entity_id,
                        conversation=conversation,
                        tool_call=tool_call,
                        result=result,
                        artifact_path=persist_result.record_path,
                    )
                else:
                    results[tool_call.id] = result
                    conversation.messages.append(
                        Message(role="tool", content=result, tool_call_id=tool_call.id)
                    )

                if self._registry is not None and plan_name is not None:
                    if success:
                        updates: dict[str, str] = {
                            "status": "running",
                            "last_tool_call_id": tool_call.id,
                        }
                        if persisted_record_path is not None:
                            updates["last_tool_record_path"] = persisted_record_path
                        await self._registry.update_boulder(
                            plan_name=plan_name, updates=updates
                        )
                    else:
                        await self._registry.update_boulder(
                            plan_name=plan_name,
                            updates={
                                "status": "tool_failed",
                                "last_error": result,
                            },
                        )

            world.remove_component(entity_id, PendingToolCallsComponent)
            if results:
                world.add_component(entity_id, ToolResultsComponent(results=results))

    async def _execute_tool_call(
        self,
        entity_id: EntityId,
        world: World,
        tool_call: ToolCall,
        handlers: dict[str, Callable[..., Awaitable[str]]],
    ) -> str:
        logger.info(
            "tool_called", tool_name=tool_call.name, arguments=tool_call.arguments
        )

        handler = handlers.get(tool_call.name)
        if handler is None:
            reason = f"Error: unknown tool '{tool_call.name}'"
            logger.error("tool_failed", tool_name=tool_call.name, reason=reason)
            return reason

        start_time = time.monotonic()
        try:
            arguments = tool_call.arguments
            sandbox_config = world.get_component(entity_id, SandboxConfigComponent)
            context = ToolExecutionContext(
                world=world,
                entity_id=entity_id,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
            )
            with use_tool_context(context):
                if sandbox_config is None:
                    result = await handler(**arguments)
                else:
                    result = await sandboxed_execute(
                        handler,
                        arguments,
                        timeout=sandbox_config.timeout,
                        max_output_size=sandbox_config.max_output_size,
                    )

            duration_ms = (time.monotonic() - start_time) * 1000
            result_str = str(result)
            result_tail = "\n".join(result_str.splitlines()[-10:])
            logger.debug(
                "tool_result",
                tool_name=tool_call.name,
                success=True,
                duration_ms=duration_ms,
                result_tail=result_tail,
            )
            return result_str
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            reason = f"Error executing tool '{tool_call.name}': {exc}"
            logger.error(
                "tool_failed",
                tool_name=tool_call.name,
                reason=str(exc),
                duration_ms=duration_ms,
            )
            return reason

    async def _cache_overflowed_tool_result(
        self,
        *,
        world: World,
        entity_id: EntityId,
        conversation: ConversationComponent,
        tool_call: ToolCall,
        result: str,
        artifact_path: str | None,
    ) -> None:
        budget = world.get_component(entity_id, ContextTrimConfig)
        if budget is None or artifact_path is None or budget.max_tokens is None:
            # Model-window-derived budgets (max_tokens=None) do not drive this
            # transient overflow cache; the CompactionSystem trim step handles them.
            return

        estimated_tokens = self._estimate_conversation_tokens(
            conversation,
            chars_per_token=budget.token_estimation_chars_per_token,
        )
        if estimated_tokens <= budget.max_tokens:
            return

        cache = world.get_component(entity_id, ContextCacheComponent)
        if cache is None:
            cache = ContextCacheComponent()
            world.add_component(entity_id, cache)

        cached_hint = (
            f"[Tool result cached - retrieve full content from {artifact_path}]"
        )
        if all(ref.tool_call_id != tool_call.id for ref in cache.cached_tool_results):
            cache.cached_tool_results.append(
                CachedToolResultRef(
                    tool_call_id=tool_call.id,
                    artifact_path=artifact_path,
                    summary=cached_hint,
                    original_content=result,
                )
            )

        for index in range(len(conversation.messages) - 1, -1, -1):
            message = conversation.messages[index]
            if message.role != "tool" or message.tool_call_id != tool_call.id:
                continue
            conversation.messages[index] = Message(
                role="tool",
                content=cached_hint,
                parts=message.parts,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                compaction_metadata=message.compaction_metadata,
            )
            break

        logger.info(
            "tool_result_cached",
            entity_id=entity_id,
            tool_call_id=tool_call.id,
            artifact_path=artifact_path,
        )
        await world.event_bus.publish(
            ToolResultCachedEvent(
                entity_id=entity_id,
                tool_call_id=tool_call.id,
                artifact_path=artifact_path,
            )
        )

    def _estimate_conversation_tokens(
        self,
        conversation: ConversationComponent,
        *,
        chars_per_token: float,
    ) -> int:
        # Real BPE count when tiktoken is available; the CJK-aware fallback
        # reduces to ceil(total_chars / chars_per_token) for ASCII (ISSUE-8).
        text = "".join(
            message.content or "" for message in conversation.messages
        )
        return count_tokens(text, fallback_chars_per_token=chars_per_token)
