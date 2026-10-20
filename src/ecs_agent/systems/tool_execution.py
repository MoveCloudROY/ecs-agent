"""Tool call dispatch system."""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from ecs_agent.components import (
    AgentTurnComponent,
    ConversationComponent,
    ErrorComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    PendingToolCallsComponent,
    PlanComponent,
    SandboxConfigComponent,
    ToolExecutionConfigComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    TerminalComponent,
)
from ecs_agent.core.world import World
from ecs_agent.history import append_conversation_message, record_conversation_message, record_session_event, replace_conversation_history, tool_record_id
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook import ArtifactRegistry, ScratchbookService, ToolResultsSink
from ecs_agent.token_counting import count_tokens
from ecs_agent.systems._plan_utils import derive_plan_name
from ecs_agent.tools.authorization import (
    AuthorizationBatch,
    authorization_batch,
    authorization_fingerprint,
    authorize_tool_call,
    publish_tool_denial,
)
from ecs_agent.tools.context import ToolExecutionContext, use_tool_context
from ecs_agent.tools.sandbox import sandboxed_execute
from ecs_agent.tools.validation import tool_call_error
from ecs_agent.types import (
    CachedToolResultRef,
    EntityId,
    Message,
    ToolCall,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolResultCachedEvent,
    ToolSchema,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class _ToolOutcome:
    """A call result awaiting ordered, durable publication."""

    call: ToolCall
    result: str
    started_at: datetime | None = None
    duration_seconds: float = 0.0
    denial_reason: str | None = None


def _plan_execution_groups(
    tool_calls: list[ToolCall],
    tools: dict[str, ToolSchema],
) -> list[list[ToolCall]]:
    """Split a batch into ordered groups; calls within a group may run concurrently.

    Consecutive calls whose schema declares ``concurrency_safe`` share a group.
    Every other call (mutating tools, unknown tools) forms a singleton group,
    acting as a barrier: it starts only after earlier calls finished and blocks
    later calls until it returns.
    """
    groups: list[list[ToolCall]] = []
    safe_run: list[ToolCall] = []
    for tool_call in tool_calls:
        schema = tools.get(tool_call.name)
        if schema is not None and schema.concurrency_safe:
            safe_run.append(tool_call)
            continue
        if safe_run:
            groups.append(safe_run)
            safe_run = []
        groups.append([tool_call])
    if safe_run:
        groups.append(safe_run)
    return groups


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
        try:
            await self._process_pending(world)
        except (Exception, asyncio.CancelledError) as exc:
            # Once a tool might have run, leaving it queued would repeat side
            # effects on another tick. A durable start without a result means
            # "completion unconfirmed", never "safe to rerun".
            for entity_id, _ in world.query_active(PendingToolCallsComponent, ToolRegistryComponent, ConversationComponent):
                turn = world.get_component(entity_id, AgentTurnComponent)
                if turn is not None:
                    # Cancellation while awaiting approval has not started a
                    # side effect. Keep the queue for safe explicit continuation.
                    if isinstance(exc, asyncio.CancelledError) and not turn.unconfirmed_tool_calls:
                        continue
                    reason = "tool_execution_cancelled" if isinstance(exc, asyncio.CancelledError) else "tool_execution_error"
                    world.add_component(entity_id, ErrorComponent(error=str(exc) or reason, system_name="ToolExecutionSystem", timestamp=time.time()))
                    world.add_component(entity_id, TerminalComponent(reason=reason, error=str(exc) or reason))
                elif world.session_service is not None:
                    reason = "tool_execution_cancelled" if isinstance(exc, asyncio.CancelledError) else "tool_execution_error"
                    world.add_component(entity_id, ErrorComponent(error=str(exc) or reason, system_name="ToolExecutionSystem", timestamp=time.time()))
                    world.add_component(entity_id, TerminalComponent(reason=reason, error=str(exc) or reason))
                    world.remove_component(entity_id, PendingToolCallsComponent)
            logger.error("tool_execution_stopped", exception=str(exc) or type(exc).__name__)
            raise

    async def _process_pending(self, world: World) -> None:
        for entity_id, components in world.query_active(
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        ):
            pending, registry, conversation = components
            assert isinstance(pending, PendingToolCallsComponent)
            assert isinstance(registry, ToolRegistryComponent)
            assert isinstance(conversation, ConversationComponent)

            batch = authorization_batch(world, entity_id, pending)
            async with batch.execution_lock:
                if batch.completed or world.get_component(entity_id, PendingToolCallsComponent) is not pending:
                    continue
                await self._process_entity(world, entity_id, pending, registry, conversation, batch)

    async def _process_entity(
        self,
        world: World,
        entity_id: EntityId,
        pending: PendingToolCallsComponent,
        registry: ToolRegistryComponent,
        conversation: ConversationComponent,
        batch: AuthorizationBatch,
    ) -> None:
        ids = [call.id for call in pending.tool_calls]
        if any(not call.id or not call.name for call in pending.tool_calls) or len(set(ids)) != len(ids):
            error = "Invalid or duplicate tool call identity in execution queue"
            logger.error("tool_batch_rejected", entity_id=int(entity_id), error=error)
            world.add_component(entity_id, ErrorComponent(
                error=error, system_name="ToolExecutionSystem", timestamp=time.time(),
            ))
            world.add_component(entity_id, TerminalComponent(reason="tool_execution_error", error=error))
            world.remove_component(entity_id, PendingToolCallsComponent)
            return

        results: dict[str, str] = {}
        plan = world.get_component(entity_id, PlanComponent)
        plan_name: str | None = None
        if self._registry is not None and plan is not None:
            plan_name = derive_plan_name(
                plan=plan,
                conversation=conversation,
                entity_id=entity_id,
            )

        config = world.get_component(entity_id, ToolExecutionConfigComponent)
        if config is None:
            config = ToolExecutionConfigComponent()
        max_concurrency = max(1, config.max_concurrency)

        # Execute group by group: within a group calls run concurrently,
        # groups are barriers.  Results land strictly in the original
        # tool_calls order so conversation appends stay deterministic.
        for group in _plan_execution_groups(pending.tool_calls, registry.tools):
            group_results = await self._execute_group(
                entity_id,
                world,
                group,
                registry.handlers,
                max_concurrency,
                batch,
            )
            for source_call, outcome in zip(group, group_results, strict=True):
                tool_call = outcome.call
                schema = registry.tools.get(tool_call.name)
                await self._land_result(
                    world=world,
                    entity_id=entity_id,
                    conversation=conversation,
                    tool_call=tool_call,
                    source_call=source_call,
                    outcome=outcome,
                    batch=batch,
                    results=results,
                    plan_name=plan_name,
                    inline_result=bool(schema and schema.inline_result),
                )

        batch.completed = True
        if world.get_component(entity_id, PendingToolCallsComponent) is pending:
            world.remove_component(entity_id, PendingToolCallsComponent)
        if results:
            world.add_component(entity_id, ToolResultsComponent(results=results))

    async def _execute_group(
        self,
        entity_id: EntityId,
        world: World,
        group: list[ToolCall],
        handlers: dict[str, Callable[..., Awaitable[str]]],
        max_concurrency: int,
        batch: AuthorizationBatch,
    ) -> list[_ToolOutcome]:
        """Execute one group, concurrently when it holds multiple safe calls."""
        if len(group) == 1 or max_concurrency == 1:
            return [
                await self._execute_with_events(entity_id, world, tool_call, handlers, batch)
                for tool_call in group
            ]

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded(tool_call: ToolCall) -> _ToolOutcome:
            async with semaphore:
                return await self._execute_with_events(
                    entity_id, world, tool_call, handlers, batch
                )

        async with asyncio.TaskGroup() as task_group:
            tasks = [task_group.create_task(bounded(tool_call)) for tool_call in group]
        return [task.result() for task in tasks]

    async def _execute_with_events(
        self,
        entity_id: EntityId,
        world: World,
        tool_call: ToolCall,
        handlers: dict[str, Callable[..., Awaitable[str]]],
        batch: AuthorizationBatch,
    ) -> _ToolOutcome:
        state = batch.call_state(tool_call)
        if state.consumed:
            raise RuntimeError("Tool call already consumed; refusing to repeat execution")
        registry = world.get_component(entity_id, ToolRegistryComponent)
        invalid = tool_call_error(
            [tool_call], list(registry.tools.values()) if registry else None,
        )
        if tool_call.name not in handlers:
            invalid = f"unknown tool '{tool_call.name}'"
        if invalid is not None:
            logger.error("tool_failed", tool_name=tool_call.name, reason=invalid)
            state.consumed = True
            return _ToolOutcome(deepcopy(tool_call), f"Error: {invalid}")

        decision = await authorize_tool_call(world, entity_id, batch.pending, tool_call)
        snapshot = deepcopy(tool_call)
        reason = decision.reason
        if reason is None and decision.fingerprint != authorization_fingerprint(world, entity_id, tool_call):
            reason = "Tool authorization changed before execution"
        # Approval handlers may have changed arguments while awaiting a decision.
        invalid = tool_call_error([snapshot], list(registry.tools.values()) if registry else None)
        state.consumed = True
        if reason is not None:
            return _ToolOutcome(snapshot, f"Error: {reason}", denial_reason=reason)
        if invalid is not None:
            return _ToolOutcome(snapshot, f"Error: {invalid}")

        # Authorization commits at execution start. Freeze payload/handler before
        # any await; subsequent policy changes apply to calls not yet started.
        selected_handlers = dict(handlers)
        start_monotonic = time.monotonic()
        tool_start_time = datetime.now(timezone.utc)
        start_id = tool_record_id(world, entity_id, snapshot.id, "started")
        if start_id is not None and world.session_service is not None:
            try:
                world.session_service.store.get_record(start_id)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("Tool already started in this session; execution must not be repeated, completion may be unconfirmed")
        record_session_event(world, entity_id, "tool_started", {"tool_call_id": snapshot.id, "tool_name": snapshot.name, "arguments": snapshot.arguments, "status": "started"}, record_id=start_id)
        turn = world.get_component(entity_id, AgentTurnComponent)
        if turn is not None:
            if snapshot.id in turn.unconfirmed_tool_calls:
                raise RuntimeError("Tool completion is unconfirmed; refusing to repeat execution")
            turn.unconfirmed_tool_calls.append(snapshot.id)
        start_publication = asyncio.create_task(world.event_bus.publish(ToolExecutionStartedEvent(
            entity_id=entity_id, tool_call=deepcopy(snapshot), start_time=tool_start_time,
        )))
        try:
            # Enter the handler before yielding to observation subscribers. A
            # subscriber cannot alter an approved call between the gate and start.
            result = await self._execute_tool_call(entity_id, world, snapshot, selected_handlers)
            duration = time.monotonic() - start_monotonic
            await start_publication
            return _ToolOutcome(snapshot, result, tool_start_time, duration)
        finally:
            if not start_publication.done():
                start_publication.cancel()
                await asyncio.gather(start_publication, return_exceptions=True)

    async def _land_result(
        self,
        *,
        world: World,
        entity_id: EntityId,
        conversation: ConversationComponent,
        tool_call: ToolCall,
        source_call: ToolCall,
        outcome: _ToolOutcome,
        batch: AuthorizationBatch,
        results: dict[str, str],
        plan_name: str | None,
        inline_result: bool = False,
    ) -> None:
        """Record one result: persistence, conversation append, plan updates.

        ``inline_result`` tools (e.g. ``ask_question``) keep their result
        verbatim in the conversation even when the sink is externalizing other
        results — the model must always see them, so they are never replaced by
        an artifact path it might not fetch.
        """
        result = outcome.result
        record_conversation_message(
            world, entity_id, Message(role="tool", content=result, tool_call_id=tool_call.id),
            record_id=tool_record_id(world, entity_id, tool_call.id, "result"),
        )
        success = not result.startswith("Error")
        persisted_record_path: str | None = None
        if self.tool_sink is not None and not inline_result:
            persist_result = self.tool_sink.persist_tool_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                arguments=tool_call.arguments,
            )
            persisted_record_path = persist_result.record_path
            results[tool_call.id] = persist_result.record_path
            append_conversation_message(world, entity_id,
                Message(
                    role="tool",
                    content=persist_result.record_path if success else result,
                    tool_call_id=tool_call.id,
                ),
                recorded_record_id=tool_record_id(world, entity_id, tool_call.id, "result"),
            )
            self._confirm_managed_result(world, entity_id, tool_call.id, source_call, batch.pending)
            if success:
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
            append_conversation_message(world, entity_id,
                Message(role="tool", content=result, tool_call_id=tool_call.id),
                recorded_record_id=tool_record_id(world, entity_id, tool_call.id, "result"),
            )
            self._confirm_managed_result(world, entity_id, tool_call.id, source_call, batch.pending)

        if outcome.denial_reason is not None:
            await publish_tool_denial(
                world, entity_id, batch.call_state(tool_call), tool_call, outcome.denial_reason,
            )
        else:
            await world.event_bus.publish(ToolExecutionCompletedEvent(
                entity_id=entity_id, tool_call_id=tool_call.id, result=result,
                success=success, tool_name=tool_call.name,
                duration_seconds=outcome.duration_seconds,
                start_time=outcome.started_at, end_time=datetime.now(timezone.utc),
                arguments=deepcopy(tool_call.arguments),
            ))

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

    @staticmethod
    def _confirm_managed_result(
        world: World, entity: EntityId, call_id: str,
        source_call: ToolCall, original_pending: PendingToolCallsComponent,
    ) -> None:
        turn = world.get_component(entity, AgentTurnComponent)
        if turn is None:
            return
        if call_id in turn.unconfirmed_tool_calls:
            turn.unconfirmed_tool_calls.remove(call_id)
        pending = world.get_component(entity, PendingToolCallsComponent)
        if pending is original_pending:
            # Remove exactly the admitted object, even if an observer changed
            # its id after start. Do not consume a replacement queue or call.
            pending.tool_calls = [call for call in pending.tool_calls if call is not source_call]
            if not pending.tool_calls:
                world.remove_component(entity, PendingToolCallsComponent)

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

        registry = world.get_component(entity_id, ToolRegistryComponent)
        invalid = tool_call_error(
            [tool_call], list(registry.tools.values()) if registry is not None else None,
        )
        if invalid is not None:
            logger.error("tool_failed", tool_name=tool_call.name, reason=invalid)
            return f"Error: {invalid}"

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
            updated_messages = list(conversation.messages)
            updated_messages[index] = Message(
                role="tool",
                content=cached_hint,
                parts=message.parts,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                compaction_metadata=message.compaction_metadata,
            )
            replace_conversation_history(world, entity_id, updated_messages, reason="tool_result_cached")
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
