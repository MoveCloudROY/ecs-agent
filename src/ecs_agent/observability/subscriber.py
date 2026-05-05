"""EventBus subscriber that maps runner lifecycle events to telemetry records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil
import uuid
from typing import Any

from ecs_agent.accounting.models import LLMRetryEvent
from ecs_agent.components import ConversationComponent, ContextBudgetConfig
from ecs_agent.observability.context import current_run_id, current_trace_id
from ecs_agent.observability.events import (
    LLMObservationCompletedEvent,
    LLMObservationStartedEvent,
    UserInputReceivedEvent,
)
from ecs_agent.observability.schema import (
    TelemetryRecord,
    TelemetryRecordKind,
    TelemetryScore,
    TelemetryStatus,
)
from ecs_agent.observability.sinks import TelemetrySink
from ecs_agent.types import (
    BranchCreatedEvent,
    CompactionCompleteEvent,
    ContextPrunedEvent,
    ErrorOccurredEvent,
    MessageDeliveredEvent,
    PlanRevisedEvent,
    PlanStepCompletedEvent,
    RAGRetrievalCompletedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
    PromptReplacementEvent,
    StreamContentDeltaEvent,
    StreamContentStartEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
    SystemExecutionCompletedEvent,
    SystemExecutionStartedEvent,
    ToolCall,
    ToolDeniedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolResultCachedEvent,
    WorkflowStateEvaluatedEvent,
)


_SUPPRESSED_SYSTEM_SPANS = {
    "ecs_agent.systems.reasoning.ReasoningSystem",
    "ecs_agent.systems.subagent.SubagentSystem",
    "ecs_agent.systems.user_input.UserInputSystem",
    "ecs_agent.systems.tool_execution.ToolExecutionSystem",
    "ecs_agent.systems.terminal_cleanup.TerminalCleanupSystem",
    "ecs_agent.systems.compaction.CompactionSystem",
    "ecs_agent.systems.workflow_state.WorkflowStateSystem",
    "ecs_agent.systems.system_prompt_render_system.SystemPromptRenderSystem",
    "ecs_agent.systems.user_prompt_normalization_system.UserPromptNormalizationSystem",
}


@dataclass(slots=True)
class TraceState:
    """Per-run trace state tracked by run ID."""

    trace_id: str
    run_id: str
    observation_id: str
    root_record: TelemetryRecord
    started_at: datetime
    max_ticks: int | None
    start_tick: int
    active_entities_start: int
    tick_count: int = 0
    error_count: int = 0
    closed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    active_tools: dict[tuple[int, str], ToolCall] = field(default_factory=dict)
    active_generations: dict[int, str] = field(default_factory=dict)
    stream_sequences: dict[int, int] = field(default_factory=dict)


class ObservabilitySubscriber:
    """Adapter-neutral telemetry subscriber for ECS event lifecycle mapping."""

    def __init__(self, sink: TelemetrySink, world: Any | None = None) -> None:
        self.sink = sink
        self.world = world
        self.trace_states: dict[str, TraceState] = {}

    def subscriptions(self) -> tuple[tuple[type[Any], Any], ...]:
        """Return the EventBus subscriptions for this subscriber."""
        return (
            (RunStartedEvent, self.handle_run_started),
            (RunnerTickStartedEvent, self.handle_runner_tick_started),
            (RunnerTickCompletedEvent, self.handle_runner_tick_completed),
            (RunCompletedEvent, self.handle_run_completed),
            (SystemExecutionStartedEvent, self.handle_system_execution_started),
            (SystemExecutionCompletedEvent, self.handle_system_execution_completed),
            (PromptReplacementEvent, self.handle_prompt_replacement),
            (WorkflowStateEvaluatedEvent, self.handle_workflow_state_evaluated),
            (ErrorOccurredEvent, self.handle_error_occurred),
            (PlanStepCompletedEvent, self.handle_plan_step_completed),
            (PlanRevisedEvent, self.handle_plan_revised),
            (ContextPrunedEvent, self.handle_context_pruned),
            (CompactionCompleteEvent, self.handle_compaction_complete),
            (RAGRetrievalCompletedEvent, self.handle_rag_retrieval_completed),
            (MessageDeliveredEvent, self.handle_message_delivered),
            (BranchCreatedEvent, self.handle_branch_created),
            (SubagentStreamStartEvent, self.handle_subagent_stream_start),
            (SubagentStreamDeltaEvent, self.handle_subagent_stream_delta),
            (SubagentStreamEndEvent, self.handle_subagent_stream_end),
            (LLMObservationStartedEvent, self.handle_llm_observation_started),
            (LLMObservationCompletedEvent, self.handle_llm_observation_completed),
            (StreamStartEvent, self.handle_stream_start),
            (StreamReasoningDeltaEvent, self.handle_stream_reasoning_delta),
            (StreamReasoningEndEvent, self.handle_stream_reasoning_end),
            (StreamContentStartEvent, self.handle_stream_content_start),
            (StreamContentDeltaEvent, self.handle_stream_content_delta),
            (StreamEndEvent, self.handle_stream_end),
            (LLMRetryEvent, self.handle_llm_retry),
            (UserInputReceivedEvent, self.handle_user_input_received),
            (ToolExecutionStartedEvent, self.handle_tool_execution_started),
            (ToolExecutionCompletedEvent, self.handle_tool_execution_completed),
            (ToolDeniedEvent, self.handle_tool_denied),
            (ToolResultCachedEvent, self.handle_tool_result_cached),
        )

    async def handle_llm_observation_started(
        self, event: LLMObservationStartedEvent
    ) -> None:
        """Track the active generation observation ID for child events."""
        state = self._state_for_current_run()
        if state is None:
            return

        state.active_generations[int(event.entity_id)] = uuid.uuid4().hex

    async def handle_llm_observation_completed(
        self, event: LLMObservationCompletedEvent
    ) -> None:
        """Map completed raw LLM observations to generation records."""
        state = self._state_for_current_run()
        if state is None:
            return

        entity_id = int(event.entity_id)
        observation_id = state.active_generations.pop(entity_id, uuid.uuid4().hex)

        await self.sink.emit(
            TelemetryRecord(
                trace_id=state.trace_id,
                run_id=state.run_id,
                observation_id=observation_id,
                parent_observation_id=state.observation_id,
                name=f"llm.{event.operation}",
                kind="generation",
                status=self._status(event.status),
                entity_id=entity_id,
                start_time=event.start_time,
                end_time=event.end_time,
                latency_ms=_record_latency_ms(
                    event.duration_seconds,
                    event.start_time,
                    event.end_time,
                ),
                input={
                    "messages": event.messages,
                    "tools": event.tools,
                    "streaming": event.streaming,
                },
                output=(
                    {
                        "message": event.response_message,
                        "reasoning_content": event.reasoning_content,
                        "response_id": event.response_id,
                    }
                    if event.response_message is not None
                    else None
                ),
                metadata={
                    "provider_id": event.provider_id,
                    "model_id": event.model,
                    "operation": event.operation,
                    "response_id": event.response_id,
                    "streaming": event.streaming,
                    **self._context_pressure_metadata(event),
                },
                error=event.error,
                model=_provider_model_label(event.provider_id, event.model),
                model_parameters=event.model_parameters,
                usage_details=self._usage_details(event),
                cost_details=dict(event.cost_details),
            )
        )

    async def handle_run_started(self, event: RunStartedEvent) -> None:
        """Create one trace root for the active run without emitting it yet."""
        run_id = current_run_id()
        trace_id = current_trace_id()
        if run_id is None or trace_id is None or run_id in self.trace_states:
            return

        started_at = datetime.now(timezone.utc)
        observation_id = uuid.uuid4().hex
        root_record = TelemetryRecord(
            trace_id=trace_id,
            run_id=run_id,
            observation_id=observation_id,
            name="runner.run",
            kind="trace",
            status="running",
            start_time=started_at,
            metadata={
                "max_ticks": event.max_ticks,
                "start_tick": event.start_tick,
                "active_entities_start": event.active_entities,
            },
        )
        self.trace_states[run_id] = TraceState(
            trace_id=trace_id,
            run_id=run_id,
            observation_id=observation_id,
            root_record=root_record,
            started_at=started_at,
            max_ticks=event.max_ticks,
            start_tick=event.start_tick,
            active_entities_start=event.active_entities,
            metadata={
                "max_ticks": event.max_ticks,
                "start_tick": event.start_tick,
                "active_entities_start": event.active_entities,
            },
        )
        await self._emit_initial_user_messages(self.trace_states[run_id])

    async def handle_user_input_received(self, event: UserInputReceivedEvent) -> None:
        """Map resolved user input text to a user-input event record."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="user.input",
                kind="event",
                entity_id=int(event.entity_id),
                input={"text": event.text},
                metadata={"prompt": event.prompt, "source": "user_input_system"},
            )
        )

    async def handle_tool_execution_started(
        self, event: ToolExecutionStartedEvent
    ) -> None:
        """Remember raw tool-call arguments for the completion observation."""
        state = self._state_for_current_run()
        if state is None:
            return
        state.active_tools[(int(event.entity_id), event.tool_call.id)] = event.tool_call

    async def handle_tool_execution_completed(
        self, event: ToolExecutionCompletedEvent
    ) -> None:
        """Map completed tool execution to a tool observation."""
        state = self._state_for_current_run()
        if state is None:
            return

        tool_call = state.active_tools.pop((int(event.entity_id), event.tool_call_id), None)
        tool_name = event.tool_name or (tool_call.name if tool_call is not None else "")
        arguments = tool_call.arguments if tool_call is not None else {}
        result = event.result
        await self.sink.emit(
            TelemetryRecord(
                trace_id=state.trace_id,
                run_id=state.run_id,
                observation_id=uuid.uuid4().hex,
                parent_observation_id=state.observation_id,
                name=f"tool.{tool_name}" if tool_name else "tool.unknown",
                kind="tool",
                status="success" if event.success else "error",
                entity_id=int(event.entity_id),
                latency_ms=(
                    event.duration_seconds * 1000
                    if event.duration_seconds is not None
                    else None
                ),
                input={
                    "tool_call_id": event.tool_call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                output={"result": result},
                metadata={"tool_call_id": event.tool_call_id, "tool_name": tool_name},
                error=None if event.success else result,
            )
        )

    async def handle_tool_denied(self, event: ToolDeniedEvent) -> None:
        """Map permission denials to error tool observations."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            TelemetryRecord(
                trace_id=state.trace_id,
                run_id=state.run_id,
                observation_id=uuid.uuid4().hex,
                parent_observation_id=state.observation_id,
                name=f"tool.{event.tool_name}" if event.tool_name else "tool.denied",
                kind="tool",
                status="error",
                entity_id=int(event.entity_id),
                input={
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                },
                metadata={
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "reason": event.reason,
                },
                error=event.reason,
            )
        )

    async def handle_tool_result_cached(self, event: ToolResultCachedEvent) -> None:
        """Map cached tool result references to tool cache observations."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            TelemetryRecord(
                trace_id=state.trace_id,
                run_id=state.run_id,
                observation_id=uuid.uuid4().hex,
                parent_observation_id=state.observation_id,
                name="tool.cache",
                kind="tool",
                status="success",
                entity_id=int(event.entity_id),
                input={"tool_call_id": event.tool_call_id},
                output={"artifact_path": event.artifact_path},
                metadata={
                    "tool_call_id": event.tool_call_id,
                    "artifact_path": event.artifact_path,
                    "cache_status": event.status,
                },
            )
        )

    async def handle_runner_tick_started(self, event: RunnerTickStartedEvent) -> None:
        """Suppress empty tick-start spans while preserving event consumption."""
        _ = event

    async def handle_runner_tick_completed(self, event: RunnerTickCompletedEvent) -> None:
        """Update run counters without emitting empty runner.tick spans."""
        state = self._state_for_current_run()
        if state is None:
            return

        state.tick_count += 1

    async def handle_run_completed(self, event: RunCompletedEvent) -> None:
        """Close the active trace, emit summary scores, and clear state."""
        run_id = current_run_id()
        if run_id is None:
            return

        state = self.trace_states.get(run_id)
        if state is None or state.closed:
            return

        ended_at = datetime.now(timezone.utc)
        root_record = state.root_record
        root_record.end_time = ended_at
        root_record.latency_ms = (ended_at - state.started_at).total_seconds() * 1000
        root_record.status = self._status(event.status)
        root_record.metadata = {
            **state.metadata,
            "active_entities_end": event.active_entities,
            "reason": event.reason,
            "ticks": event.ticks,
        }

        try:
            await self.sink.emit(root_record)
            await self.sink.score(
                TelemetryScore(
                    trace_id=state.trace_id,
                    run_id=state.run_id,
                    observation_id=state.observation_id,
                    name="agent_tick_count",
                    value=state.tick_count,
                )
            )
            await self.sink.score(
                TelemetryScore(
                    trace_id=state.trace_id,
                    run_id=state.run_id,
                    observation_id=state.observation_id,
                    name="agent_latency_ms",
                    value=root_record.latency_ms or 0.0,
                )
            )
            await self.sink.score(
                TelemetryScore(
                    trace_id=state.trace_id,
                    run_id=state.run_id,
                    observation_id=state.observation_id,
                    name="agent_error_count",
                    value=state.error_count,
                )
            )
            await self.sink.score(
                TelemetryScore(
                    trace_id=state.trace_id,
                    run_id=state.run_id,
                    observation_id=state.observation_id,
                    name="estimated_context_pressure",
                    value=self._estimated_context_pressure_score(),
                )
            )
            await self.sink.score(
                TelemetryScore(
                    trace_id=state.trace_id,
                    run_id=state.run_id,
                    observation_id=state.observation_id,
                    name="max_ticks_reached",
                    value=event.reason == "max_ticks",
                )
            )
        finally:
            state.closed = True
            self.trace_states.pop(run_id, None)

    async def handle_stream_start(self, event: StreamStartEvent) -> None:
        """Map stream start to a child event under the active generation."""
        state = self._state_for_current_run()
        if state is None:
            return

        entity_id = int(event.entity_id)
        state.stream_sequences[entity_id] = 0
        await self.sink.emit(
            self._stream_event_record(
                state,
                entity_id=entity_id,
                name="stream.start",
                status="running",
                metadata={
                    "provider_id": event.provider_id,
                    "model": event.model,
                    "operation": event.operation,
                    "timestamp": event.timestamp,
                },
            )
        )

    async def handle_stream_reasoning_delta(
        self, event: StreamReasoningDeltaEvent
    ) -> None:
        """Map streamed reasoning deltas without duplicating final output."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._stream_event_record(
                state,
                entity_id=int(event.entity_id),
                name="stream.reasoning.delta",
                output={"reasoning_delta": event.reasoning_delta},
                metadata={
                    "provider_id": event.provider_id,
                    "model": event.model,
                    "operation": event.operation,
                    "first_delta_seconds": event.first_delta_seconds,
                },
            )
        )

    async def handle_stream_reasoning_end(
        self, event: StreamReasoningEndEvent
    ) -> None:
        """Map the end of the reasoning stream phase."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._stream_event_record(
                state,
                entity_id=int(event.entity_id),
                name="stream.reasoning.end",
            )
        )

    async def handle_stream_content_start(
        self, event: StreamContentStartEvent
    ) -> None:
        """Map the start of the assistant content stream phase."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._stream_event_record(
                state,
                entity_id=int(event.entity_id),
                name="stream.content.start",
            )
        )

    async def handle_stream_content_delta(
        self, event: StreamContentDeltaEvent
    ) -> None:
        """Map streamed content deltas as child events only."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._stream_event_record(
                state,
                entity_id=int(event.entity_id),
                name="stream.content.delta",
                output={"delta": event.delta},
                metadata={
                    "provider_id": event.provider_id,
                    "model": event.model,
                    "operation": event.operation,
                    "first_delta_seconds": event.first_delta_seconds,
                },
            )
        )

    async def handle_stream_end(self, event: StreamEndEvent) -> None:
        """Map stream end and clear stream sequence state."""
        state = self._state_for_current_run()
        if state is None:
            return

        entity_id = int(event.entity_id)
        try:
            await self.sink.emit(
                self._stream_event_record(
                    state,
                    entity_id=entity_id,
                    name="stream.end",
                    status=self._status(event.status),
                    latency_ms=(
                        event.duration_seconds * 1000
                        if event.duration_seconds is not None
                        else None
                    ),
                    metadata={
                        "provider_id": event.provider_id,
                        "model": event.model,
                        "operation": event.operation,
                        "timestamp": event.timestamp,
                        "duration_seconds": event.duration_seconds,
                        "first_delta_seconds": event.first_delta_seconds,
                    },
                )
            )
        finally:
            state.stream_sequences.pop(entity_id, None)

    async def handle_llm_retry(self, event: LLMRetryEvent) -> None:
        """Map retry attempts under the current generation when available."""
        state = self._state_for_current_run()
        if state is None:
            return

        parent_observation_id = state.observation_id
        entity_id: int | None = None
        if len(state.active_generations) == 1:
            entity_id, parent_observation_id = next(iter(state.active_generations.items()))

        await self.sink.emit(
            TelemetryRecord(
                trace_id=state.trace_id,
                run_id=state.run_id,
                observation_id=uuid.uuid4().hex,
                parent_observation_id=parent_observation_id,
                name="llm.retry",
                kind="event",
                status="running",
                entity_id=entity_id,
                metadata={
                    "provider_id": event.provider_id,
                    "model": event.model,
                    "reason": event.reason,
                    "attempt": event.attempt,
                    "status": "retrying",
                },
            )
        )

    async def handle_system_execution_started(self, event: SystemExecutionStartedEvent) -> None:
        """Suppress empty system-start spans."""
        _ = event

    async def handle_system_execution_completed(
        self, event: SystemExecutionCompletedEvent
    ) -> None:
        """Emit a system execution completion event."""
        state = self._state_for_current_run()
        if state is None:
            return
        if event.system in _SUPPRESSED_SYSTEM_SPANS:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name=event.system,
                kind="span",
                status=self._status(event.status),
                system_name=event.system,
                latency_ms=event.duration_seconds * 1000,
            )
        )

    async def handle_prompt_replacement(self, event: PromptReplacementEvent) -> None:
        """Emit prompt replacement telemetry only when prompt text changed."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name=f"prompt.{event.prompt_kind}.replacement",
                kind="event",
                entity_id=int(event.entity_id),
                input={"text": event.source_text},
                output={"text": event.rendered_text},
                metadata={
                    "prompt_kind": event.prompt_kind,
                    "replacements": dict(event.replacements),
                    **event.metadata,
                },
            )
        )

    async def handle_workflow_state_evaluated(
        self, event: WorkflowStateEvaluatedEvent
    ) -> None:
        """Emit workflow state and transition payloads."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="workflow.state",
                kind="event",
                status="error" if event.status == "ambiguous" else "success",
                entity_id=int(event.entity_id),
                tick=event.tick,
                input={
                    "workflow_id": event.workflow_id,
                    "state_id": event.state_id,
                    "matched_transition_ids": list(event.matched_transition_ids),
                },
                output={
                    "current_state_id": event.current_state_id,
                    "committed_transition_id": event.committed_transition_id,
                    "from_state_id": event.from_state_id,
                    "to_state_id": event.to_state_id,
                    "transition_history": list(event.transition_history),
                },
                metadata={"status": event.status},
                error=event.error,
            )
        )

    async def handle_error_occurred(self, event: ErrorOccurredEvent) -> None:
        """Emit an error event and increment the current run's error count."""
        state = self._state_for_current_run()
        if state is None:
            return

        state.error_count += 1
        await self.sink.emit(
            self._event_record(
                state,
                name="error.occurred",
                kind="event",
                status="error",
                entity_id=int(event.entity_id),
                system_name=event.system_name,
                error=event.error,
            )
        )

    async def handle_plan_step_completed(self, event: PlanStepCompletedEvent) -> None:
        """Emit a plan step completion event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="plan.step.completed",
                kind="event",
                status=self._status(event.status),
                entity_id=int(event.entity_id),
                metadata={
                    "step_index": event.step_index,
                    "step_description": event.step_description,
                    "operation": event.operation,
                },
            )
        )

    async def handle_plan_revised(self, event: PlanRevisedEvent) -> None:
        """Emit a plan revision event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="plan.revised",
                kind="event",
                status=self._status(event.status),
                entity_id=int(event.entity_id),
                metadata={"old_steps": event.old_steps, "new_steps": event.new_steps},
            )
        )

    async def handle_context_pruned(self, event: ContextPrunedEvent) -> None:
        """Emit a context pruning event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="context.pruned",
                kind="event",
                entity_id=int(event.entity_id),
                metadata={
                    "reason": event.reason,
                    "tool_call_id": event.tool_call_id,
                    "artifact_path": event.artifact_path,
                    "source_label": event.source_label,
                },
            )
        )

    async def handle_compaction_complete(self, event: CompactionCompleteEvent) -> None:
        """Emit a compaction completion event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="compaction.complete",
                kind="event",
                status=self._status(event.status),
                entity_id=int(event.entity_id),
                metadata={
                    "original_tokens": event.original_tokens,
                    "compacted_tokens": event.compacted_tokens,
                    "operation": event.operation,
                },
            )
        )

    async def handle_rag_retrieval_completed(
        self, event: RAGRetrievalCompletedEvent
    ) -> None:
        """Emit a RAG retrieval event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="rag.retrieval.completed",
                kind="event",
                entity_id=int(event.entity_id),
                metadata={"query": event.query, "num_results": event.num_results},
            )
        )

    async def handle_message_delivered(self, event: MessageDeliveredEvent) -> None:
        """Emit a message delivery event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="message.delivered",
                kind="event",
                metadata={
                    "from_entity": int(event.from_entity),
                    "to_entity": int(event.to_entity),
                    "message": event.message,
                },
            )
        )

    async def handle_branch_created(self, event: BranchCreatedEvent) -> None:
        """Emit a branch creation event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="conversation.branch.created",
                kind="event",
                entity_id=int(event.entity_id),
                metadata={
                    "branch_id": event.branch_id,
                    "parent_message_id": event.parent_message_id,
                },
            )
        )

    async def handle_subagent_stream_start(self, event: SubagentStreamStartEvent) -> None:
        """Emit a subagent stream start event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="subagent.stream.start",
                kind="event",
                metadata={
                    "session_id": event.session_id,
                    "parent_entity_id": int(event.parent_entity_id),
                    "category": event.category,
                    "child_world_name": event.child_world_name,
                    "seq": event.seq,
                    "timestamp": event.timestamp,
                },
            )
        )

    async def handle_subagent_stream_delta(self, event: SubagentStreamDeltaEvent) -> None:
        """Emit a subagent stream delta event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="subagent.stream.delta",
                kind="event",
                metadata={
                    "session_id": event.session_id,
                    "parent_entity_id": int(event.parent_entity_id),
                    "category": event.category,
                    "child_world_name": event.child_world_name,
                    "seq": event.seq,
                    "timestamp": event.timestamp,
                    "delta": event.delta,
                    "reasoning_delta": event.reasoning_delta,
                },
            )
        )

    async def handle_subagent_stream_end(self, event: SubagentStreamEndEvent) -> None:
        """Emit a subagent stream end event."""
        state = self._state_for_current_run()
        if state is None:
            return

        await self.sink.emit(
            self._event_record(
                state,
                name="subagent.stream.end",
                kind="event",
                metadata={
                    "session_id": event.session_id,
                    "parent_entity_id": int(event.parent_entity_id),
                    "category": event.category,
                    "child_world_name": event.child_world_name,
                    "seq": event.seq,
                    "timestamp": event.timestamp,
                    "total_tokens": event.total_tokens,
                },
            )
        )

    async def _emit_initial_user_messages(self, state: TraceState) -> None:
        """Emit user-input records for existing conversation user messages."""
        if self.world is None:
            return

        for entity_id, components in self.world.query(ConversationComponent):
            conversation = components[0]
            assert isinstance(conversation, ConversationComponent)
            for message in conversation.messages:
                if message.role != "user":
                    continue
                await self.sink.emit(
                    self._event_record(
                        state,
                        name="user.input",
                        kind="event",
                        entity_id=int(entity_id),
                        input={"message": message},
                        metadata={"source": "initial_conversation"},
                    )
                )

    def _state_for_current_run(self) -> TraceState | None:
        run_id = current_run_id()
        if run_id is None:
            return None
        return self.trace_states.get(run_id)

    def _event_record(
        self,
        state: TraceState,
        *,
        name: str,
        kind: TelemetryRecordKind,
        status: TelemetryStatus = "success",
        entity_id: int | None = None,
        tick: int | None = None,
        system_name: str | None = None,
        latency_ms: float | None = None,
        input: Any = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        error: str | dict[str, Any] | None = None,
    ) -> TelemetryRecord:
        return TelemetryRecord(
            trace_id=state.trace_id,
            run_id=state.run_id,
            observation_id=uuid.uuid4().hex,
            parent_observation_id=state.observation_id,
            name=name,
            kind=kind,
            status=status,
            entity_id=entity_id,
            tick=tick,
            system_name=system_name,
            latency_ms=latency_ms,
            input=input,
            output=output,
            metadata=metadata,
            error=error,
        )

    def _stream_event_record(
        self,
        state: TraceState,
        *,
        entity_id: int,
        name: str,
        status: TelemetryStatus = "success",
        latency_ms: float | None = None,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> TelemetryRecord:
        seq = state.stream_sequences.get(entity_id, 0)
        state.stream_sequences[entity_id] = seq + 1
        parent_observation_id = state.active_generations.get(
            entity_id,
            state.observation_id,
        )
        event_metadata = {"seq": seq}
        if metadata is not None:
            event_metadata.update(metadata)
        return TelemetryRecord(
            trace_id=state.trace_id,
            run_id=state.run_id,
            observation_id=uuid.uuid4().hex,
            parent_observation_id=parent_observation_id,
            name=name,
            kind="event",
            status=status,
            entity_id=entity_id,
            latency_ms=latency_ms,
            output=output,
            metadata=event_metadata,
        )

    def _status(self, status: str) -> TelemetryStatus:
        if status == "error":
            return "error"
        if status == "cancelled":
            return "cancelled"
        if status == "running":
            return "running"
        if status in {"success", "terminal_component", "max_ticks", "interruption_component"}:
            return "success"
        return "unknown"

    def _usage_details(self, event: LLMObservationCompletedEvent) -> dict[str, Any]:
        usage = event.usage
        if usage is None:
            return {}

        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_creation_tokens": usage.cache_creation_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "image_count": usage.image_count,
            "audio_seconds": usage.audio_seconds,
            "provider_id": usage.provider_id,
            "model": usage.model,
            "stream_completeness": usage.stream_completeness.value,
        }

    def _context_pressure_metadata(
        self, event: LLMObservationCompletedEvent
    ) -> dict[str, Any]:
        prompt_char_count = sum(len(message.content or "") for message in event.messages)
        metadata: dict[str, Any] = {
            "message_count": len(event.messages),
            "prompt_char_count": prompt_char_count,
            "estimated_prompt_tokens": ceil(prompt_char_count / 4),
            "provider_prompt_tokens": (
                event.usage.prompt_tokens if event.usage is not None else None
            ),
        }
        if self.world is not None:
            budget = self.world.get_component(int(event.entity_id), ContextBudgetConfig)
            if budget is not None:
                metadata["context_budget"] = self._context_budget_payload(budget)
        return metadata

    def _context_budget_payload(self, budget: ContextBudgetConfig) -> dict[str, Any]:
        return {
            "max_tokens": budget.max_tokens,
            "prune_tool_results": budget.prune_tool_results,
            "prune_reasoning": budget.prune_reasoning,
            "token_estimation_chars_per_token": budget.token_estimation_chars_per_token,
            "overflow_behavior": budget.overflow_behavior,
        }

    def _estimated_context_pressure_score(self) -> float:
        if self.world is None:
            return 0.0

        max_pressure = 0.0
        for entity_id, components in self.world.query(ConversationComponent):
            conversation = components[0]
            assert isinstance(conversation, ConversationComponent)
            budget = self.world.get_component(entity_id, ContextBudgetConfig)
            if budget is None or budget.max_tokens <= 0:
                continue
            prompt_char_count = sum(
                len(message.content or "") for message in conversation.messages
            )
            estimated_tokens = ceil(
                prompt_char_count / budget.token_estimation_chars_per_token
            )
            max_pressure = max(max_pressure, estimated_tokens / budget.max_tokens)
        return max_pressure


def _provider_model_label(provider_id: str, model: str) -> str:
    if "/" in model:
        return model
    if provider_id:
        return f"{provider_id}/{model}"
    return model


def _record_latency_ms(
    duration_seconds: float | None,
    start_time: datetime | None,
    end_time: datetime | None,
) -> float | None:
    if start_time is not None and end_time is not None:
        return (end_time - start_time).total_seconds() * 1000
    if duration_seconds is not None:
        return duration_seconds * 1000
    return None


__all__ = ["ObservabilitySubscriber", "TraceState"]
