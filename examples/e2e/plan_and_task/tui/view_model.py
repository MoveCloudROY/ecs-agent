"""Pure view state for the plan-and-task TUI.

``PlanTaskViewModel.apply_event`` folds ECS events into renderable state and
returns the list of screen sections that changed. This module deliberately
imports nothing from ``textual`` so the reducer is unit-testable without a
terminal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.logging import get_logger
from ecs_agent.types import (
    CompactionCompleteEvent,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    EntityId,
    ErrorOccurredEvent,
    PhaseChangedEvent,
    PromptReplacementEvent,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    UserInputReceivedEvent,
)

if TYPE_CHECKING:
    from examples.e2e.plan_and_task.state_models import RuntimeState

logger = get_logger(__name__)

TranscriptKind = Literal[
    "user",
    "assistant",
    "reasoning",
    "command",
    "tool_call",
    "tool_result",
    "subagent",
    "system",
    "error",
]

Section = Literal[
    "transcript",
    "live",
    "phases",
    "tasks",
    "usage",
    "subagents",
    "notify",
    "input",
]

_PREVIEW_CHARS = 160


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


@dataclass(slots=True)
class TranscriptEntry:
    """One append-only line/block in the conversation log."""

    kind: TranscriptKind
    text: str
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class UiChange:
    """A screen section invalidation, optionally carrying appended entries."""

    section: Section
    entries: list[TranscriptEntry] = field(default_factory=list)
    notification: str | None = None
    severity: Literal["information", "warning", "error"] = "information"


@dataclass(slots=True)
class SubagentRun:
    """Lifecycle of one delegated subagent, keyed by correlation id."""

    correlation_id: str
    name: str
    task_preview: str
    status: Literal["running", "completed", "failed"] = "running"
    stream_chars: int = 0


@dataclass(slots=True)
class TaskRow:
    """One row of the task-queue table."""

    task_id: str
    title: str
    status: str
    retry_count: int


@dataclass(slots=True)
class UsageTotals:
    """Cumulative LLM token/cost accounting for the session."""

    invocations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost: float = 0.0
    last_model: str = ""

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cache_read_tokens / self.prompt_tokens


class PlanTaskViewModel:
    """Folds ECS events into renderable TUI state."""

    def __init__(self, agent_id: EntityId, phase_ids: tuple[str, ...]) -> None:
        self.agent_id = agent_id
        self.phase_ids = phase_ids
        self.transcript: list[TranscriptEntry] = []
        self.live_reasoning: str = ""
        self.live_content: str = ""
        self.streaming: bool = False
        self.current_phase: str = phase_ids[0] if phase_ids else ""
        self.subagent_runs: list[SubagentRun] = []
        self.workflow_id: str | None = None
        self.workflow_status: str | None = None
        self.current_task_id: str | None = None
        self.tasks: list[TaskRow] = []
        self.review_verdicts: list[tuple[str, str]] = []
        self.usage = UsageTotals()

    # -- reducers ---------------------------------------------------------

    def apply_event(self, event: object) -> list[UiChange]:
        """Fold one ECS event into the view state; returns changed sections."""
        match event:
            case StreamStartEvent() if event.entity_id == self.agent_id:
                self.streaming = True
                self.live_reasoning = ""
                self.live_content = ""
                return [UiChange(section="live")]
            case StreamReasoningDeltaEvent() if event.entity_id == self.agent_id:
                self.live_reasoning += event.reasoning_delta
                return [UiChange(section="live")]
            case StreamReasoningEndEvent() if event.entity_id == self.agent_id:
                return self._flush_reasoning()
            case StreamContentDeltaEvent() if event.entity_id == self.agent_id:
                self.live_content += event.delta
                return [UiChange(section="live")]
            case StreamEndEvent() if event.entity_id == self.agent_id:
                self.streaming = False
                changes = self._flush_reasoning()
                if self.live_content:
                    changes += self._append(
                        TranscriptEntry(kind="assistant", text=self.live_content)
                    )
                    self.live_content = ""
                changes.append(UiChange(section="live"))
                return changes
            case UserInputReceivedEvent() if event.entity_id == self.agent_id:
                return self._append(TranscriptEntry(kind="user", text=event.text))
            case PromptReplacementEvent() if (
                event.entity_id == self.agent_id
                and event.prompt_kind == "user"
                and event.metadata.get("trigger_action") == "script"
            ):
                return self._append(
                    TranscriptEntry(
                        kind="command",
                        text=event.rendered_text,
                        meta={"command": event.source_text.split()[0]},
                    )
                )
            case ToolExecutionStartedEvent() if event.entity_id == self.agent_id:
                args = json.dumps(event.tool_call.arguments, ensure_ascii=False)
                return self._append(
                    TranscriptEntry(
                        kind="tool_call",
                        text=f"{event.tool_call.name} {_preview(args, 120)}",
                        meta={"tool": event.tool_call.name},
                    )
                )
            case ToolExecutionCompletedEvent() if event.entity_id == self.agent_id:
                duration = (
                    f"{event.duration_seconds:.1f}s"
                    if event.duration_seconds is not None
                    else "?"
                )
                marker = "✓" if event.success else "✗"
                return self._append(
                    TranscriptEntry(
                        kind="tool_result",
                        text=(
                            f"{marker} {event.tool_name or event.tool_call_id}"
                            f" ({duration}) {_preview(event.result)}"
                        ),
                        meta={
                            "tool": event.tool_name,
                            "success": "true" if event.success else "false",
                        },
                    )
                )
            case DelegationStartedEvent() if event.entity_id == self.agent_id:
                self.subagent_runs.append(
                    SubagentRun(
                        correlation_id=event.correlation_id,
                        name=event.subagent_name,
                        task_preview=_preview(event.task, 80),
                    )
                )
                changes = self._append(
                    TranscriptEntry(
                        kind="subagent",
                        text=f"◇ {event.subagent_name}: {_preview(event.task, 120)}",
                        meta={"subagent": event.subagent_name, "status": "running"},
                    )
                )
                changes.append(UiChange(section="subagents"))
                return changes
            case DelegationCompletedEvent() if event.entity_id == self.agent_id:
                status: Literal["completed", "failed"] = (
                    "completed" if event.success else "failed"
                )
                run = self._find_run(event.correlation_id, event.subagent_name)
                if run is not None:
                    run.status = status
                marker = "✓" if event.success else "✗"
                summary = event.result if event.success else (event.error or "failed")
                changes = self._append(
                    TranscriptEntry(
                        kind="subagent",
                        text=f"{marker} {event.subagent_name}: {_preview(summary)}",
                        meta={"subagent": event.subagent_name, "status": status},
                    )
                )
                changes.append(UiChange(section="subagents"))
                return changes
            case SubagentStreamDeltaEvent() if (
                event.parent_entity_id == self.agent_id
            ):
                run = self._latest_running(event.category)
                if run is not None:
                    run.stream_chars += len(event.delta) + len(
                        event.reasoning_delta or ""
                    )
                    return [UiChange(section="subagents")]
                return []
            case SubagentStreamEndEvent() if (
                event.parent_entity_id == self.agent_id
            ):
                return [UiChange(section="subagents")]
            case PhaseChangedEvent() if event.entity_id == self.agent_id:
                self.current_phase = event.to_phase
                changes = self._append(
                    TranscriptEntry(
                        kind="system",
                        text=(
                            f"phase {event.from_phase} → {event.to_phase}"
                            f" ({event.reason})"
                        ),
                    )
                )
                changes.append(UiChange(section="phases"))
                return changes
            case LLMInvocationEvent():
                usage = event.usage
                self.usage.invocations += 1
                prompt = usage.prompt_tokens or 0
                completion = usage.completion_tokens or 0
                self.usage.prompt_tokens += prompt
                self.usage.completion_tokens += completion
                self.usage.total_tokens += usage.total_tokens or (prompt + completion)
                self.usage.cache_read_tokens += (
                    usage.cache_read_tokens or usage.cached_input_tokens or 0
                )
                if event.cost is not None:
                    self.usage.total_cost += event.cost.total_cost or 0.0
                self.usage.last_model = event.model
                return [UiChange(section="usage")]
            case CompactionCompleteEvent() if event.entity_id == self.agent_id:
                message = (
                    f"compaction: {event.original_tokens} → "
                    f"{event.compacted_tokens} tokens"
                )
                changes = self._append(
                    TranscriptEntry(kind="system", text=message)
                )
                changes.append(
                    UiChange(section="notify", notification=message)
                )
                return changes
            case ErrorOccurredEvent() if event.entity_id == self.agent_id:
                message = f"{event.system_name}: {event.error}"
                changes = self._append(TranscriptEntry(kind="error", text=message))
                changes.append(
                    UiChange(
                        section="notify", notification=message, severity="error"
                    )
                )
                return changes
            case _:
                return []

    def refresh_runtime(self, state: RuntimeState) -> UiChange:
        """Re-read workflow/task-queue facts from the persisted runtime state."""
        self.workflow_id = state.workflow_id
        self.workflow_status = state.status
        self.current_task_id = state.current_task_id
        self.review_verdicts = [
            (verdict.phase, verdict.verdict) for verdict in state.review_verdicts
        ]
        self.tasks = [
            TaskRow(
                task_id=task.task_id,
                title=task.title,
                status=task.status,
                retry_count=task.retry_count,
            )
            for task in state.tasks
        ]
        return UiChange(section="tasks")

    # -- helpers ----------------------------------------------------------

    def _append(self, entry: TranscriptEntry) -> list[UiChange]:
        self.transcript.append(entry)
        return [UiChange(section="transcript", entries=[entry])]

    def _flush_reasoning(self) -> list[UiChange]:
        if not self.live_reasoning:
            return []
        entry = TranscriptEntry(kind="reasoning", text=self.live_reasoning)
        self.live_reasoning = ""
        changes = self._append(entry)
        changes.append(UiChange(section="live"))
        return changes

    def _find_run(self, correlation_id: str, name: str) -> SubagentRun | None:
        for run in reversed(self.subagent_runs):
            if correlation_id and run.correlation_id == correlation_id:
                return run
        for run in reversed(self.subagent_runs):
            if run.name == name and run.status == "running":
                return run
        return None

    def _latest_running(self, category: str) -> SubagentRun | None:
        for run in reversed(self.subagent_runs):
            if run.name == category and run.status == "running":
                return run
        return None
