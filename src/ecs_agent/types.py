"""Core type definitions for ECS-based LLM Agent."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Literal, NewType

from ecs_agent.accounting.models import LLMInvocationEvent, UsageRecord

EntityId = NewType("EntityId", int)
SystemHandle = NewType("SystemHandle", str)


@dataclass(slots=True)
class ToolCall:
    """Represents a call to a tool/function."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class TextPart:
    """A plain text message part."""

    text: str


@dataclass(slots=True)
class ImageUrlPart:
    """An image URL message part."""

    url: str
    detail: str | None = None


@dataclass(slots=True)
class FileRefPart:
    """A file reference message part."""

    file_id: str
    filename: str | None = None


MessagePart = TextPart | ImageUrlPart | FileRefPart


@dataclass(slots=True)
class Message:
    """Represents a message in the conversation."""

    role: str
    content: str
    parts: list[MessagePart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class ConversationMessage:
    """A message node in a conversation tree."""

    id: str
    parent_message_id: str | None
    role: str  # 'system' | 'user' | 'assistant' | 'tool'
    content: str | None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    created_at: str = ""  # ISO timestamp
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationBranch:
    """A named branch pointing to a leaf message."""

    branch_id: str
    leaf_message_id: str


@dataclass(slots=True)
class ToolSchema:
    """Describes the schema of a tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    sandbox_compatible: bool = False


Usage = UsageRecord


@dataclass(slots=True)
class CompletionResult:
    """Result from LLM completion."""

    message: Message
    usage: Usage | None = None
    response_id: str | None = None


@dataclass(slots=True)
class ConversationTruncatedEvent:
    entity_id: EntityId
    removed_count: int


@dataclass(slots=True)
class ErrorOccurredEvent:
    entity_id: EntityId
    error: str
    system_name: str


@dataclass(slots=True)
class MessageDeliveredEvent:
    from_entity: EntityId
    to_entity: EntityId
    message: Message


@dataclass(slots=True)
class PlanStepCompletedEvent:
    entity_id: EntityId
    step_index: int
    step_description: str


@dataclass(slots=True)
class PlanRevisedEvent:
    """Event emitted when the plan is dynamically revised during execution."""

    entity_id: EntityId
    old_steps: list[str]
    new_steps: list[str]


@dataclass(slots=True)
class StreamDelta:
    """Represents a chunk of streamed response data."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    response_id: str | None = None


@dataclass(slots=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    multiplier: float = 1.0
    min_wait: float = 4.0
    max_wait: float = 60.0
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


class ToolTimeoutError(Exception):
    """Raised when a sandboxed tool exceeds its timeout."""

    pass


class ApprovalPolicy(Enum):
    """Policy for tool approval decisions."""

    ALWAYS_APPROVE = "always_approve"
    ALWAYS_DENY = "always_deny"
    REQUIRE_APPROVAL = "require_approval"


class InterruptionReason(Enum):
    """Reason for agent interruption."""

    USER_REQUESTED = "user_requested"
    SYSTEM_PAUSE = "system_pause"
    ERROR = "error"
    COMPLETION = "completion"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(slots=True)
class ScratchbookRef:
    """Reference to a scratchbook artifact."""

    artifact_id: str
    category: str
    content_hash: str
    timestamp: str


@dataclass(slots=True)
class ToolApprovalRequestedEvent:
    """Event emitted when a tool call requires approval."""

    entity_id: EntityId
    tool_call: ToolCall
    approval_future: asyncio.Future[bool]


@dataclass(slots=True)
class ToolApprovedEvent:
    """Event emitted when a tool call is approved."""

    entity_id: EntityId
    tool_call_id: str


@dataclass(slots=True)
class ToolDeniedEvent:
    """Event emitted when a tool call is denied."""

    entity_id: EntityId
    tool_call_id: str
    reason: str


@dataclass(slots=True)
class MCTSNodeScoredEvent:
    """Event emitted when an MCTS node is scored."""

    entity_id: EntityId
    node_id: int
    score: float


@dataclass(slots=True)
class StreamStartEvent:
    """Event emitted when streaming starts."""

    entity_id: EntityId
    timestamp: float


@dataclass(slots=True)
class StreamContentDeltaEvent:
    """Event emitted when a streaming content delta is received."""

    entity_id: EntityId
    delta: str


@dataclass(slots=True)
class StreamReasoningDeltaEvent:
    """Event emitted when a streaming reasoning delta is received."""

    entity_id: EntityId
    reasoning_delta: str


@dataclass(slots=True)
class StreamReasoningEndEvent:
    """Event emitted when reasoning stream phase ends."""

    entity_id: EntityId


@dataclass(slots=True)
class StreamContentStartEvent:
    """Event emitted when assistant content stream phase starts."""

    entity_id: EntityId


@dataclass(slots=True)
class StreamEndEvent:
    """Event emitted when streaming ends."""

    entity_id: EntityId
    timestamp: float


@dataclass(slots=True)
class CheckpointCreatedEvent:
    """Event emitted when a checkpoint is created."""

    entity_id: EntityId
    checkpoint_id: int
    timestamp: float


@dataclass(slots=True)
class CheckpointRestoredEvent:
    """Event emitted when a checkpoint is restored."""

    entity_id: EntityId
    checkpoint_id: int
    timestamp: float


@dataclass(slots=True)
class CompactionCompleteEvent:
    """Event emitted when context compaction completes."""

    entity_id: EntityId
    original_tokens: int
    compacted_tokens: int


@dataclass(slots=True)
class RAGRetrievalCompletedEvent:
    """Event emitted when RAG retrieval completes."""

    entity_id: EntityId
    query: str
    num_results: int


@dataclass(slots=True)
class UserInputRequestedEvent:
    """Event emitted when a system needs user input.

    External code should subscribe to this event, present the prompt to
    the user, and resolve ``input_future`` with the user's text.
    """

    entity_id: EntityId
    prompt: str
    input_future: asyncio.Future[str]


@dataclass(slots=True)
class ToolExecutionStartedEvent:
    """Event emitted when tool execution starts."""

    entity_id: EntityId
    tool_call: ToolCall


@dataclass(slots=True)
class ToolExecutionCompletedEvent:
    """Event emitted when tool execution completes."""

    entity_id: EntityId
    tool_call_id: str
    result: str
    success: bool


@dataclass(slots=True)
class SkillInstalledEvent:
    """Event emitted when a skill is installed."""

    entity_id: EntityId
    skill_name: str
    tool_names: list[str]


@dataclass(slots=True)
class SkillUninstalledEvent:
    """Event emitted when a skill is uninstalled."""

    entity_id: EntityId
    skill_name: str


@dataclass(slots=True)
class SkillDiscoveryEvent:
    """Event emitted when skill discovery completes."""

    source: str
    skills_found: list[str]
    errors: list[str]


@dataclass(slots=True)
class MCPConnectedEvent:
    """Event emitted when MCP server connects."""

    server_name: str


@dataclass(slots=True)
class MCPDisconnectedEvent:
    """Event emitted when MCP server disconnects."""

    server_name: str


@dataclass(slots=True)
class MCPToolCallEvent:
    """Event emitted when an MCP tool is called."""

    server_name: str
    tool_name: str
    success: bool


@dataclass(slots=True)
class ResponsesAPICallEvent:
    """Event emitted when a Responses API call completes."""

    entity_id: EntityId
    response_id: str
    model: str


@dataclass(slots=True)
class BranchCreatedEvent:
    """Event emitted when a conversation branch is created."""

    entity_id: EntityId
    branch_id: str
    parent_message_id: str


@dataclass(slots=True)
class InheritancePolicy:
    """Policy for parent-to-child attribute inheritance in subagent delegation.

    Controls which attributes are inherited from parent entity and how conflicts are resolved.
    """

    enabled: bool = True
    inherit_system_prompt: bool = True
    inherit_tools: list[str] = field(default_factory=list)  # whitelist of tool names
    inherit_permissions: bool = False
    allow_delegate_tool: bool = False
    tool_conflict_policy: str = "skip"  # skip|error|override
    missing_skill_policy: str = "warn"  # warn|error


@dataclass(slots=True)
class SubagentConfig:
    """Configuration for a named subagent."""

    name: str
    provider: Any  # LLMProvider (can't reference Protocol in dataclass field type)
    model: str
    description: str = ""
    system_prompt: str = ""
    skills: list[str] = field(default_factory=list)  # skill names to install
    max_ticks: int | None = None
    inheritance_policy: InheritancePolicy = field(default_factory=InheritancePolicy)


SubagentLifecycleStatus = Literal[
    "Idle",
    "Working",
    "Dead",
    "Timeout",
    "Cancelled",
]


@dataclass(slots=True)
class SubagentSessionRecord:
    """Serializable session metadata for a subagent delegation."""

    session_id: str
    category: str
    prompt: str
    parent_entity_id: EntityId
    created_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp
    load_skills: list[str] = field(default_factory=list)
    background: bool = False
    status: SubagentLifecycleStatus = "Idle"
    correlation_id: str = ""
    traceparent: str = ""
    timeout_seconds: float | None = None
    deadline_at: str | None = None  # ISO timestamp
    result_excerpt: str | None = None
    error: str | None = None


def validate_subagent_lifecycle_transition(
    current: SubagentLifecycleStatus,
    next_status: SubagentLifecycleStatus,
) -> None:
    allowed_transitions: dict[SubagentLifecycleStatus, set[SubagentLifecycleStatus]] = {
        "Idle": {"Working"},
        "Working": {"Idle", "Dead", "Timeout", "Cancelled"},
        "Dead": set(),
        "Timeout": set(),
        "Cancelled": set(),
    }
    if next_status not in allowed_transitions[current]:
        raise ValueError(
            f"Invalid subagent lifecycle transition from '{current}' to '{next_status}'"
        )


def render_subagent_session_reminder_table(
    sessions: dict[str, SubagentSessionRecord],
) -> list[str]:
    """Render subagent session reminder table rows sorted by updated_at desc, then session_id asc."""
    if not sessions:
        return []

    # Sort by updated_at descending, then session_id ascending
    sorted_sessions = sorted(
        sessions.items(),
        key=lambda item: (-_iso_timestamp_to_sortable(item[1].updated_at), item[0]),
    )

    rows = []
    for session_id, record in sorted_sessions:
        # Format: session_id | status | category | updated_at
        row = (
            f"{session_id} | {record.status} | {record.category} | {record.updated_at}"
        )
        rows.append(row)

    return rows


def _iso_timestamp_to_sortable(timestamp: str) -> float:
    """Convert ISO timestamp to sortable float (seconds since epoch)."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, AttributeError):
        # Invalid timestamp, return 0 so it sorts last
        return 0.0


@dataclass(slots=True)
class DelegationStartedEvent:
    """Event emitted when subagent delegation starts."""

    entity_id: EntityId
    subagent_name: str
    task: str
    correlation_id: str
    traceparent: str
    child_world_name: str | None = None


@dataclass(slots=True)
class DelegationCompletedEvent:
    """Event emitted when subagent delegation completes."""

    entity_id: EntityId
    subagent_name: str
    result: str
    success: bool = True
    error: str | None = None
    correlation_id: str = ""
    traceparent: str = ""
    child_world_name: str | None = None


@dataclass(slots=True)
class TaskCreatedEvent:
    """Event emitted when a task is created."""

    entity_id: EntityId
    task_id: str
    description: str


@dataclass(slots=True)
class TaskStateChangedEvent:
    """Event emitted when task status changes."""

    entity_id: EntityId
    task_id: str
    old_status: "TaskStatus"
    new_status: "TaskStatus"


@dataclass(slots=True)
class TaskBlockedEvent:
    """Event emitted when a task becomes blocked."""

    entity_id: EntityId
    task_id: str
    reason: str
    blocked_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskCompletedEvent:
    """Event emitted when a task completes successfully."""

    entity_id: EntityId
    task_id: str
    result: str


@dataclass(slots=True)
class TaskFailedEvent:
    """Event emitted when a task fails."""

    entity_id: EntityId
    task_id: str
    error: str
    retry_count: int = 0


@dataclass(slots=True, frozen=True)
class TaskReadyEvent:
    """Event emitted when a task becomes ready for execution.

    Fired when all upstream dependencies are resolved and the task
    transitions from PENDING to READY status.
    """

    entity_id: EntityId
    task_id: str
    dependencies_resolved: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""


@dataclass(slots=True, frozen=True)
class TaskRunningEvent:
    """Event emitted when a task begins execution.

    Fired when task transitions from READY/BLOCKED to RUNNING status.
    Includes backend assignment and agent information.
    """

    entity_id: EntityId
    task_id: str
    backend: str  # e.g., 'fetch', 'dispatch', 'completion'
    assigned_agent: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""


@dataclass(slots=True, frozen=True)
class TaskBlockedUpdatedEvent:
    """Event emitted when a task becomes or remains blocked.

    Fired when task transitions to BLOCKED status due to upstream failures
    or missing dependencies. Includes context about what is blocking it.
    """

    entity_id: EntityId
    task_id: str
    blocking_reasons: list[str] = field(default_factory=list)
    upstream_failures: list[str] = field(
        default_factory=list
    )  # task_ids of failed dependencies
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""


@dataclass(slots=True, frozen=True)
class TaskCompletedWithMetadataEvent:
    """Event emitted when a task completes successfully with correlation metadata.

    Fired when task transitions to COMPLETED status. Includes result references
    and duration information for observability.
    """

    entity_id: EntityId
    task_id: str
    result_refs: list[str] = field(default_factory=list)  # ScratchbookRef artifact IDs
    duration_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""


@dataclass(slots=True, frozen=True)
class TaskFailedWithReasonEvent:
    """Event emitted when a task fails with detailed reason.

    Fired when task transitions to FAILED status. Includes error details
    and exception information for debugging.
    """

    entity_id: EntityId
    task_id: str
    error_reason: str
    exception_details: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""


@dataclass(slots=True, frozen=True)
class TaskUnblockedEvent:
    """Event emitted when a task transitions from BLOCKED to ready state.

    Fired when upstream dependencies are resolved or manual override is applied.
    Enables tracking of task unblocking events in the execution pipeline.
    """

    entity_id: EntityId
    task_id: str
    unblock_reason: str  # e.g., 'dependency_resolved', 'manual_override'
    manual_override: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""


@dataclass(slots=True)
class MessageBusEnvelope:
    """CloudEvents-aligned message bus envelope with correlation and tracing extensions.

    Enforces CloudEvents required fields (id, source, type, specversion) plus
    correlation extension (correlationid, causationid) and distributed tracing
    extension (traceparent, tracestate).
    """

    # CloudEvents required fields
    id: str
    source: str
    type: str
    specversion: str
    # Correlation extension (required for message bus)
    correlationid: str
    # Distributed tracing extension (required for message bus)
    traceparent: str
    # Optional CloudEvents fields
    datacontenttype: str = "application/json"
    subject: str | None = None
    time: datetime | None = None
    data: Any | None = None
    # Correlation extension (optional)
    causationid: str | None = None
    # Distributed tracing extension (optional)
    tracestate: str | None = None
    # Message bus extension (optional)
    expirytime: datetime | None = None


@dataclass(slots=True)
class MessageBusPublishedEvent:
    """Event emitted when a message is published to a topic."""

    entity_id: EntityId
    envelope: MessageBusEnvelope
    topic: str


@dataclass(slots=True)
class MessageBusDeliveredEvent:
    """Event emitted when a message is delivered to a subscriber."""

    entity_id: EntityId
    subscriber_id: EntityId
    envelope: MessageBusEnvelope


@dataclass(slots=True)
class MessageBusTimeoutEvent:
    """Event emitted when a request times out waiting for response."""

    entity_id: EntityId
    correlation_id: str


@dataclass(slots=True)
class MessageBusResponseEvent:
    """Event emitted when a response is received for a request."""

    entity_id: EntityId
    correlation_id: str
    envelope: MessageBusEnvelope


@dataclass(slots=True)
class ReasoningCompleteEvent:
    """Event emitted when ReasoningSystem produces a final text response (no tool calls)."""

    entity_id: EntityId
    model: str
    duration_ms: float


@dataclass(slots=True)
class RevertRequest:
    """Request to revert conversation to a specific branch."""

    entity_id: EntityId
    target_branch_id: str


@dataclass(slots=True)
class RevertResult:
    """Result of a conversation revert operation."""

    entity_id: EntityId
    success: bool
    new_branch_id: str | None = None
    message: str = ""


__all__ = [
    "ApprovalPolicy",
    "BranchCreatedEvent",
    "CheckpointCreatedEvent",
    "CheckpointRestoredEvent",
    "CompactionCompleteEvent",
    "CompletionResult",
    "ConversationBranch",
    "ConversationMessage",
    "ConversationTruncatedEvent",
    "DelegationCompletedEvent",
    "DelegationStartedEvent",
    "EntityId",
    "ErrorOccurredEvent",
    "FileRefPart",
    "ImageUrlPart",
    "InheritancePolicy",
    "InterruptionReason",
    "LLMInvocationEvent",
    "MCPConnectedEvent",
    "MCPDisconnectedEvent",
    "MCPToolCallEvent",
    "MCTSNodeScoredEvent",
    "Message",
    "MessagePart",
    "MessageDeliveredEvent",
    "MessageBusDeliveredEvent",
    "MessageBusEnvelope",
    "MessageBusPublishedEvent",
    "MessageBusResponseEvent",
    "MessageBusTimeoutEvent",
    "PlanRevisedEvent",
    "PlanStepCompletedEvent",
    "ReasoningCompleteEvent",
    "RAGRetrievalCompletedEvent",
    "ResponsesAPICallEvent",
    "RetryConfig",
    "RevertRequest",
    "RevertResult",
    "ScratchbookRef",
    "SkillDiscoveryEvent",
    "SkillInstalledEvent",
    "SkillUninstalledEvent",
    "StreamDelta",
    "StreamContentDeltaEvent",
    "StreamContentStartEvent",
    "StreamReasoningDeltaEvent",
    "StreamReasoningEndEvent",
    "StreamEndEvent",
    "StreamStartEvent",
    "SubagentConfig",
    "SubagentLifecycleStatus",
    "SubagentSessionRecord",
    "SystemHandle",
    "TaskBlockedEvent",
    "TaskBlockedUpdatedEvent",
    "TaskCompletedEvent",
    "TaskCompletedWithMetadataEvent",
    "TaskCreatedEvent",
    "TaskFailedEvent",
    "TaskFailedWithReasonEvent",
    "TaskReadyEvent",
    "TaskRunningEvent",
    "TaskStateChangedEvent",
    "TaskStatus",
    "TaskUnblockedEvent",
    "TextPart",
    "ToolApprovalRequestedEvent",
    "ToolApprovedEvent",
    "ToolCall",
    "ToolDeniedEvent",
    "ToolExecutionCompletedEvent",
    "ToolExecutionStartedEvent",
    "ToolSchema",
    "ToolTimeoutError",
    "Usage",
    "UserInputRequestedEvent",
    "render_subagent_session_reminder_table",
    "validate_subagent_lifecycle_transition",
]
