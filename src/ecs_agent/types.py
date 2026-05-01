"""Core type definitions for ECS-based LLM Agent."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, NewType, cast, get_args

from ecs_agent.accounting.models import LLMInvocationEvent, LLMRetryEvent, UsageRecord

EntityId = NewType("EntityId", int)
SystemHandle = NewType("SystemHandle", str)


@dataclass(slots=True)
class ToolCall:
    """Represents a call to a tool/function."""

    id: str
    name: str
    arguments: dict[str, Any]


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


MessagePart = ImageUrlPart | FileRefPart
DroppableContextKind = Literal["tool_result", "reasoning"]
MessageRole = Literal["system", "user", "assistant", "tool"]
LegacyMessageRole = Literal["compaction"]
AcceptedMessageRole = MessageRole | LegacyMessageRole
CompactionMethod = Literal["full_history", "predrop_then_compact"]


@dataclass(slots=True)
class CachedToolResultRef:
    tool_call_id: str
    artifact_path: str
    summary: str | None = None
    original_content: str | None = None


@dataclass(slots=True)
class ToolResultCachedEvent:
    entity_id: EntityId
    tool_call_id: str
    artifact_path: str
    status: str = "cached"


@dataclass(slots=True)
class ContextPrunedEvent:
    entity_id: EntityId
    reason: str
    tool_call_id: str | None = None
    artifact_path: str | None = None
    source_label: str | None = None


@dataclass(slots=True)
class Message:
    """Represents a message in the conversation.

    Content/parts contract
    ----------------------
    ``content`` is the canonical text body of the message.  It must always
    be set for user and assistant messages (empty string is fine for tool
    messages that carry only ``tool_call_id``).

    ``parts`` carries *non-text* media attachments only: ``ImageUrlPart``
    ``parts`` carries *non-text* media attachments only: ``ImageUrlPart``
    and ``FileRefPart``.
    Doing so causes text to be sent twice to the LLM and breaks prompt
    normalisation (``UserPromptNormalizationSystem`` only reads
    ``content``, not ``parts``).

    Correct usage for a multimodal message::

        Message(
            role="user",
            content="Describe this image.",   # text goes here
            parts=[ImageUrlPart(url="...")],  # media goes here
        )
    """

    role: AcceptedMessageRole
    content: str
    parts: list[MessagePart] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    compaction_metadata: dict[str, Any] | None = field(default=None)


@dataclass(slots=True)
class ConversationMessage:
    """A message node in a conversation tree."""

    id: str
    parent_message_id: str | None
    role: str  # 'system' | 'user' | 'assistant' | 'tool' | 'compaction'
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

RunnerLifecycleStatus = Literal[
    "success",
    "terminal_component",
    "max_ticks",
    "interruption_component",
    "error",
]
SystemExecutionStatus = Literal["success", "error"]
ToolExecutionStatus = Literal["success", "error"]


@dataclass(slots=True)
class RunStartedEvent:
    """Event emitted when a runner starts processing a world."""

    max_ticks: int | None
    start_tick: int
    active_entities: int


@dataclass(slots=True)
class RunnerTickStartedEvent:
    """Event emitted when a runner tick starts."""

    tick: int
    active_entities: int


@dataclass(slots=True)
class RunnerTickCompletedEvent:
    """Event emitted when a runner tick completes."""

    tick: int
    status: RunnerLifecycleStatus
    duration_seconds: float
    active_entities: int


@dataclass(slots=True)
class RunCompletedEvent:
    """Event emitted when a runner stops processing a world."""

    status: RunnerLifecycleStatus
    reason: str
    duration_seconds: float
    ticks: int
    active_entities: int


@dataclass(slots=True)
class SystemExecutionStartedEvent:
    """Event emitted when a system execution starts."""

    system: str


@dataclass(slots=True)
class SystemExecutionCompletedEvent:
    """Event emitted when a system execution completes."""

    system: str
    status: SystemExecutionStatus
    duration_seconds: float


@dataclass(slots=True)
class CompletionResult:
    """Result from LLM completion."""

    message: Message
    usage: Usage | None = None
    response_id: str | None = None
    reasoning_content: str | None = None


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
    operation: str = "execute"
    status: str = "success"


@dataclass(slots=True)
class PlanRevisedEvent:
    """Event emitted when the plan is dynamically revised during execution."""

    entity_id: EntityId
    old_steps: list[str]
    new_steps: list[str]
    operation: str = "revise"
    status: str = "success"


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


@dataclass(slots=True)
class ScratchbookRef:
    """Reference to a scratchbook artifact."""

    artifact_id: str
    category: str
    content_hash: str
    timestamp: str
    record_path: str | None = None


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
    tool_name: str = ""
    policy: str = "unknown"


@dataclass(slots=True)
class ToolDeniedEvent:
    """Event emitted when a tool call is denied."""

    entity_id: EntityId
    tool_call_id: str
    reason: str
    tool_name: str = ""


@dataclass(slots=True)
class MCTSNodeScoredEvent:
    """Event emitted when an MCTS node is scored."""

    entity_id: EntityId
    node_id: int
    score: float
    phase: str = "score"
    status: str = "success"


@dataclass(slots=True)
class StreamStartEvent:
    """Event emitted when streaming starts."""

    entity_id: EntityId
    timestamp: float
    provider_id: str = "unknown"
    model: str = "unknown"
    operation: str = "completion"


@dataclass(slots=True)
class StreamContentDeltaEvent:
    """Event emitted when a streaming content delta is received."""

    entity_id: EntityId
    delta: str
    provider_id: str = "unknown"
    model: str = "unknown"
    operation: str = "completion"
    first_delta_seconds: float | None = None


@dataclass(slots=True)
class StreamReasoningDeltaEvent:
    """Event emitted when a streaming reasoning delta is received."""

    entity_id: EntityId
    reasoning_delta: str
    provider_id: str = "unknown"
    model: str = "unknown"
    operation: str = "completion"
    first_delta_seconds: float | None = None


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
    provider_id: str = "unknown"
    model: str = "unknown"
    operation: str = "completion"
    status: str = "success"
    duration_seconds: float | None = None
    first_delta_seconds: float | None = None


@dataclass(slots=True)
class SubagentStreamStartEvent:
    session_id: str
    parent_entity_id: EntityId
    category: str
    child_world_name: str
    seq: int
    timestamp: str


@dataclass(slots=True)
class SubagentStreamDeltaEvent:
    session_id: str
    parent_entity_id: EntityId
    category: str
    child_world_name: str
    seq: int
    timestamp: str
    delta: str
    reasoning_delta: str | None = None


@dataclass(slots=True)
class SubagentStreamEndEvent:
    session_id: str
    parent_entity_id: EntityId
    category: str
    child_world_name: str
    seq: int
    timestamp: str
    total_tokens: int | None = None


@dataclass(slots=True)
class CheckpointCreatedEvent:
    """Event emitted when a checkpoint is created."""

    entity_id: EntityId
    checkpoint_id: int
    timestamp: float
    operation: str = "save"
    status: str = "success"


@dataclass(slots=True)
class CheckpointRestoredEvent:
    """Event emitted when a checkpoint is restored."""

    entity_id: EntityId
    checkpoint_id: int
    timestamp: float
    operation: str = "restore"
    status: str = "success"


@dataclass(slots=True)
class CompactionCompleteEvent:
    """Event emitted when context compaction completes."""

    entity_id: EntityId
    original_tokens: int
    compacted_tokens: int
    operation: str = "compact"
    status: str = "success"


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
    tool_name: str = ""
    status: ToolExecutionStatus = "success"
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        self.status = "success" if self.success else "error"


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
    tool_conflict_policy: str = "skip"  # skip|error|override
    missing_skill_policy: str = "warn"  # warn|error


@dataclass(slots=True)
class SubagentConfig:
    """Configuration for a named subagent."""

    name: str
    model: Any  # LLMModel — using Any to avoid Protocol import in slots dataclass
    description: str = ""
    system_prompt: str = ""
    skills: list[str] = field(default_factory=list)  # skill names to install
    max_ticks: int | None = None
    inheritance_policy: InheritancePolicy = field(default_factory=InheritancePolicy)


SubagentLifecycleStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "timed_out",
    "cancelled",
]


def is_wake_worthy(status: SubagentLifecycleStatus) -> bool:
    return status in {"succeeded", "failed", "timed_out"}


def _normalize_subagent_lifecycle_status(status: str) -> SubagentLifecycleStatus:
    legacy_status_map: dict[str, SubagentLifecycleStatus] = {
        "Idle": "succeeded",
        "Working": "running",
        "Dead": "failed",
        "Timeout": "timed_out",
        "Cancelled": "cancelled",
    }
    if status in legacy_status_map:
        return legacy_status_map[status]
    if status not in get_args(SubagentLifecycleStatus):
        raise ValueError(f"Invalid subagent lifecycle status: {status}")
    return cast(SubagentLifecycleStatus, status)


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
    stream: bool = False
    background: bool = False
    status: SubagentLifecycleStatus = "queued"
    correlation_id: str = ""
    traceparent: str = ""
    timeout_seconds: float | None = None
    deadline_at: str | None = None  # ISO timestamp
    result_excerpt: str | None = None
    result_summary: str | None = None
    artifact_id: str | None = None
    artifact_record_path: str | None = None
    artifact_inline_content: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        self.status = _normalize_subagent_lifecycle_status(self.status)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "status" and isinstance(value, str):
            object.__setattr__(
                self,
                name,
                _normalize_subagent_lifecycle_status(value),
            )
            return
        object.__setattr__(self, name, value)


@dataclass(slots=True)
class SubagentNotificationRecord:
    notification_id: str
    session_id: str
    parent_entity_id: int
    terminal_status: Literal["succeeded", "failed", "timed_out"]
    summary: str | None
    error: str | None
    created_at: str
    delivered_at: str | None


def validate_subagent_lifecycle_transition(
    current: SubagentLifecycleStatus,
    next_status: SubagentLifecycleStatus,
) -> None:
    allowed_transitions: dict[SubagentLifecycleStatus, set[SubagentLifecycleStatus]] = {
        "queued": {"running", "cancelled"},
        "running": {"succeeded", "failed", "timed_out", "cancelled"},
        "succeeded": set(),
        "failed": set(),
        "timed_out": set(),
        "cancelled": set(),
    }
    current = _normalize_subagent_lifecycle_status(current)
    next_status = _normalize_subagent_lifecycle_status(next_status)
    if next_status not in allowed_transitions[current]:
        raise ValueError(
            f"Invalid subagent lifecycle transition from '{current}' to '{next_status}'"
        )


def render_subagent_session_reminder_table(
    sessions: dict[str, SubagentSessionRecord],
) -> str:
    """Render subagent session reminder table rows sorted by updated_at desc, then session_id asc."""
    if not sessions:
        return "No active subagent sessions."

    # Sort by updated_at descending, then session_id ascending
    sorted_sessions = sorted(
        sessions.items(),
        key=lambda item: (-_iso_timestamp_to_sortable(item[1].updated_at), item[0]),
    )

    rows = [
        "Session ID       | Category        | Status    | Updated At          | Last Message",
        "-" * 95,
    ]
    for session_id, record in sorted_sessions:
        result_excerpt = record.result_excerpt or ""
        if len(result_excerpt) > 50:
            result_excerpt = result_excerpt[:47] + "..."
        row = (
            f"{session_id:16} | {record.category:15} | {record.status:9} | "
            f"{record.updated_at:19} | {result_excerpt}"
        )
        rows.append(row)

    return "\n".join(rows)


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
    phase: str = "running"
    status: str = "running"


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
    phase: str = "completed"
    status: str = ""


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
    "LLMRetryEvent",
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
    "RunCompletedEvent",
    "RunnerLifecycleStatus",
    "RunnerTickCompletedEvent",
    "RunnerTickStartedEvent",
    "RunStartedEvent",
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
    "SubagentStreamDeltaEvent",
    "SubagentStreamEndEvent",
    "SubagentStreamStartEvent",
    "SubagentSessionRecord",
    "SystemExecutionCompletedEvent",
    "SystemExecutionStartedEvent",
    "SystemExecutionStatus",
    "SystemHandle",
    "ToolApprovalRequestedEvent",
    "ToolApprovedEvent",
    "ToolCall",
    "ToolDeniedEvent",
    "ToolExecutionCompletedEvent",
    "ToolExecutionStartedEvent",
    "ToolExecutionStatus",
    "ToolSchema",
    "ToolTimeoutError",
    "Usage",
    "UserInputRequestedEvent",
    "render_subagent_session_reminder_table",
    "validate_subagent_lifecycle_transition",
]
