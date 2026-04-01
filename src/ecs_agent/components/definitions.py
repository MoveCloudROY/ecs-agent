"""Component dataclass definitions for ECS-based LLM Agent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import asyncio
import time

from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.types import (
    ApprovalPolicy,
    ConversationBranch,
    ConversationMessage,
    EntityId,
    InterruptionReason,
    Message,
    SubagentConfig,
    SubagentSessionRecord,
    ToolCall,
    ToolSchema,
    TaskStatus,
    ScratchbookRef,
)

if TYPE_CHECKING:
    from ecs_agent.core.world import World

ScriptHandler = Callable[["World", EntityId, str], Awaitable[str | None]]

try:
    from ecs_agent.providers.protocol import LLMProvider
except ImportError:
    # TYPE_CHECKING workaround: if LLMProvider not yet implemented, use Any
    LLMProvider = Any  # type: ignore[assignment,misc]


@dataclass(slots=True)
class LLMComponent:
    """Links Agent to LLM provider."""

    provider: LLMProvider
    model: str
    system_prompt: str = ""
    pending_provider: LLMProvider | None = None
    pending_model: str | None = None


@dataclass(slots=True)
class ConversationComponent:
    """Conversation history."""

    messages: list[Message]
    max_messages: int = 100


@dataclass(slots=True)
class ConversationTreeComponent:
    """Tree-structured conversation with branching support."""

    messages: dict[str, ConversationMessage] = field(default_factory=dict)
    current_branch_id: str | None = None
    branches: dict[str, ConversationBranch] = field(default_factory=dict)


@dataclass(slots=True)
class KVStoreComponent:
    """Simple key-value memory."""

    store: dict[str, Any]


@dataclass(slots=True)
class ToolRegistryComponent:
    """Registered tools and their handlers."""

    tools: dict[str, ToolSchema]
    handlers: dict[str, Callable[..., Awaitable[str]]]


@dataclass(slots=True)
class SkillMetadata:
    """Tier-1 metadata manifest for an installed skill (two-phase loading)."""

    # --- Core identity (required) ---
    name: str
    description: str
    tool_names: list[str]
    has_system_prompt: bool
    activated: bool = False

    # --- Invocation controls ---
    user_invocable: bool = True
    disable_model_invocation: bool = False

    # --- Argument passing ---
    argument_hint: str = ""

    # --- Tool filtering ---
    allowed_tools: list[str] = field(default_factory=list)

    # --- Advanced routing metadata ---
    context: str | None = None
    agent: str | None = None
    model: str | None = None
    hooks: dict[str, Any] = field(default_factory=dict)

    # --- Skill location ---
    skill_dir_path: str | None = None

    # --- Slash command identity ---
    slash_command: str = ""

    # --- Substitution variables available at metadata build time ---
    substitution_variables: list[str] = field(
        default_factory=lambda: [
            "$ARGUMENTS",
            "$ARGUMENTS[0]",
            "$1",
            "${CLAUDE_SESSION_ID}",
            "${CLAUDE_SKILL_DIR}",
        ]
    )


@dataclass(slots=True)
class SkillComponent:
    skills: dict[str, SkillMetadata]


@dataclass(slots=True)
class PendingToolCallsComponent:
    """Pending tool calls."""

    tool_calls: list[ToolCall]


@dataclass(slots=True)
class ToolResultsComponent:
    """Tool call results (id → result string)."""

    results: dict[str, str]


@dataclass(slots=True)
class PlanComponent:
    """ReAct plan."""

    steps: list[str]
    current_step: int = 0
    completed: bool = False


@dataclass(slots=True)
class OwnerComponent:
    """Entity ownership relationship."""

    owner_id: EntityId


@dataclass(slots=True)
class ErrorComponent:
    """Error information."""

    error: str
    system_name: str
    timestamp: float


@dataclass(slots=True)
class TerminalComponent:
    """Marks Agent completion."""

    reason: str


@dataclass(slots=True)
class SystemPromptComponent:
    """System prompt assembly inputs and rendered output."""

    template: str = ""
    content: str = ""


@dataclass(slots=True)
class ToolApprovalComponent:
    """Tool approval policy configuration."""

    policy: ApprovalPolicy
    timeout: float | None = 30.0
    approved_calls: list[str] = field(default_factory=list)
    denied_calls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PermissionComponent:
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SandboxConfigComponent:
    """Sandbox execution limits."""

    timeout: float = 30.0
    max_output_size: int = 10_000
    sandbox_mode: str = "asyncio"


@dataclass(slots=True)
class PlanSearchComponent:
    """MCTS tree search configuration."""

    max_depth: int = 5
    max_branching: int = 3
    exploration_weight: float = 1.414
    best_plan: list[str] = field(default_factory=list)
    search_active: bool = False


@dataclass(slots=True)
class RAGTriggerComponent:
    """RAG retrieval trigger and results."""

    query: str = ""
    top_k: int = 5
    retrieved_docs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResponsesAPIStateComponent:
    """Tracks OpenAI Responses API state for conversation threading."""

    previous_response_id: str | None = None


@dataclass(slots=True)
class EmbeddingComponent:
    """Embedding provider reference."""

    provider: Any
    dimension: int = 0


@dataclass(slots=True)
class VectorStoreComponent:
    """Vector store reference."""

    store: Any


@dataclass(slots=True)
class StreamingComponent:
    """Streaming output configuration."""

    enabled: bool = False
    non_blocking_delta_publish: bool = False


@dataclass(slots=True)
class CheckpointComponent:
    """Checkpoint snapshots storage."""

    snapshots: list[dict[str, Any]] = field(default_factory=list)
    max_snapshots: int = 10


@dataclass(slots=True)
class CompactionConfigComponent:
    """Configuration for context compaction."""

    threshold_tokens: int
    summary_model: str


@dataclass(slots=True)
class ConversationArchiveComponent:
    """Archive of past conversation summaries."""

    archived_summaries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunnerStateComponent:
    """Maintains runner execution state."""

    current_tick: int
    is_paused: bool = False
    checkpoint_path: str | None = None


@dataclass(slots=True)
class UserInputComponent:
    """Awaits external user input via an asyncio Future.

    Attach this component to an entity that requires user input.
    A UserInputSystem (or external code) creates the future and
    publishes a UserInputRequestedEvent.  External code resolves
    the future via ``future.set_result(user_text)``.

    Attributes:
        prompt: Prompt text shown to the user.
        future: An asyncio Future that will be resolved with the user's input.
                ``None`` means no pending request yet.
        timeout: Seconds to wait for input.  ``None`` means wait indefinitely.
        result: The resolved user input (set by UserInputSystem).
    """

    prompt: str = ""
    future: asyncio.Future[str] | None = field(default=None, repr=False)
    timeout: float | None = None
    result: str | None = None


@dataclass(slots=True)
class MessageBusConfigComponent:
    """Bus configuration (buffer sizes, timeouts)."""

    max_queue_size: int = 1000
    publish_timeout: float = 2.0
    request_timeout: float = 30.0
    cleanup_interval: float = 60.0
    max_pending_requests: int = 10000


@dataclass(slots=True)
class MessageBusSubscriptionComponent:
    """Topic subscriptions per entity."""

    subscriptions: dict[str, set[str]] = field(default_factory=dict)


@dataclass(slots=True)
class MessageBusConversationComponent:
    """Bounded conversation retention metadata."""

    entity_id: EntityId
    messages: list[Message] = field(default_factory=list)
    max_messages: int = 1000


@dataclass(slots=True)
class SubagentRegistryComponent:
    """Registry of named subagents available for delegation."""

    subagents: dict[str, SubagentConfig] = field(default_factory=dict)


@dataclass(slots=True)
class WorkspaceBindingComponent:
    workspace_root: Path | str


@dataclass(slots=True)
class EntityRegistryComponent:
    """Entity registry metadata for runtime naming and tagging."""

    entity_id: EntityId
    name: str
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InterruptionComponent:
    """Marks an entity as interrupted with reason."""

    reason: InterruptionReason
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class TaskComponent:
    """Task orchestration component."""

    # REQUIRED (user-confirmed mandatory fields)
    description: str
    expected_output: str
    assigned_agent: EntityId | str | None
    tools: list[str]
    context_dependencies: list[str]

    # RUNTIME (state machine metadata)
    task_id: str
    status: TaskStatus
    priority: int = 0

    # OPTIONAL (advanced features)
    output_schema: dict[str, Any] | None = None
    max_retries: int = 0


@dataclass(slots=True)
class ScratchbookRefComponent:
    """Reference to a scratchbook artifact."""

    artifact_id: str
    category: str
    content_hash: str
    timestamp: str
    record_path: str | None = None


@dataclass(slots=True)
class ScratchbookIndexComponent:
    """Index of scratchbook artifacts."""

    artifacts: dict[str, ScratchbookRef] = field(default_factory=dict)


@dataclass(slots=True)
class SubagentSessionTableComponent:
    """Table of active and recent subagent sessions."""

    sessions: dict[str, "SubagentSessionRecord"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt normalization components (Task-1)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UserPromptConfigComponent:
    """Opts an entity into the prompt normalization pipeline."""

    triggers: list[TriggerSpec] = field(default_factory=list)
    # whether to enable stage-2 context pool injection
    enable_context_pool: bool = False
    # max characters for context pool rendering (overflow = drop lowest priority)
    context_pool_max_chars: int = 8192
    script_handlers: dict[str, ScriptHandler] = field(default_factory=dict)


@dataclass(slots=True)
class ContextEntry:
    entry_id: str
    priority: int
    source_label: str
    content: str
    registration_order: int


@dataclass(slots=True)
class PromptContextQueueComponent:
    entries: list[ContextEntry] = field(default_factory=list)


@dataclass(slots=True)
class PromptContextReservationComponent:
    reservation_id: str
    created_at_tick: int
    reserved_entries: list[ContextEntry] = field(default_factory=list)


@dataclass(slots=True)
class RenderedSystemPromptComponent:
    text: str
    placeholder_snapshot: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RenderedUserPromptComponent:
    text: str


@dataclass(slots=True)
class ChildStubComponent:
    """Marker: this entity is a parent-world stub tracking a delegated child subagent.

    Entities with this component should be skipped by ReasoningSystem so that
    the parent world does not attempt LLM inference on delegation stubs.
    """
