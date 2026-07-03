"""Component dataclass definitions for ECS-based LLM Agent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import asyncio
import time

from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.types import (
    ApprovalPolicy,
    CachedToolResultRef,
    CompactionMethod,
    ConversationBranch,
    ConversationMessage,
    DroppableContextKind,
    EntityId,
    FreeSubagentConfig,
    InterruptionReason,
    Message,
    SubagentConfig,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    ToolCall,
    ToolSchema,
    ScratchbookRef,
)

if TYPE_CHECKING:
    from ecs_agent.core.world import World
    from ecs_agent.workflows.compiler import CompiledWorkflow

ScriptHandler = Callable[["World", EntityId, str], Awaitable[str | None]]

@dataclass(slots=True)
class LLMComponent:
    """Links Agent to an LLM model implementation."""

    model: Any  # LLMModel — using Any to avoid Protocol import in slots dataclass
    system_prompt: str = ""
    pending_model: Any | None = None  # LLMModel | None


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
class ToolStateNamespace:
    """Namespaced internal runtime state shared by related tools."""

    values: dict[str, object] = field(default_factory=dict)
    version: int = 0


@dataclass(slots=True)
class ToolRuntimeStateComponent:
    """Entity-scoped internal state shared across tool calls."""

    namespaces: dict[str, ToolStateNamespace] = field(default_factory=dict)


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
class ContextBudgetConfig:
    max_tokens: int
    prune_tool_results: bool = True
    prune_reasoning: bool = False
    token_estimation_chars_per_token: float = 4.0
    overflow_behavior: str = "error"


@dataclass(slots=True)
class CompactionConfigComponent:
    """Configuration for context compaction."""

    threshold_tokens: int
    summary_model: str | None = None
    compaction_method: CompactionMethod = "full_history"
    summary_model_id: str | None = None
    compaction_prompt_template: str | None = None


@dataclass(slots=True)
class ContextCacheComponent:
    cached_tool_results: list[CachedToolResultRef] = field(default_factory=list)


@dataclass(slots=True)
class ConversationArchiveComponent:
    """Archive of past conversation summaries."""

    archived_summaries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CurrentCompactionSummaryComponent:
    summary: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class RunnerStateComponent:
    """Maintains runner execution state."""

    current_tick: int
    is_paused: bool = False
    checkpoint_path: str | None = None


@dataclass(slots=True)
class WorkflowDefinitionComponent:
    """Holds the compiled workflow definition for an entity."""

    compiled: "CompiledWorkflow"


@dataclass(slots=True)
class WorkflowRuntimeComponent:
    """Holds the mutable runtime workflow state for an entity."""

    current_state_id: str
    transition_history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkflowBindingComponent:
    """Binds an agent key to the workflow for this entity."""

    agent_key: str


@dataclass(slots=True)
class WorkflowGateSnapshotComponent:
    """Records the last evaluated gate snapshot (for debugging/logging)."""

    state_id: str
    evaluated_at_tick: int
    matched_transition_id: str | None = None


@dataclass(slots=True)
class WorkflowLastTransitionComponent:
    """Records the most recent committed transition (for history / exact-once semantics)."""

    from_state_id: str
    to_state_id: str
    transition_id: str
    tick: int


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
    """Registry and opt-in free-form defaults for subagent delegation."""

    subagents: dict[str, SubagentConfig] = field(default_factory=dict)
    free_subagent_config: FreeSubagentConfig = field(default_factory=FreeSubagentConfig)


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
class ScratchbookRefComponent(ScratchbookRef):
    """Reference to a scratchbook artifact (ECS component form)."""


@dataclass(slots=True)
class ScratchbookIndexComponent:
    """Index of scratchbook artifacts."""

    artifacts: dict[str, ScratchbookRef] = field(default_factory=dict)


@dataclass(slots=True)
class SubagentSessionTableComponent:
    """Table of active and recent subagent sessions."""

    sessions: dict[str, "SubagentSessionRecord"] = field(default_factory=dict)


@dataclass(slots=True)
class SubagentNotificationQueueComponent:
    notifications: list[SubagentNotificationRecord] = field(default_factory=list)


@dataclass(slots=True)
class SubagentWaitComponent:
    session_ids: list[str] | None = None
    timeout: float | None = None
    future: Any | None = field(default=None, repr=False)
    started_at: str | None = None
    resolved_session_ids: list[str] | None = None
    auto_restart_budget: int = 0
    restart_counts: dict[str, int] = field(default_factory=dict)


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
    droppable_kind: DroppableContextKind | None = field(default=None)


@dataclass(slots=True)
class PromptContextQueueComponent:
    entries: list[ContextEntry] = field(default_factory=list)


@dataclass(slots=True)
class PromptContextReservationComponent:
    reservation_id: str
    created_at_tick: int
    reserved_entries: list[ContextEntry] = field(default_factory=list)


@dataclass(slots=True)
class TokenUsageComponent:
    """Actual token usage reported by the LLM API for an entity.

    Populated by ``ReasoningSystem`` after each invocation from the provider's
    usage response — the ground truth, more accurate than any local estimate.
    ``last_*`` reflects the most recent call; ``total_*`` accumulates across all
    calls. Absent until the entity has completed at least one LLM call."""

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    last_cache_read_tokens: int = 0
    last_cache_creation_tokens: int = 0

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0

    call_count: int = 0

    # Number of conversation messages that had accumulated when the last call was
    # made (its input basis). Lets compaction calibrate: real last_prompt_tokens
    # plus a local estimate of only the messages appended since. -1 means "no
    # valid anchor" (fall back to a pure local estimate).
    last_prompt_message_count: int = -1


@dataclass(slots=True)
class RenderedSystemPromptComponent:
    text: str
    placeholder_snapshot: dict[str, str] = field(default_factory=dict)
    stable_text: str = ""
    """Cache-stable prefix: base template with volatile placeholders emptied.

    Byte-stable across turns given fixed stable-provider fingerprints, so it can
    serve as an Anthropic prompt-cache prefix. Empty for legacy render paths that
    predate the split (consumers fall back to ``text``)."""
    volatile_text: str = ""
    """Volatile tail: compaction summary + workflow state, rendered after the
    cache breakpoint. ``text`` == ``stable_text`` + this tail (when non-empty)."""


@dataclass(slots=True)
class RenderedUserPromptComponent:
    text: str
    source_fingerprint: str | None = None
    trigger_key: str | None = None
    source_message_index: int | None = None


@dataclass(slots=True)
class ChildStubComponent:
    """Marker: this entity is a parent-world stub tracking a delegated child subagent.

    Entities with this component should be skipped by ReasoningSystem so that
    the parent world does not attempt LLM inference on delegation stubs.
    """
