"""ECS-based LLM Agent framework."""

__version__ = "0.1.0"

from ecs_agent.types import (
    ApprovalPolicy,
    BranchCreatedEvent,
    CachedToolResultRef,
    CheckpointCreatedEvent,
    CheckpointRestoredEvent,
    CompactionCompleteEvent,
    ConversationBranch,
    ConversationMessage,
    CompletionResult,
    DelegationCompletedEvent,
    DelegationStartedEvent,
    DroppableContextKind,
    EntityId,
    MCPConnectedEvent,
    MCPDisconnectedEvent,
    MCPToolCallEvent,
    Message,
    MessageBusDeliveredEvent,
    MessageBusEnvelope,
    MessageBusPublishedEvent,
    MessageBusResponseEvent,
    MessageBusTimeoutEvent,
    ResponsesAPICallEvent,
    RetryConfig,
    SkillDiscoveryEvent,
    SkillInstalledEvent,
    SkillUninstalledEvent,
    StreamDelta,
    StreamContentStartEvent,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamReasoningEndEvent,
    StreamStartEvent,
    SubagentConfig,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolDeniedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolSchema,
    ToolTimeoutError,
    UserInputRequestedEvent,
)
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.tools import (
    bwrap_execute,
    sandboxed_execute,
    scan_module,
    tool,
    wrap_sandbox_handler,
)
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.discovery import DiscoveryManager, DiscoveryReport, SkillDiscovery
from ecs_agent.skills.web_search import WebSearchSkill
from ecs_agent.skills.skill import Skill
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tree_search import TreeSearchSystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.serialization import WorldSerializer
from ecs_agent.dsl import (
    AgentSpec,
    compile_agent_specs,
    discover_agent_sources,
    load_json_agents,
    load_markdown_agent,
    resolve_agent_specs,
    validate_agent_spec,
)
from ecs_agent.logging import (
    configure_logging,
    get_logger,
    log_bus_deliver,
    log_bus_publish,
    log_bus_response,
    log_bus_timeout,
)
from ecs_agent.observability import (
    extract_parent_id,
    extract_trace_id,
    generate_traceparent,
    propagate_trace_context,
)

from ecs_agent.components.definitions import (
    CheckpointComponent,
    CompactionConfigComponent,
    ContextBudgetConfig,
    ContextCacheComponent,
    ConversationArchiveComponent,
    ConversationTreeComponent,
    PermissionComponent,
    ResponsesAPIStateComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    RunnerStateComponent,
    SandboxConfigComponent,
    StreamingComponent,
    SubagentRegistryComponent,
    UserInputComponent,
)

from ecs_agent.providers import (
    ApiFormat,
    ClaudeModel,
    FakeModel,
    LLMModel,
    Model,
    ModelId,
    ModelType,
    OpenAIModel,
    ProviderConfig,
    ProviderEntry,
    ProviderRegistry,
    RetryModel,
    format_model_id,
    get_model,
    parse_model_id,
)

try:
    from ecs_agent.providers import LiteLLMModel
except ImportError:
    LiteLLMModel = None  # type: ignore[assignment, misc]


from ecs_agent.systems import (
    CheckpointSystem,
    CompactionSystem,
    UserInputSystem,
    MessageBusSystem,
)
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.conversation_tree import (
    add_message,
    create_branch,
    linearize,
    switch_branch,
)


__all__ = [
    "AgentSpec",
    "ApprovalPolicy",
    "ApiFormat",
    "BranchCreatedEvent",
    "CachedToolResultRef",
    "BuiltinToolsSkill",
    "CheckpointComponent",
    "CheckpointCreatedEvent",
    "CheckpointRestoredEvent",
    "CheckpointSystem",
    "ClaudeModel",
    "compile_agent_specs",
    "CompactionCompleteEvent",
    "CompactionConfigComponent",
    "ContextBudgetConfig",
    "ContextCacheComponent",
    "CompactionSystem",
    "CompletionResult",
    "ConversationArchiveComponent",
    "ConversationBranch",
    "ConversationMessage",
    "ConversationTreeComponent",
    "DelegationCompletedEvent",
    "DelegationStartedEvent",
    "DiscoveryManager",
    "discover_agent_sources",
    "DiscoveryReport",
    "DroppableContextKind",
    "EntityId",
    "extract_parent_id",
    "extract_trace_id",
    "FakeEmbeddingProvider",
    "FakeModel",
    "get_model",
    "LLMModel",
    "LiteLLMModel",
    "load_json_agents",
    "load_markdown_agent",
    "MCPConnectedEvent",
    "MCPDisconnectedEvent",
    "MCPToolCallEvent",
    "Message",
    "MessageBusDeliveredEvent",
    "MessageBusEnvelope",
    "MessageBusPublishedEvent",
    "MessageBusResponseEvent",
    "MessageBusSystem",
    "MessageBusTimeoutEvent",
    "Model",
    "ModelId",
    "ModelType",
    "PermissionComponent",
    "PermissionSystem",
    "ProviderConfig",
    "ProviderEntry",
    "ProviderRegistry",
    "RAGSystem",
    "resolve_agent_specs",
    "ResponsesAPICallEvent",
    "ResponsesAPIStateComponent",
    "RenderedSystemPromptComponent",
    "RenderedUserPromptComponent",
    "RetryConfig",
    "OpenAIModel",
    "RetryModel",
    "RunnerStateComponent",
    "SandboxConfigComponent",
    "Skill",
    "ScriptSkill",
    "SkillComponent",
    "SkillDiscovery",
    "SkillDiscoveryEvent",
    "SkillInstalledEvent",
    "SkillManager",
    "SkillMetadata",
    "SkillUninstalledEvent",
    "StreamDelta",
    "StreamContentStartEvent",
    "StreamContentDeltaEvent",
    "StreamEndEvent",
    "StreamReasoningDeltaEvent",
    "StreamReasoningEndEvent",
    "StreamStartEvent",
    "StreamingComponent",
    "SubagentConfig",
    "SubagentStreamDeltaEvent",
    "SubagentStreamEndEvent",
    "SubagentStreamStartEvent",
    "SubagentRegistryComponent",
    "SubagentSystem",
    "ToolApprovalRequestedEvent",
    "ToolApprovalSystem",
    "ToolApprovedEvent",
    "ToolDeniedEvent",
    "ToolExecutionCompletedEvent",
    "ToolExecutionStartedEvent",
    "ToolSchema",
    "ToolTimeoutError",
    "TreeSearchSystem",
    "UserInputComponent",
    "UserInputRequestedEvent",
    "UserInputSystem",
    "validate_agent_spec",
    "WebSearchSkill",
    "WorldSerializer",
    "__version__",
    "add_message",
    "bwrap_execute",
    "configure_logging",
    "create_branch",
    "get_logger",
    "generate_traceparent",
    "linearize",
    "log_bus_deliver",
    "log_bus_publish",
    "log_bus_response",
    "log_bus_timeout",
    "sandboxed_execute",
    "scan_module",
    "parse_model_id",
    "get_model",
    "switch_branch",
    "tool",
    "format_model_id",
    "propagate_trace_context",
    "wrap_sandbox_handler",
]

# MCP (optional dependency)
try:
    from ecs_agent.mcp.client import MCPClient as MCPClient
    from ecs_agent.mcp.adapter import MCPSkillAdapter as MCPSkillAdapter
    from ecs_agent.mcp.components import (
        MCPClientComponent as MCPClientComponent,
        MCPConfigComponent as MCPConfigComponent,
    )

    __all__.extend(
        ["MCPClient", "MCPSkillAdapter", "MCPConfigComponent", "MCPClientComponent"]
    )
except ImportError:
    pass
