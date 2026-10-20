"""ECS-based LLM Agent framework."""

__version__ = "0.1.0"

from ecs_agent.core.agent import AgentHandle
from ecs_agent.authoring import (
    AgentDefinition, AssemblyReport, ModelRef, RuntimeConfig, SkillRef, SystemStep,
    build_agent, build_agent_from_specs, inspect_runtime, restore_runtime,
)

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
    FinishReason,
    GenerationStatus,
    ProviderOutput,
    FreeSubagentConfig,
    LLMRetryEvent,
    MCPConnectedEvent,
    MCPDisconnectedEvent,
    MCPToolCallEvent,
    Message,
    MessageBusDeliveredEvent,
    MessageBusEnvelope,
    MessageBusPublishedEvent,
    MessageBusResponseEvent,
    MessageBusTimeoutEvent,
    PhaseChangedEvent,
    ResponsesAPICallEvent,
    RetryConfig,
    RunCompletedEvent,
    RunnerTickCompletedEvent,
    RunnerTickStartedEvent,
    RunStartedEvent,
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
    SystemExecutionCompletedEvent,
    SystemExecutionStartedEvent,
    TodoItem,
    TodoListUpdatedEvent,
    TodoStatus,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolDeniedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolSchema,
    ToolTimeoutError,
    UserInputReceivedEvent,
    UserInputRequestedEvent,
)
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.tools import (
    ToolDefinition,
    bwrap_execute,
    get_tool_definition,
    register_tools,
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
from ecs_agent.skills.todo import TodoSkill
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.components.definitions import TodoListComponent
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.components import SessionPolicyComponent, SessionRefComponent
from ecs_agent.history import append_conversation_message
from ecs_agent.sessions import (
    InMemorySessionStore,
    SQLiteSessionStore,
    SessionService,
    SessionStore,
    install_session_recording,
    release_session_writer,
)
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
from ecs_agent.plugins import (
    CompositeTelemetrySink,
    ObservabilityPlugin,
    PluginsHandle,
    TelemetrySinkPlugin,
    install_plugins,
    propagate_plugins,
    uninstall_plugins,
)

from ecs_agent.components.definitions import (
    CheckpointComponent,
    CompactionConfigComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    ConversationArchiveComponent,
    ConversationTreeComponent,
    PermissionComponent,
    PhaseApprovalsComponent,
    PhaseComponent,
    PhaseDefinitionComponent,
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

from ecs_agent.providers import LiteLLMModel


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
    "AgentDefinition", "AssemblyReport", "ModelRef", "RuntimeConfig", "SkillRef", "SystemStep",
    "build_agent", "build_agent_from_specs", "inspect_runtime", "restore_runtime",
    "AgentHandle",
    "SessionService",
    "SessionStore",
    "SQLiteSessionStore",
    "InMemorySessionStore",
    "SessionRefComponent",
    "SessionPolicyComponent",
    "install_session_recording",
    "release_session_writer",
    "append_conversation_message",
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
    "ContextTrimConfig",
    "ContextCacheComponent",
    "CompactionSystem",
    "CompositeTelemetrySink",
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
    "uninstall_plugins",
    "extract_parent_id",
    "extract_trace_id",
    "FakeEmbeddingProvider",
    "FakeModel",
    "FinishReason",
    "GenerationStatus",
    "ProviderOutput",
    "FreeSubagentConfig",
    "get_model",
    "LLMModel",
    "LLMRetryEvent",
    "LiteLLMModel",
    "install_plugins",
    "load_json_agents",
    "load_markdown_agent",
    "MCPConnectedEvent",
    "MCPDisconnectedEvent",
    "MCPToolCallEvent",
    "TelemetrySinkPlugin",
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
    "ObservabilityPlugin",
    "PermissionComponent",
    "PermissionSystem",
    "PhaseApprovalsComponent",
    "PhaseChangedEvent",
    "PhaseComponent",
    "PhaseDefinitionComponent",
    "PluginsHandle",
    "ProviderConfig",
    "ProviderEntry",
    "ProviderRegistry",
    "RAGSystem",
    "propagate_plugins",
    "resolve_agent_specs",
    "ResponsesAPICallEvent",
    "ResponsesAPIStateComponent",
    "RenderedSystemPromptComponent",
    "RenderedUserPromptComponent",
    "RetryConfig",
    "RunCompletedEvent",
    "RunnerTickCompletedEvent",
    "RunnerTickStartedEvent",
    "RunStartedEvent",
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
    "SystemExecutionCompletedEvent",
    "SystemExecutionStartedEvent",
    "TodoItem",
    "TodoListComponent",
    "TodoListUpdatedEvent",
    "TodoSkill",
    "TodoStatus",
    "ToolApprovalRequestedEvent",
    "ToolApprovalSystem",
    "ToolApprovedEvent",
    "ToolDeniedEvent",
    "ToolExecutionCompletedEvent",
    "ToolExecutionStartedEvent",
    "ToolSchema",
    "ToolDefinition",
    "get_tool_definition",
    "register_tools",
    "ToolTimeoutError",
    "TreeSearchSystem",
    "UserInputComponent",
    "UserInputReceivedEvent",
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
    "OpenAIEmbeddingProvider",
    "OpenAIModel",
    "sandboxed_execute",
    "scan_module",
    "parse_model_id",
    "switch_branch",
    "tool",
    "format_model_id",
    "propagate_trace_context",
    "wrap_sandbox_handler",
]

# Optional integrations load only when their exported symbol is requested.
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
from typing import Any as _Any

_MCP_EXPORTS = {
    "MCPClient": "ecs_agent.mcp.client",
    "MCPSkillAdapter": "ecs_agent.mcp.adapter",
    "MCPClientComponent": "ecs_agent.mcp.components",
    "MCPConfigComponent": "ecs_agent.mcp.components",
}
if _find_spec("mcp") is not None:
    __all__.extend(_MCP_EXPORTS)


def __getattr__(name: str) -> _Any:
    """Resolve optional public MCP symbols without eager SDK imports."""
    if name not in _MCP_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(_MCP_EXPORTS[name]), name)
    globals()[name] = value
    return value
