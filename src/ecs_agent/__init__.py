"""ECS-based LLM Agent framework."""

__version__ = "0.1.0"

from ecs_agent.types import (
    ApprovalPolicy,
    CheckpointCreatedEvent,
    CheckpointRestoredEvent,
    CompactionCompleteEvent,
    CompletionResult,
    EntityId,
    MCPConnectedEvent,
    MCPDisconnectedEvent,
    MCPToolCallEvent,
    Message,
    RetryConfig,
    SkillDiscoveryEvent,
    SkillInstalledEvent,
    SkillUninstalledEvent,
    StreamDelta,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolDeniedEvent,
    ToolExecutionCompletedEvent,
    ToolExecutionStartedEvent,
    ToolSchema,
    ToolTimeoutError,
    UserInputRequestedEvent,
)
from ecs_agent.providers.retry_provider import RetryProvider
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.tools import (
    bwrap_execute,
    sandboxed_execute,
    scan_module,
    tool,
    wrap_sandbox_handler,
)
from ecs_agent.skills.protocol import Skill
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.discovery import DiscoveryManager, DiscoveryReport, SkillDiscovery
from ecs_agent.skills.web_search import WebSearchSkill
from ecs_agent.components.definitions import SkillComponent, SkillMetadata
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tree_search import TreeSearchSystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.serialization import WorldSerializer
from ecs_agent.logging import configure_logging, get_logger

from ecs_agent.components.definitions import (
    CheckpointComponent,
    CompactionConfigComponent,
    ConversationArchiveComponent,
    PermissionComponent,
    RunnerStateComponent,
    SandboxConfigComponent,
    StreamingComponent,
    UserInputComponent,
)

from ecs_agent.providers import ClaudeProvider

try:
    from ecs_agent.providers import LiteLLMProvider
except ImportError:
    LiteLLMProvider = None  # type: ignore[assignment, misc]


from ecs_agent.systems import CheckpointSystem, CompactionSystem, UserInputSystem


__all__ = [
    "__version__",
    "ApprovalPolicy",
    "BuiltinToolsSkill",
    "CheckpointComponent",
    "CheckpointCreatedEvent",
    "CheckpointRestoredEvent",
    "CheckpointSystem",
    "ClaudeProvider",
    "CompactionCompleteEvent",
    "CompactionConfigComponent",
    "CompactionSystem",
    "CompletionResult",
    "ConversationArchiveComponent",
    "DiscoveryManager",
    "DiscoveryReport",
    "configure_logging",
    "EntityId",
    "FakeEmbeddingProvider",
    "get_logger",
    "LiteLLMProvider",
    "MCPConnectedEvent",
    "MCPDisconnectedEvent",
    "MCPToolCallEvent",
    "Message",
    "OpenAIEmbeddingProvider",
    "PermissionComponent",
    "PermissionSystem",
    "RAGSystem",
    "RetryConfig",
    "RetryProvider",
    "RunnerStateComponent",
    "SandboxConfigComponent",
    "Skill",
    "SkillComponent",
    "SkillDiscovery",
    "SkillDiscoveryEvent",
    "SkillInstalledEvent",
    "SkillManager",
    "SkillMetadata",
    "SkillUninstalledEvent",
    "StreamDelta",
    "StreamDeltaEvent",
    "StreamEndEvent",
    "StreamingComponent",
    "StreamStartEvent",
    "ToolApprovedEvent",
    "ToolApprovalRequestedEvent",
    "ToolApprovalSystem",
    "ToolDeniedEvent",
    "ToolExecutionCompletedEvent",
    "ToolExecutionStartedEvent",
    "ToolSchema",
    "ToolTimeoutError",
    "TreeSearchSystem",
    "UserInputComponent",
    "UserInputRequestedEvent",
    "UserInputSystem",
    "WorldSerializer",
    "bwrap_execute",
    "sandboxed_execute",
    "scan_module",
    "tool",
    "wrap_sandbox_handler",
    "WebSearchSkill",
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
