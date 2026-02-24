"""ECS-based LLM Agent framework."""

__version__ = "0.1.0"

from ecs_agent.types import (
    ApprovalPolicy,
    CompletionResult,
    EntityId,
    Message,
    RetryConfig,
    StreamDelta,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolDeniedEvent,
    ToolSchema,
    ToolTimeoutError,
)

from ecs_agent.providers.retry_provider import RetryProvider
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.tools import scan_module, sandboxed_execute, tool
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tree_search import TreeSearchSystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.serialization import WorldSerializer
from ecs_agent.logging import configure_logging, get_logger

__all__ = [
    "__version__",
    "ApprovalPolicy",
    "CompletionResult",
    "configure_logging",
    "EntityId",
    "FakeEmbeddingProvider",
    "get_logger",
    "Message",
    "OpenAIEmbeddingProvider",
    "RAGSystem",
    "RetryConfig",
    "RetryProvider",
    "StreamDelta",
    "ToolApprovedEvent",
    "ToolApprovalRequestedEvent",
    "ToolDeniedEvent",
    "ToolSchema",
    "ToolTimeoutError",
    "ToolApprovalSystem",
    "TreeSearchSystem",
    "WorldSerializer",
    "sandboxed_execute",
    "scan_module",
    "tool",
]
