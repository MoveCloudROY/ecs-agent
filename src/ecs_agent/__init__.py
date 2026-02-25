"""ECS-based LLM Agent framework."""

__version__ = "0.1.0"

from ecs_agent.types import (
    ApprovalPolicy,
    CheckpointCreatedEvent,
    CheckpointRestoredEvent,
    CompactionCompleteEvent,
    CompletionResult,
    EntityId,
    Message,
    RetryConfig,
    StreamDelta,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolDeniedEvent,
    ToolSchema,
    ToolTimeoutError,
    UserInputRequestedEvent,
)
from typing import Any
from ecs_agent.providers.retry_provider import RetryProvider
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.tools import scan_module, sandboxed_execute, tool
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tree_search import TreeSearchSystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.serialization import WorldSerializer
from ecs_agent.logging import configure_logging, get_logger

from ecs_agent.components.definitions import (
    CheckpointComponent,
    CompactionConfigComponent,
    ConversationArchiveComponent,
    RunnerStateComponent,
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
    "CheckpointComponent",
    "CheckpointCreatedEvent",
    "CheckpointRestoredEvent",
    "CheckpointSystem",
    "ClaudeProvider",
    "CompactionCompleteEvent",
    "CompactionConfigComponent",
    "CompactionSystem",

    "ConversationArchiveComponent",
    "configure_logging",
    "EntityId",
    "FakeEmbeddingProvider",
    "get_logger",
    "LiteLLMProvider",
    "Message",
    "OpenAIEmbeddingProvider",
    "RAGSystem",
    "RetryConfig",
    "RetryProvider",
    "RunnerStateComponent",
    "StreamDelta",
    "StreamDeltaEvent",
    "StreamEndEvent",
    "StreamingComponent",
    "StreamStartEvent",
    "ToolApprovedEvent",
    "ToolApprovalRequestedEvent",
    "ToolApprovalSystem",
    "ToolDeniedEvent",
    "ToolSchema",
    "ToolTimeoutError",
    "TreeSearchSystem",
    "UserInputComponent",
    "UserInputRequestedEvent",
    "UserInputSystem",
    "WorldSerializer",
    "sandboxed_execute",
    "scan_module",
    "tool",
]
