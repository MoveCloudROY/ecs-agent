"""Component dataclass definitions for ECS-based LLM Agent."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ecs_agent.types import EntityId, Message, ToolCall, ToolSchema

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


@dataclass(slots=True)
class ConversationComponent:
    """Conversation history."""

    messages: list[Message]
    max_messages: int = 100


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
class CollaborationComponent:
    """Multi-agent messaging."""

    peers: list[EntityId]
    inbox: list[tuple[EntityId, Message]]


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
    """LLM system prompt."""

    content: str
