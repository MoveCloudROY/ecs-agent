"""Real ECS scenarios for tool authorization tests, with deterministic models."""

from dataclasses import dataclass, field
from typing import Any

from ecs_agent.components import ConversationComponent, LLMComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult, EntityId, Message, ToolCall, ToolSchema,
    ToolApprovedEvent, ToolApprovalRequestedEvent, ToolDeniedEvent,
    ToolExecutionStartedEvent, ToolExecutionCompletedEvent,
)


@dataclass(slots=True)
class ToolScenario:
    """Observable handler effects and events for a complete model/tool turn."""

    world: World
    entity: EntityId
    model: FakeModel
    called: list[tuple[str, int]] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)


def scenario(calls: list[ToolCall] | None = None) -> ToolScenario:
    """Create a reasoning-capable World; callers install the execution systems."""
    if calls is None:
        calls = [ToolCall(id="read-1", name="read", arguments={"value": 1}),
                 ToolCall(id="write-1", name="write", arguments={"value": 2})]
    model = FakeModel([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=calls)),
        CompletionResult(message=Message(role="assistant", content="done")),
    ])
    world = World()
    entity = world.create_entity()
    result = ToolScenario(world, entity, model)
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(entity, ConversationComponent(messages=[Message(role="user", content="Use the tools.")]))
    registry = ToolRegistryComponent(tools={}, handlers={})
    for name in ("read", "write"):
        async def handler(value: int = 0, *, tool_name: str = name) -> str:
            result.called.append((tool_name, value))
            return f"{tool_name}:{value}"
        registry.tools[name] = ToolSchema(
            name=name, description=name, concurrency_safe=True,
            parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
        )
        registry.handlers[name] = handler
    world.add_component(entity, registry)
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ErrorHandlingSystem(), priority=99)

    async def observe(event: Any) -> None:
        result.events.append(event)
    for event_type in (ToolApprovalRequestedEvent, ToolApprovedEvent, ToolDeniedEvent,
                       ToolExecutionStartedEvent, ToolExecutionCompletedEvent):
        # Approval observers are installed explicitly by each test, since the
        # existence of an approval subscriber is itself part of the contract.
        if event_type is not ToolApprovalRequestedEvent:
            world.event_bus.subscribe(event_type, observe)
    return result
