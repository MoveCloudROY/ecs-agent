# mypy: disable-error-code=import-untyped

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent, TerminalComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, ToolSchema


def make_recording_fake_provider(
    responses: list[CompletionResult],
) -> tuple[FakeProvider, dict[str, int]]:
    provider = FakeProvider(responses=responses)
    call_counts = {"attempt": 0, "success": 0}
    original_complete = provider.complete

    async def wrapped_complete(
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        call_counts["attempt"] += 1
        result = await original_complete(messages, tools=tools)
        call_counts["success"] += 1
        return result

    provider.complete = wrapped_complete  # type: ignore[method-assign]
    return provider, call_counts


class CounterSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority
        self.run_count = 0

    async def process(self, world: World) -> None:
        _ = world
        self.run_count += 1


@pytest.mark.asyncio
async def test_simple_chat_agent_end_to_end() -> None:
    world = World()
    provider, call_counts = make_recording_fake_provider(
        [
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Hello! I am an assistant.",
                )
            )
        ]
    )

    eid = world.create_entity()
    world.add_component(
        eid,
        LLMComponent(
            provider=provider,
            model="fake",
            system_prompt="You are helpful.",
        ),
    )
    world.add_component(
        eid,
        ConversationComponent(messages=[Message(role="user", content="Hi there!")]),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    conversation = world.get_component(eid, ConversationComponent)
    assert conversation is not None
    assert len(conversation.messages) >= 2
    assistant_messages = [
        msg for msg in conversation.messages if msg.role == "assistant"
    ]
    assert assistant_messages
    assert assistant_messages[0].content == "Hello! I am an assistant."
    assert call_counts["success"] == 1


@pytest.mark.asyncio
async def test_terminal_on_provider_exhausted() -> None:
    world = World()
    provider, call_counts = make_recording_fake_provider(
        [CompletionResult(message=Message(role="assistant", content="Only once"))]
    )

    eid = world.create_entity()
    world.add_component(
        eid,
        LLMComponent(
            provider=provider,
            model="fake",
            system_prompt="You are helpful.",
        ),
    )
    world.add_component(
        eid,
        ConversationComponent(messages=[Message(role="user", content="Hi there!")]),
    )

    counter = CounterSystem()
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(counter, priority=50)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=5)

    terminal = world.get_component(eid, TerminalComponent)
    assert terminal is not None
    assert terminal.reason == "provider_exhausted"
    assert call_counts["attempt"] == 2
    assert call_counts["success"] == 1
    assert counter.run_count == 2


@pytest.mark.asyncio
async def test_max_ticks_enforcement() -> None:
    world = World()
    counter = CounterSystem()
    world.register_system(counter, priority=0)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    terminal_components = list(world.query(TerminalComponent))
    assert len(terminal_components) == 1
    _, (terminal,) = terminal_components[0]
    assert terminal.reason == "max_ticks"
    assert counter.run_count == 1
