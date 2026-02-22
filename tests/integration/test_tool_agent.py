from __future__ import annotations

import json

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema


@pytest.mark.asyncio
async def test_tool_call_execution_end_to_end() -> None:
    executed_args: list[tuple[str, str]] = []

    async def add(a: str, b: str) -> str:
        executed_args.append((a, b))
        return str(int(a) + int(b))

    tool_call_response = CompletionResult(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="add",
                    arguments=json.dumps({"a": "2", "b": "3"}),
                )
            ],
        )
    )
    final_response = CompletionResult(
        message=Message(role="assistant", content="The answer is 5")
    )
    provider = FakeProvider(responses=[tool_call_response, final_response])

    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="What is 2 + 3?")]
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "add": ToolSchema(
                    name="add",
                    description="Add two numbers",
                    parameters={},
                )
            },
            handlers={"add": add},
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=5)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert any(message.role == "tool" for message in conversation.messages)
    assert any(
        message.role == "tool" and message.content == "5"
        for message in conversation.messages
    )
    assert any(
        message.role == "assistant" and "The answer is 5" in message.content
        for message in conversation.messages
    )
    assert executed_args == [("2", "3")]


@pytest.mark.asyncio
async def test_unknown_tool_graceful_handling() -> None:
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc2",
                            name="nonexistent_tool",
                            arguments="{}",
                        )
                    ],
                )
            )
        ]
    )

    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Use a missing tool")]
        ),
    )
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=5)

    expected_error = "Error: unknown tool 'nonexistent_tool'"
    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert results.results["tc2"] == expected_error

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert any(message.role == "tool" for message in conversation.messages)
    assert any(
        message.role == "tool" and message.content == expected_error
        for message in conversation.messages
    )

    error_component = world.get_component(entity_id, ErrorComponent)
    assert error_component is None
