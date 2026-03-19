"""Tests for TaskExecutor mixed backend routing."""

from __future__ import annotations

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PromptConfigComponent,
    SubagentRegistryComponent,
    TurnStateComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.task.executor import ExecutionResult, TaskExecutor
from ecs_agent.task.fetching_unit import DispatchRequest
from ecs_agent.types import (
    CompletionResult,
    Message,
    SubagentConfig,
    ToolCall,
    ToolSchema,
)


class RecordingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        return await super().complete(messages, tools=tools)


class FlakyRecordingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            responses=[
                CompletionResult(message=Message(role="assistant", content="Local OK"))
            ]
        )
        self.calls: list[list[Message]] = []
        self._attempt = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        self.calls.append(list(messages))
        self._attempt += 1
        if self._attempt == 1:
            raise RuntimeError("provider exploded")
        return await super().complete(messages, tools=tools)


@pytest.mark.asyncio
async def test_route_backend_local_with_entity_id() -> None:
    """Verify local backend selected when assigned_agent is EntityId."""
    world = World()
    executor = TaskExecutor()

    # Create entity with local agent reference
    agent = world.create_entity()
    local_executor = world.create_entity()  # This is the "assigned_agent" EntityId

    # Set up minimal components for local execution
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Local result"))
        ]
    )
    world.add_component(
        agent, LLMComponent(provider=provider, model="fake", system_prompt="")
    )
    world.add_component(agent, ConversationComponent(messages=[]))
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))

    request = DispatchRequest(
        task_id="task-1",
        wave_number=0,
        sequence_number=0,
        description="Test local task",
        expected_output="Output",
        assigned_agent=local_executor,  # EntityId (int)
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-1"
    assert result.backend_type == "local"
    assert result.success is True
    assert "Local result" in result.result_content


@pytest.mark.asyncio
async def test_route_backend_subagent_with_string_name() -> None:
    """Verify subagent backend selected when assigned_agent is string."""
    world = World()
    executor = TaskExecutor()

    # Create entity with subagent configuration
    agent = world.create_entity()
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Subagent done"))
        ]
    )

    # Register subagent
    config = SubagentConfig(
        name="researcher",
        provider=provider,
        model="fake",
        system_prompt="",
        max_ticks=5,
    )
    registry = SubagentRegistryComponent(subagents={"researcher": config})
    world.add_component(agent, registry)

    # Install delegate tool (minimal mock)
    async def mock_delegate_handler(subagent_name: str, task: str) -> str:
        return f"Subagent {subagent_name} completed: {task}"

    tool_registry = ToolRegistryComponent(
        tools={
            "delegate": ToolSchema(
                name="delegate",
                description="Delegate task",
                parameters={},
            )
        },
        handlers={"delegate": mock_delegate_handler},
    )
    world.add_component(agent, tool_registry)

    request = DispatchRequest(
        task_id="task-2",
        wave_number=0,
        sequence_number=1,
        description="Research quantum physics",
        expected_output="Report",
        assigned_agent="researcher",  # String subagent name
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-2"
    assert result.backend_type == "subagent"
    assert result.success is True
    assert "researcher" in result.result_content


@pytest.mark.asyncio
async def test_route_backend_local_with_none_default() -> None:
    """Verify local backend (default policy) when assigned_agent is None."""
    world = World()
    executor = TaskExecutor()

    agent = world.create_entity()

    # Set up minimal components for local execution
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Default local result")
            )
        ]
    )
    world.add_component(
        agent, LLMComponent(provider=provider, model="fake", system_prompt="")
    )
    world.add_component(agent, ConversationComponent(messages=[]))
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))

    request = DispatchRequest(
        task_id="task-3",
        wave_number=0,
        sequence_number=2,
        description="Default task",
        expected_output="Output",
        assigned_agent=None,  # None → default to local
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-3"
    assert result.backend_type == "local"
    assert result.success is True
    assert "Default local result" in result.result_content


@pytest.mark.asyncio
async def test_invalid_backend_config_type() -> None:
    """Verify clear validation error for invalid assigned_agent type."""
    world = World()
    executor = TaskExecutor()

    agent = world.create_entity()

    # Create request with invalid assigned_agent type (list instead of EntityId/str/None)
    request = DispatchRequest(
        task_id="task-invalid",
        wave_number=0,
        sequence_number=0,
        description="Invalid task",
        expected_output="Output",
        assigned_agent=["bad", "type"],  # type: ignore[arg-type]  # Invalid type
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    with pytest.raises(ValueError, match="Invalid assigned_agent type"):
        await executor.execute_dispatch_request(world, agent, request)


@pytest.mark.asyncio
async def test_subagent_backend_missing_registry() -> None:
    """Verify error when subagent backend selected but no SubagentRegistryComponent."""
    world = World()
    executor = TaskExecutor()

    agent = world.create_entity()

    request = DispatchRequest(
        task_id="task-missing-registry",
        wave_number=0,
        sequence_number=0,
        description="Task",
        expected_output="Output",
        assigned_agent="researcher",  # String → subagent, but no registry
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-missing-registry"
    assert result.backend_type == "subagent"
    assert result.success is False
    assert "missing SubagentRegistryComponent" in result.result_content


@pytest.mark.asyncio
async def test_subagent_backend_unknown_subagent_name() -> None:
    """Verify error when subagent name not in registry."""
    world = World()
    executor = TaskExecutor()

    agent = world.create_entity()

    provider = FakeProvider(responses=[])
    config = SubagentConfig(
        name="known",
        provider=provider,
        model="fake",
    )
    registry = SubagentRegistryComponent(subagents={"known": config})
    world.add_component(agent, registry)
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))

    request = DispatchRequest(
        task_id="task-unknown-subagent",
        wave_number=0,
        sequence_number=0,
        description="Task",
        expected_output="Output",
        assigned_agent="unknown",  # Not in registry
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-unknown-subagent"
    assert result.backend_type == "subagent"
    assert result.success is False
    assert "Unknown subagent 'unknown'" in result.result_content


@pytest.mark.asyncio
async def test_local_backend_missing_llm_component() -> None:
    """Verify error when local backend selected but no LLMComponent."""
    world = World()
    executor = TaskExecutor()

    agent = world.create_entity()
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))

    request = DispatchRequest(
        task_id="task-missing-llm",
        wave_number=0,
        sequence_number=0,
        description="Task",
        expected_output="Output",
        assigned_agent=None,  # Local backend
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-missing-llm"
    assert result.backend_type == "local"
    assert result.success is False
    assert "missing LLMComponent" in result.result_content


@pytest.mark.asyncio
async def test_local_backend_with_tool_calls() -> None:
    """Verify local backend executes tool calls and returns aggregated results."""
    world = World()
    executor = TaskExecutor()

    agent = world.create_entity()

    # Create tool handler
    async def mock_tool_handler(param: str) -> str:
        return f"Tool executed with: {param}"

    # Set up provider that returns tool call
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="mock_tool",
                            arguments={"param": "test-value"},
                        )
                    ],
                )
            )
        ]
    )

    world.add_component(
        agent, LLMComponent(provider=provider, model="fake", system_prompt="")
    )
    world.add_component(agent, ConversationComponent(messages=[]))
    world.add_component(
        agent,
        ToolRegistryComponent(
            tools={
                "mock_tool": ToolSchema(
                    name="mock_tool",
                    description="Test tool",
                    parameters={},
                )
            },
            handlers={"mock_tool": mock_tool_handler},
        ),
    )

    request = DispatchRequest(
        task_id="task-with-tools",
        wave_number=0,
        sequence_number=0,
        description="Execute tool",
        expected_output="Output",
        assigned_agent=None,  # Local backend
        tools=("mock_tool",),
        context_dependencies=tuple(),
        priority=0,
    )

    result = await executor.execute_dispatch_request(world, agent, request)

    assert result.task_id == "task-with-tools"
    assert result.backend_type == "local"
    assert result.success is True
    assert "mock_tool:" in result.result_content
    assert "test-value" in result.result_content


@pytest.mark.asyncio
async def test_normalized_results_both_backends() -> None:
    """Verify both backends return ExecutionResult with same contract."""
    world = World()
    executor = TaskExecutor()

    # LOCAL BACKEND TEST
    local_agent = world.create_entity()
    provider_local = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Local OK"))
        ]
    )
    world.add_component(
        local_agent,
        LLMComponent(provider=provider_local, model="fake", system_prompt=""),
    )
    world.add_component(local_agent, ConversationComponent(messages=[]))
    world.add_component(local_agent, ToolRegistryComponent(tools={}, handlers={}))

    local_request = DispatchRequest(
        task_id="local-task",
        wave_number=0,
        sequence_number=0,
        description="Local task",
        expected_output="Output",
        assigned_agent=None,
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    local_result = await executor.execute_dispatch_request(
        world, local_agent, local_request
    )

    # SUBAGENT BACKEND TEST
    subagent_agent = world.create_entity()
    provider_subagent = FakeProvider(responses=[])

    async def mock_delegate(subagent_name: str, task: str) -> str:
        return f"Subagent {subagent_name} OK"

    config = SubagentConfig(
        name="test-sub",
        provider=provider_subagent,
        model="fake",
    )
    world.add_component(
        subagent_agent, SubagentRegistryComponent(subagents={"test-sub": config})
    )
    world.add_component(
        subagent_agent,
        ToolRegistryComponent(
            tools={
                "delegate": ToolSchema(
                    name="delegate", description="Delegate", parameters={}
                )
            },
            handlers={"delegate": mock_delegate},
        ),
    )

    subagent_request = DispatchRequest(
        task_id="subagent-task",
        wave_number=0,
        sequence_number=1,
        description="Subagent task",
        expected_output="Output",
        assigned_agent="test-sub",
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    subagent_result = await executor.execute_dispatch_request(
        world, subagent_agent, subagent_request
    )

    # VERIFY BOTH RETURN SAME CONTRACT
    assert isinstance(local_result, ExecutionResult)
    assert isinstance(subagent_result, ExecutionResult)

    assert local_result.task_id == "local-task"
    assert local_result.backend_type == "local"
    assert local_result.success is True
    assert hasattr(local_result, "result_content")

    assert subagent_result.task_id == "subagent-task"
    assert subagent_result.backend_type == "subagent"
    assert subagent_result.success is True
    assert hasattr(subagent_result, "result_content")


@pytest.mark.asyncio
async def test_local_backend_prompt_context_injection_is_transient() -> None:
    world = World()
    executor = TaskExecutor()
    agent = world.create_entity()

    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Local OK"))
        ]
    )
    world.add_component(
        agent,
        LLMComponent(provider=provider, model="fake", system_prompt=""),
    )
    world.add_component(agent, ConversationComponent(messages=[]))
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(agent, PromptConfigComponent(enable_context_pool=True))
    world.add_component(
        agent,
        OneShotContextPoolComponent(
            items=[(30, 0, "tool:search", "source: tool\nresult: local facts")]
        ),
    )

    request = DispatchRequest(
        task_id="local-context",
        wave_number=0,
        sequence_number=0,
        description="Do local work",
        expected_output="Output",
        assigned_agent=None,
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    _ = await executor.execute_dispatch_request(world, agent, request)

    sent_messages = provider.calls[0]
    assert sent_messages[-1].role == "user"
    assert "[PROMPT_CONTEXT_POOL]" in sent_messages[-1].content
    assert sent_messages[-1].content.endswith("Do local work")

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Do local work"


@pytest.mark.asyncio
async def test_local_backend_retry_reuses_reserved_context_then_commits_on_success() -> (
    None
):
    world = World()
    executor = TaskExecutor()
    agent = world.create_entity()

    provider = FlakyRecordingProvider()
    world.add_component(
        agent,
        LLMComponent(provider=provider, model="fake", system_prompt=""),
    )
    world.add_component(agent, ConversationComponent(messages=[]))
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(agent, PromptConfigComponent(enable_context_pool=True))
    world.add_component(
        agent,
        OneShotContextPoolComponent(
            items=[(30, 0, "tool:search", "source: tool\nresult: local facts")],
            _counter=1,
        ),
    )
    world.add_component(agent, TurnStateComponent(current_turn_id="turn-1"))

    request = DispatchRequest(
        task_id="local-context-retry",
        wave_number=0,
        sequence_number=0,
        description="Do local work",
        expected_output="Output",
        assigned_agent=None,
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    first = await executor.execute_dispatch_request(world, agent, request)
    assert first.success is False

    pool = world.get_component(agent, OneShotContextPoolComponent)
    assert pool is not None
    assert pool.state == "reserved"
    assert pool.items != []

    pool.items.append((20, 1, "subagent:writer", "source: subagent\nresult: draft"))
    pool._counter += 1

    second = await executor.execute_dispatch_request(world, agent, request)
    assert second.success is True

    first_user = provider.calls[0][-1].content
    second_user = provider.calls[1][-1].content
    assert first_user == second_user
    assert "source: subagent" not in second_user

    assert pool.items == []
    assert pool.state == "committed"


@pytest.mark.asyncio
async def test_local_backend_event_trigger_injection_is_transient() -> None:
    world = World()
    executor = TaskExecutor()
    agent = world.create_entity()

    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Local OK"))
        ]
    )
    world.add_component(
        agent,
        LLMComponent(provider=provider, model="fake", system_prompt=""),
    )
    world.add_component(agent, ConversationComponent(messages=[]))
    world.add_component(agent, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        agent,
        PromptConfigComponent(
            trigger_templates={"event:tool_success": "Prefer successful tool outputs"},
            enable_context_pool=True,
        ),
    )
    world.add_component(
        agent,
        OneShotContextPoolComponent(
            items=[
                (
                    30,
                    0,
                    "tool:search",
                    "source: tool:search\nstatus: success\nresult: local facts\nerror: ",
                )
            ]
        ),
    )

    request = DispatchRequest(
        task_id="local-context-event",
        wave_number=0,
        sequence_number=0,
        description="Do local work",
        expected_output="Output",
        assigned_agent=None,
        tools=tuple(),
        context_dependencies=tuple(),
        priority=0,
    )

    _ = await executor.execute_dispatch_request(world, agent, request)

    sent_messages = provider.calls[0]
    assert sent_messages[-1].role == "user"
    assert sent_messages[-1].content.startswith(
        "[PROMPT_INJECT:event:tool_success]\nPrefer successful tool outputs"
    )
    assert sent_messages[-1].content.endswith("Do local work")

    conversation = world.get_component(agent, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Do local work"
