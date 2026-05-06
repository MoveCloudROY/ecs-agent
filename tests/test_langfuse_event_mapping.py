"""Langfuse-oriented raw LLM, tool, user, and context mapping tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

import pytest

from ecs_agent.components import (
    ConversationComponent,
    ContextBudgetConfig,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PermissionComponent,
    StreamingComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    TerminalComponent,
    ToolRegistryComponent,
    UserInputComponent,
)
from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.core import Runner, World
from ecs_agent.observability import RecordingTelemetrySink, install_observability
from ecs_agent.providers import FakeModel
from ecs_agent.providers.retry_model import RetryModel
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.accounting.models import LLMRetryEvent
from ecs_agent.types import (
    CompletionResult,
    Message,
    RetryConfig,
    StreamDelta,
    SubagentConfig,
    ToolCall,
    ToolSchema,
    Usage,
    UserInputRequestedEvent,
)


class ErrorFakeModel(FakeModel):
    """Fake model that raises on completion."""

    async def complete(
        self,
        messages: list[Message],
        tools: object = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        """Raise a provider error after accepting the request shape."""
        _ = messages
        _ = tools
        _ = stream
        _ = response_format
        raise RuntimeError("provider exploded")


class TwoChunkStreamingFakeModel(FakeModel):
    """Fake model that emits two content deltas plus final usage metadata."""

    async def _stream_complete(
        self,
        result: CompletionResult,
    ) -> AsyncIterator[StreamDelta]:
        """Yield split content so final generation output is assembled once."""
        yield StreamDelta(content="Telemetry ")
        yield StreamDelta(content="ready")
        yield StreamDelta(
            finish_reason="stop",
            usage=result.usage,
            response_id=result.response_id,
        )


class ReasoningAndContentStreamingFakeModel(FakeModel):
    """Fake model that emits reasoning deltas before content deltas."""

    async def _stream_complete(
        self,
        result: CompletionResult,
    ) -> AsyncIterator[StreamDelta]:
        """Yield interleaved reasoning and content chunks."""
        yield StreamDelta(reasoning_content="thinking-1")
        yield StreamDelta(reasoning_content="thinking-2")
        yield StreamDelta(content="answer-1")
        yield StreamDelta(content="answer-2")
        yield StreamDelta(
            finish_reason="stop",
            usage=result.usage,
            response_id=result.response_id,
        )


class ExplodingStreamingFakeModel(FakeModel):
    """Fake model that emits a partial chunk and then fails."""

    async def _stream_complete(
        self,
        result: CompletionResult,
    ) -> AsyncIterator[StreamDelta]:
        """Yield one content chunk then raise."""
        _ = result
        yield StreamDelta(content="partial")
        raise RuntimeError("stream exploded")


class FlakyRetryModel:
    """Model that fails twice before succeeding for retry mapping tests."""

    def __init__(self) -> None:
        self.model_id = "flaky-retry-model"
        self.provider_id = "flaky-provider"
        self.attempts = 0

    async def complete(
        self,
        messages: list[Message],
        tools: object = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        del messages, tools, stream, response_format
        self.attempts += 1
        if self.attempts < 3:
            response = httpx.Response(500, request=httpx.Request("POST", "https://example.invalid"))
            raise httpx.HTTPStatusError("retry me", request=response.request, response=response)
        return CompletionResult(message=Message(role="assistant", content="retry success"))


class ExhaustingRetryModel:
    """Model that always fails so retry exhaustion can be observed."""

    def __init__(self) -> None:
        self.model_id = "exhausting-retry-model"
        self.provider_id = "exhausting-provider"
        self.attempts = 0

    async def complete(
        self,
        messages: list[Message],
        tools: object = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        del messages, tools, stream, response_format
        self.attempts += 1
        response = httpx.Response(500, request=httpx.Request("POST", "https://example.invalid"))
        raise httpx.HTTPStatusError("still failing", request=response.request, response=response)


class EmitRetryEventSystem:
    """System that emits a retry event against the current run."""

    async def process(self, world: World) -> None:
        """Publish a retry event and terminate the run."""
        await world.event_bus.publish(
            LLMRetryEvent(
                provider_id="manual-provider",
                model="manual-model",
                reason="request_error",
                attempt=2,
            )
        )
        world.add_component(world.create_entity(), TerminalComponent(reason="done"))


class DirectLLMStreamingSystem:
    """System that exercises the accounting instrumentation helper directly."""

    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.entity_id: int | None = None

    async def process(self, world: World) -> None:
        """Call the LLM helper and consume the returned stream."""
        entity_id = world.create_entity()
        self.entity_id = int(entity_id)
        result = await complete_with_llm_invocation_event(
            event_bus=world.event_bus,
            entity_id=entity_id,
            model=self.model,
            messages=[Message(role="user", content="Hello Langfuse")],
            operation="direct_helper",
            stream=True,
        )
        if isinstance(result, CompletionResult):
            raise RuntimeError("expected stream result")
        async for _ in result:
            pass
        world.add_component(entity_id, TerminalComponent(reason="direct_helper_done"))


def _tool_schema(name: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=f"Run {name}",
        parameters={"type": "object"},
    )


@pytest.mark.asyncio
async def test_llm_generation_includes_raw_messages_and_output() -> None:
    """Completed LLM observations map raw prompts and assistant output."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Telemetry ready"),
                usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                response_id="resp-langfuse-success",
                reasoning_content="observed reasoning",
            )
        ],
        model_id="fake-generation-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello Langfuse")]),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation.name == "llm.reasoning"
    assert generation.status == "success"
    assert generation.model == "fake-generation-model"
    assert generation.start_time is not None
    assert generation.end_time is not None
    assert generation.latency_ms is not None
    assert generation.latency_ms == pytest.approx(
        (generation.end_time - generation.start_time).total_seconds() * 1000
    )
    assert generation.input == {
        "messages": [Message(role="user", content="Hello Langfuse")],
        "tools": None,
        "streaming": False,
    }
    assert generation.output == {
        "message": Message(role="assistant", content="Telemetry ready"),
        "reasoning_content": "observed reasoning",
        "response_id": "resp-langfuse-success",
    }
    assert generation.metadata is not None
    assert generation.metadata["provider_id"] == "FakeModel"
    assert generation.metadata["operation"] == "reasoning"
    assert generation.metadata["response_id"] == "resp-langfuse-success"
    assert generation.usage_details == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    assert generation.cost_details == {}


@pytest.mark.asyncio
async def test_llm_generation_error_records_status_and_error_message() -> None:
    """Provider errors map to generation errors and keep agent error behavior."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = ErrorFakeModel(responses=[], model_id="fake-error-model")
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello Langfuse")]),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation.status == "error"
    assert generation.error == "provider exploded"
    assert generation.input == {
        "messages": [Message(role="user", content="Hello Langfuse")],
        "tools": None,
        "streaming": False,
    }
    assert generation.output is None
    assert generation.usage_details == {}
    assert generation.cost_details == {}

    error = world.get_component(entity_id, ErrorComponent)
    assert error is not None
    assert error.error == "provider exploded"
    assert error.system_name == "ReasoningSystem"


@pytest.mark.asyncio
async def test_streaming_generation_records_final_output_once() -> None:
    """Streaming generation maps assembled final output without duplicating deltas."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = TwoChunkStreamingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="ignored by stream"),
                usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                response_id="resp-stream-final",
            )
        ],
        model_id="fake-stream-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello Langfuse")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation.input == {
        "messages": [Message(role="user", content="Hello Langfuse")],
        "tools": None,
        "streaming": True,
    }
    assert generation.output == {
        "message": Message(role="assistant", content="Telemetry ready"),
        "reasoning_content": None,
        "response_id": "resp-stream-final",
    }
    assert [record for record in sink.records if record.output == generation.output] == [
        generation
    ]


@pytest.mark.asyncio
async def test_streaming_deltas_are_ordered_without_final_output_duplication() -> None:
    """Stream deltas map as ordered child telemetry under the active generation."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = ReasoningAndContentStreamingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="ignored by stream"),
                usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                response_id="resp-stream-ordered",
            )
        ],
        model_id="fake-stream-order-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello Langfuse")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generation = next(record for record in sink.records if record.kind == "generation")
    stream_records = [
        record
        for record in sink.records
        if record.run_id == generation.run_id and record.name.startswith("stream.")
    ]

    assert [record.name for record in stream_records] == [
        "stream.start",
        "stream.reasoning.delta",
        "stream.reasoning.delta",
        "stream.reasoning.end",
        "stream.content.start",
        "stream.content.delta",
        "stream.content.delta",
        "stream.end",
    ]
    assert all(record.parent_observation_id == generation.observation_id for record in stream_records)
    assert [record.metadata["seq"] for record in stream_records] == list(range(8))
    assert generation.output == {
        "message": Message(role="assistant", content="answer-1answer-2"),
        "reasoning_content": "thinking-1thinking-2",
        "response_id": "resp-stream-ordered",
    }


@pytest.mark.asyncio
async def test_streaming_error_after_partial_output_keeps_stream_end_error() -> None:
    """Stream errors after partial output still close the stream with error status."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = ExplodingStreamingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="ignored by stream"),
                response_id="resp-stream-error",
            )
        ],
        model_id="fake-stream-error-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello Langfuse")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generation = next(record for record in sink.records if record.kind == "generation")
    stream_end = next(record for record in sink.records if record.name == "stream.end")
    assert stream_end.status == "error"
    assert stream_end.parent_observation_id == generation.observation_id
    assert generation.status == "error"
    assert generation.output is None


@pytest.mark.asyncio
async def test_retry_attempts_are_mapped_with_reason_and_status() -> None:
    """Retry attempts map under the active generation with provider/model metadata."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = RetryModel(model=FlakyRetryModel(), retry_config=RetryConfig(max_attempts=3, min_wait=0, multiplier=0, max_wait=0))
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Retry please")]),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generation = next(record for record in sink.records if record.kind == "generation")
    retry_records = [record for record in sink.records if record.name == "llm.retry"]
    assert len(retry_records) == 2
    assert all(record.parent_observation_id == generation.observation_id for record in retry_records)
    assert [record.metadata["attempt"] for record in retry_records] == [1, 2]
    assert [record.metadata["status"] for record in retry_records] == ["retrying", "retrying"]
    assert [record.metadata["provider_id"] for record in retry_records] == ["flaky-provider", "flaky-provider"]
    assert [record.metadata["model"] for record in retry_records] == ["flaky-retry-model", "flaky-retry-model"]
    assert [record.metadata["reason"] for record in retry_records] == ["http_500", "http_500"]
    assert generation.output == {"message": Message(role="assistant", content="retry success"), "reasoning_content": None, "response_id": None}


@pytest.mark.asyncio
async def test_retry_exhaustion_maps_trace_level_retry_event_and_generation_error() -> None:
    """Retry exhaustion still emits retry telemetry before the generation fails."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = RetryModel(model=ExhaustingRetryModel(), retry_config=RetryConfig(max_attempts=2, min_wait=0, multiplier=0, max_wait=0))
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Retry please")]),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generation = next(record for record in sink.records if record.kind == "generation")
    retry_records = [record for record in sink.records if record.name == "llm.retry"]
    assert len(retry_records) == 1
    assert retry_records[0].parent_observation_id == generation.observation_id
    assert retry_records[0].metadata["attempt"] == 1
    assert retry_records[0].metadata["status"] == "retrying"
    assert generation.status == "error"
    assert generation.error == "still failing"


@pytest.mark.asyncio
async def test_retry_event_without_active_generation_maps_to_trace_level_event() -> None:
    """Retry events without an active generation fall back to the run trace."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    world.register_system(EmitRetryEventSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    retry_record = next(record for record in sink.records if record.name == "llm.retry")
    trace_record = next(record for record in sink.records if record.kind == "trace")
    assert retry_record.parent_observation_id == trace_record.observation_id
    assert retry_record.metadata == {
        "provider_id": "manual-provider",
        "model": "manual-model",
        "reason": "request_error",
        "attempt": 2,
        "status": "retrying",
    }


@pytest.mark.asyncio
async def test_llm_generation_missing_usage_maps_to_empty_usage_details() -> None:
    """Missing provider usage maps to empty usage details without errors."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="No usage"))],
        model_id="fake-no-usage-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hello Langfuse")]),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 1
    assert generations[0].usage_details == {}
    assert generations[0].cost_details == {}


@pytest.mark.asyncio
async def test_llm_generation_records_context_pressure_without_provider_usage() -> None:
    """Generation metadata includes prompt pressure even without provider usage."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="No usage"))],
        model_id="fake-pressure-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ContextBudgetConfig(
            max_tokens=128,
            prune_tool_results=False,
            prune_reasoning=True,
            token_estimation_chars_per_token=2.0,
            overflow_behavior="warn",
        ),
    )
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="system", content="system prompt"),
                Message(role="user", content="Hello Langfuse"),
            ]
        ),
    )
    world.register_system(ReasoningSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 1
    metadata = generations[0].metadata
    assert metadata is not None
    assert metadata["message_count"] == 2
    assert metadata["prompt_char_count"] == len("system prompt") + len("Hello Langfuse")
    assert metadata["estimated_prompt_tokens"] == 7
    assert metadata["provider_prompt_tokens"] is None
    assert metadata["context_budget"] == {
        "max_tokens": 128,
        "prune_tool_results": False,
        "prune_reasoning": True,
        "token_estimation_chars_per_token": 2.0,
        "overflow_behavior": "warn",
    }


@pytest.mark.asyncio
async def test_initial_user_messages_are_recorded_at_run_start() -> None:
    """Existing conversation user messages are mapped as user-input events."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="system", content="ignore"),
                Message(role="user", content="Initial question"),
            ]
        ),
    )
    world.add_component(entity_id, TerminalComponent(reason="done"))

    await Runner().run(world, max_ticks=1)

    user_records = [
        record for record in sink.records if record.kind == "event" and record.name == "user.input"
    ]
    assert len(user_records) == 1
    assert user_records[0].entity_id == int(entity_id)
    assert user_records[0].input == {"message": Message(role="user", content="Initial question")}
    assert user_records[0].metadata == {"source": "initial_conversation"}


@pytest.mark.asyncio
async def test_user_input_received_records_resolved_text() -> None:
    """Resolved UserInputSystem text is mapped without changing conversation append."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()
    world.add_component(entity_id, UserInputComponent(prompt="Name?"))
    world.add_component(entity_id, ConversationComponent(messages=[]))

    async def provide_input(event: UserInputRequestedEvent) -> None:
        event.input_future.set_result("Alice")

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)
    world.register_system(UserInputSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    user_records = [
        record for record in sink.records if record.kind == "event" and record.name == "user.input"
    ]
    assert len(user_records) == 1
    assert user_records[0].entity_id == int(entity_id)
    assert user_records[0].input == {"text": "Alice"}
    assert user_records[0].metadata == {"prompt": "Name?", "source": "user_input_system"}

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages == [Message(role="user", content="Alice")]


@pytest.mark.asyncio
async def test_tool_success_records_arguments_result_and_latency() -> None:
    """Tool success maps raw arguments, result, status, and latency."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()

    async def lookup_weather(city: str) -> str:
        return f"sunny in {city}"

    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"lookup_weather": _tool_schema("lookup_weather")},
            handlers={"lookup_weather": lookup_weather},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="weather-1",
                    name="lookup_weather",
                    arguments={"city": "Paris"},
                )
            ]
        ),
    )
    world.register_system(ToolExecutionSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    tool_records = [record for record in sink.records if record.kind == "tool"]
    assert len(tool_records) == 1
    tool_record = tool_records[0]
    assert tool_record.name == "tool.lookup_weather"
    assert tool_record.status == "success"
    assert tool_record.entity_id == int(entity_id)
    assert tool_record.input == {
        "tool_call_id": "weather-1",
        "tool_name": "lookup_weather",
        "arguments": {"city": "Paris"},
    }
    assert tool_record.output == {"result": "sunny in Paris"}
    assert tool_record.start_time is not None
    assert tool_record.end_time is not None
    assert tool_record.latency_ms is not None
    assert tool_record.latency_ms >= 0
    assert tool_record.latency_ms == pytest.approx(
        (tool_record.end_time - tool_record.start_time).total_seconds() * 1000
    )


@pytest.mark.asyncio
async def test_tool_error_records_arguments_and_error_result() -> None:
    """Tool handler errors map as error tool observations with raw arguments."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()

    async def exploding_tool(city: str) -> str:
        _ = city
        raise RuntimeError("boom")

    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"lookup_weather": _tool_schema("lookup_weather")},
            handlers={"lookup_weather": exploding_tool},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="weather-boom",
                    name="lookup_weather",
                    arguments={"city": "Paris"},
                )
            ]
        ),
    )
    world.register_system(ToolExecutionSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    tool_records = [record for record in sink.records if record.kind == "tool"]
    assert len(tool_records) == 1
    assert tool_records[0].status == "error"
    assert tool_records[0].input == {
        "tool_call_id": "weather-boom",
        "tool_name": "lookup_weather",
        "arguments": {"city": "Paris"},
    }
    assert tool_records[0].output == {
        "result": "Error executing tool 'lookup_weather': boom"
    }
    assert tool_records[0].error == "Error executing tool 'lookup_weather': boom"


@pytest.mark.asyncio
async def test_unknown_tool_records_error_observation() -> None:
    """Unknown tools map as error tool observations with attempted arguments."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="missing-1",
                    name="does_not_exist",
                    arguments={"query": "x"},
                )
            ]
        ),
    )
    world.register_system(ToolExecutionSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    tool_records = [record for record in sink.records if record.kind == "tool"]
    assert len(tool_records) == 1
    assert tool_records[0].name == "tool.does_not_exist"
    assert tool_records[0].status == "error"
    assert tool_records[0].input == {
        "tool_call_id": "missing-1",
        "tool_name": "does_not_exist",
        "arguments": {"query": "x"},
    }
    assert tool_records[0].error == "Error: unknown tool 'does_not_exist'"


@pytest.mark.asyncio
async def test_tool_permission_denied_records_error_observation() -> None:
    """Permission denial maps to an error tool observation with the reason."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="deny-1", name="bash", arguments={"cmd": "pwd"})]
        ),
    )
    world.add_component(entity_id, PermissionComponent(denied_tools=["bash"]))
    world.register_system(PermissionSystem(), priority=0)

    await Runner().run(world, max_ticks=1)

    tool_records = [record for record in sink.records if record.kind == "tool"]
    assert len(tool_records) == 1
    assert tool_records[0].name == "tool.bash"
    assert tool_records[0].status == "error"
    assert tool_records[0].input == {"tool_call_id": "deny-1", "tool_name": "bash"}
    assert tool_records[0].error is not None
    assert "denied by permission policy" in tool_records[0].error


@pytest.mark.asyncio
async def test_cached_tool_result_records_artifact_observation(tmp_path) -> None:
    """Tool result caching maps artifact metadata without changing cache behavior."""
    scratchbook_root = tmp_path / ".scratchbook"
    registry = ArtifactRegistry(root=scratchbook_root)
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    entity_id = world.create_entity()

    async def verbose_tool() -> str:
        return "cached payload " * 80

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="run the tool")]),
    )
    world.add_component(
        entity_id,
        ContextBudgetConfig(
            max_tokens=5,
            token_estimation_chars_per_token=1.0,
            overflow_behavior="warn",
        ),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={"verbose_tool": _tool_schema("verbose_tool")},
            handlers={"verbose_tool": verbose_tool},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="overflow-1", name="verbose_tool", arguments={})]
        ),
    )
    world.register_system(ToolExecutionSystem(registry=registry), priority=0)

    await Runner().run(world, max_ticks=1)

    cache_records = [
        record for record in sink.records if record.kind == "tool" and record.name == "tool.cache"
    ]
    assert len(cache_records) == 1
    cache_record = cache_records[0]
    assert cache_record.status == "success"
    assert cache_record.input == {"tool_call_id": "overflow-1"}
    assert cache_record.output is not None
    assert "scratchbook/records/tool/tool_" in cache_record.output["artifact_path"]


@pytest.mark.asyncio
async def test_complete_with_llm_invocation_event_streaming_maps_final_output_once() -> None:
    """The direct instrumentation helper maps consumed streams once."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    model = TwoChunkStreamingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="ignored by stream"),
                usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                response_id="resp-direct-stream",
            )
        ],
        model_id="fake-direct-stream-model",
    )
    system = DirectLLMStreamingSystem(model)
    world.register_system(system, priority=0)

    await Runner().run(world, max_ticks=1)

    generations = [record for record in sink.records if record.kind == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation.name == "llm.direct_helper"
    assert generation.entity_id == system.entity_id
    assert generation.input == {
        "messages": [Message(role="user", content="Hello Langfuse")],
        "tools": None,
        "streaming": True,
    }
    assert generation.output == {
        "message": Message(role="assistant", content="Telemetry ready"),
        "reasoning_content": None,
        "response_id": "resp-direct-stream",
    }
    assert [record for record in sink.records if record.output == generation.output] == [
        generation
    ]


@pytest.mark.asyncio
async def test_subagent_run_records_span_with_child_generation() -> None:
    """A subagent run is a trace span and its internal LLM call is a child generation."""
    world = World()
    sink = RecordingTelemetrySink()
    install_observability(world, sink)
    parent_model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-subagent",
                            name="subagent",
                            arguments={"category": "worker", "prompt": "Do work"},
                        )
                    ],
                ),
                usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
            ),
            CompletionResult(
                message=Message(role="assistant", content="Parent done"),
                usage=Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
            ),
        ],
        model_id="parent-model",
    )
    child_model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Child done"),
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        ],
        model_id="child-model",
    )
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=parent_model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Delegate work")]),
    )
    world.add_component(entity_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        entity_id,
        SubagentRegistryComponent(
            subagents={
                "worker": SubagentConfig(
                    name="worker",
                    model=child_model,
                    system_prompt="You are a worker.",
                    max_ticks=2,
                )
            }
        ),
    )
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))
    world.register_system(SubagentSystem(priority=-1), priority=-1)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)

    await Runner().run(world, max_ticks=3)

    traces = [record for record in sink.records if record.kind == "trace"]
    assert len(traces) == 1
    subagent_spans = [
        record
        for record in sink.records
        if record.kind == "span" and record.name == "subagent.worker"
    ]
    assert len(subagent_spans) == 1
    subagent_span = subagent_spans[0]
    assert subagent_span.parent_observation_id == traces[0].observation_id
    assert subagent_span.status == "success"
    assert subagent_span.input == {
        "category": "worker",
        "prompt": "Do work",
    }
    assert subagent_span.output == {"result": "Child done"}

    child_generation = next(
        record for record in sink.records if record.model == "child-model"
    )
    assert child_generation.kind == "generation"
    assert child_generation.trace_id == subagent_span.trace_id
    assert child_generation.parent_observation_id == subagent_span.observation_id
