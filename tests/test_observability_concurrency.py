"""Observability concurrency isolation tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent, StreamingComponent
from ecs_agent.core import Runner, World
from ecs_agent.observability import RecordingTelemetrySink, install_observability
from ecs_agent.providers import FakeModel
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, StreamDelta, Usage


class BarrierStreamingFakeModel(FakeModel):
    """Fake streaming model that waits until both concurrent runs are active."""

    def __init__(
        self,
        *,
        responses: list[CompletionResult],
        model_id: str,
        release_event: asyncio.Event,
        entered: list[str],
    ) -> None:
        super().__init__(responses=responses, model_id=model_id)
        self._release_event = release_event
        self._entered = entered

    async def _stream_complete(
        self,
        result: CompletionResult,
    ) -> AsyncIterator[StreamDelta]:
        """Wait for the shared release event, then emit one content delta."""
        self._entered.append(self.model_id)
        if len(self._entered) == 2:
            self._release_event.set()
        await self._release_event.wait()
        yield StreamDelta(content=result.message.content)
        yield StreamDelta(
            finish_reason="stop",
            usage=result.usage,
            response_id=result.response_id,
        )


def _build_streaming_world(
    *,
    model_id: str,
    output: str,
    sink: RecordingTelemetrySink,
    release_event: asyncio.Event,
    entered: list[str],
) -> World:
    world = World(name=model_id)
    install_observability(world, sink)
    entity_id = world.create_entity()
    model = BarrierStreamingFakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content=output),
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                response_id=f"resp-{model_id}",
            )
        ],
        model_id=model_id,
        release_event=release_event,
        entered=entered,
    )
    world.add_component(entity_id, LLMComponent(model=model))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content=f"Hello {model_id}")]),
    )
    world.add_component(entity_id, StreamingComponent(enabled=True))
    world.register_system(ReasoningSystem(), priority=0)
    return world


@pytest.mark.asyncio
async def test_concurrent_runs_have_isolated_trace_ids() -> None:
    """Concurrent Runner.run calls keep run/trace IDs and stream sequences isolated."""
    sink = RecordingTelemetrySink()
    release_event = asyncio.Event()
    entered: list[str] = []
    world_a = _build_streaming_world(
        model_id="model-a",
        output="alpha",
        sink=sink,
        release_event=release_event,
        entered=entered,
    )
    world_b = _build_streaming_world(
        model_id="model-b",
        output="beta",
        sink=sink,
        release_event=release_event,
        entered=entered,
    )

    await asyncio.gather(
        Runner().run(world_a, max_ticks=1),
        Runner().run(world_b, max_ticks=1),
    )

    generations = [record for record in sink.records if record.kind == "generation"]
    assert {record.model for record in generations} == {
        "model-a",
        "model-b",
    }
    assert len({record.run_id for record in generations}) == 2
    assert len({record.trace_id for record in generations}) == 2

    generation_by_model = {record.model: record for record in generations}
    for model_id, generation in generation_by_model.items():
        run_records = [record for record in sink.records if record.run_id == generation.run_id]
        assert {record.trace_id for record in run_records} == {generation.trace_id}
        assert all(record.run_id == generation.run_id for record in run_records)
        stream_records = [record for record in run_records if record.name.startswith("stream.")]
        assert [record.metadata["seq"] for record in stream_records] == [0, 1, 2, 3]
        assert all(record.parent_observation_id == generation.observation_id for record in stream_records)

    assert generation_by_model["model-a"].output == {
        "message": Message(role="assistant", content="alpha"),
        "reasoning_content": None,
        "response_id": "resp-model-a",
    }
    assert generation_by_model["model-b"].output == {
        "message": Message(role="assistant", content="beta"),
        "reasoning_content": None,
        "response_id": "resp-model-b",
    }
