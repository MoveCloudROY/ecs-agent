from __future__ import annotations

import pytest

from ecs_agent.components import ConversationComponent, LLMComponent, PlanComponent
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.types import CompletionResult, Message, PlanRevisedEvent

pytestmark = pytest.mark.asyncio


class RecordingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
        return await super().complete(messages, tools=None)


def _create_entity(
    world: World,
    provider: FakeProvider,
    *,
    steps: list[str],
    current_step: int,
    completed: bool = False,
    messages: list[Message] | None = None,
) -> int:
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=messages
            or [
                Message(role="user", content="objective"),
                Message(role="assistant", content="finished first step"),
            ]
        ),
    )
    world.add_component(
        entity_id,
        PlanComponent(steps=steps, current_step=current_step, completed=completed),
    )
    return entity_id


async def test_replanning_skip_completed_plan() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant", content='{"revised_steps": ["unused"]}'
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2"],
        current_step=1,
        completed=True,
    )

    await ReplanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == ["step 1", "step 2"]
    assert provider.calls == []


async def test_replanning_skip_no_completed_steps() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant", content='{"revised_steps": ["unused"]}'
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2"],
        current_step=0,
    )

    await ReplanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == ["step 1", "step 2"]
    assert provider.calls == []


async def test_replanning_skip_already_replanned() -> None:
    world = World()
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["step 2", "step 3"]}',
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2", "step 3"],
        current_step=1,
    )
    system = ReplanningSystem()

    await system.process(world)
    await system.process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == ["step 1", "step 2", "step 3"]
    assert len(provider.calls) == 1


async def test_replanning_revises_steps() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["new step 2", "new step 3"]}',
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "old step 2", "old step 3"],
        current_step=1,
    )

    await ReplanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == ["step 1", "new step 2", "new step 3"]


async def test_replanning_publishes_event() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["new step 2", "new step 3"]}',
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "old step 2", "old step 3"],
        current_step=1,
    )

    seen: list[PlanRevisedEvent] = []

    async def handler(event: PlanRevisedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(PlanRevisedEvent, handler)

    await ReplanningSystem().process(world)

    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].old_steps == ["step 1", "old step 2", "old step 3"]
    assert seen[0].new_steps == ["step 1", "new step 2", "new step 3"]


async def test_replanning_no_event_when_steps_unchanged() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content='{"revised_steps": ["step 2", "step 3"]}',
                )
            )
        ]
    )
    _ = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2", "step 3"],
        current_step=1,
    )

    seen: list[PlanRevisedEvent] = []

    async def handler(event: PlanRevisedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(PlanRevisedEvent, handler)

    await ReplanningSystem().process(world)

    assert seen == []


async def test_replanning_graceful_on_invalid_json() -> None:
    world = World()
    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="this is not json")
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2", "step 3"],
        current_step=1,
    )

    await ReplanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == ["step 1", "step 2", "step 3"]


async def test_replanning_graceful_on_provider_exhausted() -> None:
    world = World()
    provider = FakeProvider(responses=[])
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2", "step 3"],
        current_step=1,
    )

    await ReplanningSystem().process(world)

    plan = world.get_component(entity_id, PlanComponent)
    assert plan is not None
    assert plan.steps == ["step 1", "step 2", "step 3"]


async def test_parse_revised_steps_extracts_json_from_text() -> None:
    content = (
        "analysis complete\n"
        'result: {"revised_steps": ["new step 2", "new step 3"]}\n'
        "done"
    )

    revised = ReplanningSystem._parse_revised_steps(content)

    assert revised == ["new step 2", "new step 3"]


async def test_parse_revised_steps_returns_none_for_empty() -> None:
    revised = ReplanningSystem._parse_revised_steps("")

    assert revised is None
