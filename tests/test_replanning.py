from __future__ import annotations

import pytest

from ecs_agent.components import (
    ContextEntry,
    ConversationComponent,
    LLMComponent,
    PlanComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    UserPromptConfigComponent,
)
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.systems.replanning import ReplanningSystem
from ecs_agent.types import CompletionResult, Message, PlanRevisedEvent

pytestmark = pytest.mark.asyncio


class RecordingFakeModel(FakeModel):
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


class FlakyRecordingProvider(FakeModel):
    def __init__(self) -> None:
        super().__init__(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant", content='{"revised_steps": ["step 2"]}'
                    )
                )
            ]
        )
        self.calls: list[list[Message]] = []
        self._attempt = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
        self._attempt += 1
        if self._attempt == 1:
            raise RuntimeError("provider exploded")
        return await super().complete(messages, tools=None)


def _create_entity(
    world: World,
    provider: FakeModel,
    *,
    steps: list[str],
    current_step: int,
    completed: bool = False,
    messages: list[Message] | None = None,
) -> int:
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=provider))
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
    provider = RecordingFakeModel(
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
    provider = RecordingFakeModel(
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
    provider = RecordingFakeModel(
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
    provider = FakeModel(
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
    provider = FakeModel(
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
    provider = FakeModel(
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
    provider = FakeModel(
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
    provider = FakeModel(responses=[])
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


async def test_prompt_context_injection_is_transient_for_replanning_provider_call() -> (
    None
):
    world = World()
    provider = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant", content='{"revised_steps": ["step 2"]}'
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2"],
        current_step=1,
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool\nresult: citations",
                )
            ]
        ),
    )

    await ReplanningSystem().process(world)

    sent = provider.calls[0]
    assert sent[-1].role == "user"
    assert "[PROMPT_CONTEXT_POOL]" in sent[-1].content
    assert "source: tool" in sent[-1].content

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "objective"


async def test_replanning_retry_reuses_reserved_context_then_commits_on_success() -> (
    None
):
    world = World()
    provider = FlakyRecordingProvider()
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2"],
        current_step=1,
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool\nresult: citations",
                )
            ]
        ),
    )

    system = ReplanningSystem()
    await system.process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    reservation = world.get_component(entity_id, PromptContextReservationComponent)
    assert queue is not None
    assert reservation is not None
    assert reservation.reserved_entries != []
    reserved_ids = {entry.entry_id for entry in reservation.reserved_entries}

    queue.entries.append(
        ContextEntry(
            entry_id="subagent-writer-1",
            priority=20,
            registration_order=1,
            source_label="subagent:writer",
            content="source: subagent\nresult: draft",
        )
    )

    await system.process(world)

    first_user = provider.calls[0][-1].content
    second_user = provider.calls[1][-1].content
    assert first_user == second_user
    assert "source: subagent" not in second_user

    assert world.get_component(entity_id, PromptContextReservationComponent) is None
    remaining_ids = {entry.entry_id for entry in queue.entries}
    assert reserved_ids.isdisjoint(remaining_ids)
    assert "subagent-writer-1" in remaining_ids


async def test_event_trigger_injection_is_transient_for_replanning_provider_call() -> (
    None
):
    world = World()
    provider = RecordingFakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant", content='{"revised_steps": ["step 2"]}'
                )
            )
        ]
    )
    entity_id = _create_entity(
        world,
        provider,
        steps=["step 1", "step 2"],
        current_step=1,
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[TriggerSpec(pattern="objective", match_mode="keyword", action="inject", content="Prefer successful tool context", priority=0)],
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-search-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool:search\nstatus: success\nresult: citations\nerror: ",
                )
            ]
        ),
    )

    await ReplanningSystem().process(world)

    sent = provider.calls[0]
    assert sent[-1].role == "user"
    assert sent[-1].content.startswith(
        "[PROMPT_INJECT:objective]\nPrefer successful tool context"
    )

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "objective"
