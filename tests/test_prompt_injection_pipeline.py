import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PromptConfigComponent,
    TurnStateComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.message_assembly import (
    commit_context_pool_reservation,
    reserve_context_pool_items,
)
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, ToolSchema


class FlakyRecordingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            responses=[
                CompletionResult(message=Message(role="assistant", content="ok"))
            ]
        )
        self.calls: list[list[Message]] = []
        self._attempt = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
        self._attempt += 1
        if self._attempt == 1:
            raise RuntimeError("provider exploded")
        return await super().complete(messages, tools)


def test_reserve_context_pool_reuses_existing_reservation_for_retry() -> None:
    pool = OneShotContextPoolComponent(
        items=[(30, 0, "tool:search", "source: tool\nresult: facts")],
        _counter=1,
    )

    first = reserve_context_pool_items(pool=pool, turn_id="turn-1")

    pool.items.append((20, 1, "subagent:writer", "source: subagent\nresult: draft"))
    pool._counter += 1
    second = reserve_context_pool_items(pool=pool, turn_id="turn-1")

    assert first == second
    assert len(second) == 1
    assert "source: tool" in second[0][3]


def test_commit_context_pool_is_idempotent() -> None:
    pool = OneShotContextPoolComponent(
        items=[(30, 0, "tool:search", "source: tool\nresult: facts")],
        _counter=1,
    )

    _ = reserve_context_pool_items(pool=pool, turn_id="turn-1")
    commit_context_pool_reservation(pool=pool, turn_id="turn-1")

    assert pool.items == []
    first_state = pool.state

    commit_context_pool_reservation(pool=pool, turn_id="turn-1")

    assert pool.items == []
    assert pool.state == first_state


@pytest.mark.asyncio
async def test_retry_uses_reserved_payload_then_commit_clears_once_on_success() -> None:
    world = World()
    provider = FlakyRecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    world.add_component(entity_id, PromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(
            items=[(30, 0, "tool:search", "source: tool\nresult: facts")],
            _counter=1,
        ),
    )
    world.add_component(entity_id, TurnStateComponent(current_turn_id="turn-1"))

    await ReasoningSystem().process(world)

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert pool.state == "reserved"
    assert pool.items != []

    pool.items.append((20, 1, "subagent:writer", "source: subagent\nresult: draft"))
    pool._counter += 1

    await ReasoningSystem().process(world)

    first_user = provider.calls[0][-1].content
    second_user = provider.calls[1][-1].content
    assert first_user == second_user
    assert "source: subagent" not in second_user

    assert pool.items == []
    assert pool.state == "committed"
