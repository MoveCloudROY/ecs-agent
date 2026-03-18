import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PromptConfigComponent,
    ToolResultsComponent,
    TurnStateComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.message_assembly import (
    commit_context_pool_reservation,
    reserve_context_pool_items,
)
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.prompt_context_collector import (
    CONTEXT_ENTRY_DELIMITER,
    CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX,
    PromptContextCollectorSystem,
)
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import (
    CompletionResult,
    DelegationCompletedEvent,
    Message,
    ToolExecutionCompletedEvent,
    ToolSchema,
)


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


class RecordingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            responses=[
                CompletionResult(message=Message(role="assistant", content="ok")),
            ]
        )
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> CompletionResult:
        _ = tools
        self.calls.append(list(messages))
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
        ConversationComponent(
            messages=[Message(role="user", content="@code Need summary")]
        ),
    )
    world.add_component(
        entity_id,
        PromptConfigComponent(
            keyword_templates={"@code": "Use code-first reasoning"},
            enable_context_pool=True,
        ),
    )
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
    assert first_user.startswith("[PROMPT_INJECT:@code]\nUse code-first reasoning")
    assert "[PROMPT_CONTEXT_POOL]" in first_user
    assert first_user.endswith("@code Need summary")
    assert first_user.index("source: tool") < first_user.index("@code Need summary")
    assert "source: subagent" not in second_user

    assert pool.items == []
    assert pool.state == "committed"


@pytest.mark.asyncio
async def test_event_collector_feeds_keyword_and_context_injection_end_to_end() -> None:
    world = World()
    provider = RecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Please @code summarize findings")]
        ),
    )
    world.add_component(
        entity_id,
        PromptConfigComponent(
            keyword_templates={"@code": "Use code-first reasoning"},
            enable_context_pool=True,
            context_pool_max_chars=10000,
        ),
    )
    world.add_component(entity_id, OneShotContextPoolComponent())
    world.add_component(entity_id, TurnStateComponent(current_turn_id="turn-ctx-1"))

    collector = PromptContextCollectorSystem()
    await collector.process(world)
    await world.event_bus.publish(
        ToolExecutionCompletedEvent(
            entity_id=entity_id,
            tool_call_id="tool-1",
            result="tool facts",
            success=True,
        )
    )
    await world.event_bus.publish(
        DelegationCompletedEvent(
            entity_id=entity_id,
            subagent_name="researcher",
            result="subagent synthesis",
            success=True,
        )
    )
    await collector.process(world)

    await ReasoningSystem().process(world)

    sent_user = provider.calls[0][-1].content
    assert sent_user.startswith("[PROMPT_INJECT:@code]\nUse code-first reasoning")
    assert sent_user.index("Use code-first reasoning") < sent_user.index(
        "[PROMPT_CONTEXT_POOL]"
    )
    assert sent_user.index("source: tool:tool-1") < sent_user.index(
        "source: subagent:researcher"
    )
    assert sent_user.endswith("Please @code summarize findings")

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert pool.state == "committed"
    assert pool.items == []

    turn_state = world.get_component(entity_id, TurnStateComponent)
    assert turn_state is not None
    assert turn_state.last_injected_turn_id == "turn-ctx-1"
    assert turn_state.current_turn_id == ""

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Please @code summarize findings"


@pytest.mark.asyncio
async def test_non_opt_in_reasoning_path_leaves_user_prompt_and_pool_unchanged() -> (
    None
):
    world = World()
    provider = RecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Please @code summarize findings")]
        ),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(
            items=[(30, 0, "tool:seed", "source: tool\nresult: keep")],
            state="idle",
        ),
    )

    await ReasoningSystem().process(world)

    sent_user = provider.calls[0][-1].content
    assert sent_user == "Please @code summarize findings"
    assert "[PROMPT_INJECT:" not in sent_user
    assert "[PROMPT_CONTEXT_POOL]" not in sent_user

    pool = world.get_component(entity_id, OneShotContextPoolComponent)
    assert pool is not None
    assert pool.state == "idle"
    assert len(pool.items) == 1


@pytest.mark.asyncio
async def test_overflow_footer_is_injected_when_context_pool_entries_are_dropped() -> (
    None
):
    world = World()
    provider = RecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    keep_entry = "source: tool:keep\nresult: keep"
    footer = f"{CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX} dropped_entries=2"
    max_chars = len(keep_entry) + len(CONTEXT_ENTRY_DELIMITER) + len(footer)
    world.add_component(
        entity_id,
        PromptConfigComponent(
            enable_context_pool=True,
            context_pool_max_chars=max_chars,
        ),
    )
    world.add_component(
        entity_id,
        OneShotContextPoolComponent(
            items=[
                (30, 0, "tool:keep", keep_entry),
                (20, 1, "subagent:drop", "source: subagent:drop\nresult: drop"),
            ],
            _counter=2,
        ),
    )
    world.add_component(
        entity_id, ToolResultsComponent(results={"result-1": "drop-me"})
    )
    world.add_component(
        entity_id, TurnStateComponent(current_turn_id="turn-overflow-1")
    )

    collector = PromptContextCollectorSystem()
    await collector.process(world)
    await ReasoningSystem().process(world)

    sent_user = provider.calls[0][-1].content
    assert "[PROMPT_CONTEXT_POOL]" in sent_user
    assert "source: tool:keep" in sent_user
    assert "source: subagent:drop" not in sent_user
    assert "structured_output:result-1" not in sent_user
    assert footer in sent_user
    assert sent_user.endswith("Need summary")
