import pytest

from ecs_agent.components import (
    ConversationComponent,
    ContextEntry,
    LLMComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    RenderedUserPromptComponent,
    UserPromptConfigComponent,
    SystemPromptComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.message_assembly import (
    commit_prompt_context_reservation,
    prepare_outbound_messages,
    reserve_prompt_context_reservation,
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


def test_reserve_prompt_context_reuses_existing_reservation_on_retry() -> None:
    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-tool-1",
                priority=30,
                source_label="tool:search",
                content="source: tool\nresult: facts",
                registration_order=0,
            )
        ]
    )

    first = reserve_prompt_context_reservation(
        queue=queue, reservation=None, current_tick=10
    )

    queue.entries.append(
        ContextEntry(
            entry_id="entry-subagent-1",
            priority=20,
            source_label="subagent:writer",
            content="source: subagent\nresult: draft",
            registration_order=1,
        )
    )

    second = reserve_prompt_context_reservation(
        queue=queue,
        reservation=first,
        current_tick=11,
    )

    assert second.reservation_id == first.reservation_id
    assert second.created_at_tick == first.created_at_tick
    assert [entry.entry_id for entry in second.reserved_entries] == ["entry-tool-1"]


def test_commit_prompt_context_reservation_removes_only_matching_entry_id() -> None:
    duplicated_content = "same content, different identity"
    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-1",
                priority=30,
                source_label="tool:a",
                content=duplicated_content,
                registration_order=0,
            ),
            ContextEntry(
                entry_id="entry-2",
                priority=30,
                source_label="tool:b",
                content=duplicated_content,
                registration_order=1,
            ),
        ]
    )
    reservation = PromptContextReservationComponent(
        reservation_id="reservation-1",
        created_at_tick=7,
        reserved_entries=[queue.entries[0]],
    )

    commit_prompt_context_reservation(queue=queue, reservation=reservation)

    assert [entry.entry_id for entry in queue.entries] == ["entry-2"]


def test_reserve_prompt_context_is_isolated_from_post_failure_queue_append() -> None:
    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-initial",
                priority=10,
                source_label="tool:initial",
                content="initial",
                registration_order=0,
            )
        ]
    )
    reservation = reserve_prompt_context_reservation(
        queue=queue,
        reservation=None,
        current_tick=3,
    )

    queue.entries.append(
        ContextEntry(
            entry_id="entry-late",
            priority=99,
            source_label="tool:late",
            content="late",
            registration_order=1,
        )
    )

    assert [entry.entry_id for entry in reservation.reserved_entries] == [
        "entry-initial"
    ]


def test_prepare_outbound_messages_no_context_pool_uses_rendered_user_prompt() -> None:
    world = World()
    entity_id = world.create_entity()
    conversation = ConversationComponent(
        messages=[Message(role="user", content="raw user prompt")]
    )
    world.add_component(entity_id, conversation)
    world.add_component(
        entity_id,
        RenderedUserPromptComponent(text="normalized user prompt", turn_id="turn-1"),
    )

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        system_prompt="system prompt",
        prefix_messages=[Message(role="system", content="prefix context")],
        current_tick=7,
    )

    assert reservation is None
    assert [message.role for message in messages] == ["system", "system", "user"]
    assert messages[-1].content == "normalized user prompt"
    assert conversation.messages[-1].content == "raw user prompt"


def test_prepare_outbound_messages_reserves_reuses_and_commits_context_pool() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Need @code summary")]
        ),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers={"@code": "Use code-first reasoning"},
            enable_context_pool=True,
        ),
    )
    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-1",
                priority=30,
                source_label="tool:search",
                content="source: tool\nstatus: success\nresult: facts",
                registration_order=0,
            )
        ]
    )
    world.add_component(entity_id, queue)

    first_messages, first_reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=10,
    )

    assert first_reservation is not None
    world.add_component(entity_id, first_reservation)
    first_user_text = first_messages[-1].content
    assert "[PROMPT_INJECT:@code]" in first_user_text
    assert "[PROMPT_CONTEXT_POOL]" in first_user_text
    assert "source: tool" in first_user_text

    queue.entries.append(
        ContextEntry(
            entry_id="entry-2",
            priority=20,
            source_label="subagent:writer",
            content="source: subagent\nstatus: success\nresult: draft",
            registration_order=1,
        )
    )

    second_messages, second_reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=11,
    )

    assert second_reservation is not None
    assert second_reservation.reservation_id == first_reservation.reservation_id
    assert second_messages[-1].content == first_user_text
    assert "source: subagent" not in second_messages[-1].content

    commit_prompt_context_reservation(queue=queue, reservation=second_reservation)
    assert [entry.entry_id for entry in queue.entries] == ["entry-2"]


def test_prepare_outbound_messages_uses_existing_stale_reservation_not_live_queue() -> (
    None
):
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))

    stale_entry = ContextEntry(
        entry_id="entry-stale",
        priority=10,
        source_label="tool:stale",
        content="source: stale",
        registration_order=0,
    )
    stale_reservation = PromptContextReservationComponent(
        reservation_id="reservation-stale",
        created_at_tick=1,
        reserved_entries=[stale_entry],
    )
    world.add_component(entity_id, stale_reservation)
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="entry-live",
                    priority=100,
                    source_label="tool:live",
                    content="source: live",
                    registration_order=0,
                )
            ]
        ),
    )

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=50,
    )

    assert reservation is not None
    assert reservation.reservation_id == "reservation-stale"
    assert "source: stale" in messages[-1].content
    assert "source: live" not in messages[-1].content


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
        UserPromptConfigComponent(
            triggers={"@code": "Use code-first reasoning"},
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="entry-tool-1",
                    priority=30,
                    source_label="tool:search",
                    content="source: tool\nresult: facts",
                    registration_order=0,
                )
            ]
        ),
    )

    await ReasoningSystem().process(world)

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    reservation = world.get_component(entity_id, PromptContextReservationComponent)
    assert reservation is not None
    assert [entry.entry_id for entry in reservation.reserved_entries] == [
        "entry-tool-1"
    ]

    queue.entries.append(
        ContextEntry(
            entry_id="entry-subagent-1",
            priority=20,
            source_label="subagent:writer",
            content="source: subagent\nresult: draft",
            registration_order=1,
        )
    )

    await ReasoningSystem().process(world)

    first_user = provider.calls[0][-1].content
    second_user = provider.calls[1][-1].content
    assert first_user == second_user
    assert first_user.startswith("[PROMPT_INJECT:@code]\nUse code-first reasoning")
    assert "[PROMPT_CONTEXT_POOL]" in first_user
    assert first_user.endswith("@code Need summary")
    assert first_user.index("source: tool") < first_user.index("@code Need summary")
    assert "source: subagent" not in second_user

    assert [entry.entry_id for entry in queue.entries] == ["entry-subagent-1"]
    assert world.get_component(entity_id, PromptContextReservationComponent) is None


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
        UserPromptConfigComponent(
            triggers={
                "@code": "Use code-first reasoning",
                "event:tool_success": "Prioritize successful tool evidence",
            },
            enable_context_pool=True,
            context_pool_max_chars=10000,
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptComponent(
            content=(
                "# Markdown System Prompt\n\n"
                "## toolSelection\n\n"
                "Prefer deterministic tool-first synthesis.\n\n"
                "## exploreSection\n\n"
                "Capture concrete evidence from tool outputs.\n\n"
                "## librarianSection\n\n"
                "Reference exact snippets in final answer."
            ),
        ),
    )
    world.add_component(entity_id, PromptContextQueueComponent())

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

    sent_system = provider.calls[0][0]
    assert sent_system.role == "system"
    assert sent_system.content == (
        "# Markdown System Prompt\n\n"
        "## toolSelection\n\n"
        "Prefer deterministic tool-first synthesis.\n\n"
        "## exploreSection\n\n"
        "Capture concrete evidence from tool outputs.\n\n"
        "## librarianSection\n\n"
        "Reference exact snippets in final answer."
    )

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert queue.entries == []
    assert world.get_component(entity_id, PromptContextReservationComponent) is None

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[0].content == "Please @code summarize findings"


@pytest.mark.asyncio
async def test_event_trigger_injection_uses_context_signal_and_preserves_user_tail() -> (
    None
):
    world = World()
    provider = RecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Need concise summary")]
        ),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers={"event:tool_success": "Prefer successful tool context"},
            enable_context_pool=True,
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="entry-tool",
                    priority=30,
                    registration_order=0,
                    source_label="tool:search",
                    content="source: tool:search\nstatus: success\nresult: citations\nerror: ",
                ),
                ContextEntry(
                    entry_id="entry-subagent",
                    priority=20,
                    registration_order=1,
                    source_label="subagent:researcher",
                    content="source: subagent:researcher\nstatus: success\nresult: synthesis\nerror: ",
                ),
            ],
        ),
    )

    await ReasoningSystem().process(world)

    sent_user = provider.calls[0][-1].content
    assert sent_user.startswith(
        "[PROMPT_INJECT:event:tool_success]\nPrefer successful tool context"
    )
    assert sent_user.index("[PROMPT_INJECT:event:tool_success]") < sent_user.index(
        "[PROMPT_CONTEXT_POOL]"
    )
    assert sent_user.index("source: tool:search") < sent_user.index(
        "source: subagent:researcher"
    )
    assert sent_user.endswith("Need concise summary")


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
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="entry-seed",
                    priority=30,
                    registration_order=0,
                    source_label="tool:seed",
                    content="source: tool\nresult: keep",
                )
            ],
        ),
    )

    await ReasoningSystem().process(world)

    sent_user = provider.calls[0][-1].content
    assert sent_user == "Please @code summarize findings"
    assert "[PROMPT_INJECT:" not in sent_user
    assert "[PROMPT_CONTEXT_POOL]" not in sent_user

    queue = world.get_component(entity_id, PromptContextQueueComponent)
    assert queue is not None
    assert len(queue.entries) == 1


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
        UserPromptConfigComponent(
            enable_context_pool=True,
            context_pool_max_chars=max_chars,
        ),
    )
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="entry-keep",
                    priority=30,
                    registration_order=0,
                    source_label="tool:keep",
                    content=keep_entry,
                ),
                ContextEntry(
                    entry_id="entry-drop",
                    priority=20,
                    registration_order=1,
                    source_label="subagent:drop",
                    content="source: subagent:drop\nresult: drop",
                ),
            ],
        ),
    )
    world.add_component(
        entity_id, ToolResultsComponent(results={"result-1": "drop-me"})
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
