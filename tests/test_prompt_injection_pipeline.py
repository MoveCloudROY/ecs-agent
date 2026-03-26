import pytest

from ecs_agent.components import (
    ConversationComponent,
    ContextEntry,
    LLMComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    PlanComponent,
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
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.systems.planning import PlanningSystem
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
        RenderedUserPromptComponent(text="normalized user prompt"),
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
            triggers=[
                TriggerSpec(
                    pattern="@code",
                    match_mode="keyword",
                    action="skill",
                    content="Use code-first reasoning",
                    priority=0,
                )
            ],
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
            triggers=[
                TriggerSpec(
                    pattern="@code",
                    match_mode="keyword",
                    action="skill",
                    content="Use code-first reasoning",
                    priority=0,
                )
            ],
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
            triggers=[
                TriggerSpec(
                    pattern="@code",
                    match_mode="keyword",
                    action="skill",
                    content="Use code-first reasoning",
                    priority=0,
                )
            ],
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
async def test_keyword_trigger_injection_with_context_pool_preserves_user_tail() -> (
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
            triggers=[
                TriggerSpec(
                    pattern="summary",
                    match_mode="keyword",
                    action="skill",
                    content="Prefer successful tool context",
                    priority=0,
                )
            ],
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
        "[PROMPT_INJECT:summary]\nPrefer successful tool context"
    )
    assert sent_user.index("[PROMPT_INJECT:summary]") < sent_user.index(
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


@pytest.mark.asyncio
async def test_reasoning_uses_prepare_outbound_messages_shared_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    provider = RecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Original user text")]
        ),
    )
    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-1",
                priority=30,
                registration_order=0,
                source_label="tool:search",
                content="source: tool\nresult: evidence",
            )
        ]
    )
    world.add_component(entity_id, queue)
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))

    reservation = PromptContextReservationComponent(
        reservation_id="reservation-shared-path",
        created_at_tick=0,
        reserved_entries=list(queue.entries),
    )

    def fake_prepare_outbound_messages(
        world_obj: World,
        target_entity_id: int,
        *,
        system_prompt: str | None = None,
        prefix_messages: list[Message] | None = None,
        current_tick: int,
        conversation_override: list[Message] | None = None,
    ) -> tuple[list[Message], PromptContextReservationComponent | None]:
        _ = world_obj
        _ = system_prompt
        _ = prefix_messages
        _ = current_tick
        assert target_entity_id == entity_id
        return [
            Message(role="user", content="[shared-path] reasoning payload")
        ], reservation

    monkeypatch.setattr(
        "ecs_agent.systems.reasoning.prepare_outbound_messages",
        fake_prepare_outbound_messages,
        raising=False,
    )

    await ReasoningSystem().process(world)

    assert provider.calls[0][-1].content == "[shared-path] reasoning payload"
    assert world.get_component(entity_id, PromptContextReservationComponent) is None
    assert queue.entries == []


@pytest.mark.asyncio
async def test_planning_uses_prepare_outbound_messages_shared_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    provider = RecordingProvider()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Plan this")]),
    )
    world.add_component(entity_id, PlanComponent(steps=["step one"], current_step=0))
    queue = PromptContextQueueComponent(
        entries=[
            ContextEntry(
                entry_id="entry-1",
                priority=30,
                registration_order=0,
                source_label="tool:search",
                content="source: tool\nresult: evidence",
            )
        ]
    )
    world.add_component(entity_id, queue)
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))

    reservation = PromptContextReservationComponent(
        reservation_id="reservation-plan-shared-path",
        created_at_tick=0,
        reserved_entries=list(queue.entries),
    )

    def fake_prepare_outbound_messages(
        world_obj: World,
        target_entity_id: int,
        *,
        system_prompt: str | None = None,
        prefix_messages: list[Message] | None = None,
        current_tick: int,
        conversation_override: list[Message] | None = None,
    ) -> tuple[list[Message], PromptContextReservationComponent | None]:
        _ = world_obj
        _ = system_prompt
        _ = current_tick
        assert target_entity_id == entity_id
        assert prefix_messages is not None
        return [
            *prefix_messages,
            Message(role="user", content="[shared-path] planning payload"),
        ], reservation

    monkeypatch.setattr(
        "ecs_agent.systems.planning.prepare_outbound_messages",
        fake_prepare_outbound_messages,
        raising=False,
    )

    await PlanningSystem().process(world)

    assert provider.calls[0][-1].content == "[shared-path] planning payload"
    assert world.get_component(entity_id, PromptContextReservationComponent) is None
    assert queue.entries == []


def test_conversation_override_bypasses_world_conversation() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="world user"),
                Message(role="assistant", content="world assistant"),
            ]
        ),
    )

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
        conversation_override=[Message(role="user", content="override user")],
    )

    assert reservation is None
    assert [(message.role, message.content) for message in messages] == [
        ("user", "override user")
    ]


def test_conversation_override_skips_rendered_user_prompt_substitution() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="world user")]),
    )
    world.add_component(entity_id, RenderedUserPromptComponent(text="rendered prompt"))

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
        conversation_override=[Message(role="user", content="override user")],
    )

    assert reservation is None
    assert messages[-1].content == "override user"
    assert "rendered prompt" not in messages[-1].content


def test_conversation_override_applies_config_triggers_inline() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(
                    pattern="@greet",
                    match_mode="keyword",
                    action="skill",
                    content="Be greeting",
                    priority=0,
                )
            ]
        ),
    )

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
        conversation_override=[Message(role="user", content="please @greet now")],
    )

    assert reservation is None
    assert messages[-1].content.startswith("[PROMPT_INJECT:@greet]\nBe greeting")
    assert messages[-1].content.endswith("please @greet now")


def test_conversation_override_no_triggers_when_no_config() -> None:
    world = World()
    entity_id = world.create_entity()

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
        conversation_override=[Message(role="user", content="please @greet now")],
    )

    assert reservation is None
    assert messages[-1].content == "please @greet now"


def test_slash_skill_context_injected_via_prepare_outbound_messages() -> None:
    """Red test: slash skill context WILL be injected into prepare_outbound_messages output.

    When prepare_outbound_messages is called with an entity that has a slash command
    in its last user message, the returned message should contain the injected
    slash skill context before the original text (once implemented).
    """
    from ecs_agent.components import (
        SkillComponent,
        SkillMetadata,
        ConversationComponent,
    )

    world = World()
    entity_id = world.create_entity()

    # Install a skill with slash command on the entity
    skill_metadata = SkillMetadata(
        name="helpskill",
        description="Help skill with documentation",
        tool_names=[],
        has_system_prompt=False,
        user_invocable=True,
        slash_command="/helpskill",
    )
    skill_component = SkillComponent(
        skills={"helpskill": skill_metadata},
    )
    world.add_component(entity_id, skill_component)

    # Add user message with slash command
    original_text = "please /helpskill explain this feature"
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
    )

    assert reservation is None
    # RED ASSERTION: The slash context WILL be injected
    user_msg = messages[-1]
    assert "调用 skill: helpskill" in user_msg.content, (
        "Slash skill context should be injected once feature is implemented"
    )
    # RED ASSERTION: Original text WILL be preserved
    assert user_msg.content.endswith(original_text), (
        f"Final message should end with original text '{original_text}'"
    )


def test_slash_command_via_conversation_override_does_not_auto_inject() -> None:
    from ecs_agent.components import (
        SkillComponent,
        SkillMetadata,
        ToolRegistryComponent,
    )

    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ToolRegistryComponent(tools={}, handlers={}),
    )

    skill_metadata = SkillMetadata(
        name="searchskill",
        description="Search skill",
        tool_names=[],
        has_system_prompt=False,
        user_invocable=True,
        slash_command="/searchskill",
    )
    skill_component = SkillComponent(
        skills={"searchskill": skill_metadata},
    )
    world.add_component(entity_id, skill_component)

    original_text = "please /searchskill for resources about AI"
    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
        conversation_override=[Message(role="user", content=original_text)],
    )

    assert reservation is None
    assert messages[-1].content == original_text
    assert "Skill: searchskill" not in messages[-1].content


def test_context_pool_injection_is_call_time_not_normalization_time() -> None:
    # Contract: ContextPool injection is call-time (prepare_outbound_messages), not normalization-time (UserPromptNormalizationSystem)
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need summary")]),
    )
    world.add_component(entity_id, UserPromptConfigComponent(enable_context_pool=True))
    world.add_component(
        entity_id,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="tool-one-0",
                    priority=30,
                    registration_order=0,
                    source_label="tool:one",
                    content="source: tool:one\nresult: A",
                ),
            ]
        ),
    )
    world.add_component(entity_id, RenderedUserPromptComponent(text="Need summary"))

    # 1. Verify ContextPool content is NOT in RenderedUserPromptComponent
    rendered = world.get_component(entity_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert "source: tool:one" not in rendered.text

    # 2. Verify ContextPool content IS present in final assembled messages
    messages, reservation = prepare_outbound_messages(
        world,
        entity_id,
        current_tick=1,
    )

    assert reservation is not None
    user_msg = messages[-1]
    assert "source: tool:one" in user_msg.content
    assert user_msg.content.endswith("Need summary")


def test_slash_context_is_transient_not_persisted_to_queue() -> None:
    """Hardening test: slash skill context is transient, never persisted to PromptContextQueueComponent.entries.

    When a user message contains a slash command, the injected skill context
    should appear in the outbound message but NOT be permanently stored in
    the queue component. The queue entries remain unchanged after assembly.
    """
    from ecs_agent.components import (
        ConversationComponent,
        ContextEntry,
        PromptContextQueueComponent,
        SkillComponent,
        SkillMetadata,
        UserPromptConfigComponent,
    )
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()

    # Set up skill with slash command
    world.add_component(
        entity,
        SkillComponent(
            skills={
                "docskill": SkillMetadata(
                    name="docskill",
                    description="A documentation skill",
                    tool_names=[],
                    has_system_prompt=False,
                    user_invocable=True,
                    slash_command="/docskill",
                )
            }
        ),
    )

    # Add one context pool entry (reserved)
    initial_entry = ContextEntry(
        entry_id="reserved-ctx-0",
        priority=10,
        registration_order=0,
        source_label="tool:resolver",
        content="RESERVED_CONTEXT_DATA",
    )
    world.add_component(
        entity,
        PromptContextQueueComponent(entries=[initial_entry]),
    )

    world.add_component(
        entity,
        UserPromptConfigComponent(enable_context_pool=True),
    )

    original_text = "/docskill explain this component"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    # Call prepare_outbound_messages
    messages, reservation = prepare_outbound_messages(
        world,
        entity,
        current_tick=1,
    )

    # Verify slash context IS in the outbound message
    user_msg = messages[-1]
    assert "调用 skill: docskill" in user_msg.content, (
        "Slash context should be injected in outbound message"
    )
    assert "RESERVED_CONTEXT_DATA" in user_msg.content, (
        "Reserved context should also be present"
    )
    assert original_text in user_msg.content, (
        "Original slash command text should be preserved"
    )

    # HARDENING: Verify slash context is NOT persisted in the queue component
    queue_after = world.get_component(entity, PromptContextQueueComponent)
    assert queue_after is not None, "Queue component should still exist"
    # The queue should still contain only the original reserved entry, NOT any slash-derived entries
    assert len(queue_after.entries) >= 1, "At least the original entry should remain"

    # Check that no entries mention the slash skill name (i.e., slash context was transient)
    for entry in queue_after.entries:
        assert "docskill" not in entry.content, (
            f"Slash skill context should be transient and not persisted to queue. "
            f"Found 'docskill' in entry {entry.entry_id}: {entry.content[:50]}"
        )


def test_repeated_prepare_outbound_messages_with_slash_produces_stable_context() -> (
    None
):
    """Hardening test: repeated prepare_outbound_messages calls with the same slash command
    produce identical slash context across multiple invocations.

    This ensures slash context synthesis is deterministic and does not depend on
    mutable internal state or non-deterministic ordering.
    """
    from ecs_agent.components import (
        ConversationComponent,
        ContextEntry,
        PromptContextQueueComponent,
        SkillComponent,
        SkillMetadata,
        UserPromptConfigComponent,
    )
    from ecs_agent.prompts.message_assembly import prepare_outbound_messages
    from ecs_agent.types import Message

    world = World()
    entity = world.create_entity()

    # Set up skill with slash command
    world.add_component(
        entity,
        SkillComponent(
            skills={
                "querygen": SkillMetadata(
                    name="querygen",
                    description="A query generation skill",
                    tool_names=["query_builder"],
                    has_system_prompt=False,
                    user_invocable=True,
                    slash_command="/querygen",
                )
            }
        ),
    )

    # Add context pool entry
    world.add_component(
        entity,
        PromptContextQueueComponent(
            entries=[
                ContextEntry(
                    entry_id="ctx-stable-0",
                    priority=20,
                    registration_order=0,
                    source_label="tool:cache",
                    content="STABLE_CONTEXT_DATA",
                )
            ]
        ),
    )

    world.add_component(
        entity,
        UserPromptConfigComponent(enable_context_pool=True),
    )

    original_text = "/querygen build complex SQL"
    world.add_component(
        entity,
        ConversationComponent(messages=[Message(role="user", content=original_text)]),
    )

    # First call to prepare_outbound_messages
    first_messages, first_reservation = prepare_outbound_messages(
        world,
        entity,
        current_tick=1,
    )

    first_user_msg = first_messages[-1]
    first_slash_context_start = first_user_msg.content.find("调用 skill: querygen")
    first_slash_context_end = first_user_msg.content.find("STABLE_CONTEXT_DATA")
    first_slash_context = first_user_msg.content[
        first_slash_context_start:first_slash_context_end
    ]

    # Simulate a retry by installing the reservation and calling again
    if first_reservation is not None:
        world.add_component(entity, first_reservation)

    # Second call to prepare_outbound_messages (simulating a retry)
    second_messages, second_reservation = prepare_outbound_messages(
        world,
        entity,
        current_tick=2,
    )

    second_user_msg = second_messages[-1]
    second_slash_context_start = second_user_msg.content.find("调用 skill: querygen")
    second_slash_context_end = second_user_msg.content.find("STABLE_CONTEXT_DATA")
    second_slash_context = second_user_msg.content[
        second_slash_context_start:second_slash_context_end
    ]

    # HARDENING: Verify both calls produce identical slash context
    assert first_slash_context == second_slash_context, (
        f"Slash context should be stable across repeated calls. "
        f"First: {first_slash_context[:100]}\n"
        f"Second: {second_slash_context[:100]}"
    )

    # Verify the reservation IDs match (same transaction)
    if first_reservation is not None and second_reservation is not None:
        assert first_reservation.reservation_id == second_reservation.reservation_id, (
            "Repeated calls should reuse the same reservation ID"
        )

    # Verify both messages end with the original text
    assert first_user_msg.content.endswith(original_text)
    assert second_user_msg.content.endswith(original_text)
