"""Opt-in Responses API coverage of typed tools and managed multi-run turns.

Select a profile with --live; see docs/testing.md.
"""

import asyncio
from typing import Literal

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PermissionComponent,
    ToolApprovalComponent,
)
from ecs_agent.core import AgentHandle, RunStatus, World
from ecs_agent.providers._message_contract import tool_span_error
from ecs_agent.sessions import (
    InMemorySessionStore,
    SessionService,
    install_session_recording,
    release_session_writer,
)
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools import register_tools, tool
from ecs_agent.types import ApprovalPolicy, RunCompletedEvent

from tests.support.models import LiveModelFactory


@pytest.mark.live(
    formats=["openai_responses"], capabilities=["text", "tools", "streaming"]
)
@pytest.mark.authoring
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", ["allow", "permission_deny", "approval_deny"])
async def test_responses_typed_tools_budget_resume_and_next_turn(
    stream: bool, policy: str, *, live_model_factory: LiveModelFactory
) -> None:
    model = live_model_factory.create()
    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    executed = []

    @tool(
        description="Compute a sum or count of integer values in batches. This is an in-memory test tool."
    )
    async def tally(
        batches: list[dict[str, int]],
        mode: Literal["sum", "count"],
        note: str | None,
        label: str = "default",
    ) -> str:
        executed.append((batches, mode, note, label))
        return "TALLY_RESULT=5"

    register_tools(world, entity, [tally])
    if policy == "permission_deny":
        world.add_component(entity, PermissionComponent(allowed_tools=[]))
    elif policy == "approval_deny":
        world.add_component(
            entity, ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY)
        )
    service = SessionService(InMemorySessionStore())
    ref = install_session_recording(
        world,
        entity,
        service,
        workspace_id="p2-live",
        agent_name="typed-probe",
        register_tools=False,
    )
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)
    agent = AgentHandle(world, entity)
    prompt = (
        'Call tally exactly once with batches=[{"a":2,"b":3}], mode="sum", note=null. '
        "Omit label so its Python default applies. You must call the actual tool. "
        "After its result, give a short final answer. If denied, acknowledge that and stop; never retry."
    )
    envelopes = []
    try:
        async with asyncio.timeout(live_model_factory.profile.timeouts.case_seconds):
            if stream:
                async with agent.stream_turn(prompt, max_steps=1) as events:
                    async for envelope in events:
                        envelopes.append(envelope)
                first = agent.last_result
            else:
                first = await agent.run_turn(prompt, max_steps=1)
            assert first is not None and first.status is RunStatus.MAX_TICKS
            if stream:
                async with agent.stream_resume(max_steps=3) as events:
                    async for envelope in events:
                        envelopes.append(envelope)
                resumed = agent.last_result
            else:
                resumed = await agent.resume(max_steps=3)
            assert resumed is not None and resumed.success, resumed
            assert first.turn_id == resumed.turn_id and first.run_id != resumed.run_id
            expected = (
                [([{"a": 2, "b": 3}], "sum", None, "default")]
                if policy == "allow"
                else []
            )
            assert executed == expected
            second = await agent.run_turn(
                "Reply exactly SECOND_TURN. Do not call any tool.", max_steps=2
            )
            assert second.success and second.turn_id != first.turn_id
            assert executed == expected
            conversation = world.get_component(entity, ConversationComponent)
            assert (
                conversation is not None
                and tool_span_error(conversation.messages) is None
            )
            inputs = [
                message for message in conversation.messages if message.role == "user"
            ]
            assert len(inputs) == 2
            calls = [
                call
                for message in conversation.messages
                for call in message.tool_calls or []
            ]
            assert len(calls) == 1 and calls[0].name == "tally"
            for message in conversation.messages:
                assert message.turn_id in {first.turn_id, second.turn_id}
                if message.role == "assistant":
                    assert message.provider_output is not None
                    assert message.provider_output.protocol == "openai_responses"
            records = list(service.store.records(ref.session_id))
            assert (
                len(
                    [
                        record
                        for record in records
                        if record.kind == "message"
                        and record.payload.get("role") == "user"
                    ]
                )
                == 2
            )
            assert all(
                record.payload.get("turn_id")
                for record in records
                if record.kind in {"message", "run_started", "run_finished"}
            )
            if stream:
                finished = [
                    item
                    for item in envelopes
                    if isinstance(item.event, RunCompletedEvent)
                ]
                assert len(finished) == 2
                assert {item.turn_id for item in envelopes} == {first.turn_id}
                assert {item.run_id for item in envelopes} == {
                    first.run_id,
                    resumed.run_id,
                }
    finally:
        release_session_writer(world, entity)
        service.close()
