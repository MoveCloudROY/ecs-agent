"""Real compaction and trim preserve evidence and commit summaries before mutation."""

import pytest

from ecs_agent.components import (
    CompactionConfigComponent, ContextTrimConfig, ConversationComponent,
    CurrentCompactionSummaryComponent, LLMComponent, SessionPolicyComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeModel
from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall


def setup(messages):
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent(messages=messages))
    world.add_component(entity, LLMComponent(model=FakeModel([CompletionResult(message=Message(role="assistant", content="durable summary"))])))
    world.add_component(entity, CompactionConfigComponent(threshold_tokens=1))
    service = SessionService(InMemorySessionStore())
    ref = install_session_recording(world, entity, service, workspace_id="test", agent_name="coder", register_tools=False)
    return world, entity, service, ref


async def test_actual_compaction_retains_original_and_persists_summary():
    world, entity, service, ref = setup([Message(role="user", content="task"), Message(role="assistant", content="original evidence " * 500)])
    await CompactionSystem().process(world)
    assert world.get_component(entity, CurrentCompactionSummaryComponent).summary == "durable summary"
    assert len(service.search(ref.session_id, SessionPolicyComponent(), "original evidence").items) == 1
    records = list(service.store.records(ref.session_id))
    assert records[-1].payload["summary"] == "durable summary"
    assert records[-1].payload["previous_message_count"] == 2
    service.close()


async def test_compaction_write_failure_keeps_previous_summary_and_context(monkeypatch):
    world, entity, service, _ = setup([Message(role="user", content="task"), Message(role="assistant", content="original evidence " * 500)])
    before = list(world.get_component(entity, ConversationComponent).messages)

    def fail(*args, **kwargs):
        raise OSError("cannot persist summary")

    monkeypatch.setattr(service.store, "append", fail)
    with pytest.raises(OSError):
        await CompactionSystem().process(world)
    assert world.get_component(entity, CurrentCompactionSummaryComponent) is None
    assert world.get_component(entity, ConversationComponent).messages == before
    service.close()


async def test_actual_trim_retains_tool_evidence_in_records():
    world, entity, service, ref = setup([
        Message(role="user", content="old"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="old_call", name="run", arguments={})]),
        Message(role="tool", content="trimmed tool evidence " * 500, tool_call_id="old_call"),
        Message(role="user", content="continue"),
    ])
    world.add_component(entity, ContextTrimConfig(max_tokens=30, protect_recent_turns=0))
    await CompactionSystem().process(world)
    assert not any(message.role == "tool" for message in world.get_component(entity, ConversationComponent).messages)
    assert len(service.search(ref.session_id, SessionPolicyComponent(), "trimmed tool evidence").items) == 1
    assert any(record.payload.get("reason") == "compaction_trim" for record in service.store.records(ref.session_id))
    service.close()
