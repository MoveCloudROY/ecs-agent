"""Shared record services for sync, background, nested, and sibling agents."""

import asyncio
import json

import pytest

from ecs_agent.components import (
    ChildStubComponent, ConversationComponent, LLMComponent, SessionPolicyComponent,
    SessionRefComponent, SubagentRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.history import append_conversation_message
from ecs_agent.providers import FakeModel
from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.subagent_runtime import reset_global_scheduler
from ecs_agent.types import CompletionResult, Message, SubagentConfig


def root():
    world = World()
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent(messages=[Message(role="user", content="root task")]))
    world.add_component(entity, LLMComponent(model=FakeModel([])))
    service = SessionService(InMemorySessionStore())
    install_session_recording(world, entity, service, workspace_id="project", agent_name="main", register_tools=False)
    return world, entity, service


async def test_sync_child_has_own_persistent_records_and_stub_reference():
    world, entity, service = root()
    config = SubagentConfig(name="coder", model=FakeModel([
        CompletionResult(message=Message(role="assistant", content="sync evidence")),
    ]))
    world.add_component(entity, SubagentRegistryComponent(subagents={"coder": config}))
    result = await SubagentSystem()._make_subagent_handler(world, entity)(category="coder", prompt="test request")
    assert result == "sync evidence"
    parent = world.get_component(entity, SessionRefComponent)
    children = service.list_sessions(parent.session_id, SessionPolicyComponent(), parent_session_id=parent.session_id).items
    assert len(children) == 1
    child_id = children[0]["session_id"]
    assert children[0]["status"] == "completed"
    assert len(service.search(parent.session_id, SessionPolicyComponent(), "sync evidence").items) == 1
    stubs = world.query(ChildStubComponent, SessionRefComponent)
    assert len(stubs) == 1
    assert stubs[0][1][1].session_id == child_id
    # Child release must not close the service or parent's writer.
    append_conversation_message(world, entity, Message(role="assistant", content="parent still active"))
    service.close()


def test_nested_child_and_siblings_share_tree_but_have_distinct_writers():
    world, entity, service = root()
    builder = ChildWorldBuilder()
    child, child_id = builder.assemble_child_world(world, entity, SubagentConfig(name="coder", model=FakeModel([])))
    sibling, sibling_id = builder.assemble_child_world(world, entity, SubagentConfig(name="reviewer", model=FakeModel([])))
    nested, nested_id = builder.assemble_child_world(child, child_id, SubagentConfig(name="tester", model=FakeModel([])))
    append_conversation_message(nested, nested_id, Message(role="user", content="nested unique evidence"))
    sibling_ref = sibling.get_component(sibling_id, SessionRefComponent)
    child_ref = child.get_component(child_id, SessionRefComponent)
    nested_ref = nested.get_component(nested_id, SessionRefComponent)
    assert service.store.get_session(nested_ref.session_id).parent_session_id == child_ref.session_id
    assert len(service.search(sibling_ref.session_id, SessionPolicyComponent(), "nested unique evidence").items) == 1
    assert len({r.session_id for r in service.store.sessions()}) == 4
    assert child.session_service is sibling.session_service is nested.session_service is service
    service.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_background_has_transcript_identity_before_execution_and_retains_it(stream):
    reset_global_scheduler()
    world, entity, service = root()
    entered = asyncio.Event()
    release = asyncio.Event()

    class WaitingModel(FakeModel):
        async def complete(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return await super().complete(*args, **kwargs)

    model = WaitingModel([CompletionResult(message=Message(role="assistant", content="background evidence"))])
    world.add_component(entity, SubagentRegistryComponent(subagents={"coder": SubagentConfig(name="coder", model=model)}))
    system = SubagentSystem()
    response = json.loads(await system._make_subagent_handler(world, entity)(category="coder", prompt="background task", background=True, stream=stream))
    try:
        transcript_id = response["transcript_session_id"]
        assert service.store.get_session(transcript_id).parent_session_id == world.get_component(entity, SessionRefComponent).session_id
        await asyncio.wait_for(entered.wait(), timeout=3)
        release.set()
        task = await system._runtime_manager.get_task(response["session_id"])
        await asyncio.wait_for(asyncio.shield(task), timeout=3)
        assert service.store.get_session(transcript_id).status == "completed"
        metadata = await system._runtime_manager.get_session(response["session_id"])
        assert metadata.transcript_session_id == transcript_id
        assert system._session_payload(metadata, status="success")["transcript_session_id"] == transcript_id
        assert len(service.store.sessions()) == 2
    finally:
        release.set()
        await system._runtime_manager.cancel_session(response["session_id"])
        task = await system._runtime_manager.get_task(response["session_id"])
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        service.close()
        reset_global_scheduler()


async def test_queued_cancellation_updates_durable_status_without_running_child():
    from ecs_agent.systems.subagent.tools import make_cancel_handler

    reset_global_scheduler()
    world, entity, service = root()
    entered = asyncio.Event()
    release = asyncio.Event()

    class WaitingModel(FakeModel):
        async def complete(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return await super().complete(*args, **kwargs)

    world.add_component(entity, SubagentRegistryComponent(subagents={
        "blocker": SubagentConfig(name="blocker", model=WaitingModel([CompletionResult(message=Message(role="assistant", content="done"))])),
        "queued": SubagentConfig(name="queued", model=FakeModel([CompletionResult(message=Message(role="assistant", content="must not run"))])),
    }))
    system = SubagentSystem(max_background_concurrency=1)
    handler = system._make_subagent_handler(world, entity)
    blocker = json.loads(await handler(category="blocker", prompt="wait", background=True))
    await asyncio.wait_for(entered.wait(), 3)
    queued = json.loads(await handler(category="queued", prompt="queued task", background=True))
    try:
        assert service.store.get_session(queued["transcript_session_id"]).status == "queued"
        await make_cancel_handler(system, world, entity)(queued["session_id"])
        assert service.store.get_session(queued["transcript_session_id"]).status == "cancelled"
        records = list(service.store.records(queued["transcript_session_id"]))
        assert not any(record.payload.get("content") == "must not run" for record in records)
        assert records[-1].kind == "run_finished"
    finally:
        release.set()
        await system._runtime_manager.cancel_session(blocker["session_id"])
        task = await system._runtime_manager.get_task(blocker["session_id"])
        if task:
            await asyncio.gather(task, return_exceptions=True)
        service.close()
        reset_global_scheduler()


@pytest.mark.parametrize("outcome", ["cancelled", "timeout", "error"])
async def test_background_terminal_statuses_match_query_catalog(outcome):
    reset_global_scheduler()
    world, entity, service = root()
    entered = asyncio.Event()

    class InterruptedModel(FakeModel):
        async def complete(self, *args, **kwargs):
            entered.set()
            if outcome == "error":
                raise RuntimeError("provider failed")
            await asyncio.Event().wait()

    world.add_component(entity, SubagentRegistryComponent(subagents={"coder": SubagentConfig(name="coder", model=InterruptedModel([]))}))
    system = SubagentSystem()
    payload = json.loads(await system._make_subagent_handler(world, entity)(category="coder", prompt="test", background=True, timeout=0.03 if outcome == "timeout" else None))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if outcome == "cancelled":
            await system._runtime_manager.cancel_session(payload["session_id"])
            await system._runtime_manager.sync_to_component(world, entity)
        task = await system._runtime_manager.get_task(payload["session_id"])
        if task:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 3)
        assert service.store.get_session(payload["transcript_session_id"]).status == outcome
    finally:
        await system._runtime_manager.cancel_session(payload["session_id"])
        service.close()
        reset_global_scheduler()


async def test_resume_creates_new_transcript_linked_to_original():
    from ecs_agent.types import FinishReason

    reset_global_scheduler()

    world, entity, service = root()
    model = FakeModel([
        CompletionResult(message=Message(role="assistant", content="unfinished"), finish_reason=FinishReason.LENGTH),
        CompletionResult(message=Message(role="assistant", content="second run completed")),
    ])
    world.add_component(entity, SubagentRegistryComponent(subagents={"coder": SubagentConfig(name="coder", model=model)}))
    system = SubagentSystem()
    first = json.loads(await system._make_subagent_handler(world, entity)(category="coder", prompt="same task", background=True))
    task = await system._runtime_manager.get_task(first["session_id"])
    await asyncio.wait_for(task, 3)
    new_control_id = await system._resume_session(world, entity, first["session_id"])
    task = await system._runtime_manager.get_task(new_control_id)
    await asyncio.wait_for(task, 3)
    metadata = await system._runtime_manager.get_session(new_control_id)
    assert metadata.transcript_session_id != first["transcript_session_id"]
    assert service.store.get_session(metadata.transcript_session_id).source_session_id == first["transcript_session_id"]
    assert service.store.get_session(first["transcript_session_id"]).status == "incomplete"
    assert service.store.get_session(metadata.transcript_session_id).status == "completed"
    service.close()
    reset_global_scheduler()


async def test_child_setup_failure_records_error_and_releases_writer(monkeypatch):
    world, entity, service = root()
    world.add_component(entity, SubagentRegistryComponent(subagents={"coder": SubagentConfig(name="coder", model=FakeModel([]))}))
    system = SubagentSystem()

    def fail(**kwargs):
        raise RuntimeError("child plugin setup failed")

    monkeypatch.setattr(system._delegation_executor, "install_child_observability", fail)
    result = await system._make_subagent_handler(world, entity)(category="coder", prompt="task")
    assert "child plugin setup failed" in result
    parent = world.get_component(entity, SessionRefComponent)
    child = service.list_sessions(parent.session_id, SessionPolicyComponent(), parent_session_id=parent.session_id).items[0]
    assert child["status"] == "error"
    token = service.claim_writer(child["session_id"])
    service.store.release_writer(child["session_id"], token)
    service.close()


async def test_running_child_append_keeps_existing_transcript_pages_stable():
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.sessions.resources import read_session_resource
    from ecs_agent.types import InheritancePolicy, ToolCall, ToolSchema

    reset_global_scheduler()
    world, entity, service = root()
    waiting = asyncio.Event()
    proceed = asyncio.Event()

    class Model(FakeModel):
        async def complete(self, *args, **kwargs):
            if self._index == 1:
                waiting.set()
                await proceed.wait()
            return await super().complete(*args, **kwargs)

    async def probe():
        return "probe completed"

    world.add_component(entity, ToolRegistryComponent(tools={"probe": ToolSchema(name="probe", description="test", parameters={"type": "object"})}, handlers={"probe": probe}))
    model = Model([
        CompletionResult(message=Message(role="assistant", content="stable old line\n" * 180, tool_calls=[ToolCall(id="probe_call", name="probe", arguments={})])),
        CompletionResult(message=Message(role="assistant", content="newly appended final evidence")),
    ])
    world.add_component(entity, SubagentRegistryComponent(subagents={"coder": SubagentConfig(name="coder", model=model, inheritance_policy=InheritancePolicy(inherit_tools=["probe"]))}))
    system = SubagentSystem()
    payload = json.loads(await system._make_subagent_handler(world, entity)(category="coder", prompt="test", background=True))
    try:
        await asyncio.wait_for(waiting.wait(), 3)
        info = service.store.get_session(payload["transcript_session_id"])
        assert info.status == "running"
        uri = service.transcript_uri(info)
        caller = world.get_component(entity, SessionRefComponent).session_id
        first = read_session_resource(service, caller, SessionPolicyComponent(), uri, limit=20, line_numbers=True)
        next_offset = json.loads(first.split("\n---\n")[0])["next_offset"]
        assert next_offset is not None
        before_next = read_session_resource(service, caller, SessionPolicyComponent(), uri, offset=next_offset, limit=20)
        proceed.set()
        task = await system._runtime_manager.get_task(payload["session_id"])
        await asyncio.wait_for(task, 3)
        assert first == read_session_resource(service, caller, SessionPolicyComponent(), uri, limit=20, line_numbers=True)
        assert before_next == read_session_resource(service, caller, SessionPolicyComponent(), uri, offset=next_offset, limit=20)
        latest = service.transcript_uri(service.store.get_session(info.session_id))
        assert latest != uri
        assert "newly appended final evidence" in read_session_resource(service, caller, SessionPolicyComponent(), latest, offset=175, limit=100)
    finally:
        proceed.set()
        await system._runtime_manager.cancel_session(payload["session_id"])
        service.close()
        reset_global_scheduler()


async def test_restored_running_background_session_is_recorded_as_failed():
    from ecs_agent.components import SubagentSessionTableComponent
    from ecs_agent.types import SubagentSessionRecord

    reset_global_scheduler()

    world, entity, service = root()
    parent = world.get_component(entity, SessionRefComponent)
    info = service.create_session(agent_name="coder", workspace_id="project", parent_session_id=parent.session_id)
    writer = service.claim_writer(info.session_id)
    service.append(SessionRefComponent(info.session_id), "run_started", {"status": "running"}, writer_id=writer)
    service.store.release_writer(info.session_id, writer)
    metadata = SubagentSessionRecord(session_id="ses_restored", category="coder", prompt="lost running task", parent_entity_id=entity, created_at="2026-10-19T00:00:00Z", updated_at="2026-10-19T00:00:00Z", status="running", transcript_session_id=info.session_id)
    world.add_component(entity, SubagentSessionTableComponent(sessions={metadata.session_id: metadata}))
    system = SubagentSystem()
    await system._reconcile_restored_sessions(world, entity, SubagentRegistryComponent())
    assert service.store.get_session(info.session_id).status == "error"
    assert metadata.error == "restored_without_live_task_handle"
    service.close()
    reset_global_scheduler()


async def test_two_level_delegation_and_sibling_tool_read_use_real_callers(tmp_path):
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.tools.builtins import BuiltinToolsSkill
    from ecs_agent.tools.context import current_tool_context
    from ecs_agent.types import InheritancePolicy, ToolCall, ToolSchema

    world, entity, service = root()
    read_schema, read_handler = BuiltinToolsSkill().bind_workspace(str(tmp_path)).tools()["read_file"]

    async def delegate_deeper():
        context = current_tool_context()
        context.world.add_component(context.entity_id, SubagentRegistryComponent(subagents={
            "tester": SubagentConfig(name="tester", model=FakeModel([CompletionResult(message=Message(role="assistant", content="deep unique test evidence"))])),
        }))
        return await SubagentSystem()._make_subagent_handler(context.world, context.entity_id)(category="tester", prompt="nested test")

    world.add_component(entity, ToolRegistryComponent(
        tools={"read_file": read_schema, "delegate_deeper": ToolSchema(name="delegate_deeper", description="Delegate nested test", parameters={"type": "object"})},
        handlers={"read_file": read_handler, "delegate_deeper": delegate_deeper},
    ))
    install_session_recording(world, entity, service, workspace_id="project", agent_name="main")
    coder = SubagentConfig(name="coder", model=FakeModel([
        CompletionResult(message=Message(role="assistant", content="delegate", tool_calls=[ToolCall(id="nested", name="delegate_deeper", arguments={})])),
        CompletionResult(message=Message(role="assistant", content="coder complete")),
    ]), inheritance_policy=InheritancePolicy(inherit_tools=["delegate_deeper"]))
    registry = SubagentRegistryComponent(subagents={"coder": coder})
    world.add_component(entity, registry)
    system = SubagentSystem()
    assert await system._make_subagent_handler(world, entity)(category="coder", prompt="code and delegate") == "coder complete"
    parent = world.get_component(entity, SessionRefComponent)
    coder_info = next(info for info in service.store.sessions() if info.agent_name == "coder")
    nested_info = next(info for info in service.store.sessions() if info.agent_name == "tester")
    assert nested_info.parent_session_id == coder_info.session_id
    hit = service.search(parent.session_id, SessionPolicyComponent(), "deep unique test evidence", agent_name="tester").items[0]
    registry.subagents["reviewer"] = SubagentConfig(name="reviewer", model=FakeModel([
        CompletionResult(message=Message(role="assistant", content="find evidence", tool_calls=[ToolCall(id="search_nested", name="sessions_search", arguments={"query": "deep unique test evidence", "agent_name": "tester"})])),
        CompletionResult(message=Message(role="assistant", content="read evidence", tool_calls=[ToolCall(id="read_nested", name="read_file", arguments={"file_path": hit["read_uri"]})])),
        CompletionResult(message=Message(role="assistant", content=f"Reviewed {hit['session_id']} {hit['record_id']}")),
    ]), inheritance_policy=InheritancePolicy(inherit_tools=["read_file", "sessions_search"]))
    review = await system._make_subagent_handler(world, entity)(category="reviewer", prompt="review nested test")
    assert nested_info.session_id in review
    reviewer_info = next(info for info in service.store.sessions() if info.agent_name == "reviewer")
    read_result = next(record for record in service.store.records(reviewer_info.session_id) if record.payload.get("tool_call_id") == "read_nested" and record.payload.get("role") == "tool")
    assert "deep unique test evidence" in read_result.payload["content"]
    assert nested_info.session_id in read_result.payload["content"]
    service.close()
