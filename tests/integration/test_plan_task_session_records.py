"""Plan/task records persist across workflow resets and application restart."""

import pytest

from ecs_agent.components import SessionPolicyComponent, SessionRefComponent, ToolRegistryComponent
from ecs_agent.history import append_conversation_message
from ecs_agent.providers import FakeModel
from ecs_agent.types import Message
from examples.e2e.plan_and_task.controller import PlanController
from examples.e2e.plan_and_task.main import build_plan_task_world, resume_workflow, _reset_workflow_boundary_state
from examples.e2e.plan_and_task.phase_sync import save_state
from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter


async def test_plan_task_enables_record_tools_and_preserves_workflow_evidence(tmp_path):
    world, entity, _, _ = await build_plan_task_world(FakeModel([]), tmp_path)
    assert world.session_service is not None
    tools = world.get_component(entity, ToolRegistryComponent)
    assert {"sessions_list", "sessions_search", "read_file"} <= set(tools.tools)
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="evidence-flow")
    state = await PlanController(world, entity).handle_plan_start(adapter, "session evidence task")
    original = world.get_component(entity, SessionRefComponent).session_id
    append_conversation_message(world, entity, Message(role="assistant", content="original workflow evidence"))
    save_state(world, entity, state, adapter)
    _reset_workflow_boundary_state(world, entity)
    next_id = world.get_component(entity, SessionRefComponent).session_id
    assert next_id != original
    save_state(world, entity, state, adapter)
    assert set(adapter.read_state().record_session_ids) == {original, next_id}
    from examples.e2e.plan_and_task.session_records import close_session_records

    await close_session_records(world)
    resumed, resumed_entity, _, _ = await build_plan_task_world(FakeModel([]), tmp_path)
    try:
        await resume_workflow(resumed, resumed_entity, "evidence-flow", base_dir=tmp_path)
        ref = resumed.get_component(resumed_entity, SessionRefComponent)
        policy = resumed.get_component(resumed_entity, SessionPolicyComponent)
        hits = resumed.session_service.search(ref.session_id, policy, "original workflow evidence").items
        assert len(hits) == 1 and hits[0]["session_id"] == original
        assert (tmp_path / "sessions.sqlite3").is_file()
    finally:
        await close_session_records(resumed)


async def test_workflow_resume_does_not_share_unrelated_task_roots(tmp_path):
    world, entity, _, _ = await build_plan_task_world(FakeModel([]), tmp_path)
    assert world.session_service is not None
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="selected-flow")
    await PlanController(world, entity).handle_plan_start(adapter, "selected task")
    unrelated = world.session_service.create_session(agent_name="unrelated", workspace_id=str(tmp_path.resolve()))
    token = world.session_service.claim_writer(unrelated.session_id)
    world.session_service.append_message(SessionRefComponent(unrelated.session_id), Message(role="user", content="unrelated private evidence"), writer_id=token)
    await resume_workflow(world, entity, "selected-flow", base_dir=tmp_path)
    ref = world.get_component(entity, SessionRefComponent)
    policy = world.get_component(entity, SessionPolicyComponent)
    assert world.session_service.search(ref.session_id, policy, "unrelated private evidence").items == []
    from examples.e2e.plan_and_task.session_records import close_session_records

    await close_session_records(world)
