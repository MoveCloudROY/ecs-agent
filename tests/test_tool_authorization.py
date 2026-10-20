"""End-to-end contracts for mandatory tool authorization and ordered results."""

import asyncio
import json
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent, PendingToolCallsComponent, PermissionComponent,
    SubagentRegistryComponent, ToolApprovalComponent, ToolRegistryComponent,
)
from ecs_agent.core import Runner
from ecs_agent.history import append_conversation_message
from ecs_agent.dsl import compile_agent_specs, load_json_agents, load_markdown_agent, resolve_agent_specs
from ecs_agent.providers._message_contract import tool_span_error
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.tool_approval import ToolApprovalSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    ApprovalPolicy, InheritancePolicy, Message, SubagentConfig, ToolApprovalRequestedEvent,
    ToolCall, ToolDeniedEvent, ToolExecutionStartedEvent,
)
from tests.authorization_support import scenario


@pytest.mark.parametrize("priority", [None, -10, 1, 5, 10])
@pytest.mark.parametrize("restriction", ["permission", "approval"])
async def test_model_generated_calls_cannot_bypass_policy(priority: int | None, restriction: str) -> None:
    case = scenario()
    if restriction == "permission":
        case.world.add_component(case.entity, PermissionComponent(allowed_tools=["read"]))
        preflight = PermissionSystem()
    else:
        case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY))
        preflight = ToolApprovalSystem()
    if priority is not None:
        case.world.register_system(preflight, priority=priority)
    case.world.register_system(ToolExecutionSystem(), priority=5)
    outcome = await Runner().run(case.world, max_ticks=4, target_entity=case.entity)
    assert outcome.success
    assert case.called == ([("read", 1)] if restriction == "permission" else [])
    conversation = case.world.get_component(case.entity, ConversationComponent)
    assert conversation is not None
    assert tool_span_error(conversation.messages) is None
    assert [message.tool_call_id for message in conversation.messages if message.role == "tool"] == ["read-1", "write-1"]
    denials = [event for event in case.events if isinstance(event, ToolDeniedEvent)]
    starts = [event for event in case.events if isinstance(event, ToolExecutionStartedEvent)]
    assert len(denials) == (1 if restriction == "permission" else 2)
    assert len(starts) == len(case.called)


@pytest.mark.parametrize("reply", ["allow", "deny", "failure", "allow_then_failure", "timeout", "no_handler"])
async def test_required_approval_failures_never_execute(reply: str) -> None:
    case = scenario([ToolCall(id="call", name="read", arguments={})])
    case.world.add_component(case.entity, ToolApprovalComponent(
        policy=ApprovalPolicy.REQUIRE_APPROVAL, timeout=0.02 if reply == "timeout" else None,
    ))
    requests = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event)
        if reply == "timeout":
            await asyncio.sleep(1)
        elif reply == "failure":
            raise RuntimeError("approval UI failed")
        else:
            event.approval_future.set_result(reply in {"allow", "allow_then_failure"})
            if reply == "allow_then_failure":
                raise RuntimeError("failed after answering")
    if reply != "no_handler":
        case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    case.world.register_system(ToolExecutionSystem(), priority=5)
    result = await asyncio.wait_for(Runner().run(case.world, max_ticks=4), 1)
    assert result.success
    assert case.called == ([("read", 0)] if reply == "allow" else [])
    assert len(requests) == (0 if reply == "no_handler" else 1)
    assert all(event.approval_future.done() for event in requests)


async def test_concurrent_preflights_and_executors_share_approval_and_result() -> None:
    case = scenario([ToolCall(id="call", name="read", arguments={})])
    case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))
    requests = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event)
        await asyncio.sleep(0.01)
        event.approval_future.set_result(True)
    case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    for system in (PermissionSystem(), ToolApprovalSystem(), ToolApprovalSystem(), ToolExecutionSystem(), ToolExecutionSystem()):
        case.world.register_system(system, priority=5)
    assert (await Runner().run(case.world, max_ticks=4)).success
    assert len(requests) == 1
    assert case.called == [("read", 0)]
    conversation = case.world.get_component(case.entity, ConversationComponent)
    assert conversation is not None
    assert tool_span_error(conversation.messages) is None
    assert len([message for message in conversation.messages if message.role == "tool"]) == 1


@pytest.mark.parametrize("change", ["arguments", "permission", "approval_policy", "approval_component"])
async def test_preapproved_call_is_rechecked_after_changes(change: str) -> None:
    case = scenario([ToolCall(id="call", name="read", arguments={"value": 1})])
    approval = ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL)
    case.world.add_component(case.entity, approval)
    requests = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event.tool_call.arguments.copy())
        event.approval_future.set_result(True)
    case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    await ReasoningSystem().process(case.world)
    await ToolApprovalSystem().process(case.world)
    pending = case.world.get_component(case.entity, PendingToolCallsComponent)
    assert pending is not None
    if change == "arguments":
        pending.tool_calls[0].arguments["value"] = 2
    elif change == "permission":
        case.world.add_component(case.entity, PermissionComponent(allowed_tools=[]))
    elif change == "approval_policy":
        approval.policy = ApprovalPolicy.ALWAYS_DENY
    else:
        case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))
    await ToolExecutionSystem().process(case.world)
    assert case.called == ({"arguments": [("read", 2)], "approval_component": [("read", 1)]}.get(change, []))
    assert len(requests) == (2 if change in {"arguments", "approval_component"} else 1)


async def test_changed_arguments_during_approval_require_a_new_decision() -> None:
    case = scenario([ToolCall(id="call", name="read", arguments={"value": 1})])
    case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))
    requests = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event.tool_call.arguments["value"])
        if len(requests) == 1:
            pending = case.world.get_component(case.entity, PendingToolCallsComponent)
            pending.tool_calls[0].arguments["value"] = 2
        event.approval_future.set_result(len(requests) == 1)
    case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    case.world.register_system(ToolExecutionSystem(), priority=5)
    assert (await Runner().run(case.world, max_ticks=4)).success
    assert requests == [1, 2]
    assert case.called == []


async def test_cancellation_during_approval_never_starts_a_handler() -> None:
    case = scenario([ToolCall(id="call", name="read", arguments={})])
    case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL, timeout=None))
    requested = asyncio.Event()
    futures = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        futures.append(event.approval_future)
        requested.set()
    case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    case.world.register_system(ToolExecutionSystem(), priority=5)
    task = asyncio.create_task(Runner().run(case.world, max_ticks=4))
    await asyncio.wait_for(requested.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert case.called == []
    assert futures[0].cancelled()
    assert not any(isinstance(event, ToolExecutionStartedEvent) for event in case.events)


@pytest.mark.parametrize("format_name", ["json", "markdown"])
@pytest.mark.parametrize("tools,expected", [(None, ["read", "write"]), ({}, []), ({"write": False}, []), ({"read": True, "write": False}, ["read"])])
async def test_loaded_dsl_enforces_permissions_on_dynamically_installed_tools(tmp_path: Path, format_name: str, tools: dict[str, bool] | None, expected: list[str]) -> None:
    case = scenario()
    config = {"name": "main", "mode": "primary", "model": "fake", "prompt": "test"}
    if tools is not None:
        config["tools"] = tools
    if format_name == "json":
        path = tmp_path / "agents.json"
        path.write_text(json.dumps({"main": config}))
        specs = load_json_agents(path)
    else:
        path = tmp_path / "main.md"
        import yaml
        path.write_text("---\n" + yaml.safe_dump(config) + "---\ntest")
        specs = [load_markdown_agent(path)]
    entity, world = compile_agent_specs(resolve_agent_specs(specs), lambda *_: case.model)
    world.add_component(entity, case.world.get_component(case.entity, ConversationComponent))
    world.add_component(entity, case.world.get_component(case.entity, ToolRegistryComponent))
    world.register_system(ReasoningSystem(), priority=0)
    world.register_system(ToolExecutionSystem(), priority=5)
    assert (await Runner().run(world, max_ticks=4)).success
    assert [name for name, _ in case.called] == expected
    assert tool_span_error(world.get_component(entity, ConversationComponent).messages) is None


@pytest.mark.parametrize("version", [None, 1, 2])
async def test_checkpoint_empty_allowlist_migration_preserves_versioned_behavior(version: int | None) -> None:
    case = scenario()
    case.world.add_component(case.entity, PermissionComponent(allowed_tools=[], denied_tools=["write"]))
    data = WorldSerializer.to_dict(case.world)
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    registry = case.world.get_component(case.entity, ToolRegistryComponent)
    restored = WorldSerializer.from_dict(data, {"fake": case.model}, registry.handlers)
    restored.register_system(ReasoningSystem(), priority=0)
    restored.register_system(ToolExecutionSystem(), priority=5)
    from ecs_agent.core import AgentHandle

    handle = AgentHandle(restored, case.entity)
    handle.adopt_turn()
    assert (await handle.resume(max_steps=4)).success
    assert case.called == ([] if version == 2 else [("read", 1)])


async def test_checkpoint_restoration_does_not_reuse_approved_call_ids() -> None:
    case = scenario([ToolCall(id="call", name="read", arguments={})])
    case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))
    requests = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event)
        event.approval_future.set_result(len(requests) == 1)
    case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    await ReasoningSystem().process(case.world)
    await ToolApprovalSystem().process(case.world)
    data = WorldSerializer.to_dict(case.world)
    registry = case.world.get_component(case.entity, ToolRegistryComponent)
    restored = WorldSerializer.from_dict(data, {"fake": case.model}, registry.handlers)
    restored.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    await ToolExecutionSystem().process(restored)
    assert len(requests) == 2
    assert case.called == []
    assert tool_span_error(restored.get_component(case.entity, ConversationComponent).messages) is None


@pytest.mark.parametrize("parent_allow,local,expected", [(None, None, ["read", "write"]), ([], None, []), (None, {}, []), (["read"], {"read": True, "write": True}, ["read"])])
async def test_child_world_enforces_inherited_and_local_permissions(parent_allow: list[str] | None, local: dict[str, bool] | None, expected: list[str]) -> None:
    case = scenario()
    case.world.add_component(case.entity, PermissionComponent(allowed_tools=parent_allow))
    config = SubagentConfig(name="child", model=case.model, tools=local,
        inheritance_policy=InheritancePolicy(inherit_tools=["read", "write"], inherit_permissions=True))
    # Round-trip the config too: policy and inheritance must survive checkpoints.
    case.world.add_component(case.entity, SubagentRegistryComponent(subagents={"child": config}))
    registry = case.world.get_component(case.entity, ToolRegistryComponent)
    restored = WorldSerializer.from_dict(WorldSerializer.to_dict(case.world), {"fake": case.model}, registry.handlers)
    config = restored.get_component(case.entity, SubagentRegistryComponent).subagents["child"]
    child, entity = ChildWorldBuilder().assemble_child_world(restored, case.entity, config)
    append_conversation_message(child, entity, Message(role="user", content="Use the tools."))
    assert (await Runner().run(child, max_ticks=4, target_entity=entity)).success
    assert [name for name, _ in case.called] == expected
    assert tool_span_error(child.get_component(entity, ConversationComponent).messages) is None


@pytest.mark.parametrize("tools", [{}, {"read": False}, {"read": True, "write": False}])
async def test_dsl_child_policy_reaches_actual_child_execution(tools: dict[str, bool]) -> None:
    from ecs_agent.dsl import AgentSpec

    case = scenario()
    entity, parent = compile_agent_specs({
        "main": AgentSpec(mode="primary", model="fake", prompt="main"),
        "child": AgentSpec(mode="subagent", model="fake", prompt="child", tools=tools),
    }, lambda *_: case.model)
    parent.add_component(entity, case.world.get_component(case.entity, ToolRegistryComponent))
    config = parent.get_component(entity, SubagentRegistryComponent).subagents["child"]
    config.inheritance_policy.inherit_tools = ["read", "write"]
    child, child_entity = ChildWorldBuilder().assemble_child_world(parent, entity, config)
    append_conversation_message(child, child_entity, Message(role="user", content="Use tools."))
    assert (await Runner().run(child, max_ticks=4, target_entity=child_entity)).success
    assert [name for name, _ in case.called] == [name for name in ["read", "write"] if tools.get(name, False)]


@pytest.mark.parametrize("phase,expected", [("BLOCK", []), ("OPEN", ["read", "write"]), ("READ", ["read"])])
async def test_phase_switch_applies_none_empty_and_explicit_allowlists(phase: str, expected: list[str]) -> None:
    from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph

    case = scenario()
    graph = build_graph("policy", initial="BLOCK", phases=[
        PhaseSpec(phase_id="BLOCK", prompts={"main": "test"}, tools=(), to=("OPEN",)),
        PhaseSpec(phase_id="OPEN", prompts={"main": "test"}, tools=None, to=("READ",)),
        PhaseSpec(phase_id="READ", prompts={"main": "test"}, tools=("read",), to=("BLOCK",)),
    ])
    await bind_phase_graph(case.world, case.entity, graph)
    if phase in {"OPEN", "READ"}:
        await advance(case.world, case.entity, "OPEN", reason="test")
    if phase == "READ":
        await advance(case.world, case.entity, "READ", reason="test")
    case.world.register_system(ToolExecutionSystem(), priority=5)
    assert (await Runner().run(case.world, max_ticks=4)).success
    assert [name for name, _ in case.called] == expected


@pytest.mark.parametrize("restriction", ["permission", "approval"])
async def test_denial_is_durable_once_and_has_no_execution_start(tmp_path: Path, restriction: str) -> None:
    from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
    from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording

    case = scenario()
    service = SessionService(InMemorySessionStore(), artifact_registry=ArtifactRegistry(tmp_path))
    ref = install_session_recording(case.world, case.entity, service,
        workspace_id="tests", agent_name="agent", register_tools=False)
    if restriction == "permission":
        case.world.add_component(case.entity, PermissionComponent(allowed_tools=["read"]))
    else:
        case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY))
    observed = []
    async def on_denied(event: ToolDeniedEvent) -> None:
        records = list(service.store.records(ref.session_id))
        observed.append(sum(record.payload.get("tool_call_id") == event.tool_call_id
            and record.payload.get("role") == "tool" for record in records))
    case.world.event_bus.subscribe(ToolDeniedEvent, on_denied)
    case.world.register_system(PermissionSystem(), priority=5)
    case.world.register_system(ToolApprovalSystem(), priority=5)
    case.world.register_system(ToolExecutionSystem(), priority=5)
    try:
        assert (await Runner().run(case.world, max_ticks=4)).success
        records = list(service.store.records(ref.session_id))
        assert [record.payload["tool_call_id"] for record in records if record.payload.get("role") == "tool"] == ["read-1", "write-1"]
        starts = [record.payload["tool_call_id"] for record in records if record.kind == "tool_started"]
        assert starts == (["read-1"] if restriction == "permission" else [])
        assert observed == ([1] if restriction == "permission" else [1, 1])
    finally:
        service.close()


async def test_session_tools_cannot_be_installed_under_an_empty_allowlist(tmp_path: Path) -> None:
    from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
    from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording
    from ecs_agent.tools.builtins import BuiltinToolsSkill

    case = scenario()
    registry = case.world.get_component(case.entity, ToolRegistryComponent)
    schema, handler = BuiltinToolsSkill().bind_workspace(str(tmp_path)).tools()["read_file"]
    registry.tools["read_file"] = schema
    registry.handlers["read_file"] = handler
    case.world.add_component(case.entity, PermissionComponent(allowed_tools=[]))
    service = SessionService(InMemorySessionStore(), artifact_registry=ArtifactRegistry(tmp_path))
    try:
        with pytest.raises(PermissionError, match="read_file"):
            install_session_recording(case.world, case.entity, service, workspace_id="tests", agent_name="agent")
    finally:
        service.close()


async def test_same_call_id_in_a_new_batch_requires_fresh_approval() -> None:
    case = scenario([ToolCall(id="same", name="read", arguments={})])
    case.world.add_component(case.entity, ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL))
    requests = []
    async def answer(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event)
        event.approval_future.set_result(len(requests) == 1)
    case.world.event_bus.subscribe(ToolApprovalRequestedEvent, answer)
    await ReasoningSystem().process(case.world)
    await ToolExecutionSystem().process(case.world)
    # A new conversation may use the same provider call ID; history is replaced
    # explicitly rather than creating a duplicate ID in one conversation.
    from ecs_agent.history import replace_conversation_history
    call = ToolCall(id="same", name="read", arguments={})
    replace_conversation_history(case.world, case.entity,
        [Message(role="assistant", content="", tool_calls=[call])], reason="new turn", new_conversation=True)
    case.world.add_component(case.entity, PendingToolCallsComponent(tool_calls=[call]))
    await ToolExecutionSystem().process(case.world)
    assert len(requests) == 2
    assert case.called == [("read", 0)]
    assert tool_span_error(case.world.get_component(case.entity, ConversationComponent).messages) is None
