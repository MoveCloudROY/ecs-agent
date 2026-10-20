"""Opt-in real Responses agent loops for the P1 authorization contracts.

Set ECS_P1_LIVE=1 plus LLM_API_KEY, LLM_BASE_URL and LLM_MODEL to run.
Only an in-memory probe tool is exposed; no filesystem or network tool executes.
"""

import asyncio
import os

import pytest

from ecs_agent.components import (
    ConversationComponent, LLMComponent, PermissionComponent, StreamingComponent,
    ToolApprovalComponent, ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.dsl import AgentSpec, compile_agent_specs
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers._message_contract import tool_span_error
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    ApprovalPolicy, Message, ToolApprovalRequestedEvent, ToolDeniedEvent,
    ToolExecutionStartedEvent, ToolSchema,
)

pytestmark = pytest.mark.skipif(os.getenv("ECS_P1_LIVE") != "1", reason="ECS_P1_LIVE is not enabled")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", ["allow", "permission_deny", "approval_deny", "require_approval", "dsl_deny"])
async def test_responses_tool_authorization(live_api_key: str, stream: bool, policy: str) -> None:
    model = Model(os.environ["LLM_MODEL"], base_url=os.environ["LLM_BASE_URL"],
        api_key=live_api_key, api_format=ApiFormat.OPENAI_RESPONSES,
        timeout=60, stream_read_timeout=60, enable_store=False)
    if policy == "dsl_deny":
        entity, world = compile_agent_specs({"main": AgentSpec(
            mode="primary", model=os.environ["LLM_MODEL"], prompt="Follow the user's tool instruction.",
            tools={"probe": False},
        )}, lambda *_: model)
    else:
        world = World()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
    world.add_component(entity, ConversationComponent(messages=[Message(role="user", content=(
        "Call the probe tool exactly once with value 7. You must use the tool; do not simulate it. "
        "After receiving its result, give a short final answer. If the tool is denied, acknowledge "
        "the denial and stop. Never retry or call another tool."
    ))]))
    world.add_component(entity, StreamingComponent(enabled=stream))
    called = []
    async def probe(value: int) -> str:
        called.append(value)
        return "PROBE_RESULT=7"
    world.add_component(entity, ToolRegistryComponent(tools={"probe": ToolSchema(
        name="probe", description="Return an in-memory test value.",
        parameters={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False},
    )}, handlers={"probe": probe}))
    if policy == "permission_deny":
        world.add_component(entity, PermissionComponent(allowed_tools=[]))
    elif policy in {"approval_deny", "require_approval"}:
        world.add_component(entity, ToolApprovalComponent(policy=(
            ApprovalPolicy.ALWAYS_DENY if policy == "approval_deny" else ApprovalPolicy.REQUIRE_APPROVAL
        )))
    requests = []
    async def approve(event: ToolApprovalRequestedEvent) -> None:
        requests.append(event.tool_call.id)
        event.approval_future.set_result(True)
    if policy == "require_approval":
        world.event_bus.subscribe(ToolApprovalRequestedEvent, approve)
    starts = []
    denials = []
    async def on_start(event: ToolExecutionStartedEvent) -> None:
        starts.append(event.tool_call.id)
    async def on_denied(event: ToolDeniedEvent) -> None:
        denials.append(event.tool_call_id)
    world.event_bus.subscribe(ToolExecutionStartedEvent, on_start)
    world.event_bus.subscribe(ToolDeniedEvent, on_denied)
    world.register_system(ReasoningSystem(), priority=0)
    # Deliberately omit both preflight systems: the execution gate is mandatory.
    world.register_system(ToolExecutionSystem(), priority=5)
    try:
        result = await asyncio.wait_for(Runner().run(world, max_ticks=4, target_entity=entity), 180)
        assert result.success, f"{result.status}: {result.error}"
        conversation = world.get_component(entity, ConversationComponent)
        assert conversation is not None
        calls = [call for message in conversation.messages for call in message.tool_calls or []]
        assert len(calls) == 1
        assert calls[0].name == "probe" and calls[0].arguments == {"value": 7}
        assert tool_span_error(conversation.messages) is None
        assistant = [message for message in conversation.messages if message.role == "assistant"]
        assert all(message.provider_output is not None and message.provider_output.protocol == "openai_responses" for message in assistant)
        allowed = policy in {"allow", "require_approval"}
        assert called == ([7] if allowed else [])
        assert len(starts) == int(allowed)
        assert len(denials) == int(not allowed)
        assert len(requests) == int(policy == "require_approval")
    finally:
        await model.aclose()
