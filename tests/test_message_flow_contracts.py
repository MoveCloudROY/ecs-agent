"""Desired message-flow contracts. Failures reproduce audit findings offline."""

import json
from typing import Any

import httpx
import pytest

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.components import (
    CompactionConfigComponent, ConversationComponent, LLMComponent,
    ResponsesAPIStateComponent, StreamingComponent, TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, InheritancePolicy, Message, SubagentConfig, ToolCall, ToolSchema
from examples.e2e.plan_and_task.main import _reset_workflow_boundary_state
from tests.http_mock import MockHTTP


def reply(response_id: str, *, status: str = "completed", output: list[dict[str, Any]] | None = None) -> httpx.Response:
    """Produce a Responses payload for the real SDK to decode."""
    data = {"id": response_id, "status": status, "output": output if output is not None else [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "partial answer"}]}
    ], "usage": {"input_tokens": 2000, "output_tokens": 10, "total_tokens": 2010}}
    if status == "incomplete":
        data["incomplete_details"] = {"reason": "max_output_tokens"}
    return httpx.Response(200, json=data)


async def sdk_world(*, store: bool | None = False, chain: bool = False) -> tuple[World, Any, OpenAIModel, MockHTTP]:
    """Build a real SDK provider over a deterministic HTTP transport."""
    transport = MockHTTP()
    model = OpenAIModel(config=ProviderConfig(
        provider_id="audit", base_url="https://audit.invalid/v1", api_key="test",
        api_format=ApiFormat.OPENAI_RESPONSES, enable_store=store, enable_response_chaining=chain,
    ))
    await model._client.aclose()
    model._client = transport.client()
    world = World()
    entity = world.create_entity()
    world.add_component(entity, LLMComponent(model=model))
    world.add_component(entity, ConversationComponent(messages=[Message(role="user", content="old workflow")]))
    return world, entity, model, transport


async def test_workflow_reset_must_clear_remote_response_chain() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_old_workflow"), reply("resp_new_workflow")]
    try:
        await ReasoningSystem().process(world)
        world.remove_component(entity, TerminalComponent)
        _reset_workflow_boundary_state(world, entity, preserve_user_text="new unrelated workflow")
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)
        assert "previous_response_id" not in sent, sent
    finally:
        await model.aclose()


async def test_compaction_must_clear_remote_response_chain() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_before_summary"), reply("resp_summary"), reply("resp_after_summary")]
    try:
        await ReasoningSystem().process(world)
        world.remove_component(entity, TerminalComponent)
        world.add_component(entity, CompactionConfigComponent(threshold_tokens=1))
        await CompactionSystem().process(world)
        assert len(world.get_component(entity, ConversationComponent).messages) == 1
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)
        assert "previous_response_id" not in sent, sent
    finally:
        await model.aclose()


async def test_model_switch_must_not_reuse_another_provider_response_id() -> None:
    world, entity, first, first_transport = await sdk_world(store=True, chain=True)
    _, _, second, second_transport = await sdk_world(store=True, chain=True)
    second._base_url = "https://other-provider.invalid/v1"
    first_transport.responses = [reply("resp_provider_A")]
    second_transport.responses = [reply("resp_provider_B")]
    try:
        await ReasoningSystem().process(world)
        world.remove_component(entity, TerminalComponent)
        world.get_component(entity, LLMComponent).pending_model = second
        world.get_component(entity, ConversationComponent).messages.append(Message(role="user", content="continue"))
        await ReasoningSystem().process(world)
        sent = json.loads(second_transport.requests[-1].content)
        assert "previous_response_id" not in sent, sent
    finally:
        await first.aclose()
        await second.aclose()


async def test_stateless_responses_tool_loop_must_preserve_reasoning_item() -> None:
    world, entity, model, transport = await sdk_world()
    reasoning = {"type": "reasoning", "id": "rs_original", "summary": [], "encrypted_content": "opaque-provider-state"}
    transport.responses = [reply("resp_call", output=[reasoning, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "noop", "arguments": "{}",
    }]), reply("resp_after_tool")]
    async def noop() -> str:
        return "tool result"
    world.add_component(entity, ToolRegistryComponent(
        tools={"noop": ToolSchema(name="noop", description="Read a value", parameters={"type": "object", "properties": {}})},
        handlers={"noop": noop},
    ))
    try:
        await ReasoningSystem().process(world)
        await ToolExecutionSystem().process(world)
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)
        assert reasoning in sent["input"], sent["input"]
    finally:
        await model.aclose()


@pytest.mark.parametrize("terminal_type", ["incomplete", "cancelled", "missing"])
async def test_incomplete_stream_must_not_look_like_successful_turn(terminal_type: str) -> None:
    world, entity, model, transport = await sdk_world()
    events = [{"type": "response.output_text.delta", "delta": "half an answer"}]
    if terminal_type != "missing":
        data = reply("resp_partial", status=terminal_type).json()
        events.append({"type": f"response.{terminal_type}", "response": data})
    transport.responses = [httpx.Response(200, headers={"content-type": "text/event-stream"},
        content="".join("data: " + json.dumps(event) + "\n\n" for event in events))]
    world.add_component(entity, StreamingComponent(enabled=True))
    invocation_events: list[LLMInvocationEvent] = []
    async def record(event: LLMInvocationEvent) -> None:
        invocation_events.append(event)
    world.event_bus.subscribe(LLMInvocationEvent, record)
    try:
        await ReasoningSystem().process(world)
        terminal = world.get_component(entity, TerminalComponent)
        evidence = {"reason": terminal.reason if terminal else None,
                    "status": invocation_events[-1].status,
                    "completeness": invocation_events[-1].usage.stream_completeness.value}
        assert terminal is None or terminal.reason != "reasoning_complete", evidence
    finally:
        await model.aclose()


class FailsAfterTool(FakeModel):
    """Return an intermediate assistant message, then simulate network failure."""
    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        if self._index:
            raise httpx.ReadTimeout("upstream unavailable after tool execution")
        return await super().complete(*args, **kwargs)


async def test_failed_child_must_not_return_intermediate_message_as_success() -> None:
    child_model = FailsAfterTool(responses=[CompletionResult(message=Message(
        role="assistant", content="Still working; this is not the result.",
        tool_calls=[ToolCall(id="call_1", name="noop", arguments={})],
    ))])
    parent = World()
    entity = parent.create_entity()
    parent.add_component(entity, LLMComponent(model=child_model))
    async def noop() -> str:
        return "tool result"
    parent.add_component(entity, ToolRegistryComponent(
        tools={"noop": ToolSchema(name="noop", description="Read a value", parameters={"type": "object", "properties": {}})},
        handlers={"noop": noop},
    ))
    config = SubagentConfig(name="audit-child", model=child_model, max_ticks=3,
                            inheritance_policy=InheritancePolicy(inherit_tools=["noop"]))
    result, success, error = await SubagentSystem()._execute_subagent_core(
        parent, entity, "audit-child", "Finish the task", "audit-correlation", "", config,
    )
    assert not success, {"result": result, "success": success, "error": error}
