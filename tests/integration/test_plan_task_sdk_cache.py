"""Verify cache prefixes and accounting on the example's actual SDK HTTP path."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.components import CompactionConfigComponent, ConversationComponent, LLMComponent, PendingToolCallsComponent, StreamingComponent, ToolResultsComponent
from ecs_agent.plugins import install_plugins
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from examples.e2e.plan_and_task.billing import BillingSubscriber
from examples.e2e.plan_and_task.debug import PlanTaskDebugSession
from examples.e2e.plan_and_task.main import build_plan_task_world
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import ToolCall, DelegationCompletedEvent
from tests.http_mock import MockHTTP
from tests.test_message_flow_contracts import reply


@pytest.mark.parametrize("scenario", ["qa_approved", "review_cap"])
async def test_review_callback_context_follows_real_tool_receipts(tmp_path: Path, scenario: str) -> None:
    """Callback control prompts must never split a pending tool span."""
    transport = MockHTTP()
    model = OpenAIModel(config=ProviderConfig(
        provider_id="test", base_url="https://gateway.example/v1", api_key="test",
        api_format=ApiFormat.OPENAI_RESPONSES,
    ))
    await model._client.aclose()
    model._client = transport.client()

    def dispatch(category: str) -> httpx.Response:
        return reply(f"resp_dispatch_{category}", output=[{
            "type": "function_call", "call_id": f"call_{category}", "name": "subagent",
            "arguments": json.dumps({"category": category, "prompt": "Review the draft.",
                                     "load_skills": [], "background": False}),
        }])

    def review(category: str, verdict: str) -> httpx.Response:
        response = reply(f"resp_review_{category}", output=[{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": f"VERDICT: {verdict}"}],
        }]).json()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
            "data: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n"
        ))

    transport.responses = [_reply(0, 0, stream=False), _reply(1, 0, stream=False)]
    try:
        async with await PlanTaskDebugSession.build(model, base_dir=tmp_path, enable_tool_sink=True) as session:
            assert (await session.send("/plan:start Tiny test task", timeout=5)).ok
            if scenario == "qa_approved":
                transport.responses = [dispatch("advisor"), review("advisor", "approved"),
                                       dispatch("qa"), review("qa", "approved"), _reply(9, 0, stream=False)]
                expected = "workflow_plan.md"
                category = "qa"
            else:
                session._runtime_state_ref[0].review_rounds["DRAFT_ADVISOR_REVIEW"] = 5
                transport.responses = [dispatch("advisor"), review("advisor", "revise"), _reply(9, 0, stream=False)]
                expected = "could not be recorded"
                category = "advisor"
            result = await session.send("Review the current draft.", timeout=5)
            assert result.ok and not result.errors
            body = json.loads(transport.requests[-1].content)
            items = body["input"]
            receipts = [(i, item) for i, item in enumerate(items)
                        if item.get("type") == "function_call_output" and item["call_id"] == f"call_{category}"]
            assert len(receipts) == 1
            index, receipt = receipts[0]
            assert receipt["output"] == ("VERDICT: approved" if scenario == "qa_approved" else "VERDICT: revise")
            assert any(item.get("role") == "user" and expected in json.dumps(item)
                       for item in items[index + 1:])
            # The cap rejects the verdict before committing a phase transition.
            assert result.snapshot.phase == ("WRITE_PLAN" if scenario == "qa_approved" else "DRAFT_INTERVIEW")
    finally:
        await model.aclose()


@pytest.mark.parametrize("change", ["workflow_reset", "compaction", "model_switch"])
async def test_example_invalidates_verified_chain_after_context_boundary(tmp_path: Path, change: str) -> None:
    transport = MockHTTP()
    transport.responses = [_reply(1, 0, stream=False), _reply(2, 0, stream=False)]
    model = OpenAIModel(config=ProviderConfig(
        provider_id="test", base_url="https://gateway.example/v1", api_key="test",
        api_format=ApiFormat.OPENAI_RESPONSES, enable_store=True, enable_response_chaining=True,
    ))
    await model._client.aclose()
    model._client = transport.client()
    second = None
    try:
        async with await PlanTaskDebugSession.build(model, base_dir=tmp_path) as session:
            assert (await session.send("first turn", timeout=5)).ok
            assert (await session.send("normal append", timeout=5)).ok
            assert json.loads(transport.requests[-1].content)["previous_response_id"] == "resp_cache_1"
            target_transport = transport
            if change == "workflow_reset":
                from ecs_agent.components import ContextEntry, PromptContextQueueComponent, PromptContextReservationComponent
                entries = [ContextEntry("old", 0, "plan_task:write_plan", "OBSOLETE_WRITER_PROMPT", 0),
                           ContextEntry("caller", 0, "user_clarification", "KEEP_CALLER_CONTEXT", 1)]
                session._world.add_component(session._agent_id, PromptContextQueueComponent(entries=list(entries)))
                session._world.add_component(session._agent_id, PromptContextReservationComponent(
                    reservation_id="pending", created_at_tick=0, reserved_entries=list(entries),
                    rendered_block="OBSOLETE_WRITER_PROMPT KEEP_CALLER_CONTEXT",
                ))
                transport.responses = [_reply(0, 0, stream=False), _reply(3, 0, stream=False)]
                result = await session.send("/plan:start A new independent workflow", timeout=5)
                assert result.snapshot.phase == "DRAFT_INTERVIEW"
            elif change == "compaction":
                transport.responses = [_reply(4, 0, stream=False), _reply(3, 0, stream=False)]
                session._world.add_component(session._agent_id, CompactionConfigComponent(threshold_tokens=1))
                await CompactionSystem().process(session._world)
                session._world.remove_component(session._agent_id, CompactionConfigComponent)
                result = await session.send("after summary", timeout=5)
            else:
                target_transport = MockHTTP()
                target_transport.responses = [_reply(3, 0, stream=False)]
                second = OpenAIModel(config=ProviderConfig(
                    provider_id="other", base_url="https://other.example/v1", api_key="other",
                    api_format=ApiFormat.OPENAI_RESPONSES, enable_store=True, enable_response_chaining=True,
                ))
                await second._client.aclose()
                second._client = target_transport.client()
                session._world.get_component(session._agent_id, LLMComponent).pending_model = second
                result = await session.send("switch provider", timeout=5)
            assert result.ok and not result.errors
            body = json.loads(target_transport.requests[-1].content)
            assert body["store"] is True
            assert "previous_response_id" not in body
            if change == "workflow_reset":
                assert "OBSOLETE_WRITER_PROMPT" not in json.dumps(body)
                assert "KEEP_CALLER_CONTEXT" in json.dumps(body)
    finally:
        await model.aclose()
        if second is not None:
            await second.aclose()


@pytest.mark.parametrize("status", ["completed", "revise", "incomplete", "cancelled", "failed", "missing"])
async def test_example_review_gate_uses_child_outcome_not_partial_verdict(tmp_path: Path, status: str) -> None:
    transport = MockHTTP()
    model = OpenAIModel(config=ProviderConfig(
        provider_id="test", base_url="https://gateway.example/v1", api_key="test",
        api_format=ApiFormat.OPENAI_RESPONSES,
    ))
    await model._client.aclose()
    model._client = transport.client()
    verdict_text = ("Detailed review context. " * 40 + "Critical fix after the old cutoff.\nVERDICT: revise") if status == "revise" else "VERDICT: approved"
    protocol_status = "completed" if status == "revise" else status
    child = reply("resp_child", status=protocol_status, output=[{
        "type": "message", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": verdict_text},
        ],
    }]).json()
    if status == "failed":
        child["error"] = {"code": "server_error", "message": "child generation failed"}
    child_events = [{"type": "response.output_text.delta", "delta": verdict_text}]
    if status != "missing":
        child_events.append({"type": f"response.{protocol_status}", "response": child})
    transport.responses = [
        _reply(0, 0, stream=False), _reply(1, 0, stream=False),
        reply("resp_dispatch", output=[{"type": "function_call", "call_id": "call_review", "name": "subagent",
            "arguments": json.dumps({"category": "advisor", "prompt": "Review the current draft.",
                                      "load_skills": [], "background": False})}]),
        httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(
            "data: " + json.dumps(event) + "\n\n" for event in child_events)),
        _reply(4, 0, stream=False),
    ]
    completions: list[DelegationCompletedEvent] = []

    async def record(event: DelegationCompletedEvent) -> None:
        completions.append(event)

    try:
        async with await PlanTaskDebugSession.build(model, base_dir=tmp_path, enable_tool_sink=True) as session:
            session._world.event_bus.subscribe(DelegationCompletedEvent, record)
            first = await session.send("/plan:start Tiny test task", timeout=5)
            assert first.snapshot.phase == "DRAFT_INTERVIEW"
            result = await session.send("Ask the advisor to review the draft.", timeout=5)
            assert result.ok and not result.errors
            assert len(completions) == 1
            succeeded = status in ("completed", "revise")
            assert completions[0].success == succeeded
            assert bool(result.snapshot.review_verdicts) == succeeded
            if succeeded:
                final_body = json.loads(transport.requests[-1].content)
                outputs = [i["output"] for i in final_body["input"] if i.get("type") == "function_call_output"]
                assert outputs[-1] == verdict_text
                if status == "revise":
                    assert result.snapshot.review_verdicts[-1]["notes"] == verdict_text
                    persisted = session._adapter_ref[0].read_state()
                    assert persisted.review_verdicts[-1].notes == verdict_text
            if not succeeded:
                assert result.snapshot.phase == "DRAFT_INTERVIEW"
                assert completions[0].partial_content == "VERDICT: approved"
                final_body = json.loads(transport.requests[-1].content)
                outputs = [i["output"] for i in final_body["input"] if i.get("type") == "function_call_output"]
                assert outputs[-1].startswith("Error") and "Partial output" in outputs[-1]
    finally:
        await model.aclose()


async def test_sink_file_reads_and_archived_results_reach_the_next_sdk_request(tmp_path: Path) -> None:
    transport = MockHTTP()
    transport.responses = [_reply(1, 0, stream=False), _reply(2, 0, stream=False)]
    model = OpenAIModel(config=ProviderConfig(
        provider_id="test", base_url="https://gateway.example/v1", api_key="test",
        api_format=ApiFormat.OPENAI_RESPONSES,
    ))
    await model._client.aclose()
    model._client = transport.client()
    world, entity, _, _ = await build_plan_task_world(model, base_dir=tmp_path, enable_tool_sink=True)
    world.apply_pending_system_operations()
    from ecs_agent.types import Message
    from ecs_agent.components import TerminalComponent
    (tmp_path / "draft.md").write_text("# Scope\nUnique confirmed draft content for the model.\n")
    conv = world.get_component(entity, ConversationComponent)
    executor = next(entry.system for entry in world._systems._systems if isinstance(entry.system, ToolExecutionSystem))
    try:
        for index, (tool, arguments, expected) in enumerate([
            ("read_file", {"file_path": "draft.md", "offset": 1, "limit": 20}, "Unique confirmed draft content"),
            ("bash", {"command": "printf 'archived-tool-result'", "timeout": 5}, "archived-tool-result"),
        ]):
            world.remove_component(entity, TerminalComponent)
            call = ToolCall(id=f"call_read_{index}", name=tool, arguments=arguments)
            conv.messages.append(Message(role="assistant", content="", tool_calls=[call]))
            world.add_component(entity, PendingToolCallsComponent([call]))
            await executor.process(world)
            if tool == "bash":
                stored = world.get_component(entity, ToolResultsComponent).results[call.id]
                assert (tmp_path / stored).exists()
                fetch = ToolCall(id="call_fetch", name="read_file", arguments={"file_path": stored, "offset": 1, "limit": 100})
                conv.messages.append(Message(role="assistant", content="", tool_calls=[fetch]))
                world.add_component(entity, PendingToolCallsComponent([fetch]))
                await executor.process(world)
            await ReasoningSystem().process(world)
            body = json.loads(transport.requests[-1].content)
            results = [item["output"] for item in body["input"] if item.get("type") == "function_call_output"]
            assert expected in results[-1]
        assert len(transport.requests) == 2
    finally:
        await model.aclose()


def _reply(index: int, cached_tokens: int, *, stream: bool, ask: bool = False) -> httpx.Response:
    """Return controlled usage through real SDK response deserialization."""
    item: dict[str, Any]
    if ask:
        item = {
            "type": "function_call", "id": "fc_question", "call_id": "call_question",
            "name": "ask_question", "arguments": json.dumps({"questions": [{
                "header": "Storage", "question": "Which storage?",
                "options": [{"label": "JSON file"}, {"label": "SQLite"}],
            }]}),
        }
    else:
        item = {"type": "message", "id": f"msg_{index}", "role": "assistant",
                "content": [{"type": "output_text", "text": "sdk-cache-audit" if index == 0 else "OK"}]}
    response = {
        "id": f"resp_cache_{index}", "status": "completed", "output": [item],
        "usage": {"input_tokens": 6144, "output_tokens": 16, "total_tokens": 6160,
                  "input_tokens_details": {"cached_tokens": cached_tokens}},
    }
    if not stream:
        return httpx.Response(200, json=response)
    events = [
        {"type": "response.created", "response": {"id": response["id"]}},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content="".join("data: " + json.dumps(event) + "\n\n" for event in events),
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_plan_task_sdk_preserves_cache_prefix_and_reported_usage(
    tmp_path: Path, stream: bool,
) -> None:
    transport = MockHTTP()
    transport.responses = [
        _reply(0, 0, stream=False),  # workflow slug helper is non-streaming
        _reply(1, 0, stream=stream, ask=True),
        _reply(2, 5120, stream=stream),
        _reply(3, 0, stream=stream),  # a gateway cache miss must stay a miss
        _reply(4, 5632, stream=stream),
    ]
    model = OpenAIModel(config=ProviderConfig(
        provider_id="test", base_url="https://gateway.example/v1", api_key="test",
        api_format=ApiFormat.OPENAI_RESPONSES,
    ))
    await model._client.aclose()
    model._client = transport.client()
    billing = BillingSubscriber()
    events: list[LLMInvocationEvent] = []

    async def record(event: LLMInvocationEvent) -> None:
        events.append(event)

    try:
        async with await PlanTaskDebugSession.build(
            model, base_dir=tmp_path, enable_tool_sink=True,
        ) as session:
            if stream:
                session._world.add_component(session._agent_id, StreamingComponent(enabled=True))
            handle = await install_plugins(session._world, [billing])
            session._world.event_bus.subscribe(LLMInvocationEvent, record)
            try:
                for prompt in (
                    "/plan:start Build a minimal CLI notes tool",
                    "Keep the same plan and list its constraints.",
                    "Confirm the selected storage without changing the phase.",
                ):
                    result = await session.send(prompt, timeout=10)
                    assert result.ok, result.errors
                    assert result.snapshot.phase == "DRAFT_INTERVIEW"
                assert any(event.request_id == "resp_cache_2" for event in events)
            finally:
                await handle.shutdown()
    finally:
        await model.aclose()

    assert len(transport.requests) == 5
    # Inspect SDK-serialized HTTP bodies, not the pre-SDK request builder.
    bodies = [json.loads(request.content) for request in transport.requests[1:]]
    assert all(request.headers["user-agent"].startswith("AsyncOpenAI/") for request in transport.requests)
    assert all("previous_response_id" not in body and "store" not in body for body in bodies)
    assert all(body.get("stream", False) == stream for body in bodies)
    for previous, current in zip(bodies, bodies[1:]):
        assert current["instructions"] == previous["instructions"]
        assert current["tools"] == previous["tools"]
        assert current["input"][:len(previous["input"])] == previous["input"]
    assert any(item["type"] == "function_call_output" for item in bodies[1]["input"])
    by_id = {event.request_id: event for event in events}
    assert [by_id[f"resp_cache_{i}"].usage.cached_input_tokens for i in range(1, 5)] == [0, 5120, 0, 5632]
    assert all(event.status == "success" for event in events)
    assert billing._total_cached_input_tokens == 10752
