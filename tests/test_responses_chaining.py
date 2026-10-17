"""Storage and verified automatic chaining obey independent wire contracts."""

import json
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest

from ecs_agent.components import ConversationComponent, StreamingComponent, TerminalComponent
from ecs_agent.components import ResponsesAPIStateComponent
from ecs_agent.providers.retry_model import RetryModel
from ecs_agent.types import RetryConfig
from ecs_agent.providers.responses_state import bind_provider_output
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message
from tests.test_message_flow_contracts import reply, sdk_world
from tests.http_mock import sse_response
from tests.test_responses_replay import OUTPUT


async def test_invalid_reference_error_after_stream_delta_never_retries() -> None:
    _, _, model, transport = await sdk_world(store=True, chain=True)

    async def lines() -> AsyncIterator[str]:
        yield 'data: {"type":"response.output_text.delta","delta":"partial"}'
        request = transport.requests[0]
        raise httpx.HTTPStatusError("expired", request=request, response=httpx.Response(
            404, request=request, json={"error": {"param": "previous_response_id", "message": "not found"}},
        ))

    transport.responses = [sse_response(lines())]
    messages = [Message(role="user", content="old"), Message(role="assistant", content="answer"),
                Message(role="user", content="new")]
    try:
        stream = await model.complete(messages, stream=True, thread_response_id="resp_old", covered_message_count=2)
        assert not isinstance(stream, CompletionResult)
        assert (await anext(stream)).content == "partial"
        with pytest.raises(httpx.HTTPStatusError):
            await anext(stream)
        assert len(transport.requests) == 1
        assert transport.returned[0].is_closed
    finally:
        await model.aclose()


@pytest.mark.parametrize("stream", [False, True])
async def test_request_keeps_original_provenance_during_provider_rotation(stream: bool) -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    old_origin = model.get_response_context([]).origin
    body = reply("resp_original", output=OUTPUT[:2]).json()
    first = httpx.Response(200, json=body)
    if stream:
        first = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=
            "data: " + json.dumps({"type": "response.completed", "response": body}) + "\n\n")
        world.add_component(entity, StreamingComponent(enabled=True))
    transport.responses = [first, reply("resp_rotated")]

    async def rotate(request: httpx.Request) -> None:
        if not transport.requests:
            model._base_url = "https://rotated.invalid/v1"
            model._api_key = "rotated-test-key"
            model._provider_config.enable_store = False

    model._client.event_hooks["request"].append(rotate)
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        output = conv.messages[-1].provider_output
        assert output is not None and output.origin == old_origin and output.stored is True
        assert output.origin != model.get_response_context([]).origin
        world.remove_component(entity, TerminalComponent)
        world.remove_component(entity, StreamingComponent)
        conv.messages.append(Message(role="user", content="new source"))
        await ReasoningSystem().process(world)
        second = transport.requests[-1]
        sent = json.loads(second.content)
        assert second.url.host == "rotated.invalid"
        assert second.headers["authorization"] == "Bearer rotated-test-key"
        assert sent["store"] is False and "previous_response_id" not in sent
        assert "opaque-state" not in json.dumps(sent)
    finally:
        await model.aclose()


@pytest.mark.parametrize("store", [True, False, None])
@pytest.mark.parametrize("stream", [False, True])
async def test_store_maps_directly_to_sdk_without_controlling_explicit_reference(store: bool | None, stream: bool) -> None:
    _, _, model, transport = await sdk_world(store=store)
    data = reply("resp_current").json()
    transport.responses = [
        httpx.Response(200, json=data) if not stream else httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps({"type": "response.completed", "response": data}) + "\n\n",
        )
    ]
    try:
        result = await model.complete([Message(role="user", content="new input")], stream=stream, thread_response_id="resp_saved_earlier")
        if not isinstance(result, CompletionResult):
            _ = [delta async for delta in result]
        body = json.loads(transport.requests[0].content)
        assert body["previous_response_id"] == "resp_saved_earlier"
        if store is None:
            assert "store" not in body
        else:
            assert body["store"] is store
    finally:
        await model.aclose()


@pytest.mark.parametrize("automatic", [True, False])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_only_automatic_policy_enables_verified_chain(automatic: bool, wrapped: bool) -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=automatic)
    if wrapped:
        from ecs_agent.components import LLMComponent

        world.add_component(entity, LLMComponent(model=RetryModel(model, RetryConfig(max_attempts=1))))
    transport.responses = [reply("resp_first"), reply("resp_second")]
    try:
        await ReasoningSystem().process(world)
        world.remove_component(entity, TerminalComponent)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages.append(Message(role="user", content="follow up"))
        await ReasoningSystem().process(world)
        body = json.loads(transport.requests[-1].content)
        if automatic:
            assert body["previous_response_id"] == "resp_first"
            assert len(body["input"]) == 1 and body["input"][0]["content"][0]["text"] == "follow up"
        else:
            assert "previous_response_id" not in body
            assert len(body["input"]) == 3
    finally:
        await model.aclose()


async def test_store_false_can_use_old_anchor_but_does_not_create_next_anchor() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_saved"), reply("resp_unstored"), reply("resp_after_unstored")]
    try:
        await ReasoningSystem().process(world)
        model._provider_config.enable_store = False
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        for text in ["use old anchor", "full replay now"]:
            world.remove_component(entity, TerminalComponent)
            conv.messages.append(Message(role="user", content=text))
            await ReasoningSystem().process(world)
        second, third = [json.loads(request.content) for request in transport.requests[1:]]
        assert second["store"] is False and second["previous_response_id"] == "resp_saved"
        assert third["store"] is False and "previous_response_id" not in third
        assert len(third["input"]) == 5
    finally:
        await model.aclose()


async def test_same_count_but_edited_prefix_invalidates_anchor() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_old"), reply("resp_new")]
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages[0].content = "edited task"
        conv.messages.append(Message(role="user", content="continue"))
        world.remove_component(entity, TerminalComponent)
        await ReasoningSystem().process(world)
        body = json.loads(transport.requests[-1].content)
        assert "previous_response_id" not in body
        assert body["input"][0]["content"][0]["text"] == "edited task"
    finally:
        await model.aclose()


@pytest.mark.parametrize("status_code", [400, 404])
async def test_expired_chain_retries_full_protocol_history_once(status_code: int) -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_expired"), httpx.Response(status_code, json={"error": {
        "param": "previous_response_id", "code": "previous_response_not_found", "message": "Previous response not found",
    }}), reply("resp_rebuilt"), reply("resp_continued")]
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        for text in ["retry", "continue"]:
            world.remove_component(entity, TerminalComponent)
            conv.messages.append(Message(role="user", content=text))
            await ReasoningSystem().process(world)
        bodies = [json.loads(req.content) for req in transport.requests]
        assert len(bodies) == 4
        assert bodies[1]["previous_response_id"] == "resp_expired"
        assert "previous_response_id" not in bodies[2] and len(bodies[2]["input"]) == 3
        assert bodies[3]["previous_response_id"] == "resp_rebuilt"
        assert all(str(req.url).endswith("/responses") for req in transport.requests)
    finally:
        await model.aclose()


async def test_incremental_explicit_reference_cannot_fall_back_without_full_history() -> None:
    _, _, model, transport = await sdk_world()
    transport.responses = [httpx.Response(404, json={"error": {
        "param": "previous_response_id", "code": "previous_response_not_found", "message": "expired",
    }})]
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await model.complete([Message(role="user", content="increment only")], thread_response_id="resp_expired")
        assert len(transport.requests) == 1
    finally:
        await model.aclose()


@pytest.mark.parametrize("reported_store", [None, True, False])
async def test_unset_store_requires_reported_storage_before_automatic_chaining(reported_store: bool | None) -> None:
    world, entity, model, transport = await sdk_world(store=None, chain=True)
    first = reply("resp_default").json()
    if reported_store is not None:
        first["store"] = reported_store
    transport.responses = [httpx.Response(200, json=first), reply("resp_next")]
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages.append(Message(role="user", content="continue"))
        world.remove_component(entity, TerminalComponent)
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)
        assert "store" not in sent
        assert ("previous_response_id" in sent) is (reported_store is True)
    finally:
        await model.aclose()


async def test_verified_anchor_survives_unrelated_http_failure() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_stable"), httpx.Response(500, text="temporarily unavailable")]
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages.append(Message(role="user", content="continue"))
        world.remove_component(entity, TerminalComponent)
        await ReasoningSystem().process(world)
        state = world.get_component(entity, ResponsesAPIStateComponent)
        assert state is not None and state.previous_response_id == "resp_stable"
        assert json.loads(transport.requests[-1].content)["previous_response_id"] == "resp_stable"
    finally:
        await model.aclose()


async def test_expired_chain_with_remote_only_reasoning_does_not_claim_lossless_recovery() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    items = [{"type": "reasoning", "id": "rs_remote", "summary": []}, *reply("unused").json()["output"]]
    transport.responses = [reply("resp_remote", output=items), httpx.Response(404, json={"error": {
        "param": "previous_response_id", "message": "Previous response not found",
    }})]
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages.append(Message(role="user", content="continue"))
        world.remove_component(entity, TerminalComponent)
        await ReasoningSystem().process(world)
        assert len(transport.requests) == 2
        terminal = world.get_component(entity, TerminalComponent)
        assert terminal is not None and terminal.reason == "reasoning_error"
        assert not world.has_component(entity, ResponsesAPIStateComponent)
    finally:
        await model.aclose()


async def test_checkpoint_restored_anchor_is_revalidated_and_reused() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    transport.responses = [reply("resp_checkpoint"), reply("resp_restored")]
    try:
        await ReasoningSystem().process(world)
        restored = WorldSerializer.from_dict(WorldSerializer.to_dict(world), providers={model.model_id: model}, tool_handlers={})
        restored.remove_component(entity, TerminalComponent)
        conv = restored.get_component(entity, ConversationComponent)
        assert conv is not None
        conv.messages.append(Message(role="user", content="resume"))
        await ReasoningSystem().process(restored)
        sent = json.loads(transport.requests[-1].content)
        assert sent["previous_response_id"] == "resp_checkpoint"
        assert len(sent["input"]) == 1
    finally:
        await model.aclose()


async def test_resegmented_messages_use_protocol_prefix_not_saved_message_count() -> None:
    world, entity, model, transport = await sdk_world(store=True, chain=True)
    items = [
        {"type": "reasoning", "id": "rs_multi", "summary": [], "encrypted_content": "opaque"},
        *reply("unused").json()["output"],
    ]
    transport.responses = [reply("resp_multi", output=items), reply("resp_next")]
    try:
        await ReasoningSystem().process(world)
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        original = conv.messages[-1].provider_output
        assert original is not None
        # Same wire items, now split across two local messages instead of one.
        split = [bind_provider_output(
            model._responses_adapter._parse_responses_output([item]),
            replace(original, items=[item]),
        ) for item in original.items]
        conv.messages[1:] = split
        conv.messages.append(Message(role="user", content="new suffix"))
        world.remove_component(entity, TerminalComponent)
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)
        assert sent["previous_response_id"] == "resp_multi"
        assert len(sent["input"]) == 1
        assert sent["input"][0]["content"][0]["text"] == "new suffix"
    finally:
        await model.aclose()
