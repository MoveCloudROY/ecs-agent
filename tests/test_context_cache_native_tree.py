"""Desired native-tree cache contracts; strict xfails expose known gaps.

Run with --runxfail to reproduce the failures as ordinary failing tests. These
are not snapshot-branch tests and are never counted as successful validation.
"""

import json
from typing import Any

import pytest

from ecs_agent.components import ConversationComponent, ConversationTreeComponent
from ecs_agent.conversation_tree import add_message, create_branch, get_active_leaf, linearize, switch_branch
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.reasoning import ReasoningSystem
from tests.test_message_flow_contracts import reply, sdk_world
from tests.cache_audit.branching import corpus


async def test_native_tree_reads_long_shared_prefix_without_sibling_messages() -> None:
    world, entity, model, transport = await sdk_world()
    tree = ConversationTreeComponent()
    root = add_message(tree, "user", corpus("native-shared", 220))
    shared = add_message(tree, "assistant", "Shared answer", parent_id=root.id)
    left = add_message(tree, "user", "ONLY_LEFT_BRANCH", parent_id=shared.id)
    right = add_message(tree, "user", "ONLY_RIGHT_BRANCH", parent_id=shared.id)
    create_branch(tree, "left", left.id)
    create_branch(tree, "right", right.id)
    world.add_component(entity, tree)
    transport.responses = [reply("resp_left"), reply("resp_right")]
    try:
        switch_branch(tree, "left")
        await ReasoningSystem().process(world)
        switch_branch(tree, "right")
        await ReasoningSystem().process(world)
        bodies = [json.loads(request.content) for request in transport.requests]
        assert bodies[0]["input"][:2] == bodies[1]["input"][:2]
        assert "ONLY_LEFT_BRANCH" not in json.dumps(bodies[1])
        assert "ONLY_RIGHT_BRANCH" not in json.dumps(bodies[0])
    finally:
        await model.aclose()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Native tree is read-only in ReasoningSystem; generated messages do not advance its leaf")
@pytest.mark.parametrize("flat_component", [False, True])
async def test_native_tree_appends_generated_response_to_active_branch(flat_component: bool) -> None:
    world, entity, model, transport = await sdk_world()
    if not flat_component:
        world.remove_component(entity, ConversationComponent)
    tree = ConversationTreeComponent()
    root = add_message(tree, "user", "Remember root fact")
    create_branch(tree, "main", root.id)
    tree.current_branch_id = "main"
    world.add_component(entity, tree)
    transport.responses = [reply("resp_tree")]
    try:
        await ReasoningSystem().process(world)
        leaf = get_active_leaf(tree)
        assert leaf != root.id, "The assistant response was not written to the native branch"
        assert leaf is not None
        history = linearize(tree, leaf)
        assert history[-1].role == "assistant"
        assert history[-1].provider_output is not None
    finally:
        await model.aclose()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Native add_message/linearize roundtrip loses Responses reasoning items and original message identity")
async def test_native_tree_manual_append_preserves_sdk_output_for_next_turn() -> None:
    world, entity, model, transport = await sdk_world()
    original: list[dict[str, Any]] = [
        {"type": "reasoning", "id": "rs_native", "summary": [], "encrypted_content": "opaque_native"},
        {"type": "message", "id": "msg_native", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Remembered root fact"}]},
    ]
    transport.responses = [reply("resp_native_1", output=original), reply("resp_native_2")]
    tree = ConversationTreeComponent()
    root = add_message(tree, "user", "Remember root fact")
    create_branch(tree, "main", root.id)
    tree.current_branch_id = "main"
    world.add_component(entity, tree)
    try:
        await ReasoningSystem().process(world)
        flat = world.get_component(entity, ConversationComponent)
        assert flat is not None
        generated = flat.messages[-1]
        assert generated.provider_output is not None
        # Explicitly append using the public tree API, even though automatic
        # writing is missing. The protocol state still needs to survive.
        assistant = add_message(tree, generated.role, generated.content, parent_id=root.id,
                                tool_calls=generated.tool_calls, tool_call_id=generated.tool_call_id)
        followup = add_message(tree, "user", "Recall the root fact", parent_id=assistant.id)
        tree.branches["main"].leaf_message_id = followup.id
        await ReasoningSystem().process(world)
        sent = json.loads(transport.requests[-1].content)["input"]
        assert sent[1:3] == original, "Native tree projection discarded original SDK output items"
    finally:
        await model.aclose()


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="WorldSerializer does not reconstruct native tree message/branch dataclasses")
async def test_native_tree_snapshot_restores_a_usable_active_branch() -> None:
    world, entity, model, _ = await sdk_world()
    tree = ConversationTreeComponent()
    root = add_message(tree, "user", "Shared root")
    leaf = add_message(tree, "assistant", "Saved branch", parent_id=root.id)
    create_branch(tree, "branch-a", leaf.id)
    tree.current_branch_id = "branch-a"
    world.add_component(entity, tree)
    try:
        restored = WorldSerializer.from_dict(
            WorldSerializer.to_dict(world), providers={model.model_id: model}, tool_handlers={},
        )
        recovered = restored.get_component(entity, ConversationTreeComponent)
        assert recovered is not None
        active = get_active_leaf(recovered)
        assert active is not None
        assert [message.content for message in linearize(recovered, active)] == ["Shared root", "Saved branch"]
    finally:
        await model.aclose()
