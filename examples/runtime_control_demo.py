"""Runtime control demonstration using the ECS-based LLM Agent framework.

This example demonstrates 5 runtime control capabilities:
1. Entity registry (naming and tagging entities)
2. Dynamic system lifecycle (removing and replacing systems at runtime)
3. Per-entity multi-model switching (switching models/providers on the fly)
4. Graceful interruption (stopping processing for specific entities)
5. Conversation-tree revert (navigating and branching conversation history)
"""

import asyncio
import os
import uuid

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    InterruptionComponent,
)
from ecs_agent.conversation_tree import (
    ConversationTreeComponent,
    add_message,
    revert_to_message,
    get_active_leaf,
    linearize,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, InterruptionReason


async def main() -> None:
    """Run the runtime control demo."""
    configure_logging(json_output=False)
    world = World()
    runner = Runner()

    # --- Setup Providers ---
    api_key: str = os.environ.get("LLM_API_KEY", "")
    model: str = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    provider1 = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Response from Model A")
            )
        ]
    )
    provider2 = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Response from Model B")
            )
        ]
    )

    if api_key:
        real_provider = OpenAIProvider(api_key=api_key, model=model)
        provider1 = real_provider
        provider2 = real_provider

    print("\n--- 1. Entity Registry Demo ---")
    agent = world.create_entity()
    world.register_entity(agent, "primary-assistant", tags={"worker", "active"})

    resolved = world.resolve_entity("primary-assistant")
    print(f"Resolved 'primary-assistant' to EntityId: {int(resolved)}")

    workers = world.list_entities_by_tag("worker")
    print(f"Entities with tag 'worker': {[int(e) for e in workers]}")

    print("\n--- 2. Dynamic System Lifecycle Demo ---")
    reasoning_sys = ReasoningSystem(priority=0)
    handle = world.register_system(reasoning_sys, priority=0)
    world.apply_pending_system_operations()
    print(f"Registered ReasoningSystem with handle: {handle}")

    world.remove_system(handle)
    world.apply_pending_system_operations()
    print("Removed ReasoningSystem")

    world.register_system(reasoning_sys, priority=0)
    world.apply_pending_system_operations()
    print("Re-registered ReasoningSystem for next steps")

    print("\n--- 3. Multi-Model Switching Demo ---")
    llm = LLMComponent(provider=provider1, model="model-a")
    world.add_component(agent, llm)
    world.add_component(
        agent,
        ConversationComponent(messages=[Message(role="user", content="Test switch")]),
    )

    print(f"Current model: {llm.model}")
    llm.pending_model = "model-b"
    print(f"Requested switch to: {llm.pending_model}")

    # Run one tick to process switch and get response
    await runner.run(world, max_ticks=1)
    print(f"Model after tick: {llm.model}")

    print("\n--- 4. Graceful Interruption Demo ---")
    world.add_component(
        agent,
        InterruptionComponent(
            reason=InterruptionReason.USER_REQUEST, metadata={"reason": "Manual stop"}
        ),
    )
    print("Added InterruptionComponent to agent")

    # Run tick - ReasoningSystem should skip this entity
    await runner.run(world, max_ticks=1)

    # Remove interruption to allow further demo
    world.remove_component(agent, InterruptionComponent)
    print("Removed InterruptionComponent")

    print("\n--- 5. Conversation Tree Revert Demo ---")
    tree = ConversationTreeComponent()
    world.add_component(agent, tree)

    # Build tree
    m1 = add_message(tree, "user", "Message 1").id
    m2 = add_message(tree, "assistant", "Message 2", parent_id=m1).id
    m3 = add_message(tree, "user", "Message 3", parent_id=m2).id

    print(f"Full path length: {len(linearize(tree, m3))}")
    print(f"Active leaf before revert: {get_active_leaf(tree)}")

    revert_to_message(tree, m2)
    print(f"Reverted to: {m2}")
    print(f"Active leaf after revert: {get_active_leaf(tree)}")
    print(
        f"Linearized path length after revert: {len(linearize(tree, get_active_leaf(tree)))}"
    )

    print("\nDemo Complete.")


if __name__ == "__main__":
    asyncio.run(main())
