"""Runtime control feature demonstration.

Demonstrates:
1. Entity registry (register_entity, resolve_entity, list_entities_by_tag)
2. Dynamic system lifecycle (register, remove, replace)
3. Per-entity model switching (pending_provider, pending_model)
4. Graceful interruption (InterruptionComponent, partial content)
5. Conversation tree revert (revert_to_message, tree-aware reasoning)
"""

import asyncio
import os

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.components.definitions import InterruptionComponent
from ecs_agent.conversation_tree import (
    ConversationTreeComponent,
    add_message,
    create_branch,
    get_active_leaf,
    revert_to_message,
    switch_branch,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, InterruptionReason, Message


async def main() -> None:
    """Demonstrate runtime control features with dual-mode provider."""
    print("=== ECS-Agent Runtime Control Demo ===\n")

    # Dual-mode provider selection
    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        base_url = os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = os.getenv("LLM_MODEL", "qwen3.5-flash")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
        print(f"[Provider] OpenAI-compatible: {model}")
    else:
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant", content="Hello! I'm ready to help."
                    )
                ),
                CompletionResult(
                    message=Message(
                        role="assistant", content="This is the second response."
                    )
                ),
            ]
        )
        print("[Provider] FakeProvider (demo mode)")

    world = World()
    runner = Runner()

    # [1] Entity Registry
    print("\n[1] Entity Registry")
    agent = world.create_entity()
    world.register_entity(agent, "demo-agent", tags={"demo", "primary"})
    print(f"  ✓ Registered entity as 'demo-agent' (EntityId: {agent})")
    print(f"  ✓ Tags: {{'demo', 'primary'}}")

    resolved_id = world.resolve_entity("demo-agent")
    print(f"  ✓ Resolved 'demo-agent' → EntityId: {resolved_id}")

    demo_entities = world.list_entities_by_tag("demo")
    print(f"  ✓ Entities with tag 'demo': {demo_entities}")

    # [2] System Lifecycle
    print("\n[2] Dynamic System Lifecycle")
    reasoning = ReasoningSystem(priority=0)
    handle = world.register_system(reasoning, priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)
    print(f"  ✓ Registered ReasoningSystem (handle: {handle})")

    world.apply_pending_system_operations()
    print("  ✓ Applied pending system operations (tick boundary)")

    # [3] Model Switching
    print("\n[3] Per-Entity Model Switching")
    llm = LLMComponent(provider=provider, model="fake" if not api_key else model)
    world.add_component(agent, llm)
    world.add_component(
        agent,
        ConversationComponent(messages=[Message(role="user", content="Hi there!")]),
    )
    print(f"  ✓ Initial model: {llm.model}")

    llm.pending_model = "fake-switched" if not api_key else f"{model}-switched"
    print(f"  ✓ Queued model switch: {llm.pending_model}")

    # Run one tick to apply model switch
    await runner.run(world, max_ticks=1)
    print(f"  ✓ Model switch applied during reasoning")

    # [4] Graceful Interruption
    print("\n[4] Graceful Interruption")
    world.add_component(
        agent,
        InterruptionComponent(
            reason=InterruptionReason.USER_REQUESTED,
            metadata={"partial": True, "reason_detail": "User clicked stop button"},
        ),
    )
    print("  ✓ Added InterruptionComponent (reason: USER_REQUESTED)")

    interrupt = world.get_component(agent, InterruptionComponent)
    if interrupt:
        print(f"  ✓ Interruption metadata: {interrupt.metadata}")

    # [5] Conversation Tree Revert
    print("\n[5] Conversation Tree Revert")
    tree = ConversationTreeComponent()
    msg1 = add_message(tree, role="user", content="First message", parent_id=None)
    msg2 = add_message(
        tree, role="assistant", content="Second message", parent_id=msg1.id
    )
    msg3 = add_message(tree, role="user", content="Third message", parent_id=msg2.id)
    
    # Create and activate branch to enable get_active_leaf/revert
    create_branch(tree, "main", msg3.id)
    switch_branch(tree, "main")
    
    print(f"  ✓ Built tree: root → msg1 → msg2 → msg3")
    
    active_before = get_active_leaf(tree)
    print(f"  ✓ Active leaf before revert: {active_before}")
    
    revert_to_message(tree, msg2.id)
    active_after = get_active_leaf(tree)
    print(f"  ✓ Reverted to msg2")
    print(f"  ✓ Active leaf after revert: {active_after}")

    print("\n=== All Runtime Control Features Demonstrated ===")


if __name__ == "__main__":
    asyncio.run(main())
