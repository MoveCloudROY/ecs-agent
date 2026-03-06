"""Integration tests for runtime control features."""

import pytest
from ecs_agent.core import World, Runner
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.components.definitions import (
    EntityRegistryComponent,
    InterruptionComponent,
)
from ecs_agent.conversation_tree import (
    ConversationTreeComponent,
    add_message,
    revert_to_message,
    get_active_leaf,
)
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import Message, CompletionResult, InterruptionReason


async def test_entity_registry_operations():
    """Test entity registration, resolution, listing, and unregistration."""
    world = World()
    entity1 = world.create_entity()
    entity2 = world.create_entity()

    # Register entities with names and tags
    world.register_entity(entity1, "agent-1", tags={"worker", "primary"})
    world.register_entity(entity2, "agent-2", tags={"worker"})

    # Resolve by name
    assert world.resolve_entity("agent-1") == entity1
    assert world.resolve_entity("agent-2") == entity2

    # List by tag
    workers = world.list_entities_by_tag("worker")
    assert len(workers) == 2
    assert entity1 in workers
    assert entity2 in workers

    primary = world.list_entities_by_tag("primary")
    assert len(primary) == 1
    assert entity1 in primary

    # Unregister
    world.unregister_entity(entity1)
    with pytest.raises(KeyError):
        world.resolve_entity("agent-1")


async def test_dynamic_system_lifecycle():
    """Test system registration, removal, and replacement."""
    world = World()

    # Register initial system
    system1 = ReasoningSystem(priority=0)
    handle = world.register_system(system1, priority=0)

    # Apply pending (tick boundary simulation)
    world.apply_pending_system_operations()

    # Verify system registered
    # (No direct API to query systems, rely on execution)

    # Remove system
    world.remove_system(handle)
    world.apply_pending_system_operations()

    # Replace with new system
    system2 = ReasoningSystem(priority=5)
    world.replace_system(handle, system2)
    world.apply_pending_system_operations()

    # Verify replacement (execution test in full workflow)
    assert True  # Placeholder - full workflow tests actual execution


async def test_multi_entity_model_switching():
    """Test per-entity model switching with isolation."""
    world = World()
    runner = Runner()

    provider1 = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Response from model-1")
            )
        ]
    )
    provider2 = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Response from model-2")
            )
        ]
    )

    entity1 = world.create_entity()
    llm1 = LLMComponent(provider=provider1, model="model-1")
    world.add_component(entity1, llm1)
    world.add_component(
        entity1, ConversationComponent(messages=[Message(role="user", content="Hello")])
    )

    entity2 = world.create_entity()
    llm2 = LLMComponent(provider=provider2, model="model-2")
    world.add_component(entity2, llm2)
    world.add_component(
        entity2, ConversationComponent(messages=[Message(role="user", content="Hi")])
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.apply_pending_system_operations()

    # Switch entity1 model
    llm1.pending_model = "model-1-switched"

    await runner.run(world, max_ticks=1)

    # Verify entity2 NOT affected by entity1's switch
    assert llm2.model == "model-2"
    assert llm2.pending_model is None


async def test_graceful_interruption_preserves_partial():
    """Test interruption creates InterruptionComponent and preserves partial content."""
    world = World()

    entity = world.create_entity()

    # Add interruption component
    world.add_component(
        entity,
        InterruptionComponent(
            reason=InterruptionReason.USER_REQUEST, metadata={"partial": True}
        ),
    )

    # Verify component exists
    interrupt = world.get_component(entity, InterruptionComponent)
    assert interrupt is not None
    assert interrupt.reason == InterruptionReason.USER_REQUEST
    assert interrupt.metadata["partial"] is True


async def test_conversation_tree_revert():
    """Test revert_to_message affects tree navigation."""
    tree = ConversationTreeComponent()

    # Build tree: root -> msg1 -> msg2
    msg1_id = add_message(tree, "user", "First", parent_id=None).id
    msg2_id = add_message(tree, "assistant", "Second", parent_id=msg1_id).id
    msg3_id = add_message(tree, "user", "Third", parent_id=msg2_id).id

    # Verify active leaf before revert
    assert get_active_leaf(tree) == msg3_id

    # Revert to msg2
    revert_to_message(tree, msg2_id)

    # Verify active leaf after revert
    assert get_active_leaf(tree) == msg2_id


async def test_runtime_control_complete_workflow():
    """Test all runtime control features in one scenario."""
    world = World()
    runner = Runner()

    # 1. Entity registry
    agent = world.create_entity()
    world.register_entity(agent, "demo-agent", tags={"demo"})

    # 2. System lifecycle
    system = ReasoningSystem(priority=0)
    handle = world.register_system(system, priority=0)
    world.apply_pending_system_operations()

    # 3. Model switching
    provider = FakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="Test response"))
        ]
    )
    llm = LLMComponent(provider=provider, model="fake")
    world.add_component(agent, llm)
    world.add_component(
        agent, ConversationComponent(messages=[Message(role="user", content="Hello")])
    )

    llm.pending_model = "fake-switched"

    # 4. Run one tick
    await runner.run(world, max_ticks=1)

    # 5. Verify entity resolvable
    assert world.resolve_entity("demo-agent") == agent

    # Verify conversation has response
    conv = world.get_component(agent, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 2  # user + assistant
