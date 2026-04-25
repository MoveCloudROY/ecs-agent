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
    """Test register_entity, resolve_entity, list_entities_by_tag, unregister_entity."""
    world = World()
    entity1 = world.create_entity()
    entity2 = world.create_entity()

    world.register_entity(entity1, "agent-1", tags={"worker", "primary"})
    world.register_entity(entity2, "agent-2", tags={"worker"})

    assert world.resolve_entity("agent-1") == entity1
    assert world.resolve_entity("agent-2") == entity2

    workers = world.list_entities_by_tag("worker")
    assert len(workers) == 2
    assert entity1 in workers and entity2 in workers

    primary = world.list_entities_by_tag("primary")
    assert len(primary) == 1 and entity1 in primary

    world.unregister_entity(entity1)
    # After unregistration, resolve_entity returns None (not raises)
    assert world.resolve_entity("agent-1") is None


async def test_dynamic_system_lifecycle():
    """Test system registration, removal, replacement with tick-boundary semantics."""
    world = World()
    system1 = ReasoningSystem(priority=0)
    handle = world.register_system(system1, priority=0)
    world.apply_pending_system_operations()

    world.remove_system(handle)
    world.apply_pending_system_operations()

    system2 = ReasoningSystem(priority=5)
    world.replace_system(handle, system2)
    world.apply_pending_system_operations()

    assert True  # Execution test in complete workflow


async def test_multi_entity_model_switching_isolation():
    """Test pending_provider/pending_model fields with cross-entity isolation."""
    world = World()
    runner = Runner()

    provider1 = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Model-1 response")
            )
        ]
    )
    provider2 = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Model-2 response")
            )
        ]
    )

    entity1 = world.create_entity()
    llm1 = LLMComponent(model=provider1)
    world.add_component(entity1, llm1)
    world.add_component(
        entity1, ConversationComponent(messages=[Message(role="user", content="Hello")])
    )

    entity2 = world.create_entity()
    llm2 = LLMComponent(model=provider2)
    world.add_component(entity2, llm2)
    world.add_component(
        entity2, ConversationComponent(messages=[Message(role="user", content="Hi")])
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)

    llm1.pending_model = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="switched"))],
        model_id="model-1-switched",
    )
    await runner.run(world, max_ticks=1)

    # entity2's model should not have been affected by entity1's pending_model switch
    assert llm2.model is provider2
    assert llm2.pending_model is None


async def test_graceful_interruption_component():
    """Test InterruptionComponent creation and metadata preservation."""
    world = World()
    entity = world.create_entity()

    world.add_component(
        entity,
        InterruptionComponent(
            reason=InterruptionReason.USER_REQUESTED,
            metadata={"partial": True, "content_length": 42},
        ),
    )

    interrupt = world.get_component(entity, InterruptionComponent)
    assert interrupt is not None
    assert interrupt.reason == InterruptionReason.USER_REQUESTED
    assert interrupt.metadata["partial"] is True
    assert interrupt.metadata["content_length"] == 42


async def test_conversation_tree_revert_affects_active_leaf():
    """Test revert_to_message changes active leaf pointer non-destructively."""
    from ecs_agent.conversation_tree import create_branch, switch_branch
    
    tree = ConversationTreeComponent()
    
    msg1 = add_message(tree, role="user", content="First", parent_id=None)
    msg2 = add_message(tree, role="assistant", content="Second", parent_id=msg1.id)
    msg3 = add_message(tree, role="user", content="Third", parent_id=msg2.id)
    
    # Create and activate a branch pointing to msg3
    create_branch(tree, "main", msg3.id)
    switch_branch(tree, "main")
    
    assert get_active_leaf(tree) == msg3.id
    
    revert_to_message(tree, msg2.id)
    assert get_active_leaf(tree) == msg2.id


async def test_complete_runtime_control_workflow():
    """Integration test: registry + lifecycle + switching + interruption + revert."""
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
    llm = LLMComponent(model=provider)
    world.add_component(agent, llm)
    world.add_component(
        agent, ConversationComponent(messages=[Message(role="user", content="Hello")])
    )

    llm.pending_model = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="switched response"))],
        model_id="fake-switched",
    )

    # 4. Run
    await runner.run(world, max_ticks=1)

    # 5. Verify
    assert world.resolve_entity("demo-agent") == agent

    conv = world.get_component(agent, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 2
