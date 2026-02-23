"""Serialization demo: Save and load World state with multiple entities and components.

This example demonstrates the full serialization cycle:
- Create a World with multiple entities and various component types
- Save the World state to JSON using WorldSerializer.save()
- Load the World state from JSON using WorldSerializer.load()
- Verify that loaded state matches original state

Key features:
- Multiple entities (agent, sub-agent, collaboration peer)
- Multiple component types: LLMComponent, ConversationComponent, PlanComponent, etc.
- Non-serializable fields (provider, handlers) are handled gracefully
- Provider re-injection on deserialization
- No API key required (uses FakeProvider)

Usage:
    python examples/serialization_demo.py

Environment variables:
    None required - uses FakeProvider for demonstration
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    ErrorComponent,
    KVStoreComponent,
    LLMComponent,
    OwnerComponent,
    PlanComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.serialization import WorldSerializer
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    Message,
    ToolCall,
    ToolSchema,
    Usage,
)


async def main() -> None:
    """Demonstrate World serialization and deserialization."""

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="This is a response."),
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )
        ]
    )

    async def dummy_tool(arg: str) -> str:
        return f"Tool result for {arg}"

    tool_handlers = {"dummy_tool": dummy_tool}

    print("=" * 60)
    print("STEP 1: CREATE ORIGINAL WORLD")
    print("=" * 60)

    world = World()

    main_agent = world.create_entity()
    print(f"Created main_agent (EntityId: {main_agent})")

    world.add_component(
        main_agent,
        LLMComponent(
            provider=provider, model="fake-model", system_prompt="You are helpful."
        ),
    )
    world.add_component(
        main_agent,
        ConversationComponent(
            messages=[
                Message(role="user", content="Hello, can you help?"),
                Message(role="assistant", content="Of course! What do you need?"),
            ],
            max_messages=50,
        ),
    )
    world.add_component(
        main_agent,
        PlanComponent(
            steps=["Understand the problem", "Develop solution", "Implement it"],
            current_step=1,
            completed=False,
        ),
    )
    world.add_component(
        main_agent,
        SystemPromptComponent(content="You are a helpful research assistant."),
    )
    world.add_component(
        main_agent,
        ToolRegistryComponent(
            tools={
                "dummy_tool": ToolSchema(
                    name="dummy_tool",
                    description="A dummy tool for testing",
                    parameters={
                        "type": "object",
                        "properties": {
                            "arg": {"type": "string", "description": "An argument"}
                        },
                        "required": ["arg"],
                    },
                )
            },
            handlers=tool_handlers,
        ),
    )
    world.add_component(
        main_agent,
        KVStoreComponent(store={"key1": "value1", "key2": 42}),
    )
    world.add_component(
        main_agent,
        ToolResultsComponent(results={"call_1": "Tool executed successfully"}),
    )

    second_agent = world.create_entity()
    print(f"Created second_agent (EntityId: {second_agent})")

    world.add_component(
        second_agent,
        LLMComponent(provider=provider, model="fake-model-2"),
    )
    world.add_component(
        second_agent,
        ConversationComponent(messages=[], max_messages=100),
    )
    world.add_component(
        second_agent,
        OwnerComponent(owner_id=main_agent),
    )

    peer_agent = world.create_entity()
    print(f"Created peer_agent (EntityId: {peer_agent})")

    world.add_component(
        peer_agent,
        CollaborationComponent(
            peers=[main_agent],
            inbox=[(main_agent, Message(role="user", content="Collaboration message"))],
        ),
    )
    world.add_component(
        peer_agent,
        ErrorComponent(
            error="Test error", system_name="TestSystem", timestamp=1234567890.0
        ),
    )
    world.add_component(
        peer_agent,
        TerminalComponent(reason="Completed successfully"),
    )

    print(
        f"\nOriginal World has {len([main_agent, second_agent, peer_agent])} entities"
    )
    for entity_id in [main_agent, second_agent, peer_agent]:
        llm = world.get_component(entity_id, LLMComponent)
        conv = world.get_component(entity_id, ConversationComponent)
        print(
            f"  Entity {entity_id}: "
            f"LLM={llm is not None} ({llm.model if llm else 'N/A'}), "
            f"Conv={conv is not None} ({len(conv.messages) if conv else 0} messages)"
        )

    print("\n" + "=" * 60)
    print("STEP 2: SAVE WORLD TO JSON")
    print("=" * 60)

    json_file = Path("world_state.json")
    try:
        WorldSerializer.save(world, json_file)
        print(f"✓ World saved to {json_file}")

        with open(json_file) as f:
            data = json.load(f)
        print(f"✓ JSON contains {len(data['entities'])} entities")

        for entity_id_str, components in data["entities"].items():
            print(f"  Entity {entity_id_str}: {list(components.keys())}")

        print("\n" + "=" * 60)
        print("STEP 3: LOAD WORLD FROM JSON")
        print("=" * 60)

        loaded_world = WorldSerializer.load(
            json_file,
            providers={"fake-model": provider, "fake-model-2": provider},
            tool_handlers=tool_handlers,
        )
        print(f"✓ World loaded from {json_file}")

        print("\n" + "=" * 60)
        print("STEP 4: VERIFY LOADED STATE")
        print("=" * 60)

        loaded_entities = [main_agent, second_agent, peer_agent]
        print(f"\nEntity count check:")
        print(f"  Original: {len(loaded_entities)} entities")
        print(f"  Loaded: {len(loaded_entities)} entities")

        print(f"\nComponent integrity check:")
        all_passed = True

        for entity_id in loaded_entities:
            llm = loaded_world.get_component(entity_id, LLMComponent)
            conv = loaded_world.get_component(entity_id, ConversationComponent)
            plan = loaded_world.get_component(entity_id, PlanComponent)
            kv = loaded_world.get_component(entity_id, KVStoreComponent)
            collab = loaded_world.get_component(entity_id, CollaborationComponent)
            error = loaded_world.get_component(entity_id, ErrorComponent)
            owner = loaded_world.get_component(entity_id, OwnerComponent)
            terminal = loaded_world.get_component(entity_id, TerminalComponent)

            if entity_id == main_agent:
                checks = [
                    ("LLMComponent", llm is not None and llm.model == "fake-model"),
                    (
                        "ConversationComponent",
                        conv is not None and len(conv.messages) == 2,
                    ),
                    ("PlanComponent", plan is not None and plan.current_step == 1),
                    (
                        "KVStoreComponent",
                        kv is not None and kv.store["key1"] == "value1",
                    ),
                ]
            elif entity_id == second_agent:
                checks = [
                    ("LLMComponent", llm is not None and llm.model == "fake-model-2"),
                    (
                        "ConversationComponent",
                        conv is not None and len(conv.messages) == 0,
                    ),
                    (
                        "OwnerComponent",
                        owner is not None and owner.owner_id == main_agent,
                    ),
                ]
            else:
                checks = [
                    (
                        "CollaborationComponent",
                        collab is not None and len(collab.peers) == 1,
                    ),
                    (
                        "ErrorComponent",
                        error is not None and error.error == "Test error",
                    ),
                    (
                        "TerminalComponent",
                        terminal is not None
                        and terminal.reason == "Completed successfully",
                    ),
                ]

            for check_name, result in checks:
                status = "✓" if result else "✗"
                print(f"  Entity {entity_id} - {check_name}: {status}")
                all_passed = all_passed and result

        print(f"\n{'✓ All checks passed!' if all_passed else '✗ Some checks failed'}")

    finally:
        print("\n" + "=" * 60)
        print("STEP 5: CLEANUP")
        print("=" * 60)
        if json_file.exists():
            os.remove(json_file)
            print(f"✓ Cleaned up {json_file}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
