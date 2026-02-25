"""Demonstration of context management features: undo, resume, and compaction.

This example showcases three advanced features for managing agent execution state:

1. **Undo** — Run agent for 3 ticks, then undo the last tick to revert conversation
2. **Resume** — Save checkpoint to file, load, and continue execution
3. **Compact** — Build up a long conversation and trigger LLM-based summarization

Uses FakeProvider for deterministic, reproducible output.

Usage:
    python examples/context_management_agent.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from ecs_agent.components import (
    CheckpointComponent,
    CompactionConfigComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.checkpoint import CheckpointSystem
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message, Usage


def create_fake_provider() -> FakeProvider:
    """Create a FakeProvider with deterministic responses."""
    return FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="This is response 1 from the agent.",
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="This is response 2 from the agent.",
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="This is response 3 from the agent.",
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            ),
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Summary of conversation so far.",
                ),
                usage=Usage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
            ),
        ]
    )


async def part_1_undo() -> None:
    """Part 1: Demonstrate undo functionality."""
    print("\n" + "=" * 70)
    print("PART 1: UNDO — Revert Agent State")
    print("=" * 70)

    # Create world and agent
    world = World()
    provider = create_fake_provider()

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model="fake",
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[Message(role="user", content="Hello, can you help?")]
        ),
    )
    world.add_component(agent_id, CheckpointComponent(max_snapshots=10))

    # Register systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(CheckpointSystem(), priority=1)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run for 3 ticks
    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Show conversation after 3 ticks
    conv = world.get_component(agent_id, ConversationComponent)
    print(
        f"\nAfter 3 ticks: {len(conv.messages) if conv else 0} messages in conversation"
    )
    if conv:
        for i, msg in enumerate(conv.messages):
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  [{i}] {msg.role}: {content}")

    # Undo the last tick
    print("\nUndoing last tick...")
    await CheckpointSystem.undo(world, {"fake": provider}, {})

    # Show conversation after undo
    conv = world.get_component(agent_id, ConversationComponent)
    print(f"\nAfter undo: {len(conv.messages) if conv else 0} messages in conversation")
    if conv:
        for i, msg in enumerate(conv.messages):
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  [{i}] {msg.role}: {content}")

    print("\n✓ Undo complete — conversation reverted to prior state")


async def part_2_resume() -> None:
    """Part 2: Demonstrate save/resume functionality."""
    print("\n" + "=" * 70)
    print("PART 2: RESUME — Save and Load Agent State")
    print("=" * 70)

    # Create initial world and agent
    world = World()
    provider = create_fake_provider()

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model="fake",
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="First question?"),
            ]
        ),
    )

    # Register systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run for 2 ticks
    runner = Runner()
    await runner.run(world, max_ticks=2)

    # Show initial state
    conv = world.get_component(agent_id, ConversationComponent)
    print(f"\nAfter 2 ticks: {len(conv.messages) if conv else 0} messages")
    if conv:
        for i, msg in enumerate(conv.messages):
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  [{i}] {msg.role}: {content}")

    # Save checkpoint to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        checkpoint_path = f.name

    try:
        runner.save_checkpoint(world, checkpoint_path)
        print(f"\n✓ Checkpoint saved to {checkpoint_path}")

        # Load checkpoint and resume
        loaded_world, current_tick = Runner.load_checkpoint(
            checkpoint_path, {"fake": provider}, {}
        )
        print(f"✓ Checkpoint loaded (current_tick={current_tick})")

        # Create new runner and resume from saved state
        new_runner = Runner()
        await new_runner.run(loaded_world, max_ticks=4, start_tick=current_tick)

        # Show final state
        conv = loaded_world.get_component(agent_id, ConversationComponent)
        print(
            f"\nAfter resume and 2 more ticks: {len(conv.messages) if conv else 0} messages"
        )
        if conv:
            for i, msg in enumerate(conv.messages):
                content = (
                    msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                )
                print(f"  [{i}] {msg.role}: {content}")

        print("\n✓ Resume complete — state preserved and execution continued")

    finally:
        # Cleanup
        Path(checkpoint_path).unlink(missing_ok=True)


async def part_3_compact() -> None:
    """Part 3: Demonstrate conversation compaction."""
    print("\n" + "=" * 70)
    print("PART 3: COMPACT — Summarize Long Conversations")
    print("=" * 70)

    # Create world with many initial messages to trigger compaction
    world = World()
    provider = create_fake_provider()

    agent_id = world.create_entity()

    # Build a long conversation (10+ messages to exceed threshold)
    initial_messages = [
        Message(role="user", content="What is machine learning?"),
        Message(
            role="assistant",
            content="Machine learning is a branch of artificial intelligence that focuses on the development of algorithms and statistical models.",
        ),
        Message(role="user", content="Can you explain neural networks?"),
        Message(
            role="assistant",
            content="Neural networks are computing systems inspired by the biological neural networks in animal brains.",
        ),
        Message(role="user", content="What is deep learning?"),
        Message(
            role="assistant",
            content="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        ),
        Message(role="user", content="Tell me about transformers."),
        Message(
            role="assistant",
            content="Transformers are a type of neural network architecture introduced in 2017 for natural language processing.",
        ),
        Message(role="user", content="What is a token?"),
        Message(
            role="assistant",
            content="A token is a unit of text that an LLM processes, typically a word, subword, or character.",
        ),
    ]

    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model="fake",
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(messages=initial_messages),
    )
    world.add_component(
        agent_id,
        CompactionConfigComponent(
            threshold_tokens=50,  # Low threshold to trigger compaction
            summary_model="fake",
        ),
    )

    print(f"\nBefore compaction: {len(initial_messages)} messages")

    # Register systems
    world.register_system(CompactionSystem(bisect_ratio=0.5), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run one tick to trigger compaction
    await world.process()

    # Show result
    conv = world.get_component(agent_id, ConversationComponent)
    if conv:
        print(f"\nAfter compaction: {len(conv.messages)} messages")
        print("\nCompacted conversation:")
        for i, msg in enumerate(conv.messages):
            content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
            print(f"  [{i}] {msg.role}: {content}")

        # Show if archive was created
        from ecs_agent.components import ConversationArchiveComponent

        archive = world.get_component(agent_id, ConversationArchiveComponent)
        if archive and archive.archived_summaries:
            print(
                f"\n✓ Conversation archive created with {len(archive.archived_summaries)} summary"
            )
            print(f"  Summary: {archive.archived_summaries[0][:80]}...")

    print("\n✓ Compaction complete — conversation summarized")


async def main() -> None:
    """Run all three context management demonstrations."""
    print("\n" + "=" * 70)
    print("CONTEXT MANAGEMENT FEATURES DEMO")
    print("Demonstrating Undo, Resume, and Compaction")
    print("=" * 70)

    await part_1_undo()
    await part_2_resume()
    await part_3_compact()

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
