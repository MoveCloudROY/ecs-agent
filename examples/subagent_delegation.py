"""Sub-agent delegation example using the ECS-based LLM Agent framework.

This example demonstrates:
- A manager agent that delegates a research task to a sub-agent
- OwnerComponent linking the sub-agent to its parent
- CollaborationComponent inbox messaging for result delivery
- Multi-phase execution: manager delegates → sub-agent works → manager summarizes

Dual-mode operation:
- Without LLM_API_KEY: Uses FakeProvider for demonstration
- With LLM_API_KEY: Uses OpenAIProvider for real LLM interaction
"""

import asyncio
import os

from ecs_agent.components import (
    CollaborationComponent,
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    TerminalComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.systems.collaboration import CollaborationSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a sub-agent delegation example.

    Flow (3 phases):
      Phase 1 — Manager receives user question, decides to delegate.
      Phase 2 — Sub-agent is spawned, researches the topic, reports findings.
      Phase 3 — Sub-agent's findings are delivered to manager's inbox.
                Manager reads them and produces a final summary.
    """
    world = World()

    # ── LLM Provider Configuration ──────────────────────────────────
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        print(f"Base URL: {base_url}")
        print()
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()

    # ── Systems ──────────────────────────────────────────────────────
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(CollaborationSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()

    # ================================================================
    # Phase 1 — Manager decides to delegate
    # ================================================================
    if api_key:
        manager_provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        manager_provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "Good question. I'll delegate the research on quantum "
                            "computing applications to my sub-agent and summarize "
                            "the findings once they report back."
                        ),
                    )
                ),
            ]
        )

    manager_id = world.create_entity()
    world.add_component(
        manager_id,
        LLMComponent(
            provider=manager_provider,
            model=model if api_key else "fake-manager",
            system_prompt=(
                "You are a manager agent. When given a complex question, "
                "delegate research to your sub-agent and then synthesize "
                "the results into a concise summary."
            ),
        ),
    )
    world.add_component(
        manager_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "What are the most promising near-term applications "
                        "of quantum computing?"
                    ),
                )
            ]
        ),
    )

    # Run one tick — manager produces its delegation response, then
    # FakeProvider is exhausted → TerminalComponent is added.
    await runner.run(world, max_ticks=1)

    print("=" * 60)
    print("Phase 1: Manager decided to delegate")
    print("=" * 60)
    _print_conversation("Manager", manager_id, world)

    # Remove the terminal so the manager can continue later.
    world.remove_component(manager_id, TerminalComponent)

    # ================================================================
    # Phase 2 — Spawn sub-agent to do research
    # ================================================================
    if api_key:
        subagent_provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        subagent_provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "After researching quantum computing applications, I found "
                            "three promising areas: (1) drug discovery through molecular "
                            "simulation, (2) combinatorial optimization for logistics, "
                            "and (3) post-quantum cryptography. Each has active research "
                            "programs and early commercial prototypes."
                        ),
                    )
                ),
            ]
        )

    subagent_id = world.create_entity()
    world.add_component(
        subagent_id,
        LLMComponent(
            provider=subagent_provider,
            model=model if api_key else "fake-researcher",
            system_prompt=(
                "You are a research sub-agent. Investigate the given topic "
                "thoroughly and report your findings back to the manager."
            ),
        ),
    )
    world.add_component(
        subagent_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Research the most promising near-term applications "
                        "of quantum computing. Report your findings."
                    ),
                )
            ]
        ),
    )
    # Link sub-agent to its parent.
    world.add_component(subagent_id, OwnerComponent(owner_id=manager_id))

    # Temporarily remove manager's LLM so only the sub-agent runs
    # reasoning this tick.
    manager_llm = world.get_component(manager_id, LLMComponent)
    assert manager_llm is not None
    world.remove_component(manager_id, LLMComponent)

    await runner.run(world, max_ticks=1)

    print()
    print("=" * 60)
    print("Phase 2: Sub-agent completed research")
    print("=" * 60)
    _print_conversation("Sub-agent", subagent_id, world)

    # ================================================================
    # Phase 3 — Deliver findings to manager → manager summarizes
    # ================================================================

    # Restore manager's LLM with a fresh provider for the summary.
    if api_key:
        summary_provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        summary_provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "Based on my sub-agent's research, here is the summary:\n\n"
                            "Quantum computing has three key near-term applications:\n"
                            "1. Drug discovery — simulating molecular interactions\n"
                            "2. Optimization — logistics and supply-chain routing\n"
                            "3. Cryptography — post-quantum encryption standards\n\n"
                            "These areas are expected to see practical impact within "
                            "the next 5–10 years."
                        ),
                    )
                ),
            ]
        )
    world.add_component(
        manager_id,
        LLMComponent(
            provider=summary_provider,
            model=model if api_key else "fake-manager",
            system_prompt=manager_llm.system_prompt,
        ),
    )

    # Deliver sub-agent's last message into manager's collaboration inbox.
    subagent_conv = world.get_component(subagent_id, ConversationComponent)
    assert subagent_conv is not None
    last_msg = subagent_conv.messages[-1]

    world.add_component(
        manager_id,
        CollaborationComponent(
            peers=[subagent_id],
            inbox=[(subagent_id, last_msg)],
        ),
    )

    # Mark sub-agent as done.
    world.add_component(
        subagent_id,
        TerminalComponent(reason="delegation_complete"),
    )
    # Remove sub-agent's TerminalComponent added by provider exhaustion
    # so our explicit terminal takes effect cleanly.

    # Run final tick: CollaborationSystem delivers message → Reasoning
    # produces the summary.
    await runner.run(world, max_ticks=1)

    print()
    print("=" * 60)
    print("Phase 3: Manager received findings and produced summary")
    print("=" * 60)
    _print_conversation("Manager (final)", manager_id, world)

    # Show the parent-child relationship.
    owner = world.get_component(subagent_id, OwnerComponent)
    if owner is not None:
        print(
            f"\n[OwnerComponent] Sub-agent (entity {subagent_id}) "
            f"→ Manager (entity {owner.owner_id})"
        )


def _print_conversation(label: str, entity_id: int, world: World) -> None:
    """Pretty-print an entity's conversation."""
    print(f"\n--- {label} (entity {entity_id}) ---")
    conv = world.get_component(entity_id, ConversationComponent)
    if conv is None:
        print("  (no conversation)")
        return
    for msg in conv.messages:
        role = msg.role.upper()
        lines = (msg.content or "").split("\n")
        first, rest = lines[0], lines[1:]
        print(f"  [{role}] {first}")
        for line in rest:
            print(f"         {line}")


if __name__ == "__main__":
    asyncio.run(main())
