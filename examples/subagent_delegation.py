"""Sub-agent delegation example using the ECS-based LLM Agent framework.

This example demonstrates:
- A manager agent that delegates a research task to a sub-agent using the 'delegate' tool
- SubagentSystem auto-registration of the delegate tool
- Tool-driven roundtrip workflow: manager calls delegate → SubagentSystem executes child → result returned as tool response
- OwnerComponent linking the sub-agent to its parent

Dual-mode operation:
- Without LLM_API_KEY: Uses FakeProvider for demonstration
- With LLM_API_KEY: Uses OpenAIProvider for real LLM interaction
"""

import asyncio
import os

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OwnerComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
    LLMComponent,
    SubagentRegistryComponent,
    ToolRegistryComponent,
    OwnerComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.openai_provider import OpenAIProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    InheritancePolicy,
    Message,
    SubagentConfig,
    ToolCall,
)


async def main() -> None:
    """Run a sub-agent delegation example using delegate-tool workflow.

    Flow:
      1. Manager receives user question
      2. Manager calls delegate tool
      3. SubagentSystem creates and executes child entity
      4. Tool result delivered to manager conversation
      5. Manager synthesizes final summary
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

    # ── Provider Setup ──────────────────────────────────────────────
    if api_key:
        manager_provider = OpenAIProvider(
            api_key=api_key, base_url=base_url, model=model
        )
        subagent_provider = OpenAIProvider(
            api_key=api_key, base_url=base_url, model=model
        )
    else:
        # FakeProvider for manager: first response calls delegate tool, second produces summary
        manager_provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I'll delegate this research to my subagent.",
                        tool_calls=[
                            ToolCall(
                                id="call_001",
                                name="delegate",
                                arguments={
                                    "subagent_name": "researcher",
                                    "task": (
                                        "Research the most promising near-term applications "
                                        "of quantum computing. Report your findings."
                                    ),
                                },
                            )
                        ],
                    )
                ),
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

        # FakeProvider for subagent (used by SubagentSystem when delegate tool is called)
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

    # ── Manager Entity Setup ────────────────────────────────────────
    manager_id = world.create_entity()

    world.add_component(
        manager_id,
        LLMComponent(
            provider=manager_provider,
            model=model if api_key else "fake-manager",
            system_prompt=(
                "You are a manager agent. When given a complex question, "
                "use the 'delegate' tool to assign research to your 'researcher' subagent. "
                "After receiving the results, synthesize them into a concise summary."
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

    # Configure subagent registry with explicit inheritance policy
    world.add_component(
        manager_id,
        SubagentRegistryComponent(
            subagents={
                "researcher": SubagentConfig(
                    name="researcher",
                    provider=subagent_provider,
                    model=model if api_key else "fake-researcher",
                    system_prompt=(
                        "You are a research sub-agent. Investigate the given topic "
                        "thoroughly and report your findings back to the manager."
                    ),
                    max_ticks=10,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=True,  # Child will see manager's system prompt too
                        inherit_tools=[],  # No parent tools inherited in this example
                        allow_delegate_tool=True,  # Default, allows child to have its own delegate tool
                    ),
                )
            }
        ),
    )

    # ToolRegistryComponent required for delegate tool registration
    world.add_component(manager_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(
        manager_id,
        SubagentRegistryComponent(
            subagents={
                "researcher": SubagentConfig(
                    name="researcher",
                    provider=subagent_provider,
                    model=model if api_key else "fake-researcher",
                    system_prompt=(
                        "You are a research sub-agent. Investigate the given topic "
                        "thoroughly and report your findings back to the manager."
                    ),
                    max_ticks=10,
                    skills=[],  # Could inherit skills like 'web_search' if parent had them
                )
            }
        ),
    )
    world.add_component(
        manager_id,
        SubagentRegistryComponent(
            subagents={
                "researcher": SubagentConfig(
                    name="researcher",
                    provider=subagent_provider,
                    model=model if api_key else "fake-researcher",
                    system_prompt=(
                        "You are a research sub-agent. Investigate the given topic "
                        "thoroughly and report your findings back to the manager."
                    ),
                    max_ticks=10,
                    skills=[],
                )
            }
        ),
    )

    # ToolRegistryComponent required for SubagentSystem to auto-register delegate tool
    world.add_component(manager_id, ToolRegistryComponent(tools={}, handlers={}))

    # ── Systems Registration ────────────────────────────────────────
    subagent_system = SubagentSystem(priority=-1)
    world.register_system(subagent_system, priority=-1)

    # Explicitly install delegate tool (demonstrates installer API)
    # Note: SubagentSystem would also auto-register this if we skipped this call.
    subagent_system.install_delegate_tool(world, manager_id, tool_name="delegate")
    # SubagentSystem priority=-1 ensures delegate tool registered BEFORE ReasoningSystem runs
    world.register_system(SubagentSystem(priority=-1), priority=-1)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # ── Run Agent ───────────────────────────────────────────────────
    runner = Runner()
    await runner.run(world, max_ticks=20)

    # ── Print Results ───────────────────────────────────────────────
    print("=" * 60)
    print("Manager Conversation (Delegate-Tool Roundtrip)")
    print("=" * 60)
    _print_conversation("Manager", manager_id, world)

    # Show parent-child relationship
    for entity_id, components in world.query(OwnerComponent):
        (owner_comp,) = components
        print(
            f"\n[OwnerComponent] Sub-agent (entity {entity_id}) "
            f"→ Manager (entity {owner_comp.owner_id})"
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
        if msg.tool_calls:
            print(f"  [{role}] {msg.content or '(no content)'}")
            for tool_call in msg.tool_calls:
                print(f"         → Tool Call: {tool_call.name}({tool_call.arguments})")
        elif msg.role == "tool":
            print(f"  [{role}] (tool_call_id={msg.tool_call_id})")
            lines = (msg.content or "").split("\n")
            first, rest = lines[0], lines[1:]
            print(f"         {first}")
            for line in rest:
                print(f"         {line}")
        else:
            lines = (msg.content or "").split("\n")
            first, rest = lines[0], lines[1:]
            print(f"  [{role}] {first}")
            for line in rest:
                print(f"         {line}")


if __name__ == "__main__":
    asyncio.run(main())
