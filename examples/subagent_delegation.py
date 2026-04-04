"""Sub-agent delegation example using the ECS-based LLM Agent framework.

This example demonstrates:
- A manager agent that delegates a research task to a sub-agent using the 'subagent' tool.
- SystemPromptRenderSystem resolving ${name} placeholders (including ${_installed_subagents}).
- UserPromptNormalizationSystem normalizing outbound user messages.
- SubagentSystem unified API for session management and background execution.
- Tool-driven roundtrip workflow: manager calls subagent → SubagentSystem executes child → result retrieved via session tools.
- OwnerComponent linking the sub-agent to its parent.

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
    SubagentSessionTableComponent,
    ToolRegistryComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
)
from ecs_agent.providers import FakeProvider
from ecs_agent.providers import OpenAIProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    InheritancePolicy,
    Message,
    SubagentConfig,
    ToolCall,
)


async def main() -> None:
    """Run a sub-agent delegation example using the unified subagent tool.

    Flow:
      1. SystemPromptRenderSystem resolves ${_installed_subagents} and custom
         placeholders into the manager's rendered system prompt.
      2. UserPromptNormalizationSystem normalises the outbound user message.
      3. Manager receives user question and calls the 'subagent' tool.
      4. SubagentSystem creates and executes the child entity.
      5. Result is delivered back to the manager conversation.
      6. Manager synthesises a final summary.
    """
    world = World()

    # ── LLM Provider Configuration ──────────────────────────────────
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen3.5-flash"

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        print(f"Base URL: {base_url}")
        print()
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()

    # ── Provider Setup ──────────────────────────────────────────────
    manager_provider: LLMProvider
    subagent_provider: LLMProvider

    if api_key:
        manager_provider = OpenAIProvider(
            config=ProviderConfig(
                provider_id="openai",
                base_url=base_url,
                api_key=api_key,
                api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            model=model,
        )
        subagent_provider = OpenAIProvider(
            config=ProviderConfig(
                provider_id="openai",
                base_url=base_url,
                api_key=api_key,
                api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            model=model,
        )
    else:
        # FakeProvider for manager: first response calls the 'subagent' tool,
        # then produces a final summary once results are available.
        manager_provider = FakeProvider(
            responses=[
                # 1. Call 'subagent' tool (background=True)
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I'll start a deep-dive research task in the background.",
                        tool_calls=[
                            ToolCall(
                                id="call_async_001",
                                name="subagent",
                                arguments={
                                    "category": "researcher",
                                    "prompt": "Research the most promising near-term applications of quantum computing.",
                                    "background": True,
                                },
                            )
                        ],
                    )
                ),
                # 2. Final summary
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "Based on the research, here is the summary:\n\n"
                            "Quantum computing uses qubits to perform calculations impossible for classical computers.\n"
                            "Key near-term applications include:\n"
                            "1. Drug discovery — simulating molecular interactions\n"
                            "2. Optimization — logistics and supply-chain routing\n"
                            "3. Cryptography — post-quantum encryption standards"
                        ),
                    )
                ),
            ]
        )
        # Pre-wire the tool response for the subagent call
        getattr(manager_provider, "add_tool_response")(
            "subagent",
            '{"session_id": "session_001", "status": "Working", "category": "researcher"}',
        )

        # FakeProvider for the researcher subagent
        subagent_provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "After researching quantum computing applications, I found "
                            "three promising areas: (1) drug discovery through molecular "
                            "simulation, (2) combinatorial optimization for logistics, "
                            "and (3) post-quantum cryptography."
                        ),
                    )
                ),
            ]
        )

    # ── Manager Entity Setup ────────────────────────────────────────
    manager_id = world.create_entity()

    # LLMComponent — system_prompt left empty; rendered by SystemPromptRenderSystem
    world.add_component(
        manager_id,
        LLMComponent(
            provider=manager_provider,
            model=model if api_key else "fake-manager",
        ),
    )

    # SystemPromptConfigSpec — uses ${_installed_subagents} builtin placeholder
    world.add_component(
        manager_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "You are a manager agent. When given a complex question, "
                    "use the 'subagent' tool to delegate work to background workers. "
                    "After receiving the results, synthesize them into a concise summary.\n\n"
                    "Available tools:\n${_installed_tools}\n\n"
                    "Available subagents:\n${_installed_subagents}\n\n"
                    "Session: ${session_label}"
                )
            ),
            placeholders=[
                PlaceholderSpec(name="session_label", value="subagent-delegation-demo"),
            ],
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

    # UserPromptConfigComponent — opts the manager into user-prompt normalization
    world.add_component(
        manager_id,
        UserPromptConfigComponent(),
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
                    description="Research and gather information on any topic",
                    system_prompt=(
                        "You are a research sub-agent. Investigate the given topic "
                        "thoroughly and report your findings back to the manager."
                    ),
                    max_ticks=10,
                    inheritance_policy=InheritancePolicy(
                        inherit_system_prompt=True,
                        inherit_tools=[],
                    ),
                )
            }
        ),
    )

    # ToolRegistryComponent and SessionTable required for subagent tools
    world.add_component(manager_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(manager_id, SubagentSessionTableComponent(sessions={}))

    # ── Systems Registration ────────────────────────────────────────
    # Prompt rendering systems run first (negative priority)
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)

    subagent_system = SubagentSystem(priority=-1)
    world.register_system(subagent_system, priority=-1)

    # Install the unified 'subagent' tool (async/background delegation)
    subagent_system.install_subagent_tool(world, manager_id, tool_name="subagent")
    subagent_system.install_subagent_control_tools(world, manager_id)

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # ── Run Agent ───────────────────────────────────────────────────
    runner = Runner()
    await runner.run(world, max_ticks=10)

    # ── Print Results ───────────────────────────────────────────────
    print("=" * 60)
    print("Manager Conversation (Unified Subagent API)")
    print("=" * 60)
    _print_conversation("Manager", manager_id, world)

    # Show parent-child relationship
    for entity_id, components in world.query(OwnerComponent):
        (owner_comp,) = components
        print(
            f"\n[OwnerComponent] Sub-agent (entity {entity_id}) "
            f"→ Manager (entity {owner_comp.owner_id})"
        )


def _print_conversation(label: str, entity_id: EntityId, world: World) -> None:
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
