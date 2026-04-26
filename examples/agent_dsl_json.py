"""Agent DSL JSON example — Load multi-agent config from JSON and run subagent delegation.

This example demonstrates:
- Loading agent specifications from a JSON file using Agent DSL
- Compiler auto-wiring: ToolRegistryComponent, SubagentSessionTableComponent, SubagentSystem
- Manager agent delegates a research task via the 'subagent' tool
- SystemPromptRenderSystem resolving ${_installed_subagents} placeholder
- OwnerComponent parent-child relationship between manager and spawned sub-agent

Requires LLM_API_KEY to be set. Optionally set LLM_BASE_URL and LLM_MODEL.
"""

import asyncio
import os
from pathlib import Path

from ecs_agent.components import ConversationComponent, OwnerComponent
from ecs_agent.core import Runner, World
from ecs_agent.dsl import compile_agent_specs, load_json_agents, resolve_agent_specs
from ecs_agent.logging import configure_logging
from ecs_agent.providers import OpenAIModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import EntityId, Message


def create_model(model: str, system_prompt: str) -> LLMModel:
    """Create an OpenAIModel from environment variables.

    Args:
        model: Model identifier (e.g., "qwen3.5-flash")
        system_prompt: System prompt for the model (unused at construction time)

    Returns:
        OpenAIModel configured from environment.
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        print("Error: LLM_API_KEY is not set.")
        print(
            "Set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) to run this example."
        )
        raise SystemExit(1)
    return OpenAIModel(config=ProviderConfig(provider_id="openai", base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS), model=model)


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


async def main() -> None:
    """Run Agent DSL JSON subagent delegation example."""
    configure_logging(json_output=False)

    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")
    print(f"Using model: {model}")

    # Load agent specifications from JSON file
    config_path = Path(__file__).parent / "agents_config.json"
    print(f"Loading agent configuration from: {config_path}")

    spec_list = load_json_agents(config_path)
    specs = resolve_agent_specs(spec_list)
    print(f"Loaded {len(specs)} agent specification(s):")
    for name, spec in specs.items():
        print(f"  - {name}: mode={spec.mode}, model={spec.model}")

    # Compile specs into ECS World
    # compile_agent_specs auto-wires:
    #   - SystemPromptRenderSystem (priority=-20)
    #   - UserPromptNormalizationSystem (priority=-10)
    #   - ToolRegistryComponent (always)
    #   - SubagentSystem + SubagentSessionTableComponent (when subagents present)
    print("\nCompiling agent specs into ECS World...")
    primary_entity, world = compile_agent_specs(specs, model_factory=create_model)
    print(f"Created primary entity: {primary_entity}")

    # Add conversation with user's research question
    world.add_component(
        primary_entity,
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

    # Register runtime systems (prompt/subagent systems registered by compiler)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run the agent
    print("\nRunning agent...")
    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Print results
    print("\n" + "=" * 60)
    print("Manager Conversation (Agent DSL JSON)")
    print("=" * 60)
    _print_conversation("Manager", primary_entity, world)

    # Show parent-child relationship from subagent delegation
    for entity_id, components in world.query(OwnerComponent):
        (owner_comp,) = components
        print(
            f"\n[OwnerComponent] Sub-agent (entity {entity_id}) "
            f"→ Manager (entity {owner_comp.owner_id})"
        )


if __name__ == "__main__":
    asyncio.run(main())
