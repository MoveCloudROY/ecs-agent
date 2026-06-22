"""Agent DSL Markdown example — Load multi-agent config from Markdown files and run subagent delegation.

This example demonstrates:
- Loading agent specifications from Markdown files using Agent DSL
- Markdown format: YAML frontmatter (config) + markdown body (system prompt)
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
from ecs_agent.dsl import compile_agent_specs, load_markdown_agent, resolve_agent_specs
from ecs_agent.logging import configure_logging
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import EntityId, Message


def create_model(model: str, system_prompt: str) -> LLMModel:
    """Create a model from environment variables.

    Args:
        model: Model identifier (e.g., "qwen3.5-flash")
        system_prompt: System prompt for the model (unused at construction time)

    Returns:
        LLMModel configured from environment.
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
    return Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)


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
    """Run Agent DSL Markdown subagent delegation example."""
    configure_logging(json_output=False)

    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")
    print(f"Using model: {model}")

    # Load agent specifications from Markdown files
    config_path = Path(__file__).parent / "assistant.md"
    researcher_path = Path(__file__).parent / "researcher.md"
    print(f"Loading agent configuration from: {config_path}")
    print(f"Loading subagent configuration from: {researcher_path}")

    spec = load_markdown_agent(config_path)
    researcher_spec = load_markdown_agent(researcher_path)
    print("Loaded agent specifications:")
    for s in [spec, researcher_spec]:
        print(f"  - {s.name}: mode={s.mode}, model={s.model}")

    # Compile both specs into ECS World
    # compile_agent_specs auto-wires:
    #   - SystemPromptRenderSystem (priority=-20)
    #   - UserPromptNormalizationSystem (priority=-10)
    #   - ToolRegistryComponent (always)
    #   - SubagentSystem + SubagentSessionTableComponent (when subagents present)
    print("\nCompiling agent specs into ECS World...")
    specs = resolve_agent_specs([spec, researcher_spec])
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
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run the agent
    print("\nRunning agent...")
    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Print results
    print("\n" + "=" * 60)
    print("Manager Conversation (Agent DSL Markdown)")
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
