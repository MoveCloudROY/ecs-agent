"""Agent DSL JSON example - Load agent configuration from JSON file.

This example demonstrates:
- Loading agent specifications from a JSON file using Agent DSL
- Dual-mode LLM provider selection (FakeProvider or OpenAIProvider)
- Compiling DSL specs into ECS World with primary entity + subagent registry
- Running the agent with conversation history

The JSON file defines:
- "assistant": Primary agent with tool permissions
- "researcher": Subagent specialist configuration

Without LLM_API_KEY: Uses FakeProvider for deterministic testing
With LLM_API_KEY: Uses OpenAIProvider with DashScope/Qwen
"""

import asyncio
import os
from pathlib import Path

from ecs_agent.components import ConversationComponent
from ecs_agent.core import Runner
from ecs_agent.dsl import compile_agent_specs, load_json_agents, resolve_agent_specs
from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


def create_provider(model: str, system_prompt: str) -> LLMProvider:
    """Create LLM provider based on environment variables.

    Args:
        model: Model identifier (e.g., "qwen3.5-flash")
        system_prompt: System prompt for the provider

    Returns:
        LLMProvider instance (OpenAIProvider or FakeProvider)
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        return OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print(f"No LLM_API_KEY set. Using FakeProvider for {model}")
        return FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=f"Hello! I'm the {model} assistant loaded from JSON DSL. "
                        f"I'm ready to help you with your tasks. What would you like to know?",
                    )
                )
            ]
        )


async def main() -> None:
    """Run Agent DSL JSON example."""
    configure_logging(json_output=False)

    # Load agent specifications from JSON file
    config_path = Path(__file__).parent / "agents_config.json"
    print(f"Loading agent configuration from: {config_path}")

    spec_list = load_json_agents(config_path)
    specs = resolve_agent_specs(spec_list)
    print(f"Loaded {len(specs)} agent specifications:")
    for name, spec in specs.items():
        print(f"  - {name}: mode={spec.mode}, model={spec.model}")
    # Compile specs into ECS World
    print("\nCompiling agent specs into ECS World...")
    primary_entity, world = compile_agent_specs(specs, provider_factory=create_provider)
    # Note: compile_agent_specs auto-registers SystemPromptRenderSystem (priority=-20)
    # and UserPromptNormalizationSystem (priority=-10) when placeholders/triggers are present.
    print(f"Created primary entity: {primary_entity}")

    # Add conversation with initial user message
    initial_message = Message(
        role="user",
        content="Hello! Can you tell me about your capabilities?",
    )
    world.add_component(
        primary_entity,
        ConversationComponent(messages=[initial_message]),
    )

    # Register systems
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run the agent
    print("\nRunning agent...")
    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Print conversation history
    conversation = world.get_component(primary_entity, ConversationComponent)
    if conversation:
        print("\n" + "=" * 60)
        print("Conversation History:")
        print("=" * 60)
        for msg in conversation.messages:
            role_display = msg.role.upper()
            print(f"\n[{role_display}]")
            print(msg.content)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
