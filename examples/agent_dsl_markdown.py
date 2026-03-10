"""Agent DSL Markdown example - Load agent configuration from Markdown file.

This example demonstrates:
- Loading agent specification from a Markdown file using Agent DSL
- Markdown format: YAML frontmatter (config) + markdown body (prompt)
- Dual-mode LLM provider selection (FakeProvider or OpenAIProvider)
- Compiling DSL spec into ECS World with primary entity
- Running the agent with conversation history

The Markdown file (assistant.md) contains:
- YAML frontmatter: mode, model, tools, metadata
- Markdown body: System prompt with formatting

Without LLM_API_KEY: Uses FakeProvider for deterministic testing
With LLM_API_KEY: Uses OpenAIProvider with DashScope/Qwen
"""

import asyncio
import os
from pathlib import Path

from ecs_agent.components import ConversationComponent
from ecs_agent.core import Runner
from ecs_agent.dsl import compile_agent_specs, load_markdown_agent
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
                        content=f"Hello! I'm the {model} assistant loaded from Markdown DSL. "
                        f"My system prompt comes from the markdown body, which allows rich formatting. "
                        f"How can I assist you today?",
                    )
                )
            ]
        )


async def main() -> None:
    """Run Agent DSL Markdown example."""
    configure_logging(json_output=False)

    # Load agent specification from Markdown file
    config_path = Path(__file__).parent / "assistant.md"
    print(f"Loading agent configuration from: {config_path}")

    spec = load_markdown_agent(config_path)
    print("Loaded agent specification:")
    print(f"  - Name: {spec.name} (from filename)")
    print(f"  - Mode: {spec.mode}")
    print(f"  - Model: {spec.model}")
    print(f"  - Prompt length: {len(spec.prompt)} characters")
    print(f"  - Tools: {list(spec.tools.keys()) if spec.tools else 'none'}")

    # Compile spec into ECS World
    print("\nCompiling agent spec into ECS World...")
    specs = {spec.name: spec}
    primary_entity, world = compile_agent_specs(specs, provider_factory=create_provider)
    print(f"Created primary entity: {primary_entity}")

    # Add conversation with initial user message
    initial_message = Message(
        role="user",
        content="Hello! What's your purpose and how can you help me?",
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

    # Display the rich markdown prompt that was used
    print("\n" + "=" * 60)
    print("System Prompt (from Markdown body):")
    print("=" * 60)
    print(spec.prompt)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
