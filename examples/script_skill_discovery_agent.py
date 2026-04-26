"""Example demonstrating file-based skill discovery with dual-mode LLM model.

This script shows how to use SkillDiscovery to automatically find and install Skill
implementations from a filesystem path. Supports both real LLM (OpenAIModel via
LLM_API_KEY) and FakeModel (default, no credentials required).
"""

import asyncio
import os
from pathlib import Path

from ecs_agent.core import World, Runner
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.discovery import SkillDiscovery
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    world = World()
    manager = SkillManager()

    # --- Load LLM configuration ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    # Path to the directory containing skill implementations
    # In this demo, we point to the examples/skills folder
    skills_path = Path(__file__).parent / "script_skills"

    # 1. Initialize SkillDiscovery
    discovery = SkillDiscovery(skill_paths=[skills_path])

    # 2. Discover skills (returns list[Skill])
    discovered_skills = discovery.discover()
    print(
        f"Discovered {len(discovered_skills)} skills: {[s.name for s in discovered_skills]}"
    )

    # Create an agent
    agent = world.create_entity()

    # 3. Discover and install directly onto the agent
    # This registers tools, adds system prompts, and tracks metadata.
    # Note: For Markdown Skills, DiscoveryManager.auto_discover_and_install()
    # follows the lazy indexing path (metadata only), requiring an explicit
    # manager.activate() before the skill is fully operational.
    installed_names = discovery.discover_and_install(world, agent, manager)
    print(f"Installed skills: {installed_names}")
    # --- Create LLM model ---
    if api_key:
        print(f"Using model: {model}")
        print(f"Base URL: {base_url}")
        model: LLMModel = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        print("No LLM_API_KEY provided. Using FakeModel for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()
        model = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I have loaded the following skills: "
                        + ", ".join(installed_names),
                    )
                )
            ]
        )

    world.add_component(agent, LLMComponent(model=model,
))
    world.add_component(
        agent,
        ConversationComponent(
            messages=[Message(role="user", content="What skills do you have?")]
        ),
    )

    # Register core systems
    world.register_system(ReasoningSystem(), priority=0)

    # Run a tick
    runner = Runner()
    await runner.run(world, max_ticks=1)

    # Verify results
    conv = world.get_component(agent, ConversationComponent)
    if conv:
        print(f"\nFinal Assistant Response: {conv.messages[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
