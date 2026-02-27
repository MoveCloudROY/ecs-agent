"""
Example demonstrating file-based skill discovery and installation.

This script shows how to use the SkillDiscovery class to automatically find
and install Skill implementations from a filesystem path.
"""

import asyncio
from pathlib import Path

from ecs_agent.core import World, Runner
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.providers import FakeProvider
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.discovery import SkillDiscovery
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    world = World()
    manager = SkillManager()

    # Path to the directory containing skill implementations
    # In this demo, we point to the examples/skills folder
    skills_path = Path(__file__).parent / "skills"

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
    # This registers tools, adds system prompts, and tracks metadata
    installed_names = discovery.discover_and_install(world, agent, manager)
    print(f"Installed skills: {installed_names}")

    # Setup agent components
    provider = FakeProvider(
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

    world.add_component(agent, LLMComponent(provider=provider, model="fake"))
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
