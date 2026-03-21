"""UI Design Flow E2E example entrypoint.

Demonstrates a complete workflow for designing UI through an interactive
agent using ECS-based composition with dual-mode provider selection.

Tasks:
- Create World with ReasoningSystem, ToolExecutionSystem, and error handling
- Install ui-navigator and ui-prompt markdown skills (pure prompt, no script tools)
- Install BuiltinToolsSkill for read_file, write_file, edit_file, bash
- Setup interactive input handling via UserInputSystem
- Execute agent loop until TerminalComponent or max_ticks
- Output results to ui-design/ directory via builtin write_file tool
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message
from ecs_agent.skills.discovery import discover_skills
from ecs_agent.skills.manager import SkillManager
from ecs_agent.tools.builtins import BuiltinToolsSkill

from runtime import setup_interactive_input
from artifacts import ensure_output_layout

logger = get_logger(__name__)


async def main() -> None:
    """Run the UI Design Flow E2E example.

    Environment Variables:
        LLM_API_KEY: OpenAI-compatible API key (uses FakeProvider if not set)
        LLM_BASE_URL: API base URL (default: DashScope)
        LLM_MODEL: Model name (default: qwen3.5-flash)
        DEBUG: Set to '1' or 'true' to enable debug-level logging
        UI_DESIGN_FLOW_INTERACTIVE: Set to '0' or 'false' to disable interactive input.
            When not set, auto-detects based on whether stdin is a TTY.
    """
    # Configure logging
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False)

    if debug_mode:
        logger.info("debug_mode_enabled")

    # Create output directory structure
    ensure_output_layout()

    # Create World
    world = World()

    # --- Create LLM provider (dual-mode) ---
    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    provider: LLMProvider
    if api_key:
        logger.info("using_provider", provider="OpenAIProvider", model=model)
        print(f"Using OpenAIProvider with model: {model}")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        logger.info("using_provider", provider="FakeProvider")
        print("No LLM_API_KEY set. Using FakeProvider for demonstration.")
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I'll help you design a beautiful UI. Please describe what you'd like to create.",
                    )
                )
            ]
        )

    # Create Agent Entity
    agent_id = world.create_entity()

    manager = SkillManager()
    workspace_dir = Path(__file__).parent
    skills_root = workspace_dir / ".claude" / "skills"
    discovered_skills = discover_skills([skills_root])
    discovered_by_name = {skill.name: skill for skill in discovered_skills}
    required_skill_names = ("ui-navigator", "ui-prompt")

    missing_skills = [
        skill_name
        for skill_name in required_skill_names
        if skill_name not in discovered_by_name
    ]
    if missing_skills:
        logger.error(
            "required_skills_missing_or_invalid",
            missing_skills=missing_skills,
            skills_root=str(skills_root),
        )
        missing_list = ", ".join(missing_skills)
        raise ValueError(
            "Required skills are missing or invalid under "
            f"{skills_root}: {missing_list}"
        )

    workspace_root = str(workspace_dir)
    for skill_name in required_skill_names:
        skill = discovered_by_name[skill_name]
        skill.resolve_path_references(workspace_root)
        manager.install(world, agent_id, skill)

    # Install builtin file tools so the agent can write output files.
    builtin_skill = BuiltinToolsSkill()
    builtin_skill.bind_workspace(str(workspace_dir))
    manager.install(world, agent_id, builtin_skill)

    # Read initial prompt from file
    prompt_path = Path(__file__).parent / "assets/prompt.txt"
    try:
        initial_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.error("prompt_file_not_found", path=str(prompt_path))
        print(f"Error: Prompt file not found: {prompt_path}")
        sys.exit(1)

    # Add components
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model=model if api_key else "fake",
            system_prompt="You are a UI design expert. Help users create stunning interfaces.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(messages=[Message(role="user", content=initial_prompt)]),
    )

    # Register Systems (priority order: lower = earlier execution)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Setup interactive input handling (optional, based on env var)
    # Default: enabled for backward compatibility with piped input
    # Set UI_DESIGN_FLOW_INTERACTIVE=0 or false to disable for truly automated CI runs
    interactive_mode_str = os.environ.get("UI_DESIGN_FLOW_INTERACTIVE", "1")
    if interactive_mode_str.lower() in ("0", "false"):
        # Explicitly disabled via env var
        if debug_mode:
            logger.info("interactive_input_disabled", reason="env_var_set")
    else:
        # Enabled (default, or explicitly set to 1/true)
        if debug_mode:
            logger.info("interactive_input_enabled", reason="default_or_env_var_set")
        await setup_interactive_input(world, agent_id)

    # Run agent loop
    runner = Runner()
    await runner.run(world, max_ticks=10)

    # Print results
    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        logger.info("conversation_complete", message_count=len(conv.messages))
        print("\nConversation:")
        for msg in conv.messages:
            print(f"  {msg.role}: {msg.content}")
    else:
        logger.warning("no_conversation_found")
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
