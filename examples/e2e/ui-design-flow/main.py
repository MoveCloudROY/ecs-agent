"""UI Design Flow E2E example entrypoint.

Demonstrates a complete workflow for designing UI through an interactive
agent using ECS-based composition with dual-mode model selection.

Features exercised:
- SystemPromptConfigSpec with ${name} placeholder templates
- Built-in placeholders: ${_installed_tools}, ${_installed_skills}
- Progressive disclosure: skills listed by name only; full details via load_skill_details tool
- UserPromptConfigComponent with TriggerSpec-based keyword injection
- SystemPromptRenderSystem (priority -20) and UserPromptNormalizationSystem (priority -10)
- Skill discovery + SkillManager lifecycle (ui-navigator, ui-prompt, BuiltinToolsSkill)
- Interactive input handling via UserInputSystem
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.prompts.contracts import (
    PromptTemplateSource,
    SystemPromptConfigSpec,
    TriggerSpec,
)
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.skills.discovery import discover_skills
from ecs_agent.skills.manager import SkillManager
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import CompletionResult, EntityId, Message

from runtime import setup_interactive_input

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt system helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a UI design expert. Help users create stunning, modern interfaces.

## Available Tools
${_installed_tools}

## Available Skills
${_installed_skills}

To get full instructions and tool schemas for a skill, call: load_skill_details(skill_name="<name>")\
"""


def _build_system_prompt_config() -> SystemPromptConfigSpec:
    """Build a SystemPromptConfigSpec using the base template.

    Skills are listed by name/description only (progressive disclosure).
    The LLM calls load_skill_details(skill_name=...) to fetch full instructions
    and tool schemas on demand.
    """
    return SystemPromptConfigSpec(
        template_source=PromptTemplateSource(inline=_SYSTEM_PROMPT_TEMPLATE),
    )


async def main() -> None:
    """Run the UI Design Flow E2E example.

    Environment Variables:
        LLM_API_KEY: OpenAI-compatible API key (uses FakeModel if not set)
        LLM_BASE_URL: API base URL (default: DashScope)
        LLM_MODEL: Model name (default: qwen3.5-flash)
        DEBUG: Set to '1' or 'true' to enable debug-level logging
        UI_DESIGN_FLOW_INTERACTIVE: Set to '0' or 'false' to disable interactive input.
            When not set, auto-detects based on whether stdin is a TTY.
    """
    # Configure logging
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true")
    configure_logging(json_output=False, level="DEBUG" if debug_mode else None)

    if debug_mode:
        logger.info("debug_mode_enabled")

    # Create World
    world = World()

    # --- Create LLM model (dual-mode) ---
    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    model: LLMModel
    if api_key:
        logger.info("using_model", model_name=model)
        print(f"Using model: {model}")
        model = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        logger.info("using_model", model_class="FakeModel")
        print("No LLM_API_KEY set. Using FakeModel for demonstration.")
        model = FakeModel(
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

    # --- Install skills (populates ToolRegistryComponent, SkillComponent) ---
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
        descriptor = discovered_by_name[skill_name]
        skill = descriptor.materialize()
        skill.resolve_path_references(workspace_root)
        manager.install(world, agent_id, skill)

    # Install builtin file tools so the agent can write output files.
    builtin_skill = BuiltinToolsSkill()
    builtin_skill.bind_workspace(str(workspace_dir))
    manager.install(world, agent_id, builtin_skill)


    # --- Add components ---

    # LLM model (no bare system_prompt — handled by SystemPromptConfigSpec)
    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
        ),
    )

    # Conversation seed
    world.add_component(
        agent_id,
        # Conversation starts empty — UserInputSystem provides the first user message.
        ConversationComponent(messages=[]),
    )

    # System prompt: template with built-in placeholders only.
    # Skills are listed by name/description (${_installed_skills}).
    # Full details are loaded lazily via load_skill_details tool.
    world.add_component(agent_id, _build_system_prompt_config())

    # --- Register Systems (priority order: lower = earlier execution) ---
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
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
    await runner.run(world, max_ticks=None)

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
