from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.skill import Skill
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall


async def main() -> None:
    world = World()
    agent = world.create_entity()
    manager = SkillManager()

    skill_path = (
        Path(__file__).parent / "skills" / "ui-ux-reviewer" / "SKILL.md"
    )
    # Load skill object
    skill = Skill(skill_path=skill_path)

    # Note: manager.install() is a convenience for direct installation.
    # It performs both indexing (metadata) and activation (tools/prompt) in one step.
    # For automatic discovery from directories, prefer the lazy DiscoveryManager path.
    manager.install(world, agent, skill)

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")

    if api_key:
        provider: OpenAIProvider | FakeProvider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        selected_model = model
        print(f"Using OpenAIProvider with model: {model}")
    else:
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I will call build_ui_checklist for this page.",
                        tool_calls=[
                            ToolCall(
                                id="tool_call_1",
                                name="build_ui_checklist",
                                arguments={
                                    "page_type": "landing page",
                                    "emphasis": "accessibility",
                                },
                            )
                        ],
                    )
                ),
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "Checklist generated. Prioritize hierarchy, spacing, "
                            "and accessibility before final polish."
                        ),
                    )
                ),
            ]
        )
        selected_model = "fake"
        print("No LLM_API_KEY provided. Using FakeProvider.")

    world.add_component(agent, LLMComponent(provider=provider, model=selected_model))
    world.add_component(
        agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Call build_ui_checklist for a landing page with accessibility "
                        "focus, then summarize the result."
                    ),
                )
            ]
        ),
    )

    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)

    runner = Runner()
    await runner.run(world, max_ticks=6)

    skills = manager.list_skills(world, agent)
    print(f"Installed file-based skills: {[s.name for s in skills]}")

    conv = world.get_component(agent, ConversationComponent)
    if conv is not None:
        print("Conversation:")
        for msg in conv.messages:
            tool_note = ""
            if msg.tool_calls:
                tool_names = ", ".join(call.name for call in msg.tool_calls)
                tool_note = f" [tool_calls: {tool_names}]"
            if msg.tool_call_id:
                tool_note = f" [tool_result_for: {msg.tool_call_id}]"
            print(f"- {msg.role}{tool_note}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
