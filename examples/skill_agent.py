"""
Demonstrates BuiltinToolsSkill installation and tool usage lifecycle.

This example shows how to:
1. Initialize a World and an agent entity.
2. Install the BuiltinToolsSkill using SkillManager.
3. Configure the agent with Reasoning and ToolExecution systems.
4. Run a loop where the agent uses read_file and write_file.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from ecs_agent import BuiltinToolsSkill, SkillManager
from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import CompletionResult, Message, ToolCall


async def main() -> None:
    # Set up a temporary workspace for file operations
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace = Path(tmp_dir)
        test_file = workspace / "hello.txt"
        test_file.write_text("Hello from the workspace!", encoding="utf-8")

        world = World()
        agent = world.create_entity()

        # 1. Setup Provider (Use OpenAI if key is present, otherwise Fake)
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            provider = OpenAIProvider(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("LLM_MODEL", "gpt-4o"),
            )
        else:
            # Fake responses for the demo
            provider = FakeProvider(
                responses=[
                    # First response: call read_file
                    CompletionResult(
                        message=Message(
                            role="assistant",
                            content="Let me check the file.",
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="read_file",
                                    arguments={"path": "hello.txt"},
                                )
                            ],
                        )
                    ),
                    # Second response: after seeing file content, write to it
                    CompletionResult(
                        message=Message(
                            role="assistant",
                            content="I will update the file.",
                            tool_calls=[
                                ToolCall(
                                    id="call_2",
                                    name="write_file",
                                    arguments={"path": "hello.txt", "content": "Updated content!"},
                                )
                            ],
                        )
                    ),
                    # Third response: finish
                    CompletionResult(
                        message=Message(
                            role="assistant", content="All done! I've updated the file."
                        )
                    ),
                ]
            )

        # 2. Add components
        world.add_component(
            agent,
            LLMComponent(
                provider=provider,
                model="gpt-4o",
                system_prompt="You are a file manager.",
            ),
        )
        world.add_component(
            agent,
            ConversationComponent(
                messages=[
                    Message(
                        role="user",
                        content="Read hello.txt and then change its content.",
                    )
                ]
            ),
        )

        # 3. Install Skills
        manager = SkillManager()

        # The built-in file tools require a 'workspace_root' parameter.
        # We wrap the handlers to inject the workspace_root automatically.
        skill = BuiltinToolsSkill()
        original_tools = skill.tools()
        wrapped_tools = {}
        for name, (schema, handler) in original_tools.items():
            async def wrapped_handler(h=handler, **kwargs):
                return await h(workspace_root=str(workspace), **kwargs)
            wrapped_tools[name] = (schema, wrapped_handler)
        
        # Patch the skill instance for the demo
        skill.tools = lambda: wrapped_tools
        manager.install(world, agent, skill)

        # Register systems
        world.register_system(ReasoningSystem(), priority=0)
        world.register_system(ToolExecutionSystem(), priority=5)

        # 4. Run the agent
        print(f"Starting agent in workspace: {workspace}")
        runner = Runner()
        await runner.run(world, max_ticks=10)

        # Verify results
        final_content = test_file.read_text(encoding="utf-8")
        print(f"Final file content: {final_content}")

        # List installed skills
        skills = manager.list_skills(world, agent)
        print(f"Installed skills: {[s.name for s in skills]}")

        # Demo uninstallation
        manager.uninstall(world, agent, "builtin-tools")
        print("Uninstalled builtin-tools skill.")

        remaining_skills = manager.list_skills(world, agent)
        print(f"Remaining skills: {[s.name for s in remaining_skills]}")


if __name__ == "__main__":
    asyncio.run(main())
