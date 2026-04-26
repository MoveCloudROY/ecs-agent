"""Multimodal vision agent example.

This example demonstrates how to send an image URL to a vision-capable LLM
using the ECS-based agent framework, with full prompt normalization via
SystemPromptRenderSystem and UserPromptNormalizationSystem.

Dual-mode:
- Without LLM_API_KEY: Uses FakeModel with a mock image description.
- With LLM_API_KEY: Uses OpenAIModel with Chat Completions API.

Environment variables:
  LLM_API_KEY   — API key (required for real LLM mode)
  LLM_BASE_URL  — API base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  LLM_MODEL     — Model name (default: qwen3-vl-flash)
  IMAGE_URL     — Image URL to analyze (default: dog and girl demo image)
"""

import asyncio
import os

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import CompletionResult, ImageUrlPart, Message


async def main() -> None:
    """Run a multimodal vision agent example."""
    configure_logging(json_output=False)

    world = World()

    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = os.environ.get("LLM_MODEL", "qwen3-vl-flash")
    image_url: str = os.environ.get(
        "IMAGE_URL",
        "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    model: LLMModel
    if api_key:
        print(f"Using model: {model}")
        model = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        print("No LLM_API_KEY set. Using FakeModel for demonstration.")
        model = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "I see a girl playing with a dog outdoors. "
                            "The dog appears happy and the scene is cheerful."
                        ),
                    )
                )
            ]
        )

    print(f"Analyzing image URL: {image_url}")

    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            
            system_prompt="",
        ),
    )
    world.add_component(
        agent_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline="You are a helpful vision assistant."
            ),
            placeholders=[],
        ),
    )
    world.add_component(
        agent_id,
        UserPromptConfigComponent(),
    )
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Describe this image in detail.",
                    parts=[
                        ImageUrlPart(url=image_url),
                    ],
                )
            ]
        ),
    )

    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=3)

    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None and conv.messages:
        print(f"Assistant response: {conv.messages[-1].content}")
    else:
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
