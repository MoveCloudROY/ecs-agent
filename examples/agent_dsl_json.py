"""Load DSL definitions and run a completely assembled agent.

Without LLM_API_KEY this runs a deterministic delegation demo. Set LLM_MODEL,
LLM_BASE_URL and LLM_API_FORMAT=responses for a real Responses API provider.
"""

import asyncio
import os
from pathlib import Path

from ecs_agent import build_agent_from_specs
from ecs_agent.dsl import load_json_agents, load_markdown_agent, resolve_agent_specs
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import CompletionResult, Message, ToolCall


async def main() -> None:
    """Load definitions, run one turn, and report the authoritative result."""
    root = Path(__file__).parent
    loaded = load_json_agents(root / "agents_config.json")
    specs = resolve_agent_specs(loaded)
    if configured_model := os.getenv("LLM_MODEL"):
        for spec in specs.values():
            spec.model = configured_model
    child_name = next((name for name, spec in specs.items() if spec.mode == "subagent"), None)
    responses = []
    if child_name:
        responses.extend([
            CompletionResult(Message("assistant", "Delegating research.", tool_calls=[ToolCall("delegate-demo", "subagent", {
                "category": child_name, "prompt": "Research near-term quantum computing applications.", "background": False,
            })])),
            CompletionResult(Message("assistant", "Research result: chemistry simulation and optimization are promising areas.")),
        ])
    responses.append(CompletionResult(Message("assistant", "Completed research: chemistry simulation and optimization.")))
    fake = FakeModel(responses)
    providers: list[LLMModel] = []

    def create_model(model: str, prompt: str) -> LLMModel:
        if not os.getenv("LLM_API_KEY"):
            return fake
        format_name = os.getenv("LLM_API_FORMAT", "openai_chat_completions")
        api_format = ApiFormat.OPENAI_RESPONSES if format_name == "responses" else ApiFormat(format_name)
        provider = Model(model, base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), api_key=os.environ["LLM_API_KEY"], api_format=api_format)
        providers.append(provider)
        return provider

    try:
        agent = build_agent_from_specs(specs, create_model)
        result = await agent.run_turn("Delegate research on near-term quantum computing applications and summarize the findings.", max_steps=10)
        print(f"status: {result.status.value}")
        if result.final_message is not None:
            print(result.final_message.content)
        elif result.partial_message is not None:
            print(result.partial_message.content)
        if not result.success:
            raise RuntimeError(f"Agent did not complete: {result.reason}")
    finally:
        for provider in providers:
            await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
