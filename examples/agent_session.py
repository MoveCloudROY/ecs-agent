"""Multiple turns, per-run budgets, and checkpoint continuation through AgentHandle.

Run without credentials for a deterministic demo, or pass --interactive for a
stdin conversation. LLM_API_KEY, LLM_BASE_URL, LLM_MODEL and LLM_API_FORMAT select
a real provider. Use LLM_API_FORMAT=openai_responses for the Responses API.
"""

import argparse
import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ecs_agent.core import AgentHandle, Runner, RunResult
from ecs_agent import AgentDefinition, build_agent, restore_runtime
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import CompletionResult, Message, StreamDelta, ToolSchema


class EchoDemoModel(FakeModel):
    """Generate a deterministic response for any number of interactive turns."""

    def __init__(self) -> None:
        super().__init__(responses=[])

    async def complete(
        self, messages: list[Message], tools: list[ToolSchema] | None = None,
        stream: bool = False, response_format: dict[str, Any] | None = None,
        thread_response_id: str | None = None, covered_message_count: int | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        latest = next(message.content for message in reversed(messages) if message.role == "user")
        self._responses.append(CompletionResult(message=Message(role="assistant", content=f"Demo reply: {latest}")))
        return await super().complete(messages, tools, stream, response_format, thread_response_id, covered_message_count)


def show_result(result: RunResult) -> None:
    """Use the run snapshot rather than interpreting the conversation tail."""
    if result.success:
        assert result.final_message is not None
        print(f"Assistant: {result.final_message.content}")
    else:
        print(f"Run stopped: {result.status}: {result.error or result.reason}")


async def main(argv: list[str] | None = None) -> None:
    """Run a bounded checkpoint demo or an explicit user-input loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    api_key = os.getenv("LLM_API_KEY")
    model: LLMModel
    if api_key:
        model = Model(
            os.getenv("LLM_MODEL", "deepseek-flash"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=api_key,
            api_format=ApiFormat(os.getenv("LLM_API_FORMAT", "openai_responses")),
        )
    else:
        model = EchoDemoModel()
    agent = build_agent(AgentDefinition(model=model))
    world, entity = agent.world, agent.entity
    try:
        if args.interactive:
            while True:
                try:
                    text = await asyncio.to_thread(input, "You> ")
                except EOFError:
                    break
                if text.strip().lower() in {"exit", "quit"}:
                    break
                if not text.strip():
                    continue
                result = await agent.run_turn(text, max_steps=20)
                show_result(result)
                if not result.success:
                    # Leave failed/paused states explicit instead of silently
                    # clearing them and resubmitting the same work.
                    break
        else:
            show_result(await agent.run_turn("Remember that my favorite color is blue."))
            paused = await agent.run_turn("What is my favorite color?", max_steps=0)
            print(f"Paused turn: {paused.turn_id}")
            with TemporaryDirectory(prefix="ecs-agent-session-") as directory:
                checkpoint = Path(directory) / "session.json"
                Runner().save_checkpoint(world, checkpoint)
                restored, _ = Runner.load_checkpoint(checkpoint, {model.model_id: model}, {})
                restore_runtime(restored)
                resumed = await AgentHandle(restored, entity).resume(max_steps=20)
                assert resumed.turn_id == paused.turn_id
                show_result(resumed)
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
