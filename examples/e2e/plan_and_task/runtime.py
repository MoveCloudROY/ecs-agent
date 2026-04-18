"""Interactive runtime adapter for the plan-and-task example."""

from __future__ import annotations

import asyncio
import re
import re as _re
import unicodedata
from typing import TYPE_CHECKING

from ecs_agent.components import UserInputComponent
from ecs_agent.components.definitions import ConversationComponent, TerminalComponent
from ecs_agent.logging import get_logger
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems import TerminalCleanupSystem
from ecs_agent.systems.user_input import UserInputSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    ReasoningCompleteEvent,
    UserInputRequestedEvent,
)

if TYPE_CHECKING:
    from ecs_agent.core import World
    from ecs_agent.types import EntityId

logger = get_logger(__name__)

_MAX_SLUG_LENGTH = 50
_SLUG_SEPARATOR = "-"


def slug_from_description(description: str) -> str:
    """Derive a URL-safe workflow ID slug from a natural language task description.

    Returns an empty string if the description yields no usable tokens,
    leaving the caller to decide the fallback.
    """
    text = description.strip()
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)

    cjk_range = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
    )
    has_cjk = bool(cjk_range.search(normalized))

    if has_cjk:
        allowed = re.sub(
            r"[^\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7afa-z0-9\s]",
            "",
            normalized.lower(),
        )
        slug = re.sub(r"\s+", _SLUG_SEPARATOR, allowed).strip(_SLUG_SEPARATOR)
    else:
        lower = normalized.lower()
        allowed = re.sub(r"[^a-z0-9\s]", " ", lower)
        tokens = allowed.split()
        slug = _SLUG_SEPARATOR.join(tokens[:6])

    return slug[:_MAX_SLUG_LENGTH].rstrip(_SLUG_SEPARATOR)


_VALID_SLUG = _re.compile(r"^[a-z][a-z0-9-]*$")


async def derive_workflow_id_from_llm(description: str, provider: LLMProvider) -> str:
    prompt = (
        "Convert the following task description into a short, meaningful English "
        "workflow identifier. Rules: lowercase letters, digits, and hyphens only; "
        "2-6 words; max 50 characters; no spaces, no punctuation, no explanation. "
        "Return ONLY the identifier, nothing else.\n\n"
        f"Description: {description}"
    )
    try:
        result = await provider.complete(
            [Message(role="user", content=prompt)],
            stream=False,
        )
        if not isinstance(result, CompletionResult):
            return slug_from_description(description)
        raw = (result.message.content or "").strip().splitlines()[0].strip().lower()
        normalized = _re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        normalized = normalized[:_MAX_SLUG_LENGTH].rstrip("-")
        if normalized and _VALID_SLUG.match(normalized):
            return normalized
    except Exception:
        pass
    return slug_from_description(description)


async def setup_interactive_input(
    world: World,
    agent_id: EntityId,
) -> None:
    """Wire interactive stdin into the ECS world for the plan-and-task example."""
    last_printed_index: list[int] = [0]

    async def provide_input(event: UserInputRequestedEvent) -> None:
        loop = asyncio.get_running_loop()

        conv = world.get_component(event.entity_id, ConversationComponent)
        if conv is not None:
            for msg in conv.messages[last_printed_index[0] :]:
                if msg.role == "assistant" and msg.content:
                    print(f"\nAssistant: {msg.content}\n")
            last_printed_index[0] = len(conv.messages)

        while True:
            lines: list[str] = []
            prompt = event.prompt
            try:
                while True:
                    line = await loop.run_in_executor(None, input, prompt)
                    if not lines and line.lower().strip() in ("exit", "quit"):
                        logger.info(
                            "plan_task_user_exit",
                            entity_id=int(event.entity_id),
                        )
                        world.add_component(
                            event.entity_id,
                            TerminalComponent(reason="user_exit_command"),
                        )
                        if not event.input_future.done():
                            event.input_future.set_result(line)
                        return
                    if line == "":
                        break
                    lines.append(line)
                    prompt = "... "
            except EOFError:
                if not lines:
                    lines = ["exit"]

            user_text = "\n".join(lines).strip()
            if not user_text:
                continue

            if not event.input_future.done():
                event.input_future.set_result(user_text)
            return

    async def on_reasoning_complete(event: ReasoningCompleteEvent) -> None:
        if event.entity_id != agent_id:
            return

        logger.info(
            "plan_task_reasoning_complete",
            entity_id=int(agent_id),
        )
        world.add_component(agent_id, UserInputComponent(prompt="You> "))

    world.event_bus.subscribe(UserInputRequestedEvent, provide_input)
    world.event_bus.subscribe(ReasoningCompleteEvent, on_reasoning_complete)
    world.register_system(
        TerminalCleanupSystem(priority=1, clear_reasons=("reasoning_complete",)),
        priority=1,
    )
    world.register_system(UserInputSystem(priority=-5), priority=-5)

    if world.get_component(agent_id, UserInputComponent) is None:
        world.add_component(agent_id, UserInputComponent(prompt="You> "))
