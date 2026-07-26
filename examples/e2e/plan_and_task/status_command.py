"""Render read-only slash commands without an LLM round-trip.

A pure-read command like ``/plan:status`` or ``/task:status`` only shows the
user information. Routed as an ordinary script trigger it would still trigger a
full reasoning turn — the script result replaces the rendered user prompt but
does not suppress reasoning, so the model is invoked to "respond" to a status
dump (a full prompt replay of the whole history, every time, plus model chatter
the user did not ask for).

``StatusCommandSystem`` short-circuits those turns: it takes the command
handler's already-rendered result, shows it as the assistant reply directly, and
completes the turn *without* invoking the model. It mirrors the reasoning
system's end-of-turn (publish ``ReasoningCompleteEvent`` so the front end re-arms
input) but marks a transient ``TerminalComponent(reason="status_shown")`` that
``ReasoningSystem`` treats as a skip signal and a companion
``TerminalCleanupSystem`` clears within the same tick.

Register it at a priority after ``UserPromptNormalizationSystem`` (so the
rendered result exists) and before ``ReasoningSystem`` (so the skip lands), plus
``TerminalCleanupSystem(clear_reasons=("status_shown",))`` after reasoning.
"""

from __future__ import annotations

from collections.abc import Sequence

from ecs_agent.components import RenderedUserPromptComponent
from ecs_agent.components.definitions import ConversationComponent, TerminalComponent
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import EntityId, Message, ReasoningCompleteEvent

logger = get_logger(__name__)

STATUS_SHOWN_REASON = "status_shown"


class StatusCommandSystem:
    """Completes read-only slash-command turns without calling the model."""

    def __init__(
        self,
        agent_id: EntityId,
        readonly_prefixes: Sequence[str],
        *,
        priority: int = -3,
    ) -> None:
        self.priority = priority
        self._agent_id = agent_id
        self._prefixes = tuple(readonly_prefixes)

    def _is_readonly_command(self, text: str) -> bool:
        return any(
            text == prefix or text.startswith(prefix + " ")
            for prefix in self._prefixes
        )

    async def process(self, world: World) -> None:
        entity_id = self._agent_id
        conversation = world.get_component(entity_id, ConversationComponent)
        if conversation is None or not conversation.messages:
            return
        last = conversation.messages[-1]
        # Only a fresh user command triggers this; once the reply is appended the
        # last message is the assistant status, so the turn is not re-processed.
        if last.role != "user" or not self._is_readonly_command((last.content or "").strip()):
            return

        rendered = world.get_component(entity_id, RenderedUserPromptComponent)
        if rendered is None:
            # No rendered result to show — let the normal path handle it.
            return

        conversation.messages.append(
            Message(role="assistant", content=rendered.text)
        )
        # Skip reasoning this tick (a non-"reasoning_complete" terminal reason is
        # ReasoningSystem's skip signal); a companion TerminalCleanupSystem clears
        # it within the tick so the runner keeps going.
        world.add_component(
            entity_id, TerminalComponent(reason=STATUS_SHOWN_REASON)
        )
        logger.debug(
            "plan_task_status_command_shortcircuited",
            entity_id=int(entity_id),
            command=(last.content or "").strip()[:40],
        )
        # Standard end-of-turn signal so every front end re-arms user input,
        # exactly as a completed reasoning turn does.
        await world.event_bus.publish(
            ReasoningCompleteEvent(
                entity_id=entity_id, model="status_command", duration_ms=0.0
            )
        )


__all__ = ["StatusCommandSystem", "STATUS_SHOWN_REASON"]
