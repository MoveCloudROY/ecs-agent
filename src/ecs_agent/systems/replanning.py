"""ReplanningSystem for dynamic plan revision during execution.

After each plan step completes, this system asks the LLM to review
execution results and revise remaining steps if needed.
"""

from __future__ import annotations

import json
import uuid

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PlanComponent,
    PromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    ScratchbookIndexComponent,
    SystemPromptComponent,
    TurnStateComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.message_assembly import (
    assemble_messages,
    build_keyword_registry,
    build_trigger_specs,
    commit_context_pool_reservation,
    collect_active_events,
    reserve_context_pool_items,
)
from ecs_agent.scratchbook import ScratchbookService
from ecs_agent.types import CompletionResult, EntityId, Message, PlanRevisedEvent


def _substitute_last_user_message(
    messages: list[Message], new_text: str
) -> list[Message]:
    if not messages:
        return [Message(role="user", content=new_text)]

    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if result[index].role == "user":
            result[index] = Message(role="user", content=new_text)
            return result

    return [*result, Message(role="user", content=new_text)]


class ReplanningSystem:
    """System that dynamically revises plan steps based on execution results.

    After each completed step, builds a replanning prompt with:
    - Original objective (first user message)
    - Completed steps and their results
    - Remaining steps

    The LLM responds with JSON {"revised_steps": [...]} to replace remaining steps.
    Falls back gracefully on JSON parse failure (keeps existing steps).

    Priority should be higher than ToolExecutionSystem (e.g. 7) so it runs
    after tools have executed but before memory truncation.
    """

    def __init__(
        self, priority: int = 7, service: ScratchbookService | None = None
    ) -> None:
        self.priority = priority
        self._last_replanned: dict[EntityId, int] = {}
        self.service = service

    async def process(self, world: World) -> None:
        """Check each plan entity and replan if a new step was completed."""
        for entity_id, components in world.query(
            PlanComponent, LLMComponent, ConversationComponent
        ):
            plan, llm_component, conversation = components
            assert isinstance(plan, PlanComponent)
            assert isinstance(llm_component, LLMComponent)
            assert isinstance(conversation, ConversationComponent)

            # Skip if plan is done or no remaining steps to revise
            if plan.completed or plan.current_step >= len(plan.steps):
                continue

            # Only replan when a new step has completed since last replan
            last = self._last_replanned.get(entity_id, 0)
            if plan.current_step <= last:
                continue

            # Need at least one completed step to have something to review
            if plan.current_step == 0:
                continue

            # Build replanning prompt
            rendered_system_prompt = world.get_component(
                entity_id, RenderedSystemPromptComponent
            )
            system_prompt = world.get_component(entity_id, SystemPromptComponent)
            system_prompt_text = (
                rendered_system_prompt.text
                if rendered_system_prompt is not None
                else (
                    system_prompt.content
                    if system_prompt is not None
                    else (llm_component.system_prompt or None)
                )
            )
            rendered_user_prompt = world.get_component(
                entity_id, RenderedUserPromptComponent
            )
            prompt_config = world.get_component(entity_id, PromptConfigComponent)
            context_pool = world.get_component(entity_id, OneShotContextPoolComponent)
            turn_state = world.get_component(entity_id, TurnStateComponent)
            use_rendered_user_prompt = rendered_user_prompt is not None
            context_pool_enabled = False
            if not use_rendered_user_prompt:
                context_pool_enabled = (
                    prompt_config.enable_context_pool
                    if prompt_config is not None
                    else False
                )
            turn_id = ""
            reserved_context_pool_items: list[tuple[int, int, str, str]] | None = None
            if context_pool_enabled and context_pool is not None:
                if turn_state is None:
                    turn_state = TurnStateComponent()
                    world.add_component(entity_id, turn_state)
                if not turn_state.current_turn_id:
                    turn_state.current_turn_id = uuid.uuid4().hex
                turn_id = turn_state.current_turn_id
                reserved_context_pool_items = reserve_context_pool_items(
                    pool=context_pool,
                    turn_id=turn_id,
                )
            messages = self._build_replanning_messages(
                world,
                entity_id,
                plan,
                conversation,
                system_prompt_text=system_prompt_text,
                rendered_user_text=(
                    rendered_user_prompt.text
                    if rendered_user_prompt is not None
                    else None
                ),
                prompt_config=prompt_config,
                context_pool_enabled=context_pool_enabled,
                context_pool_items=reserved_context_pool_items,
                active_events=collect_active_events(reserved_context_pool_items),
            )

            try:
                result = await llm_component.provider.complete(messages)
                if not isinstance(result, CompletionResult):
                    raise RuntimeError(
                        "Provider returned stream iterator in non-streaming mode"
                    )
                if context_pool_enabled and context_pool is not None and turn_id:
                    commit_context_pool_reservation(pool=context_pool, turn_id=turn_id)
                    if turn_state is not None:
                        turn_state.last_injected_turn_id = turn_id
                        turn_state.current_turn_id = ""
                revised = self._parse_revised_steps(result.message.content)

                if revised is not None:
                    old_steps = list(plan.steps)
                    plan.steps = plan.steps[: plan.current_step] + revised
                    new_steps = list(plan.steps)

                    if old_steps != new_steps:
                        if self.service is not None:
                            scratchbook_index = world.get_component(
                                entity_id, ScratchbookIndexComponent
                            )
                            if scratchbook_index is not None:
                                artifact_id = (
                                    f"replan-delta-{entity_id}-step-{plan.current_step}"
                                )
                                delta_data = {
                                    "entity_id": entity_id,
                                    "replanned_at_step": plan.current_step,
                                    "old_steps": old_steps,
                                    "new_steps": new_steps,
                                }
                                self.service.write_artifact(
                                    artifact_id, category="replanning", data=delta_data
                                )

                        await world.event_bus.publish(
                            PlanRevisedEvent(
                                entity_id=entity_id,
                                old_steps=old_steps,
                                new_steps=new_steps,
                            )
                        )

                self._last_replanned[entity_id] = plan.current_step
            except (IndexError, StopIteration):
                # Provider exhausted — skip replanning silently
                self._last_replanned[entity_id] = plan.current_step
            except Exception:
                continue

    def _build_replanning_messages(
        self,
        world: World,
        entity_id: EntityId,
        plan: PlanComponent,
        conversation: ConversationComponent,
        *,
        system_prompt_text: str | None,
        rendered_user_text: str | None,
        prompt_config: PromptConfigComponent | None,
        context_pool_enabled: bool,
        context_pool_items: list[tuple[int, int, str, str]] | None,
        active_events: set[str],
    ) -> list[Message]:
        """Build the message list for the replanning LLM call."""
        conversation_messages: list[Message] = []

        # Extract original objective from first user message
        objective = ""
        for msg in conversation.messages:
            if msg.role == "user":
                objective = msg.content
                break

        # Build completed steps summary with results
        completed_lines: list[str] = []
        for i in range(plan.current_step):
            step_desc = plan.steps[i]
            result_text = self._find_step_result(conversation, i)
            completed_lines.append(
                f"{i + 1}. {step_desc} \u2713 \u2014 Result: {result_text}"
            )

        # Build remaining steps
        remaining_lines: list[str] = []
        for i in range(plan.current_step, len(plan.steps)):
            remaining_lines.append(f"{i + 1}. {plan.steps[i]}")

        replanning_prompt = (
            "You are a planning revision agent. Review the execution so far "
            "and revise remaining steps if needed.\n\n"
            f"## Original Objective:\n{objective}\n\n"
            f"## Completed Steps:\n" + "\n".join(completed_lines) + "\n\n"
            "## Remaining Steps:\n" + "\n".join(remaining_lines) + "\n\n"
            "## Instructions:\n"
            "Based on what you've learned from completed steps, revise the "
            "remaining steps if needed. You may add, remove, reorder, or "
            "modify steps.\n"
            'Output ONLY a JSON object: {"revised_steps": ["step 1", "step 2", ...]}\n'
            "If no changes needed, return the remaining steps as-is.\n"
            "Do NOT include completed steps in revised_steps."
        )

        conversation_messages.append(Message(role="user", content=replanning_prompt))
        if rendered_user_text is not None:
            return assemble_messages(
                system_prompt=system_prompt_text,
                conversation_messages=_substitute_last_user_message(
                    conversation_messages,
                    rendered_user_text,
                ),
                enable_context_pool=False,
            )

        keyword_registry = (
            build_keyword_registry(prompt_config.trigger_templates)
            if prompt_config is not None and prompt_config.trigger_templates
            else None
        )
        trigger_specs = (
            build_trigger_specs(prompt_config.trigger_templates)
            if prompt_config is not None and prompt_config.trigger_templates
            else None
        )
        return assemble_messages(
            system_prompt=system_prompt_text,
            conversation_messages=conversation_messages,
            enable_context_pool=context_pool_enabled,
            context_pool_items=context_pool_items,
            keyword_registry=keyword_registry,
            trigger_specs=trigger_specs,
            active_events=active_events,
        )

    def _find_step_result(
        self, conversation: ConversationComponent, step_index: int
    ) -> str:
        """Find tool results or assistant response for a given step.

        Scans conversation for tool results following the step's assistant message.
        Falls back to the assistant message content if no tool results found.
        """
        # Look for tool role messages as results
        tool_results: list[str] = []
        assistant_content = ""
        found_step_assistant = False
        step_assistant_count = 0

        for msg in conversation.messages:
            if msg.role == "assistant":
                if step_assistant_count == step_index:
                    found_step_assistant = True
                    assistant_content = msg.content or ""
                step_assistant_count += 1
            elif msg.role == "tool" and found_step_assistant:
                tool_results.append(msg.content)
            elif msg.role == "assistant" and found_step_assistant:
                break  # Next assistant message = next step

        if tool_results:
            return "; ".join(tool_results)
        if assistant_content:
            return assistant_content[:200]
        return "(no result)"

    @staticmethod
    def _parse_revised_steps(content: str) -> list[str] | None:
        """Parse LLM response for revised steps JSON.

        Returns list of step strings, or None if parsing fails.
        """
        if not content:
            return None

        # Try to extract JSON from the response
        try:
            # Try direct parse first
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON block in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end <= start:
                return None
            try:
                data = json.loads(content[start:end])
            except json.JSONDecodeError:
                return None

        if not isinstance(data, dict):
            return None

        revised = data.get("revised_steps")
        if not isinstance(revised, list):
            return None

        # Validate all items are strings
        if not all(isinstance(s, str) for s in revised):
            return None

        return revised


__all__ = ["ReplanningSystem"]
