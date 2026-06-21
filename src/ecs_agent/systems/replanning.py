"""ReplanningSystem for dynamic plan revision during execution.

After each plan step completes, this system asks the LLM to review
execution results and revise remaining steps if needed.
"""

from __future__ import annotations

import json

from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PlanComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    RenderedSystemPromptComponent,
    RunnerStateComponent,
    SystemPromptComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.message_assembly import (
    commit_prompt_context_reservation,
    prepare_outbound_messages,
)
from ecs_agent.scratchbook import ArtifactRegistry, ScratchbookService
from ecs_agent.systems._plan_utils import derive_plan_name, render_plan_markdown
from ecs_agent.types import CompletionResult, EntityId, Message, PlanRevisedEvent


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
        self,
        priority: int = 7,
        service: ScratchbookService | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        self.priority = priority
        self._last_replanned: dict[EntityId, int] = {}
        self._registry: ArtifactRegistry | None
        if registry is not None:
            self._registry = registry
        elif service is not None:
            self._registry = ArtifactRegistry(root=service.root)
        else:
            self._registry = None

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
            context_queue = world.get_component(entity_id, PromptContextQueueComponent)
            runner_state = world.get_component(entity_id, RunnerStateComponent)
            current_tick = runner_state.current_tick if runner_state is not None else 0
            replanning_prompt = self._build_replanning_prompt(
                plan=plan, conversation=conversation
            )
            messages, context_reservation = prepare_outbound_messages(
                world,
                entity_id,
                system_prompt=system_prompt_text,
                current_tick=current_tick,
                conversation_override=[Message(role="user", content=replanning_prompt)],
            )

            if context_reservation is not None:
                world.add_component(entity_id, context_reservation)

            try:
                result = await complete_with_llm_invocation_event(
                    event_bus=world.event_bus,
                    entity_id=entity_id,
                    model=llm_component.model,
                    messages=messages,
                    operation="replanning",
                )
                if not isinstance(result, CompletionResult):
                    raise RuntimeError(
                        "Provider returned stream iterator in non-streaming mode"
                    )
                if context_queue is not None and context_reservation is not None:
                    commit_prompt_context_reservation(
                        queue=context_queue,
                        reservation=context_reservation,
                    )
                    world.remove_component(entity_id, PromptContextReservationComponent)
                revised = self._parse_revised_steps(result.message.content)

                if revised is not None:
                    old_steps = list(plan.steps)
                    plan.steps = plan.steps[: plan.current_step] + revised
                    new_steps = list(plan.steps)

                    if old_steps != new_steps:
                        if self._registry is not None:
                            plan_name = derive_plan_name(
                                plan=plan,
                                conversation=conversation,
                                entity_id=entity_id,
                            )
                            self._registry.write_plan(
                                plan_name=plan_name,
                                content=render_plan_markdown(
                                    plan_name=plan_name,
                                    plan=plan,
                                ),
                            )
                            await self._registry.update_boulder(
                                plan_name=plan_name,
                                updates={
                                    "status": "replanned",
                                    "revised_steps_count": len(plan.steps),
                                },
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

    def _build_replanning_prompt(
        self,
        plan: PlanComponent,
        conversation: ConversationComponent,
    ) -> str:
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

        return replanning_prompt

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
                if found_step_assistant:
                    break
                if step_assistant_count == step_index:
                    found_step_assistant = True
                    assistant_content = msg.content or ""
                step_assistant_count += 1
            elif msg.role == "tool" and found_step_assistant:
                tool_results.append(msg.content)

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
