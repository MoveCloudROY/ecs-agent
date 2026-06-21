"""Multi-step plan execution system."""

from __future__ import annotations

import time

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    RenderedSystemPromptComponent,
    RunnerStateComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.accounting.instrumentation import complete_with_llm_invocation_event
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.message_assembly import (
    commit_prompt_context_reservation,
    prepare_outbound_messages,
)
from ecs_agent.scratchbook import ArtifactRegistry, ScratchbookService
from ecs_agent.types import CompletionResult, Message, PlanStepCompletedEvent

logger = get_logger(__name__)


class PlanningSystem:
    def __init__(
        self,
        priority: int = 0,
        service: ScratchbookService | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        self.priority = priority
        self._registry: ArtifactRegistry | None
        if registry is not None:
            self._registry = registry
        elif service is not None:
            self._registry = ArtifactRegistry(root=service.root)
        else:
            self._registry = None

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            PlanComponent, LLMComponent, ConversationComponent
        ):
            plan, llm_component, conversation = components
            assert isinstance(plan, PlanComponent)
            assert isinstance(llm_component, LLMComponent)
            assert isinstance(conversation, ConversationComponent)

            if plan.completed or not plan.steps:
                continue

            if plan.current_step >= len(plan.steps):
                plan.completed = True
                continue

            step_description = plan.steps[plan.current_step]
            plan_context = Message(
                role="system",
                content=f"Step {plan.current_step + 1}/{len(plan.steps)}: {step_description}",
            )

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
            messages, context_reservation = prepare_outbound_messages(
                world,
                entity_id,
                system_prompt=system_prompt_text,
                prefix_messages=[plan_context],
                current_tick=current_tick,
            )
            if context_reservation is not None:
                world.add_component(entity_id, context_reservation)

            tool_registry = world.get_component(entity_id, ToolRegistryComponent)
            tools = list(tool_registry.tools.values()) if tool_registry else None

            start_time = time.monotonic()
            try:
                logger.info("planning_request", message_count=len(messages))
                result = await complete_with_llm_invocation_event(
                    event_bus=world.event_bus,
                    entity_id=entity_id,
                    model=llm_component.model,
                    messages=messages,
                    tools=tools,
                    operation="planning",
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
                # Add the step description to the conversation history
                conversation.messages.append(result.message)
                if result.message.tool_calls:
                    world.add_component(
                        entity_id,
                        PendingToolCallsComponent(tool_calls=result.message.tool_calls),
                    )

                plan.current_step += 1
                completed_step_index = plan.current_step - 1

                duration_ms = (time.monotonic() - start_time) * 1000
                logger.info(
                    "planning_step_completed",
                    step_index=completed_step_index,
                    step_description=plan.steps[completed_step_index],
                    duration_ms=duration_ms,
                )

                await world.event_bus.publish(
                    PlanStepCompletedEvent(
                        entity_id=entity_id,
                        step_index=completed_step_index,
                        step_description=plan.steps[completed_step_index],
                    )
                )

                if plan.current_step >= len(plan.steps):
                    plan.completed = True

                if self._registry is not None:
                    plan_name = _derive_plan_name(
                        plan=plan,
                        conversation=conversation,
                        entity_id=entity_id,
                    )
                    self._registry.write_plan(
                        plan_name=plan_name,
                        content=_render_plan_markdown(plan_name=plan_name, plan=plan),
                    )
                    await self._registry.update_boulder(
                        plan_name=plan_name,
                        updates={
                            "status": "running",
                            "current_step": plan.current_step,
                            "last_step_description": plan.steps[completed_step_index],
                        },
                    )
            except (IndexError, StopIteration):
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="provider_exhausted"),
                )
            except Exception as exc:
                logger.error(
                    "planning_error",
                    exception=str(exc),
                    system_name="PlanningSystem",
                )
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=str(exc),
                        system_name="PlanningSystem",
                        timestamp=time.time(),
                    ),
                )
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="planning_error"),
                )


def _derive_plan_name(
    *, plan: PlanComponent, conversation: ConversationComponent, entity_id: int
) -> str:
    for message in conversation.messages:
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    if plan.steps:
        first_step = plan.steps[0].strip()
        if first_step:
            return first_step
    return f"plan-{entity_id}"


def _render_plan_markdown(*, plan_name: str, plan: PlanComponent) -> str:
    lines = [f"# Plan: {plan_name}", "", "## Steps"]

    if not plan.steps:
        lines.append("1. [ ] (no steps)")
    else:
        all_done = plan.completed and plan.current_step >= len(plan.steps)
        for index, step in enumerate(plan.steps):
            if index < plan.current_step or all_done:
                marker = "DONE"
            elif index == plan.current_step and not plan.completed:
                marker = "CURRENT"
            else:
                marker = " "
            lines.append(f"{index + 1}. [{marker}] {step}")

    lines.extend(
        [
            "",
            "## Status",
            f"Current step: {plan.current_step + 1 if not plan.completed else len(plan.steps)}",
            f"Total steps: {len(plan.steps)}",
            f"Completed: {'yes' if plan.completed else 'no'}",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = ["PlanningSystem"]
