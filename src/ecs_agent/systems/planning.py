from __future__ import annotations

import time
import uuid

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    OneShotContextPoolComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    ScratchbookIndexComponent,
    SystemPromptComponent,
    TerminalComponent,
    TurnStateComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.prompts.message_assembly import (
    assemble_messages,
    build_keyword_registry,
    build_trigger_specs,
    commit_context_pool_reservation,
    collect_active_events,
    reserve_context_pool_items,
)
from ecs_agent.scratchbook import ScratchbookService
from ecs_agent.types import CompletionResult, Message, PlanStepCompletedEvent

logger = get_logger(__name__)


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


class PlanningSystem:
    def __init__(
        self, priority: int = 0, service: ScratchbookService | None = None
    ) -> None:
        self.priority = priority
        self.service = service

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
            rendered_user_prompt = world.get_component(
                entity_id, RenderedUserPromptComponent
            )
            prompt_config = world.get_component(entity_id, PromptConfigComponent)
            context_pool = world.get_component(entity_id, OneShotContextPoolComponent)
            turn_state = world.get_component(entity_id, TurnStateComponent)
            use_rendered_user_prompt = rendered_user_prompt is not None
            keyword_registry = None
            trigger_specs = None
            context_pool_enabled = False
            if not use_rendered_user_prompt:
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
            active_events = collect_active_events(reserved_context_pool_items)
            if rendered_user_prompt is not None:
                messages = assemble_messages(
                    system_prompt=system_prompt_text,
                    prefix_messages=[plan_context],
                    conversation_messages=_substitute_last_user_message(
                        conversation.messages,
                        rendered_user_prompt.text,
                    ),
                    enable_context_pool=False,
                )
            else:
                messages = assemble_messages(
                    system_prompt=system_prompt_text,
                    prefix_messages=[plan_context],
                    conversation_messages=conversation.messages,
                    enable_context_pool=context_pool_enabled,
                    context_pool_items=reserved_context_pool_items,
                    keyword_registry=keyword_registry,
                    trigger_specs=trigger_specs,
                    active_events=active_events,
                )

            tool_registry = world.get_component(entity_id, ToolRegistryComponent)
            tools = list(tool_registry.tools.values()) if tool_registry else None

            start_time = time.monotonic()
            try:
                logger.info("planning_request", message_count=len(messages))
                result = await llm_component.provider.complete(messages, tools=tools)
                if not isinstance(result, CompletionResult):
                    raise RuntimeError(
                        "Provider returned stream iterator in non-streaming mode"
                    )
                if context_pool_enabled and context_pool is not None and turn_id:
                    commit_context_pool_reservation(pool=context_pool, turn_id=turn_id)
                    if turn_state is not None:
                        turn_state.last_injected_turn_id = turn_id
                        turn_state.current_turn_id = ""
                # Add the step description to the conversation history
                conversation.messages.append(result.message)
                if result.message.tool_calls:
                    world.add_component(
                        entity_id,
                        PendingToolCallsComponent(tool_calls=result.message.tool_calls),
                    )

                plan.current_step += 1
                completed_step_index = plan.current_step - 1

                if self.service is not None:
                    scratchbook_index = world.get_component(
                        entity_id, ScratchbookIndexComponent
                    )
                    if scratchbook_index is not None:
                        artifact_id = (
                            f"plan-snapshot-{entity_id}-step-{completed_step_index}"
                        )
                        snapshot_data = {
                            "entity_id": entity_id,
                            "step_index": completed_step_index,
                            "step_description": plan.steps[completed_step_index],
                            "current_step": plan.current_step,
                            "completed": plan.completed,
                        }
                        self.service.write_artifact(
                            artifact_id, category="planning", data=snapshot_data
                        )

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


__all__ = ["PlanningSystem"]
