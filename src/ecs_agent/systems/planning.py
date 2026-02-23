from __future__ import annotations

import time

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PendingToolCallsComponent,
    PlanComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import Message, PlanStepCompletedEvent


class PlanningSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority

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

            messages: list[Message] = []

            system_prompt = world.get_component(entity_id, SystemPromptComponent)
            if system_prompt is not None:
                messages.append(Message(role="system", content=system_prompt.content))

            messages.append(plan_context)
            messages.extend(conversation.messages)

            tool_registry = world.get_component(entity_id, ToolRegistryComponent)
            tools = list(tool_registry.tools.values()) if tool_registry else None

            try:
                print("PlanningSystem: Requesting completion for current plan step...")
                print(f"Messages sent to provider:")
                for msg in messages:
                    print(f"  {msg.role}: {msg.content}")
                
                result = await llm_component.provider.complete(messages, tools=tools)
                # print(f"PlanningSystem: Received completion result: {result}"   )# Add the step description to the conversation history
                conversation.messages.append(result.message)
                # print("-> conversation.messages: ")
                # print(conversation.messages)
                # print()

                if result.message.tool_calls:
                    world.add_component(
                        entity_id,
                        PendingToolCallsComponent(tool_calls=result.message.tool_calls),
                    )

                plan.current_step += 1
                completed_step_index = plan.current_step - 1
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
