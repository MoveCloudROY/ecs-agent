"""Shared plan helpers used by planning, replanning, and tool execution systems."""

from __future__ import annotations

from ecs_agent.components import ConversationComponent, PlanComponent
from ecs_agent.types import EntityId


def derive_plan_name(
    *, plan: PlanComponent, conversation: ConversationComponent, entity_id: EntityId
) -> str:
    for message in conversation.messages:
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    if plan.steps:
        first_step = plan.steps[0].strip()
        if first_step:
            return first_step
    return f"plan-{entity_id}"


def render_plan_markdown(*, plan_name: str, plan: PlanComponent) -> str:
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
