"""Workflow DSL example: stateful agent with gate-driven state transitions.

This example demonstrates the built-in Workflow DSL, which lets you define a
graph of states, prompt profiles, and pure transition gates, compiled onto the
existing ECS runtime.

Scenario: a two-phase writing assistant.

    DRAFT  ──(has DraftReadyMarker)──►  REVIEW  ──(has ApprovedMarker)──►  DONE

- DRAFT and REVIEW use different prompt profiles so the LLM "persona" changes
  when the workflow transitions.
- REVIEW re-uses a shared profile to show that transitions within the same
  profile cluster do NOT invalidate the rendered system prompt cache.
- Gate components are plain dataclasses attached to the entity by tool calls or
  trigger script handlers.  WorkflowStateSystem observes them once per tick.

Dual-mode:
  - Without LLM_API_KEY → FakeModel (deterministic, works offline)
  - With    LLM_API_KEY → real LLM (OpenAI-compatible or Anthropic)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.systems.workflow_state import WorkflowStateSystem
from ecs_agent.tools import tool
from ecs_agent.types import CompletionResult, EntityId, Message, ToolCall, ToolSchema
from ecs_agent.workflows import PromptProfileSpec, has, install_workflow, workflow

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Gate marker components
# These are plain dataclasses.  The workflow gates observe their presence via
# has(DraftReadyMarker) / has(ApprovedMarker).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DraftReadyMarker:
    """Attached by the agent when the draft is ready for review."""


@dataclass(slots=True)
class ApprovedMarker:
    """Attached by the agent when the review is approved."""


# ---------------------------------------------------------------------------
# Workflow spec
# ---------------------------------------------------------------------------

WRITING_WORKFLOW = workflow(
    workflow_id="writing-flow",
    initial="DRAFT",
    profiles={
        "assistant": {
            "drafter": PromptProfileSpec(
                profile_id="drafter",
                prompt=(
                    "You are a creative writing assistant in DRAFT mode.\n"
                    "Help the user write and refine their draft.\n"
                    "When the draft is ready, call the `mark_draft_ready` tool."
                ),
            ),
            "reviewer": PromptProfileSpec(
                profile_id="reviewer",
                prompt=(
                    "You are a critical reviewer in REVIEW mode.\n"
                    "Evaluate the draft for clarity, accuracy, and completeness.\n"
                    "When you approve it, call the `approve_draft` tool.\n"
                    "If you need revisions, just say so."
                ),
            ),
        }
    },
    states={
        "DRAFT": {
            "bind": {"assistant": "drafter"},
            "go": {
                # Transition fires when DraftReadyMarker is present
                "REVIEW": has(DraftReadyMarker),
            },
        },
        "REVIEW": {
            "bind": {"assistant": "reviewer"},
            "go": {
                # Transition fires when ApprovedMarker is present
                "DONE": has(ApprovedMarker),
            },
        },
        "DONE": {
            "bind": {"assistant": "reviewer"},  # terminal state — no outgoing transitions
            "go": {},
        },
    },
)

# ---------------------------------------------------------------------------
# Tools that agents call to drive transitions
# ---------------------------------------------------------------------------


def make_tool_registry(world: World, entity_id: EntityId) -> ToolRegistryComponent:
    @tool(name="mark_draft_ready", description="Signal that the draft is complete and ready for review.")
    def mark_draft_ready() -> str:
        world.add_component(entity_id, DraftReadyMarker())
        logger.info("workflow_tool_called", tool="mark_draft_ready", entity_id=entity_id)
        return "Draft marked as ready. Moving to REVIEW state."

    @tool(name="approve_draft", description="Approve the draft and complete the workflow.")
    def approve_draft() -> str:
        world.add_component(entity_id, ApprovedMarker())
        logger.info("workflow_tool_called", tool="approve_draft", entity_id=entity_id)
        return "Draft approved. Workflow complete."

    fns = [mark_draft_ready, approve_draft]
    tools_map: dict[str, ToolSchema] = {
        fn._tool_schema.name: fn._tool_schema  # type: ignore[attr-defined]
        for fn in fns
    }
    handlers_map = {
        fn._tool_schema.name: fn._tool_handler  # type: ignore[attr-defined]
        for fn in fns
    }
    return ToolRegistryComponent(tools=tools_map, handlers=handlers_map)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the workflow agent example."""
    configure_logging(json_output=False)

    # --- LLM model (dual-mode) ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_id = os.environ.get("LLM_MODEL", "qwen3.5-flash")
    api_format_env = os.environ.get("LLM_API_FORMAT", "openai_chat_completions")

    model: LLMModel
    if api_key:
        api_format = (
            ApiFormat.ANTHROPIC_MESSAGES
            if api_format_env == "anthropic_messages"
            else ApiFormat.OPENAI_CHAT_COMPLETIONS
        )
        model = Model(model_id, base_url=base_url, api_key=api_key, api_format=api_format)
        print(f"Using real LLM: {model_id} @ {base_url}")
    else:
        # Fake responses that simulate the two-phase flow:
        # Turn 1 (DRAFT): agent writes something and calls mark_draft_ready
        # Turn 2 (REVIEW): agent reviews and calls approve_draft
        # Turn 3 (DONE): agent gives a final summary
        model = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="Here is my draft: 'The quick brown fox…' I think it's ready!",
                        tool_calls=[ToolCall(id="call_1", name="mark_draft_ready", arguments={})],
                    ),
                ),
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="This draft looks good. Clear and concise. Approving it.",
                        tool_calls=[ToolCall(id="call_2", name="approve_draft", arguments={})],
                    ),
                ),
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="The draft has been approved and the workflow is complete!",
                    ),
                ),
            ]
        )
        print("No LLM_API_KEY set — using FakeModel.")

    # --- World setup ---
    world = World(name="writing-agent")
    agent = world.create_entity()

    # Attach LLM and conversation
    world.add_component(agent, LLMComponent(model=model))
    world.add_component(
        agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Please write a short paragraph about the benefits of ECS architecture.",
                )
            ]
        ),
    )

    # System prompt: the ${_workflow_state_prompt} placeholder is resolved by
    # WorkflowPromptPlaceholderProvider, which injects the active profile's text.
    world.add_component(
        agent,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_workflow_state_prompt}"),
        ),
    )

    # Register tools
    world.add_component(agent, make_tool_registry(world, agent))

    # Install the workflow (attaches WorkflowDefinitionComponent + WorkflowRuntimeComponent)
    install_workflow(world, agent, WRITING_WORKFLOW, agent_key="assistant")

    # --- Systems (order matters) ---
    # WorkflowStateSystem MUST run before SystemPromptRenderSystem so the
    # active profile is committed before the prompt is rendered.
    world.register_system(WorkflowStateSystem(priority=-25), priority=-25)
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # --- Run ---
    runner = Runner()
    await runner.run(world, max_ticks=20)

    # --- Print results ---
    from ecs_agent.workflows._components import WorkflowRuntimeComponent

    runtime = world.get_component(agent, WorkflowRuntimeComponent)
    conv = world.get_component(agent, ConversationComponent)

    print("\n" + "=" * 60)
    print(f"Final workflow state : {runtime.current_state_id if runtime else 'unknown'}")
    if runtime:
        print(f"Transition history  : {' → '.join(runtime.transition_history)}")
    print("=" * 60)

    if conv:
        print("\n--- Conversation ---")
        for msg in conv.messages:
            role = msg.role.upper()
            content = str(msg.content)[:200]
            print(f"[{role}] {content}")


if __name__ == "__main__":
    asyncio.run(main())
