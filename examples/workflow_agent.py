"""Phase graph example: stateful agent with explicit, tool-driven phase transitions.

This example demonstrates `ecs_agent.phases`: declare a graph of phases with
per-phase system prompts and allowed transitions, bind it to an entity, and
move between phases by calling `advance()` — here directly from tool handlers.

Scenario: a two-phase writing assistant.

    DRAFT  ──(tool: mark_draft_ready)──►  REVIEW  ──(tool: approve_draft)──►  DONE

- DRAFT and REVIEW carry different prompts, so the LLM "persona" changes when
  the phase changes.  DONE reuses the REVIEW prompt to show that hops between
  phases sharing a prompt keep the rendered system prompt cache stable.
- There are no gate conditions, marker components, or polling system: tools
  call `await advance(...)` themselves.  Each transition is validated against
  the graph and recorded in PhaseComponent.history.

Dual-mode:
  - Without LLM_API_KEY → FakeModel (deterministic, works offline)
  - With    LLM_API_KEY → real LLM (OpenAI-compatible or Anthropic)
"""

from __future__ import annotations

import asyncio
import os

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PhaseComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools import tool
from ecs_agent.types import CompletionResult, EntityId, Message, ToolCall, ToolSchema

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Phase graph
# ---------------------------------------------------------------------------

DRAFTER_PROMPT = (
    "You are a creative writing assistant in DRAFT mode.\n"
    "Help the user write and refine their draft.\n"
    "When the draft is ready, call the `mark_draft_ready` tool."
)
REVIEWER_PROMPT = (
    "You are a critical reviewer in REVIEW mode.\n"
    "Evaluate the draft for clarity, accuracy, and completeness.\n"
    "When you approve it, call the `approve_draft` tool.\n"
    "If you need revisions, just say so."
)

# DONE shares REVIEWER_PROMPT: transitions between phases with the same prompt
# leave the rendered system prompt cache intact.
WRITING_GRAPH = build_graph(
    "writing-flow",
    initial="DRAFT",
    phases=[
        PhaseSpec(phase_id="DRAFT", prompts={"assistant": DRAFTER_PROMPT}, to=("REVIEW",)),
        PhaseSpec(phase_id="REVIEW", prompts={"assistant": REVIEWER_PROMPT}, to=("DONE",)),
        PhaseSpec(phase_id="DONE", prompts={"assistant": REVIEWER_PROMPT}, terminal=True),
    ],
)

# ---------------------------------------------------------------------------
# Tools that agents call to drive transitions
# ---------------------------------------------------------------------------


def make_tool_registry(world: World, entity_id: EntityId) -> ToolRegistryComponent:
    @tool(name="mark_draft_ready", description="Signal that the draft is complete and ready for review.")
    async def mark_draft_ready() -> str:
        await advance(world, entity_id, "REVIEW", reason="tool:mark_draft_ready")
        logger.info("workflow_tool_called", tool="mark_draft_ready", entity_id=entity_id)
        return "Draft marked as ready. Moving to REVIEW state."

    @tool(name="approve_draft", description="Approve the draft and complete the workflow.")
    async def approve_draft() -> str:
        await advance(world, entity_id, "DONE", reason="tool:approve_draft")
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

    # System prompt: the ${_phase_prompt} placeholder resolves to the current
    # phase's prompt for this entity's bound agent key.
    world.add_component(
        agent,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline="${_phase_prompt}"),
        ),
    )

    # Register tools
    world.add_component(agent, make_tool_registry(world, agent))

    # Bind the phase graph (attaches PhaseComponent + PhaseDefinitionComponent)
    await bind_phase_graph(world, agent, WRITING_GRAPH, agent_key="assistant")

    # --- Systems (order matters) ---
    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # --- Run ---
    runner = Runner()
    await runner.run(world, max_ticks=20)

    # --- Print results ---
    component = world.get_component(agent, PhaseComponent)
    conv = world.get_component(agent, ConversationComponent)

    print("\n" + "=" * 60)
    print(f"Final workflow state : {component.phase if component else 'unknown'}")
    if component:
        history = " → ".join(f"{h['from']}→{h['to']}" for h in component.history)
        print(f"Transition history  : {history}")
    print("=" * 60)

    if conv:
        print("\n--- Conversation ---")
        for msg in conv.messages:
            role = msg.role.upper()
            content = str(msg.content)[:200]
            print(f"[{role}] {content}")


if __name__ == "__main__":
    asyncio.run(main())
