"""Sub-agent delegation — basic example.

This example shows the standard pattern for delegating a task to a sub-agent:

- Manager agent calls the ``subagent`` tool (synchronous, ``background=False``).
- ``SubagentSystem`` creates a child entity, runs it, and returns the result.
- Manager receives the result and produces a final summary.
- ``SystemPromptRenderSystem`` resolves ``${_installed_subagents}`` automatically.
- ``UserPromptNormalizationSystem`` normalises outbound user messages.

For a more advanced demo covering background queuing, FIFO scheduling, and
streaming telemetry see ``examples/subagent_delegation.py``.

Dual-mode operation:

- Without ``LLM_API_KEY``: Uses FakeModel (no network calls needed).
- With ``LLM_API_KEY``: Uses OpenAIModel for real LLM interaction.
"""

import asyncio
import os

from ecs_agent.components import ConversationComponent, OwnerComponent
from ecs_agent.core import World
from ecs_agent import AgentDefinition, build_agent
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
)
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    Message,
    ToolCall,
)


async def main() -> None:
    """Run the basic sub-agent delegation example.

    Flow:
      1. SystemPromptRenderSystem resolves ``${_installed_subagents}`` and
         custom placeholders into the manager's rendered system prompt.
      2. UserPromptNormalizationSystem normalises the outbound user message.
      3. Manager receives user question and calls the ``subagent`` tool
         synchronously (``background=False``).
      4. SubagentSystem creates and runs the child entity inline.
      5. Result is delivered back to the manager conversation.
      6. Manager synthesises a final summary.
    """

    # ── LLM Provider Configuration ──────────────────────────────────
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen3.5-flash"

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    if api_key:
        print(f"Using model: {model}")
        print(f"Base URL: {base_url}")
        print()
    else:
        print("No LLM_API_KEY provided. Using FakeModel for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
        print()

    # ── Provider Setup ──────────────────────────────────────────────
    manager_provider: LLMModel
    subagent_provider: LLMModel

    if api_key:
        manager_provider = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
        subagent_provider = Model(model, base_url=base_url, api_key=api_key, api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
    else:
        # FakeModel for manager:
        #   Turn 1 — call the 'subagent' tool synchronously (background=False).
        #   Turn 2 — produce a final summary once the result is available.
        manager_provider = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="I'll delegate the research task to the researcher sub-agent.",
                        tool_calls=[
                            ToolCall(
                                id="call_sync_001",
                                name="subagent",
                                arguments={
                                    "category": "researcher",
                                    "prompt": (
                                        "Research the most promising near-term "
                                        "applications of quantum computing."
                                    ),
                                    "background": False,
                                },
                            )
                        ],
                    )
                ),
                # Turn 2: final summary
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "Based on the research, here is the summary:\n\n"
                            "Quantum computing uses qubits to perform calculations "
                            "impossible for classical computers.\n"
                            "Key near-term applications include:\n"
                            "1. Drug discovery — simulating molecular interactions\n"
                            "2. Optimization — logistics and supply-chain routing\n"
                            "3. Cryptography — post-quantum encryption standards"
                        ),
                    )
                ),
            ]
        )

        # FakeModel for the researcher sub-agent
        subagent_provider = FakeModel(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=(
                            "After researching quantum computing applications, I found "
                            "three promising areas: (1) drug discovery through molecular "
                            "simulation, (2) combinatorial optimization for logistics, "
                            "and (3) post-quantum cryptography."
                        ),
                    )
                ),
            ]
        )

    # ── Manager Entity Setup ────────────────────────────────────────
    agent = build_agent(AgentDefinition(
        model=manager_provider,
        prompt=SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline=(
                "You are a manager agent. Delegate research to a subagent and summarize its result.\n"
                "Available tools:\n${_installed_tools}\n"
                "Available subagents:\n${_installed_subagents}\n"
                "Session: ${session_label}"
            )),
            placeholders=[PlaceholderSpec("session_label", "subagent-delegation-basic")],
        ),
        subagents={"researcher": AgentDefinition(
            model=subagent_provider, description="Research and gather information on any topic",
            prompt="Investigate the delegated topic and report your findings back to the manager.",
            max_ticks=10,
        )},
    ))
    world, manager_id = agent.world, agent.entity
    try:
        result = await agent.run_turn("What are the most promising near-term applications of quantum computing?", max_steps=10)
        print(f"status: {result.status.value}")
        if not result.success:
            raise RuntimeError(f"Delegation did not complete: {result.reason}")
    finally:
        for provider in (manager_provider, subagent_provider):
            if api_key:
                await provider.aclose()

    # ── Print Results ───────────────────────────────────────────────
    print("=" * 60)
    print("Manager Conversation (Basic Subagent Delegation)")
    print("=" * 60)
    _print_conversation("Manager", manager_id, world)

    # Show parent-child relationship
    for entity_id, components in world.query(OwnerComponent):
        (owner_comp,) = components
        print(
            f"\n[OwnerComponent] Sub-agent (entity {entity_id}) "
            f"→ Manager (entity {owner_comp.owner_id})"
        )


def _print_conversation(label: str, entity_id: EntityId, world: World) -> None:
    """Pretty-print an entity's conversation."""
    print(f"\n--- {label} (entity {entity_id}) ---")
    conv = world.get_component(entity_id, ConversationComponent)
    if conv is None:
        print("  (no conversation)")
        return
    for msg in conv.messages:
        role = msg.role.upper()
        if msg.tool_calls:
            print(f"  [{role}] {msg.content or '(no content)'}")
            for tool_call in msg.tool_calls:
                print(f"         → Tool Call: {tool_call.name}({tool_call.arguments})")
        elif msg.role == "tool":
            print(f"  [{role}] (tool_call_id={msg.tool_call_id})")
            lines = (msg.content or "").split("\n")
            first, rest = lines[0], lines[1:]
            print(f"         {first}")
            for line in rest:
                print(f"         {line}")
        else:
            lines = (msg.content or "").split("\n")
            first, rest = lines[0], lines[1:]
            print(f"  [{role}] {first}")
            for line in rest:
                print(f"         {line}")


if __name__ == "__main__":
    asyncio.run(main())
