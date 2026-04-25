"""Background subagent delegation example.

This example demonstrates a manager agent launching two background subagents in
parallel, waiting for both to finish, reading each result, and then producing a
final summary.

Dual-mode operation:

- Without ``LLM_API_KEY``: uses ``FakeModel`` and runs out of the box.
- With ``LLM_API_KEY``: uses ``OpenAIModel`` against an OpenAI-compatible
  endpoint.
"""

from __future__ import annotations

import asyncio
import os

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging, get_logger
from ecs_agent.providers import FakeModel, OpenAIModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    EntityId,
    InheritancePolicy,
    Message,
    SubagentConfig,
    ToolCall,
)

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"
DEMO_SESSION_IDS = (
    "ses_bg_analyst",
    "ses_bg_writer",
)

MANAGER_SYSTEM_PROMPT = (
    "You are a manager coordinating two specialist subagents. "
    "Use the subagent tool with background=True to launch work in parallel. "
    "Launch both workers first, then call subagent_wait once, then read each "
    'result with subagent_result(read_method="full"), and only then write a '
    "combined final summary."
)

MANAGER_USER_PROMPT = (
    "Prepare a short report about launching a new analytics newsletter. "
    "First ask the analyst subagent to identify the strongest audience and "
    "content signals. Then ask the writer subagent to draft a punchy newsletter "
    "opening based on those findings. Run both in background, wait for both, "
    "read both full results, and finish with a combined summary."
)


async def main() -> None:
    configure_logging(json_output=False, level="ERROR")

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    if api_key:
        print(f"Using OpenAIModel with model: {model}")
        print(f"Base URL: {base_url}")
    else:
        print("No LLM_API_KEY provided. Using FakeModel for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
    print()

    manager_provider, registry = _build_providers(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    world, manager_id = _build_world(
        manager_provider=manager_provider,
        registry=registry,
        model=model if api_key else "fake-manager",
    )

    runner = Runner()
    await runner.run(world, max_ticks=12)

    print("=" * 72)
    print("Background Subagent Delegation Demo")
    print("=" * 72)
    _print_conversation("Manager", manager_id, world)


def _build_providers(
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[LLMModel, SubagentRegistryComponent]:
    manager_provider: LLMModel
    if api_key:
        manager_provider = OpenAIModel(
            config=ProviderConfig(
                provider_id="openai",
                base_url=base_url,
                api_key=api_key,
                api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            model=model,
        )
        subagent_provider = OpenAIModel(
            config=ProviderConfig(
                provider_id="openai",
                base_url=base_url,
                api_key=api_key,
                api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            model=model,
        )
        return manager_provider, _build_registry(subagent_provider)

    manager_provider = FakeModel(responses=_fake_manager_responses())
    analyst_provider = FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content=(
                        "Analysis findings: the strongest near-term audience is "
                        "product and data leaders at mid-market SaaS companies. "
                        "The most compelling newsletter angles are benchmark-backed "
                        "retention insights, short operational playbooks, and one "
                        "surprising metric trend per issue."
                    ),
                )
            )
        ]
    )
    writer_provider = FakeModel(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content=(
                        "Draft opening: Welcome to Analytics Dispatch — the weekly "
                        "brief for product and data leaders who want one sharp trend, "
                        "one practical playbook, and one metric that deserves a second "
                        "look before the next planning meeting."
                    ),
                )
            )
        ]
    )
    registry = SubagentRegistryComponent(
        subagents={
            "analyst": SubagentConfig(
                name="analyst",
                model=analyst_provider,
                description="Analyze audience and content signals",
                system_prompt=(
                    "You are an analyst sub-agent. Identify the strongest audience, "
                    "content angles, and key findings for the manager."
                ),
                max_ticks=10,
                inheritance_policy=InheritancePolicy(
                    inherit_system_prompt=True,
                    inherit_tools=[],
                ),
            ),
            "writer": SubagentConfig(
                name="writer",
                model=writer_provider,
                description="Write concise launch copy from findings",
                system_prompt=(
                    "You are a writer sub-agent. Turn the available direction into "
                    "a concise, polished newsletter opening for the manager."
                ),
                max_ticks=10,
                inheritance_policy=InheritancePolicy(
                    inherit_system_prompt=True,
                    inherit_tools=[],
                ),
            ),
        }
    )
    return manager_provider, registry


def _build_registry(model: LLMModel) -> SubagentRegistryComponent:
    return SubagentRegistryComponent(
        subagents={
            "analyst": SubagentConfig(
                name="analyst",
                model=model,
                
                description="Analyze audience and content signals",
                system_prompt=(
                    "You are an analyst sub-agent. Identify the strongest audience, "
                    "content angles, and key findings for the manager."
                ),
                max_ticks=10,
                inheritance_policy=InheritancePolicy(
                    inherit_system_prompt=True,
                    inherit_tools=[],
                ),
            ),
            "writer": SubagentConfig(
                name="writer",
                model=model,
                
                description="Write concise launch copy from findings",
                system_prompt=(
                    "You are a writer sub-agent. Turn the available direction into "
                    "a concise, polished newsletter opening for the manager."
                ),
                max_ticks=10,
                inheritance_policy=InheritancePolicy(
                    inherit_system_prompt=True,
                    inherit_tools=[],
                ),
            ),
        }
    )


def _build_world(
    *,
    manager_provider: LLMModel,
    registry: SubagentRegistryComponent,
    model: str,
) -> tuple[World, EntityId]:
    world = World(name="subagent-delegation-background-demo")
    manager_id = world.create_entity()

    world.add_component(
        manager_id,
        LLMComponent(
            model=manager_provider,
            
            system_prompt=MANAGER_SYSTEM_PROMPT,
        ),
    )
    world.add_component(
        manager_id,
        ConversationComponent(
            messages=[Message(role="user", content=MANAGER_USER_PROMPT)]
        ),
    )
    world.add_component(manager_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(manager_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(manager_id, registry)

    subagent_system = SubagentSystem(priority=-1)
    subagent_system.install_subagent_tool(world, manager_id, tool_name="subagent")
    subagent_system.install_subagent_control_tools(world, manager_id)
    _install_demo_session_ids(subagent_system)

    world.register_system(SubagentWaitSystem(priority=-5), priority=-5)
    world.register_system(subagent_system, priority=-1)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)
    return world, manager_id


def _fake_manager_responses() -> list[CompletionResult]:
    return [
        CompletionResult(
            message=Message(
                role="assistant",
                content="Launching the analyst in background mode.",
                tool_calls=[
                    ToolCall(
                        id="call-bg-analyst",
                        name="subagent",
                        arguments={
                            "category": "analyst",
                            "prompt": (
                                "Identify the best audience and content signals for a "
                                "new analytics newsletter launch."
                            ),
                            "background": True,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Launching the writer in background mode.",
                tool_calls=[
                    ToolCall(
                        id="call-bg-writer",
                        name="subagent",
                        arguments={
                            "category": "writer",
                            "prompt": (
                                "Draft a compelling opening paragraph for the new "
                                "analytics newsletter using the available research."
                            ),
                            "background": True,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Waiting for both background subagents to complete.",
                tool_calls=[
                    ToolCall(
                        id="call-bg-wait",
                        name="subagent_wait",
                        arguments={"session_ids": list(DEMO_SESSION_IDS)},
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Reading the analyst result.",
                tool_calls=[
                    ToolCall(
                        id="call-bg-analyst-result",
                        name="subagent_result",
                        arguments={
                            "session_id": DEMO_SESSION_IDS[0],
                            "read_method": "full",
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Reading the writer result.",
                tool_calls=[
                    ToolCall(
                        id="call-bg-writer-result",
                        name="subagent_result",
                        arguments={
                            "session_id": DEMO_SESSION_IDS[1],
                            "read_method": "full",
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content=(
                    "Final summary: the analyst identified mid-market SaaS product "
                    "and data leaders as the clearest audience, with benchmark-driven "
                    "retention insights and practical playbooks as the strongest hooks. "
                    "The writer then turned that direction into an energetic opening "
                    "paragraph for the newsletter launch."
                ),
            )
        ),
    ]


def _install_demo_session_ids(subagent_system: SubagentSystem) -> None:
    remaining_ids = iter(DEMO_SESSION_IDS)

    def create_session() -> str:
        try:
            return next(remaining_ids)
        except StopIteration as exc:
            raise RuntimeError("Demo session id sequence exhausted") from exc

    runtime_manager = getattr(subagent_system, "_runtime_manager")
    setattr(runtime_manager, "create_session", create_session)
    reconciled_ids = getattr(subagent_system, "_reconciled_session_ids")
    reconciled_ids.update(DEMO_SESSION_IDS)


def _print_conversation(label: str, entity_id: EntityId, world: World) -> None:
    print(f"--- {label} (entity {entity_id}) ---")
    conversation = world.get_component(entity_id, ConversationComponent)
    if conversation is None:
        print("  (no conversation)")
        return

    for message in conversation.messages:
        role = message.role.upper()
        if message.tool_calls:
            print(f"  [{role}] {message.content or '(no content)'}")
            for tool_call in message.tool_calls:
                print(f"         → Tool Call: {tool_call.name}({tool_call.arguments})")
            continue

        if message.role == "tool":
            print(f"  [{role}] (tool_call_id={message.tool_call_id})")
            for line in (message.content or "").splitlines() or [""]:
                print(f"         {line}")
            continue

        for index, line in enumerate((message.content or "").splitlines() or [""]):
            prefix = f"  [{role}] " if index == 0 else "         "
            print(f"{prefix}{line}")


if __name__ == "__main__":
    asyncio.run(main())
