"""Subagent delegation example driven by ReasoningSystem and ToolExecutionSystem.

This example demonstrates three parent-controlled delegation patterns without
calling the subagent tool handlers directly from demo code:

- A synchronous subagent call.
- Two background subagent sessions that exercise queued -> running -> succeeded.
- A streamed background subagent that publishes EventBus telemetry.

It supports two modes:

- FakeProvider mode when ``LLM_API_KEY`` is unset.
- Real OpenAI-compatible mode (for Aliyun/Qwen or another compatible endpoint)
  when ``LLM_API_KEY`` is present.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.logging import configure_logging
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamDelta,
    SubagentConfig,
    SubagentSessionRecord,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
    ToolCall,
)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"
DEMO_SESSION_IDS = (
    "session-slow-worker",
    "session-queued-worker",
    "session-stream-worker",
)

SLOW_BACKGROUND_RESULT = (
    "<subagent_background_result>\n"
    "<summary>Slow background summary for the parent.</summary>\n"
    "<full_result>Slow background answer finished after the queued job waited its turn.</full_result>\n"
    "</subagent_background_result>"
)
QUEUED_BACKGROUND_RESULT = (
    "<subagent_background_result>\n"
    "<summary>Queued background summary for the parent.</summary>\n"
    "<full_result>Queued background answer completed once the slow session released the slot.</full_result>\n"
    "</subagent_background_result>"
)

SYNC_PROMPT = "Give one sentence on where early quantum value appears."
SLOW_PROMPT = "Produce the slow background answer."
QUEUED_PROMPT = "Produce the queued background answer."
STREAM_PROMPT = "Stream a concise answer back to the parent."

PARENT_SYSTEM_PROMPT = (
    "You are a delegation manager demonstrating the subagent tools. "
    "You must complete the task by calling tools, not by inventing results. "
    "Follow the requested order exactly. When a background subagent returns a "
    "session_id, launch all required background work first, then call "
    "subagent_wait() exactly once for the slow and queued sessions. After the "
    "system notification wakes you up, call subagent_result with the same "
    'session_id values and include read_method="summary" or read_method="full" '
    "as requested. Only produce a final assistant answer after every required "
    "tool call succeeds. Do not call subagent_status or subagent_cancel in this "
    "demo. When you pass timeout to subagent_result, keep it numeric rather than "
    "a quoted string."
)

PARENT_USER_PROMPT = (
    "Please demonstrate subagent delegation:\n"
    "1. call sync-worker synchronously with the prompt '" + SYNC_PROMPT + "'\n"
    "2. launch slow-worker in background with the prompt '" + SLOW_PROMPT + "'\n"
    "3. launch queued-worker in background with max_concurrency=1 still in effect "
    "using the prompt '" + QUEUED_PROMPT + "'\n"
    "4. after both background launches, call subagent_wait() once for the slow and "
    "queued session_ids, then wait for the system notification before reading "
    "results\n"
    "5. after the notification, call subagent_result for slow-worker with "
    'read_method="summary" and call subagent_result for queued-worker with '
    'read_method="full"\n'
    "6. launch stream-worker with background=True and stream=True using the prompt '"
    + STREAM_PROMPT
    + "'\n"
    "7. collect the stream-worker result via subagent_result using its returned "
    "session_id\n"
    "8. finish with a concise summary that explicitly mentions sync, background, "
    "stream, queued, running, and succeeded. Do not call subagent_status or "
    "subagent_cancel. If you include timeout, pass it as a number like 30, not a string."
)


class DelayedProvider:
    def __init__(self, provider: LLMProvider, delay_seconds: float) -> None:
        self._provider = provider
        self._delay_seconds = delay_seconds

    async def complete(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        thread_response_id: str | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        await asyncio.sleep(self._delay_seconds)
        if thread_response_id is None:
            return await self._provider.complete(
                messages,
                tools=tools,
                stream=stream,
                response_format=response_format,
            )

        return await self._provider.complete(
            messages,
            tools=tools,
            stream=stream,
            response_format=response_format,
            thread_response_id=thread_response_id,
        )


async def main() -> None:
    configure_logging(json_output=False, level="ERROR")

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        print(f"Base URL: {base_url}")
    else:
        print("No LLM_API_KEY provided. Using FakeProvider for demonstration.")
        print("To use a real API, set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL.")
    print()

    parent_provider, registry = _build_providers(api_key, base_url, model)
    world, parent_id = _build_world(
        parent_provider=parent_provider,
        registry=registry,
        model=model if api_key else "fake-parent",
    )
    stream_events = _subscribe_to_stream_events(world)

    runner = Runner()
    await runner.run(world, max_ticks=20)

    _print_demo_summary(world, parent_id, stream_events)


def _build_world(
    *,
    parent_provider: LLMProvider,
    registry: SubagentRegistryComponent,
    model: str,
) -> tuple[World, int]:
    world = World(name="subagent-delegation-demo")
    parent_id = world.create_entity()
    world.add_component(
        parent_id,
        LLMComponent(
            model=parent_provider,
            
            system_prompt=PARENT_SYSTEM_PROMPT,
        ),
    )
    world.add_component(
        parent_id,
        ConversationComponent(
            messages=[Message(role="user", content=PARENT_USER_PROMPT)]
        ),
    )
    world.add_component(parent_id, ToolRegistryComponent(tools={}, handlers={}))
    world.add_component(parent_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(parent_id, registry)

    subagent_system = SubagentSystem(max_background_concurrency=1)
    subagent_system.install_subagent_tool(world, parent_id)
    subagent_system.install_subagent_control_tools(world, parent_id)
    _install_demo_session_ids(subagent_system)
    _install_demo_tool_wrappers(world, parent_id)

    world.register_system(SubagentWaitSystem(priority=-5), priority=-5)
    world.register_system(subagent_system, priority=-1)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ToolExecutionSystem(priority=5), priority=5)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)
    return world, int(parent_id)


def _build_providers(
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[LLMProvider, SubagentRegistryComponent]:
    if api_key:
        base_provider = OpenAIProvider(
            config=ProviderConfig(
                provider_id="openai",
                base_url=base_url,
                api_key=api_key,
                api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
            ),
            model=model,
        )
        registry = _build_registry(base_provider, model=model)
        return base_provider, registry

    parent_provider = FakeProvider(responses=_fake_parent_responses())
    sync_provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content=(
                        "Synchronous subagent answer: quantum gains will arrive first "
                        "in optimization and chemistry."
                    ),
                )
            )
        ]
    )
    slow_provider = DelayedProvider(
        FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content=SLOW_BACKGROUND_RESULT,
                    )
                )
            ]
        ),
        delay_seconds=0.2,
    )
    queued_provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content=QUEUED_BACKGROUND_RESULT,
                )
            )
        ]
    )
    stream_provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="Streamed background answer delivered to the parent event bus.",
                )
            )
        ]
    )

    registry = SubagentRegistryComponent(
        subagents={
            "sync-worker": SubagentConfig(
                name="sync-worker",
                model=sync_provider,
                system_prompt="Return a concise direct answer.",
            ),
            "slow-worker": SubagentConfig(
                name="slow-worker",
                model=slow_provider,
                system_prompt="Return a concise answer after a brief pause.",
            ),
            "queued-worker": SubagentConfig(
                name="queued-worker",
                model=queued_provider,
                system_prompt="Return a short queued follow-up answer.",
            ),
            "stream-worker": SubagentConfig(
                name="stream-worker",
                model=stream_provider,
                system_prompt="Return a short answer suitable for streaming.",
            ),
        }
    )
    return parent_provider, registry


def _build_registry(provider: LLMProvider, *, model: str) -> SubagentRegistryComponent:
    return SubagentRegistryComponent(
        subagents={
            "sync-worker": SubagentConfig(
                name="sync-worker",
                model=provider,
                
                system_prompt="Return a concise direct answer.",
            ),
            "slow-worker": SubagentConfig(model=DelayedProvider(provider, delay_seconds=0.2),
name="slow-worker",
                system_prompt="Return a concise answer after a brief pause.",
            ),
            "queued-worker": SubagentConfig(
                name="queued-worker",
                model=provider,
                
                system_prompt="Return a short queued follow-up answer.",
            ),
            "stream-worker": SubagentConfig(
                name="stream-worker",
                model=provider,
                
                system_prompt="Return a short answer suitable for streaming.",
            ),
        }
    )


def _fake_parent_responses() -> list[CompletionResult]:
    return [
        CompletionResult(
            message=Message(
                role="assistant",
                content="Starting synchronous delegation.",
                tool_calls=[
                    ToolCall(
                        id="call-sync",
                        name="subagent",
                        arguments={
                            "category": "sync-worker",
                            "prompt": SYNC_PROMPT,
                            "background": False,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Launching the slow background worker.",
                tool_calls=[
                    ToolCall(
                        id="call-slow-background",
                        name="subagent",
                        arguments={
                            "category": "slow-worker",
                            "prompt": SLOW_PROMPT,
                            "background": True,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Launching the queued background worker.",
                tool_calls=[
                    ToolCall(
                        id="call-queued-background",
                        name="subagent",
                        arguments={
                            "category": "queued-worker",
                            "prompt": QUEUED_PROMPT,
                            "background": True,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Waiting for the background workers to finish.",
                tool_calls=[
                    ToolCall(
                        id="call-background-wait",
                        name="subagent_wait",
                        arguments={
                            "session_ids": [DEMO_SESSION_IDS[0], DEMO_SESSION_IDS[1]],
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Reading the cached slow background summary.",
                tool_calls=[
                    ToolCall(
                        id="call-slow-result",
                        name="subagent_result",
                        arguments={
                            "session_id": DEMO_SESSION_IDS[0],
                            "read_method": "summary",
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Reading the full queued background result.",
                tool_calls=[
                    ToolCall(
                        id="call-queued-result",
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
                content="Launching the streaming background worker.",
                tool_calls=[
                    ToolCall(
                        id="call-stream-background",
                        name="subagent",
                        arguments={
                            "category": "stream-worker",
                            "prompt": STREAM_PROMPT,
                            "background": True,
                            "stream": True,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Collecting the streaming background result.",
                tool_calls=[
                    ToolCall(
                        id="call-stream-result",
                        name="subagent_result",
                        arguments={
                            "session_id": DEMO_SESSION_IDS[2],
                            "timeout": 5.0,
                        },
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content=(
                    "Delegation complete. Sync, background, and stream runs "
                    "succeeded, and the background lifecycle progressed from "
                    "queued to running to succeeded."
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


def _install_demo_tool_wrappers(world: World, parent_id: int) -> None:
    tool_registry = world.get_component(parent_id, ToolRegistryComponent)
    assert tool_registry is not None

    result_handler = tool_registry.handlers["subagent_result"]

    async def normalized_subagent_result(
        session_id: str,
        read_method: str = "full",
        timeout: float | str | None = None,
    ) -> str:
        normalized_timeout: float | None
        if isinstance(timeout, str):
            normalized_timeout = float(timeout)
        else:
            normalized_timeout = timeout
        return await result_handler(
            session_id=session_id,
            read_method=read_method,
            timeout=normalized_timeout,
        )

    tool_registry.handlers["subagent_result"] = normalized_subagent_result


def _subscribe_to_stream_events(world: World) -> list[object]:
    received: list[object] = []

    async def on_start(event: SubagentStreamStartEvent) -> None:
        received.append(event)

    async def on_delta(event: SubagentStreamDeltaEvent) -> None:
        received.append(event)

    async def on_end(event: SubagentStreamEndEvent) -> None:
        received.append(event)

    world.event_bus.subscribe(SubagentStreamStartEvent, on_start)
    world.event_bus.subscribe(SubagentStreamDeltaEvent, on_delta)
    world.event_bus.subscribe(SubagentStreamEndEvent, on_end)
    return received


def _print_demo_summary(
    world: World, parent_id: int, stream_events: list[object]
) -> None:
    conversation = world.get_component(parent_id, ConversationComponent)
    sessions = world.get_component(parent_id, SubagentSessionTableComponent)

    print("=" * 72)
    print("TOOL CALL HISTORY")
    print("=" * 72)
    if conversation is None:
        print("No conversation found")
    else:
        for line in _conversation_history_lines(conversation.messages):
            print(line)

    print()
    print("[1/3] Synchronous subagent run")
    _print_filtered_conversation(
        conversation.messages if conversation is not None else [],
        keywords=("sync-worker", "quantum", "call-sync"),
    )

    print()
    print("[2/3] Background queue lifecycle")
    print("  lifecycle: queued -> running -> succeeded")
    _print_filtered_conversation(
        conversation.messages if conversation is not None else [],
        keywords=(
            "slow-worker",
            "queued-worker",
            DEMO_SESSION_IDS[0],
            DEMO_SESSION_IDS[1],
            "queued background",
            "slow background",
        ),
    )
    if sessions is not None:
        for line in _session_lines(sessions.sessions.values()):
            print(line)

    print()
    print("[3/3] Streamed background subagent")
    _print_filtered_conversation(
        conversation.messages if conversation is not None else [],
        keywords=("stream-worker", DEMO_SESSION_IDS[2], "stream"),
    )
    for event in stream_events:
        if isinstance(event, SubagentStreamStartEvent):
            print(
                "  stream start: "
                f"session={event.session_id} seq={event.seq} child_world={event.child_world_name}"
            )
        elif isinstance(event, SubagentStreamDeltaEvent):
            detail = event.delta if event.delta else event.reasoning_delta or ""
            print(
                f"  stream delta: session={event.session_id} seq={event.seq} {detail}"
            )
        elif isinstance(event, SubagentStreamEndEvent):
            print(f"  stream end: session={event.session_id} seq={event.seq}")


def _conversation_history_lines(messages: list[Message]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if message.role == "user":
            lines.append(f"[User] {message.content}")
            continue

        if message.role == "system":
            lines.append(f"[System] {message.content}")
            continue

        if message.tool_calls:
            for tool_call in message.tool_calls:
                lines.append(f"[Action] {tool_call.name}({tool_call.arguments})")
            continue

        if message.tool_call_id is not None:
            lines.append(f"[Result] {message.content}")
            continue

        lines.append(f"[Assistant] {message.content}")
    return lines


def _print_filtered_conversation(
    messages: list[Message], *, keywords: Iterable[str]
) -> None:
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    matched = False

    for line in _conversation_history_lines(messages):
        if any(keyword in line.lower() for keyword in lowered_keywords):
            print(f"  {line}")
            matched = True

    if not matched:
        print("  No matching conversation lines captured.")


def _session_lines(records: Iterable[SubagentSessionRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        lines.append(
            f"  session {record.session_id}: category={record.category} status={record.status}"
        )
    return lines


if __name__ == "__main__":
    asyncio.run(main())
