"""Prompt normalization demo — showcases all prompt system features.

Demonstrates:
- System prompt rendering with ${name} placeholder templates
- Built-in placeholders: ${_installed_tools}, ${_installed_skills}, ${_installed_mcps}, ${_installed_subagents}
- User-defined placeholders: static strings and callable resolvers
- Template source: inline strings and file-based loading
- Keyword trigger injection (@code → prompt text)
- Event-based triggers (event:tool_success → evidence-based reasoning)
- Context pool injection (tool results, subagent status)

Run:
    uv run python examples/prompt_normalization_demo.py
    LLM_API_KEY=your-key uv run python examples/prompt_normalization_demo.py
"""

import asyncio
import os
import tempfile
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

from ecs_agent.components import (
    ContextEntry,
    ConversationComponent,
    LLMComponent,
    PromptContextQueueComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    SkillComponent,
    SkillMetadata,
    SubagentRegistryComponent,
    ToolRegistryComponent,
    UserPromptConfigComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    PromptTemplateSource,
    SystemPromptConfigSpec,
    TriggerSpec,
)
from ecs_agent.prompts.keyword_injection import inject_triggers
from ecs_agent.prompts.message_assembly import (
    build_keyword_registry,
    build_trigger_specs,
    collect_active_events,
)
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.protocol import LLMProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import (
    CompletionResult,
    Message,
    StreamDelta,
    SubagentConfig,
    ToolSchema,
)


class RecordingProvider:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.last_messages: list[Message] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult | AsyncIterator[StreamDelta]:
        self.last_messages = list(messages)
        return await self._provider.complete(
            messages,
            tools=tools,
            stream=stream,
            response_format=response_format,
        )


def _build_provider_from_env() -> tuple[LLMProvider, str, str]:
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")

    if api_key:
        real_provider: LLMProvider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        return real_provider, model, "real"

    fake_provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content=(
                        "Fake mode response: prompt normalization wiring is active."
                    ),
                )
            )
        ]
    )
    return fake_provider, model, "fake"


def _extract_outbound_user_message(messages: list[Message]) -> Message:
    if len(messages) < 2:
        raise RuntimeError(
            "Provider did not receive expected outbound message sequence"
        )
    return messages[-1]


def _build_system_prompt_config() -> SystemPromptConfigSpec:
    return SystemPromptConfigSpec(
        template_source=PromptTemplateSource(
            inline=(
                "You are a comprehensive prompt-normalization demo agent.\n\n"
                "Installed tools:\n${_installed_tools}\n\n"
                "Installed skills:\n${_installed_skills}\n\n"
                "Installed MCP tools:\n${_installed_mcps}\n\n"
                "Installed subagents:\n${_installed_subagents}\n\n"
                "Workspace: ${workspace}\n"
                "OS: ${os_name}\n"
                "Timestamp: ${timestamp}\n"
            )
        ),
        placeholders=[
            PlaceholderSpec(name="workspace", value="/home/demo/project"),
            PlaceholderSpec(
                name="os_name",
                value=lambda: __import__("platform").system(),
            ),
            PlaceholderSpec(
                name="timestamp",
                value=lambda: __import__("datetime").datetime.now().isoformat(),
            ),
        ],
    )


def _build_match_action_demo_outputs() -> dict[str, str]:
    trigger_specs = [
        TriggerSpec(
            pattern="REPLACE_ME",
            match_mode="prefix",
            action="replace",
            content="[REPLACE_ACTION_APPLIED]",
            priority=20,
        ),
        TriggerSpec(
            pattern="@code",
            match_mode="keyword",
            action="skill",
            content="[SKILL_ACTION_APPLIED]",
            priority=10,
        ),
        TriggerSpec(
            pattern="findings",
            match_mode="contains",
            action="script",
            content="[SCRIPT_ACTION_APPLIED]",
            priority=5,
        ),
    ]

    return {
        "replace/prefix": UserPromptNormalizationSystem.apply_trigger_specs(
            user_text="REPLACE_ME this prompt should be replaced",
            trigger_specs=trigger_specs,
        ),
        "skill/keyword": UserPromptNormalizationSystem.apply_trigger_specs(
            user_text="Please @code summarize latest findings",
            trigger_specs=trigger_specs,
        ),
        "script/contains": UserPromptNormalizationSystem.apply_trigger_specs(
            user_text="Please summarize findings from notes",
            trigger_specs=trigger_specs,
        ),
    }


async def _run_file_template_demo() -> str:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(
                "Agent workspace: ${workspace}\nOS: ${os_name}\nTimestamp: ${timestamp}"
            )
            temp_path = handle.name

        world = World()
        entity = world.create_entity()
        world.add_component(
            entity,
            SystemPromptConfigSpec(
                template_source=PromptTemplateSource(file_path=temp_path),
                placeholders=[
                    PlaceholderSpec(name="workspace", value="/home/demo/project"),
                    PlaceholderSpec(
                        name="os_name",
                        value=lambda: __import__("platform").system(),
                    ),
                    PlaceholderSpec(
                        name="timestamp",
                        value=lambda: __import__("datetime").datetime.now().isoformat(),
                    ),
                ],
            ),
        )

        await SystemPromptRenderSystem(priority=-20).process(world)
        rendered = world.get_component(entity, RenderedSystemPromptComponent)
        if rendered is None:
            return "<missing rendered file-based prompt>"
        return rendered.text
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


async def main() -> None:
    world = World()
    base_provider, model, mode = _build_provider_from_env()
    provider = RecordingProvider(base_provider)

    entity = world.create_entity()
    world.add_component(entity, LLMComponent(provider=provider, model=model))
    world.add_component(
        entity,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Please @code summarize latest findings",
                )
            ]
        ),
    )

    world.add_component(entity, _build_system_prompt_config())
    world.add_component(
        entity,
        ToolRegistryComponent(
            tools={
                "read_file": ToolSchema(
                    name="read_file",
                    description="Read file contents",
                    parameters={"type": "object", "properties": {}},
                ),
                "web_search": ToolSchema(
                    name="web_search",
                    description="Search the web",
                    parameters={"type": "object", "properties": {}},
                ),
                "summarize": ToolSchema(
                    name="summarize",
                    description="Summarize text",
                    parameters={"type": "object", "properties": {}},
                ),
            },
            handlers={},
        ),
    )
    world.add_component(
        entity,
        SkillComponent(
            skills={
                "code_review": SkillMetadata(
                    name="code_review",
                    description="Review code for issues",
                    tool_names=["read_file"],
                    has_system_prompt=True,
                ),
                "web_lookup": SkillMetadata(
                    name="web_lookup",
                    description="Look up information online",
                    tool_names=["web_search"],
                    has_system_prompt=False,
                ),
            }
        ),
    )
    world.add_component(
        entity,
        SubagentRegistryComponent(
            subagents={
                "researcher": SubagentConfig(
                    name="researcher",
                    provider=base_provider,
                    model=model,
                    system_prompt="Research assistant",
                ),
                "reviewer": SubagentConfig(
                    name="reviewer",
                    provider=base_provider,
                    model=model,
                    system_prompt="Review assistant",
                ),
            }
        ),
    )

    mcp_note = "MCPClientComponent attached"
    try:
        from ecs_agent.mcp.components import MCPClientComponent

        world.add_component(
            entity,
            MCPClientComponent(
                session=None,
                connected=True,
                cached_tools=[{"name": "mcp_fetch"}, {"name": "mcp_memory"}],
            ),
        )
    except ImportError as exc:
        mcp_note = f"MCPClientComponent unavailable, skipping ({exc})"

    context_entries = [
        ContextEntry(
            entry_id="tool-search-0",
            priority=30,
            registration_order=0,
            source_label="tool:search",
            content=(
                "source: tool:search\n"
                "status: success\n"
                "result: Found 3 relevant documents"
            ),
        ),
        ContextEntry(
            entry_id="subagent-researcher-1",
            priority=20,
            registration_order=1,
            source_label="subagent:researcher",
            content=(
                "source: subagent:researcher\n"
                "status: success\n"
                "result: Synthesized findings from 5 sources"
            ),
        ),
    ]
    world.add_component(
        entity, PromptContextQueueComponent(entries=list(context_entries))
    )
    world.add_component(
        entity,
        UserPromptConfigComponent(
            triggers={
                "@code": "Prioritize deterministic code-first reasoning.",
                "event:tool_success": "Prefer successful tool outputs as evidence.",
            },
            enable_context_pool=True,
        ),
    )

    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    await Runner().run(world, max_ticks=1)

    outbound_user = _extract_outbound_user_message(provider.last_messages)
    rendered_system = world.get_component(entity, RenderedSystemPromptComponent)
    rendered_user = world.get_component(entity, RenderedUserPromptComponent)
    context_queue = world.get_component(entity, PromptContextQueueComponent)
    placeholder_snapshot = (
        rendered_system.placeholder_snapshot if rendered_system is not None else {}
    )

    triggers = {
        "@code": "Prioritize deterministic code-first reasoning.",
        "event:tool_success": "Prefer successful tool outputs as evidence.",
    }
    event_demo_text = inject_triggers(
        "Please summarize latest findings",
        build_keyword_registry(triggers),
        trigger_specs=build_trigger_specs(triggers),
        active_events=collect_active_events(context_entries),
    )
    action_demos = _build_match_action_demo_outputs()

    rendered_system_text = rendered_system.text if rendered_system is not None else ""
    rendered_user_text = rendered_user.text if rendered_user is not None else ""
    outbound_text = outbound_user.content
    original_user_text = "Please @code summarize latest findings"

    print(f"[mode] {mode}")
    print(f"[mcp] {mcp_note}")

    print("\n=== 1. System Prompt: Built-in Placeholders ===")
    print(
        f"${{_installed_tools}} ->\n{placeholder_snapshot.get('_installed_tools', '<missing>')}"
    )
    print(
        f"${{_installed_skills}} ->\n"
        f"{placeholder_snapshot.get('_installed_skills', '<missing>')}"
    )
    print(
        f"${{_installed_mcps}} ->\n"
        f"{placeholder_snapshot.get('_installed_mcps', '<missing>')}"
    )
    print(
        f"${{_installed_subagents}} ->\n"
        f"{placeholder_snapshot.get('_installed_subagents', '<missing>')}"
    )

    print("\n=== 2. System Prompt: User Placeholders ===")
    print(f"workspace: {placeholder_snapshot.get('workspace', '<missing>')}")
    print(f"os_name: {placeholder_snapshot.get('os_name', '<missing>')}")
    print(f"timestamp: {placeholder_snapshot.get('timestamp', '<missing>')}")

    print("\n=== 3. System Prompt: Full Rendered Text ===")
    print(rendered_system_text or "<missing>")

    print("\n=== 4. User Prompt: Keyword Trigger Injection ===")
    print(rendered_user_text or "<missing>")
    print("\n[event-trigger demo]")
    print(event_demo_text)
    print("\n[match-action demos]")
    for name, value in action_demos.items():
        print(f"{name}: {value}")

    print("\n=== 5. User Prompt: Outbound Message (with Context Pool) ===")
    print(outbound_text)

    print("\n=== 6. Context Pool Entries ===")
    if context_queue is None or not context_queue.entries:
        print("<empty>")
    else:
        for entry in context_queue.entries:
            print(entry.content)

    print("\n=== 7. Verification Checks ===")
    print(f"rendered system prompt present: {rendered_system is not None}")
    print(f"rendered user prompt present: {rendered_user is not None}")
    print(
        f"[PROMPT_INJECT:@code] in outbound: {'[PROMPT_INJECT:@code]' in outbound_text}"
    )
    print(
        f"[PROMPT_CONTEXT_POOL] in outbound: {'[PROMPT_CONTEXT_POOL]' in outbound_text}"
    )
    print(f"user text preserved: {outbound_text.endswith(original_user_text)}")
    print(
        "${_installed_tools} resolved (no literal ${_installed_tools} in rendered): "
        f"{'${_installed_tools}' not in rendered_system_text}"
    )
    print(
        "workspace placeholder resolved: "
        f"{'${workspace}' not in rendered_system_text and '/home/demo/project' in rendered_system_text}"
    )
    print(
        "os_name placeholder resolved: "
        f"{'${os_name}' not in rendered_system_text and bool(placeholder_snapshot.get('os_name'))}"
    )

    print("\n=== 8. File-Based Template Demo ===")
    print(await _run_file_template_demo())


if __name__ == "__main__":
    asyncio.run(main())
