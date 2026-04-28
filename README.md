<div align="center">
  <h1>ecs-agent</h1>

  <p>
    <strong>Entity-Component-System architecture for composable AI agents.</strong>
  </p>
</div>


---

Build modular, testable LLM agents by composing behavior from dataclass components, async systems, and pluggable providers. No inheritance hierarchies, just clean composition.

## Installation

```bash
# Clone and install with uv
git clone https://github.com/MoveCloudROY/ecs-agent.git
cd ecs-agent
uv sync --group dev
# Install with embeddings support (optional)
uv pip install -e ".[embeddings]"
# Install with MCP support (optional)
uv pip install -e ".[mcp]"
```

> **Requires Python ≥ 3.11**

## Quick Start

```python
import asyncio
import os

from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import Model
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.types import Message
from ecs_agent.providers.config import ApiFormat


async def main() -> None:
    world = World(name="my-agent")  # optional name — appears in all log events

    # Create a model — works with any OpenAI-compatible or Anthropic API
    model = Model(
        os.getenv("LLM_MODEL", "gpt-4o"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["LLM_API_KEY"],
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )

    # Create an agent entity and attach components
    agent = world.create_entity()
    world.add_component(agent, LLMComponent(
        model=model,
        system_prompt="You are a helpful assistant.",
    ))
    world.add_component(agent, ConversationComponent(
        messages=[Message(role="user", content="Hi there!")],
    ))

    # Register systems (priority controls execution order)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run the agent loop
    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Read results
    conv = world.get_component(agent, ConversationComponent)
    if conv:
        for msg in conv.messages:
            print(f"{msg.role}: {msg.content}")


asyncio.run(main())
```

## Features

### Composition-First Architecture
Mix 35+ components to build custom agents without inheritance bloat. The Entity-Component-System (ECS) pattern keeps logic and data strictly separated, making agents modular, serializable, and easy to test. Fully type-safe with strict mypy and `dataclass(slots=True)`.

- **Named Worlds** — Pass `name="my-agent"` to `World(name=...)` to tag every log event (`entity_created`, `component_added`, `run_start`, `tick_start`, etc.) with `world_name`. Child worlds spawned by `SubagentSystem` are automatically named `<subagent_name>-<hex8>` for end-to-end log correlation across nested agent calls.
### Multi-Agent Orchestration
- **Subagent Delegation** — Spawn child agents for subtasks with skill and permission inheritance. Control the `queued/running/succeeded/failed` lifecycle via a process-global FIFO scheduler.
- **MessageBus** — Parent-child and sibling messaging via pub/sub or request-response patterns.
- **Unified API** — Control lifecycle with `subagent`, `subagent_status`, `subagent_result`, `subagent_wait`, and `subagent_cancel` tools. Supports explicit wait/callback model for background sessions.

### Advanced Reasoning & Tree Search
- **Tree Conversations** — Branch reasoning paths, navigate multiple strategies, and linearize history for LLM compatibility.
- **Planning & ReAct** — Multi-step reasoning with dynamic replanning on errors or unexpected tool results.
- **MCTS Optimization** *(experimental)* — Find optimal execution paths using Monte Carlo Tree Search for complex goals.

### Scratchbook Artifact Registry
- **`ArtifactRegistry`** — Canonical persistence layer for durable scratchbook records and mutable plan execution state.
- **Canonical immutable records** — Tool and subagent outputs persist to `scratchbook/records/tool/tool_<uuid24>` and `scratchbook/records/subagent/subagent_<uuid24>`.
- **Canonical mutable plan state** — Plan markdown and Boulder machine state live at `scratchbook/<plan_slug>/plan.md` and `scratchbook/<plan_slug>/executes/boulder.json`.
- **Trigger-to-Boulder lifecycle** — Plan-type script triggers create Boulder; planning/replanning/tool systems update it throughout execution.
- **Inline payload policy** — Artifact inline content is populated only when UTF-8 payload size is `<= 2048` bytes. For larger results, `inline_content` is `None` and content is file-backed only. The persisted file always stores the full content — no truncation or summarisation.
- **Prompt Provider** — Injects scratchbook context into system prompts via `ScratchbookPromptConfig` component.

### Prompt Normalization & Injection
- **`SystemPromptConfigSpec`** — Declare system prompts as `${name}` placeholder templates with static strings, callable resolvers, or file paths as sources.
- **`SystemPromptRenderSystem`** — ECS system (recommended priority -20) that resolves all `${name}` placeholders and writes a `RenderedSystemPromptComponent` for LLM callers.
- **`UserPromptNormalizationSystem`** — ECS system (recommended priority -10) that injects trigger templates into outbound user messages and writes a `RenderedUserPromptComponent`. Slash-command skill context and ContextPool entries are injected later at call-time by `prepare_outbound_messages()`.
- **Workflow State & Gates** — Declarative state machine DSL for building stateful agents. Define states, prompt profiles, and transition gates (`has`, `field`, `all_of`, etc.) that drive behavior over multiple ticks.
- **Built-in Placeholders** — `${_installed_tools}`, `${_installed_skills}`, `${_installed_mcps}`, `${_installed_subagents}` automatically expand to the current inventory.
- **Provider Extension Seam** — A synchronous, narrow provider protocol (`BuiltinPlaceholderProvider`) for injecting domain-specific context into system prompts. Used by the scratchbook prompt provider.
- **Callable Placeholders** — Pass a `() -> str` callable as a placeholder resolver for dynamic content; must be side-effect-free and return a string.
- **Trigger Templates** — `@keyword` or `event:<name>` trigger patterns transform outbound user messages without mutating conversation history. Three action kinds are supported:
  - `replace` — replaces the entire user message with the trigger's `content`
  - `inject` — prepends the trigger's `content` before the user message
  - `script` — invokes a registered async Python function (`async (world, entity_id, user_text) -> str | None`). Return a string to replace the prompt; return `None` to keep the original. The handler may also mutate World state as a side effect (e.g., attach components). Register via `UserPromptConfigComponent(script_handlers={"key": fn})`. **Not available in Agent DSL** — Python API only.
- **Strict Errors** — Missing placeholder keys and resolver failures raise immediately; no silent fallbacks.

### Two-Tier Skill System
- **Markdown Skills** — Define agent capabilities via `SKILL.md` files with YAML frontmatter. System prompts are injected automatically, and `@`-prefixed relative paths are resolved to workspace-safe paths at load time.
- **Script Skills** — Extend markdown skills with Python tool handlers in a `scripts/` directory, executed as sandboxed subprocesses.
- **Built-in Tools** — `BuiltinToolsSkill` provides `read_file`, `write_file`, `edit_file`, `bash`, `glob`, and `interactive_bash` with workspace binding, path traversal protection, hash-anchored editing, and persistent tmux session support.
- **Skill Discovery** — File-based skill loading from directories with metadata-first activation and staged full-context injection via `load_skill_details`.

### Production Infrastructure
- **5 LLM Providers + Streaming** — OpenAI, Claude, LiteLLM (100+ models), Fake, and Retry providers with real-time SSE token delivery.
- **Context Management** — Checkpoints (undo/resume), conversation compaction (XML system-prompt summaries), and memory windowing.
- **Tool Ecosystem** — Auto-discovery via `@tool` decorator, manual approval flows, secure `bwrap` sandboxing, and composable skills.
- **MCP Integration** — Connect to external MCP tool servers via stdio, SSE, or HTTP transports with namespaced tool mapping.

## Architecture

```
src/ecs_agent/
├── core/
│   ├── world.py             # World, entity/component/system registry
│   ├── runner.py             # Runner, tick loop until TerminalComponent
│   ├── system.py             # System Protocol + SystemExecutor
│   ├── component.py          # ComponentStore
│   ├── entity.py             # EntityIdGenerator
│   ├── query.py              # Query engine for entity filtering
│   └── event_bus.py          # Pub/sub EventBus
├── components/
│   └── definitions.py        # 30 component dataclasses
├── providers/
│   ├── protocol.py           # LLMModel Protocol
│   ├── openai_provider.py    # OpenAI-compatible HTTP provider (httpx)
│   ├── claude_provider.py    # Anthropic Claude provider
│   ├── litellm_provider.py   # LiteLLM unified provider
│   ├── fake_provider.py      # Deterministic test provider
│   └── retry_provider.py     # Retry wrapper (tenacity)
├── systems/                  # 15 built-in systems
│   ├── reasoning.py          # LLM inference
│   ├── planning.py           # Multi-step plan execution
│   ├── replanning.py         # Dynamic plan adjustment
│   ├── tool_execution.py     # Tool call dispatch
│   ├── permission.py         # Tool whitelisting/blacklisting
│   ├── memory.py             # Conversation memory management
│   ├── collaboration.py      # (Removed in favor of MessageBusSystem)
│   ├── message_bus.py        # Pub/sub and request-response messaging
│   ├── error_handling.py     # Error capture and recovery
│   ├── tree_search.py        # MCTS plan optimization
│   ├── tool_approval.py      # Human-in-the-loop approval
│   ├── rag.py                # Retrieval-Augmented Generation
│   ├── checkpoint.py         # World state snapshots
│   ├── compaction.py         # Conversation compaction
│   ├── user_input.py         # Async user input
│   └── subagent.py           # Subagent delegation
├── tools/
│   ├── __init__.py           # Tool utilities
│   ├── discovery.py          # Auto-discovery of tools
│   ├── sandbox.py            # Secure execution environment
│   ├── bwrap_sandbox.py      # bwrap-backed isolation
│   ├── builtins/               # Standard library skills
│   │   ├── file_tools.py       # read/write/edit logic
│   │   ├── bash_tool.py        # Shell execution
│   │   ├── edit_tool.py        # Hash-anchored editing core
│   │   └── __init__.py         # BuiltinTools ScriptSkill definition
├── types.py                  # Core types (EntityId, Message, ToolCall, etc.)
├── serialization.py          # WorldSerializer for save/load
└── logging.py                # structlog configuration
├── skills/                       # Skills system
│   ├── protocol.py             # ScriptSkill Protocol definition
│   ├── manager.py              # SkillManager lifecycle handler
│   ├── discovery.py            # File-based skill discovery
│   └── web_search.py           # Brave Search integration
├── mcp/                          # MCP integration
```

The **Runner** repeatedly ticks the **World** until a `TerminalComponent` is attached to an entity. Execution also stops if `max_ticks` is reached (default 100). Pass `max_ticks=None` for infinite execution until a `TerminalComponent` is found. Each tick follows this flow:

1. Systems execute in priority order (lower = earlier).
2. Those at the same priority level run concurrently.
3. Logical operations read or write components on entities. This represents the entire data flow.

For interactive agents that must continue after a successful reasoning turn, register the opt-in `TerminalCleanupSystem` after reasoning (recommended `priority=1`). It clears selected terminal reasons—by default only `reasoning_complete`—without changing Runner's core stop semantics.

```
World
 ├── Entity 0 ── [LLMComponent, ConversationComponent, PlanComponent, ...]
 ├── Entity 1 ── [LLMComponent, ConversationComponent, MessageBusSubscriptionComponent, ...]
 └── Systems ─── [ReasoningSystem(0), PlanningSystem(0), MessageBusSystem(5), MemorySystem(10), ...]
 └── Systems ─── [ReasoningSystem(0), PlanningSystem(0), ToolExecutionSystem(5), MemorySystem(10), ...]
                          │
                    Runner.run()
                          │
              Tick 1 → Tick 2 → ... → TerminalComponent found → Done
```

### Components

README stays at the overview level. For the full component catalog and field-by-field reference, use:

- [`docs/components.md`](docs/components.md) — every component, defaults, usage, and system relationships
- [`docs/systems.md`](docs/systems.md) — which systems produce and consume each component
- [`docs/core-concepts.md`](docs/core-concepts.md) — how `World`, entities, components, and systems fit together

## Examples

The `examples/` directory contains runnable demos for the major patterns in the framework:

| Example | Description |
|---------|-------------|
| [`chat_agent.py`](examples/chat_agent.py) | Minimal agent with dual-mode model (FakeModel / OpenAIModel) |
| [`tool_agent.py`](examples/tool_agent.py) | Tool use with automatic call/result cycling |
| [`react_agent.py`](examples/react_agent.py) | ReAct pattern. Thought → Action → Observation loop |
| [`plan_and_execute_agent.py`](examples/plan_and_execute_agent.py) | Dynamic replanning with RetryModel and configurable timeouts |
| [`streaming_agent.py`](examples/streaming_agent.py) | Real-time token streaming via SSE |
| [`vision_agent.py`](examples/vision_agent.py) | Multimodal image understanding with vision-capable LLM using `ImageUrlPart` |
| [`retry_agent.py`](examples/retry_agent.py) | RetryModel with custom retry configuration |
| [`multi_agent.py`](examples/multi_agent.py) | Two agents collaborating via `MessageBusSystem` pub/sub (dual-mode) |
| [`structured_output_agent.py`](examples/structured_output_agent.py) | Pydantic schema → JSON mode for type-safe responses |
| [`serialization_demo.py`](examples/serialization_demo.py) | Save and restore World state to/from JSON |
| [`tool_approval_agent.py`](examples/tool_approval_agent.py) | Manual approval flow for sensitive tools |
| [`tree_search_agent.py`](examples/tree_search_agent.py) | MCTS-based planning for complex goals (dual-mode) |
| [`rag_agent.py`](examples/rag_agent.py) | Retrieval-Augmented Generation demo (dual-mode with real embeddings) |
| [`subagent_delegation_basic.py`](examples/subagent_delegation_basic.py) | Subagent delegation — standard pattern: sync call, `SystemPromptRenderSystem`, placeholder resolution (dual-mode) |
| [`subagent_delegation.py`](examples/subagent_delegation.py) | Subagent delegation — full feature demo: background queuing, FIFO scheduler, streaming telemetry, `queued→running→succeeded` lifecycle (dual-mode) |
| [`claude_agent.py`](examples/claude_agent.py) | Native Anthropic Claude provider usage |
| [`litellm_agent.py`](examples/litellm_agent.py) | LiteLLM unified provider for 100+ models |
| [`streaming_system_agent.py`](examples/streaming_system_agent.py) | System-level streaming with events |
| [`compaction_agent.py`](examples/compaction_agent.py) | All compaction strategies: `full_history`, `predrop_then_compact`, custom prompt template, and repeated compaction with summary folding (dual-mode) |
| [`context_management_agent.py`](examples/context_management_agent.py) | Checkpoint, undo, and compaction demo (dual-mode) |
| [`skill_agent.py`](examples/skill_agent.py) | Skill system and BuiltinTools ScriptSkill (read/write/edit) lifecycle |
| [`skill_discovery_agent.py`](examples/skill_discovery_agent.py) | File-based skill loading from folder (dual-mode) |
| [`permission_agent.py`](examples/permission_agent.py) | Permission-restricted agent with tool filtering (dual-mode) |
| [`skill_agent.py`](examples/skill_agent.py) | Load a SKILL.md Skill and install it on an agent (dual-mode) |
| [`mcp_agent.py`](examples/mcp_agent.py) | MCP server integration and namespaced tool usage |
| [`agent_dsl_json.py`](examples/agent_dsl_json.py) | Load multi-agent configuration from JSON file using Agent DSL (dual-mode) |
| [`agent_dsl_markdown.py`](examples/agent_dsl_markdown.py) | Load primary agent + subagent from Markdown files using Agent DSL; demonstrates placeholders, triggers, skills, and subagent registry (dual-mode) |
| [`workflow_agent.py`](examples/workflow_agent.py) | Workflow DSL: two-phase writing assistant with gate-driven `DRAFT→REVIEW→DONE` transitions, prompt profiles, and `@tool`-registered handlers (dual-mode) |
| [`examples/e2e/plan_and_task/`](examples/e2e/plan_and_task/) | Interactive plan→review→execute workflow; recoverable state machine, review-gated planning, artifact persistence, and slash-command dispatch |


Run any example:

```bash
# FakeModel mode (no API key needed — works out of the box)
uv run python examples/chat_agent.py
uv run python examples/tool_agent.py

# Real LLM mode (set API credentials)
LLM_API_KEY=your-api-key uv run python examples/chat_agent.py
uv run python examples/react_agent.py

# RAG with real embeddings
LLM_API_KEY=your-api-key EMBEDDING_MODEL=text-embedding-3-small uv run python examples/rag_agent.py
```

## Real LLM Configuration

The Quick Start above uses a real LLM. Set these environment variables before running it:

```bash
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen3.5-flash
```

Then run any dual-mode example, for example:

```bash
uv run python examples/chat_agent.py
```

Model setup, registry-based construction, supported protocols, and model ID rules are documented in [`docs/models.md`](docs/models.md).

## Development

### Tests

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_world.py

# Run tests matching a keyword
uv run pytest -k "streaming"

# Verbose output
uv run pytest -v
```
### Real-LLM Integration Tests

Live tests are optional and skip cleanly when `LLM_API_KEY` is not set. Run the suite you care about with the same environment variables you use for examples:

```bash
LLM_API_KEY="$LLM_API_KEY" \
  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  LLM_MODEL=qwen3.5-flash \
  uv run pytest tests/live/test_llm_api_live.py -v

LLM_API_KEY="$LLM_API_KEY" \
  uv run pytest tests/live/test_compaction_live.py -v
```

See `tests/live/` for the available live suites.

### Type Checking

```bash
# Full strict type check
uv run mypy src/ecs_agent/

# Single file
uv run mypy src/ecs_agent/core/world.py
```

### Project Configuration

- **Build**: hatchling
- **Package manager**: uv (lockfile: `uv.lock`)
- **pytest**: `asyncio_mode = "auto"`, async tests run without explicit event loop setup
- **mypy**: `strict = true`, `python_version = "3.11"`

### GitHub CI and Release

- Pushes and pull requests run `uv run pytest` via GitHub Actions.
- Tags matching `v*` trigger the release pipeline.
- Release tags must match the package version in both `pyproject.toml` and `src/ecs_agent/__init__.py`.
- Tagged releases run the full `tests/live/` suite before publishing.
- Release notes are manual and versioned at `docs/releases/vX.Y.Z.md`.
- GitHub Release body content comes directly from the matching release-notes file.

## Documentation

See [`docs/`](docs/) for detailed guides:

### Getting Started
- [Getting Started](docs/getting-started.md), Installation, first agent, key concepts
- [Architecture](docs/architecture.md), ECS pattern, data flow, system lifecycle
- [Core Concepts](docs/core-concepts.md), World, Entity, Component, System, Runner
- [API Reference](docs/api-reference.md), Complete API surface
- [Examples](docs/examples.md), Walkthrough of the example gallery

### Core Features
- [Components](docs/components.md), Complete component catalog with usage examples
- [Systems](docs/systems.md), Built-in systems and configuration details
- [Models](docs/models.md), model selection, registry routing, and built-in model implementations
- [Streaming](docs/features/streaming.md), SSE streaming setup and usage
- [Structured Output](docs/features/structured-output.md), Pydantic schema → JSON mode
- [Serialization](docs/features/serialization.md), World state persistence
- [Logging](docs/features/logging.md), structlog integration
- [Retry](docs/features/retry.md), RetryModel configuration
- [Workflow DSL](docs/features/workflows.md), Declarative state machine and transition gates

### Agent Capabilities
- [Context Management](docs/features/context-management.md), Checkpoint, undo, and compaction
- [Runtime Control](docs/features/runtime-control.md), Entity registry, system lifecycle, model switching, interruption, revert
- [Agent DSL](docs/features/agent-dsl.md), Declarative agent definition and loading
- [Agent Scratchbook](docs/features/scratchbook.md), `ArtifactRegistry` canonical paths, Boulder lifecycle, persistence APIs, and prompt provider.

### Tools & Integration
- [Skills](docs/features/skills.md), Composable capabilities and progressive disclosure
- [Built-in Tools](docs/features/builtin-tools.md), File manipulation and shell execution
- [Tool Discovery & Approval](docs/features/tool-discovery.md), Auto-discovery, sandbox, approval flow
- [MCP Integration](docs/features/mcp.md), Connecting to external MCP tool servers
- [Web Search](docs/features/web-search.md), Brave Search API integration
- [Permissions](docs/features/permissions.md), Tool filtering and bwrap sandboxing
- [User Input](docs/features/user-input.md), Async human-in-the-loop input

## License
MIT
