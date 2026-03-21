# UI Design Flow E2E — Implementation Architecture

This document details the technical architecture and implementation of the UI Design Flow E2E example, focusing on the Entity-Component-System (ECS) patterns, system orchestration, and skill-based extension.

## Architecture Overview

The example follows a strict ECS pattern where data (Components) is separated from logic (Systems). The `World` manages entities and their components, while the `Runner` executes systems in a prioritized tick loop.

### Core Components

- **LLMComponent**: Stores the provider (OpenAI or Fake), model name (defaults to `qwen3.5-flash`), and system prompt.
- **ConversationComponent**: Manages the message history between the user and the agent.
- **UserInputComponent**: Indicates that the agent is waiting for external input from the user.
- **SkillComponent**: Holds metadata for installed skills (`/ui-navigator`, `/ui-prompt`).
- **ToolRegistryComponent**: (Internal) Map of tool names to their execution logic, populated by installed skills.
- **TerminalComponent**: Attached when the user sends an "exit" command or when reasoning completes; the runtime now pairs this with the opt-in `TerminalCleanupSystem` so `reasoning_complete` does not prematurely end interactive continuation.

### System Execution Order

Systems are registered with specific priorities to ensure correct data flow within each tick. Lower priority values execute first.

| System | Priority | Purpose |
|--------|----------|---------|
| **UserInputSystem** | -5 | Processes `UserInputComponent` and triggers `UserInputRequestedEvent`. |
| **ReasoningSystem** | 0 | Calls the LLM provider to generate the next response or tool call. |
| **TerminalCleanupSystem** | 1 | Clears `TerminalComponent(reason="reasoning_complete")` so interactive turns can continue. |
| **ToolExecutionSystem** | 5 | Dispatches pending tool calls to builtin file tools (read_file, write_file, etc.). |
| **MemorySystem** | 10 | Updates conversation history and manages context window. |
| **ErrorHandlingSystem** | 99 | Captures exceptions from other systems and attaches `ErrorComponent`. |

## Interactive Input Handling

Interactive input is achieved through an event-driven adapter in `runtime.py`.

1. **Trigger**: When `UserInputSystem` detects a `UserInputComponent`, it emits a `UserInputRequestedEvent`.
2. **Subscription**: `setup_interactive_input` subscribes to this event.
3. **Async Stdin**: The subscriber uses `asyncio.run_in_executor` to call the blocking `input()` function without freezing the event loop.
4. **Resolution**: Once input is received, the subscriber resolves the event's `input_future`, allowing the system tick to complete.
5. **Termination**: Typing "exit" or "quit" attaches a `TerminalComponent` to the agent, signaling the `Runner` to stop.
6. **Opt-in Cleanup**: `TerminalCleanupSystem(priority=1, clear_reasons=("reasoning_complete",))` removes only the successful-reasoning terminal marker so multi-turn interaction can continue without changing Runner semantics.

## Skill Installation Lifecycle

Skills are loaded from Markdown files (`SKILL.md`) using the `Skill` class and installed via `SkillManager`.
Built-in file tools are provided by `BuiltinToolsSkill` with `workspace_root` injection.

```python
manager = SkillManager()
ui_nav = Skill(skill_path=Path("path/to/SKILL.md"))
if not ui_nav.valid:
    raise ValueError(f"Skill at {ui_nav.skill_path} is invalid (malformed YAML frontmatter)")
ui_nav.resolve_path_references(workspace_root)
manager.install(world, agent_id, ui_nav)

# Install builtin file tools with workspace_root injection
builtin = BuiltinToolsSkill()
# ... wrap handlers to inject workspace_root ...
manager.install(world, agent_id, builtin)
```
Invalid skills (malformed YAML frontmatter, e.g. unclosed single-quoted strings) are rejected with `ValueError` before installation.

**Critical Timing**: Skills must be installed after the agent entity is created but *before* systems are registered. This ensures the `ToolRegistryComponent` is available for the `ReasoningSystem` during the first tick.

## Artifact Management & Security

Artifacts are managed by the `artifacts.py` module for output directory setup, and by the `BuiltinToolsSkill` (`write_file` tool) for actual file writing by the agent.

- **Output Layout**: `ensure_output_layout()` creates the `ui-design/` directory and returns a dataclass with absolute paths to `draft.md` and `nano-banana-prompts.md`.
- **Path Traversal Protection**: Every read/write operation is validated against the base output directory using `Path.resolve()` and prefix checking. Any attempt to use `../` to escape the sandbox raises a `ValueError`.
- **Deterministic Resets**: In testing, the output directory is recreated to ensure a clean state for each run.
- **Explicit File Authoring**: The `ui-prompt` skill explicitly instructs the LLM to call the builtin `write_file` tool to save the generated prompt set to `ui-design/nano-banana-prompts.md`. This prevents chat-only responses from being mistaken for successful artifact authoring.

## Testing Strategy

The implementation is verified through a tiered testing approach in `tests/integration/test_ui_design_flow.py`:

1. **Deterministic E2E (FakeProvider)**: Validates the entire system loop, system priorities, and event-driven input using a mocked LLM.
2. **Error Boundary Tests**: Ensures the orchestrator handles missing prompt files or invalid skill paths gracefully without crashing.
3. **CLI Automation Tests**: Uses `subprocess` to verify that the example can be run as a standalone script with piped input.
4. **Real-LLM Gated Tests**: Optional integration tests using `OpenAIProvider` (DashScope/OpenAI-compatible). Skipped automatically when `LLM_API_KEY` is absent. When run, installs both markdown skills and `BuiltinToolsSkill` in an isolated `tmp_path` workspace, then asserts tool execution evidence (conversation `tool` messages) and artifact mutation (`ui-design/nano-banana-prompts.md` written to disk).

## Known Limitations & Future Work

- **Slash Commands**: Documentation of `/skill-name` syntax is omitted as the feature was deferred. Interaction is purely natural language or tool-driven.
- **State Persistence**: The current implementation does not persist the `World` state to disk between runs (serialization).
- **Tool Concurrency**: Tool execution is currently sequential within each tick.
