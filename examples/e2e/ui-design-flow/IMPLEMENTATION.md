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
- **TerminalComponent**: Attached when the user sends an "exit" command or the agent completes its task.

### System Execution Order

Systems are registered with specific priorities to ensure correct data flow within each tick. Lower priority values execute first.

| System | Priority | Purpose |
|--------|----------|---------|
| **UserInputSystem** | -5 | Processes `UserInputComponent` and triggers `UserInputRequestedEvent`. |
| **ReasoningSystem** | 0 | Calls the LLM provider to generate the next response or tool call. |
| **ToolExecutionSystem** | 5 | Dispatches pending tool calls to their respective skill scripts. |
| **MemorySystem** | 10 | Updates conversation history and manages context window. |
| **ErrorHandlingSystem** | 99 | Captures exceptions from other systems and attaches `ErrorComponent`. |

## Interactive Input Handling

Interactive input is achieved through an event-driven adapter in `runtime.py`.

1. **Trigger**: When `UserInputSystem` detects a `UserInputComponent`, it emits a `UserInputRequestedEvent`.
2. **Subscription**: `setup_interactive_input` subscribes to this event.
3. **Async Stdin**: The subscriber uses `asyncio.run_in_executor` to call the blocking `input()` function without freezing the event loop.
4. **Resolution**: Once input is received, the subscriber resolves the event's `input_future`, allowing the system tick to complete.
5. **Termination**: Typing "exit" or "quit" attaches a `TerminalComponent` to the agent, signaling the `Runner` to stop.

## Skill Installation Lifecycle

Skills are loaded from Markdown files (`SKILL.md`) using the `MarkdownSkill` class and installed via `SkillManager`.

```python
manager = SkillManager()
ui_nav = MarkdownSkill(skill_path=Path("path/to/SKILL.md"))
manager.install(world, agent_id, ui_nav)
```

**Critical Timing**: Skills must be installed after the agent entity is created but *before* systems are registered. This ensures the `ToolRegistryComponent` is available for the `ReasoningSystem` during the first tick.

## Artifact Management & Security

Artifacts are managed by the `artifacts.py` module, which provides a safe abstraction over filesystem operations.

- **Output Layout**: `ensure_output_layout()` creates the `ui-design/` directory and returns a dataclass with absolute paths to `draft.md` and `nano-banana-prompts.md`.
- **Path Traversal Protection**: Every read/write operation is validated against the base output directory using `Path.resolve()` and prefix checking. Any attempt to use `../` to escape the sandbox raises a `ValueError`.
- **Deterministic Resets**: In testing, the output directory is recreated to ensure a clean state for each run.

## Testing Strategy

The implementation is verified through a tiered testing approach in `tests/integration/test_ui_design_flow.py`:

1. **Deterministic E2E (FakeProvider)**: Validates the entire system loop, system priorities, and event-driven input using a mocked LLM.
2. **Error Boundary Tests**: Ensures the orchestrator handles missing prompt files or invalid skill paths gracefully without crashing.
3. **CLI Automation Tests**: Uses `subprocess` to verify that the example can be run as a standalone script with piped input.
4. **Real-LLM Gated Tests**: Optional integration tests for OpenAI-compatible providers, skipped automatically if `LLM_API_KEY` is missing.

## Known Limitations & Future Work

- **Slash Commands**: Documentation of `/skill-name` syntax is omitted as the feature was deferred. Interaction is purely natural language or tool-driven.
- **State Persistence**: The current implementation does not persist the `World` state to disk between runs (serialization).
- **Tool Concurrency**: Tool execution is currently sequential within each tick.
