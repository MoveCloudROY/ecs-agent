# UI Design Flow E2E Example

This example demonstrates an end-to-end interactive UI design workflow using the ECS-based LLM Agent framework. It showcases how to build a specialized design agent by composing skills, interactive input handling, and artifact management.

## Overview

The agent acts as a UI design expert that can:
1. Reason about design requirements through interactive dialogue.
2. Generate design drafts using the `/ui-navigator` skill.
3. Create specific styling prompts for components using the `/ui-prompt` skill.
4. Output structured artifacts to the `ui-design/` directory.

## Features

- **Dual-Mode Model**: Seamlessly switches between `FakeModel` (for testing/offline development) and `Model(...)` / `OpenAIModel` (for real LLM inference).
- **Interactive Input**: Real-time async stdin handling using `UserInputSystem` and `UserInputRequestedEvent`.
- **Skill-Based Capabilities**: Extends agent behavior using `MarkdownSkill` to install specialized tools.
- **Agent-Authored, Path-Safe Artifacts**: Artifacts are created lazily by agent `write_file` tool calls, with workspace-bounded path traversal protection.
- **Event-Driven Architecture**: Uses the internal `EventBus` to bridge the gap between ECS systems and interactive user input.

## Installation & Setup

1. **Install Dependencies**:
   Ensure you have `uv` installed and sync the development group.
   ```bash
   uv sync --group dev
   ```

2. **Environment Variables**:
   To use a real LLM, set the following environment variables. If `LLM_API_KEY` is not set, the example will fall back to `FakeModel`.
   - `LLM_API_KEY`: Your OpenAI-compatible API key.
   - `LLM_BASE_URL`: API base URL (defaults to DashScope).
   - `LLM_MODEL`: The model to use (defaults to `LLM_MODEL=qwen3.5-flash`).
   - `DEBUG`: Set to `1` or `true` to make this example call `configure_logging()` with debug-level output.
   - `UI_DESIGN_FLOW_INTERACTIVE`: Set to `0` to disable interactive stdin for automated CI runs (default: enabled).

## Usage

### Interactive Mode (Standard)

Run the entry point to start an interactive session. The agent will read its initial prompt from `assets/prompt.txt`.

```bash
# Using FakeModel (No API key needed)
uv run python main.py

# Using Real LLM
LLM_API_KEY=your-api-key uv run python main.py
```

### Automation Mode (Piped Input)

You can automate the interaction by piping commands into the agent. This is useful for CI or regression testing.

```bash
printf 'Design a calculator\ncontinue\nexit\n' | uv run python main.py
```

## Expected Outputs

The agent generates design artifacts in the `ui-design/` directory:

- `ui-design/draft.md`: The high-level UI design draft produced by the `/ui-navigator` skill.
- `ui-design/nano-banana-prompts.md`: Component-specific styling prompts produced by the `/ui-prompt` skill.

## Testing

The project includes several integration tests covering deterministic flows, error handling, and CLI automation.

```bash
# Run all integration tests
uv run pytest tests/integration/test_ui_design_flow.py

# Run specific tests
uv run pytest tests/integration/test_ui_design_flow.py -k "fake_provider"
uv run pytest tests/integration/test_ui_design_flow.py -k "cli_automation"

# Run real LLM integration test (requires LLM_API_KEY)
# Verifies tool execution evidence and artifact mutation (nano-banana-prompts.md written to disk)
# This example enables logging in-code when DEBUG / ECS_AGENT_LOG_LEVEL is set.
ECS_AGENT_LOG_LEVEL=DEBUG LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  LLM_MODEL=qwen3.5-flash LLM_API_KEY=your-api-key \
  uv run pytest tests/integration/test_ui_design_flow.py -k "real_llm" -v
```

### Non-Interactive / CI Mode

Run the example without interactive stdin for automated environments:

```bash
UI_DESIGN_FLOW_INTERACTIVE=0 uv run python main.py
```

## Known Limitations

- **Slash Commands**: Specialized syntax like `/skill-name` is currently not implemented in this example's runtime and is deferred to future enhancements.
