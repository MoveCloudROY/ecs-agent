# UI Design Flow E2E Example

## Overview

This example demonstrates an end-to-end interactive UI design workflow using
the ECS-based LLM Agent framework.

## Features

- **Dual-Mode Provider**: Uses `FakeProvider` for testing without API key, `OpenAIProvider` for real LLMs
- **Interactive Input**: Async user input handling via `UserInputSystem` and `UserInputRequestedEvent`
- **Skill Installation**: Loads and installs markdown skills for UI design reasoning
- **Artifact Management**: Safe output directory handling with traversal protection
- **Structured Output**: Generates design artifacts in `ui-design/` directory

## Usage

### Test Mode (no API key)

```bash
cd examples/e2e/ui-design-flow
uv run python main.py
```

### Real LLM Mode

```bash
cd examples/e2e/ui-design-flow
LLM_API_KEY=your-api-key LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 LLM_MODEL=qwen3.5-flash uv run python main.py
```

## Output

Artifacts are written to `examples/e2e/ui-design-flow/ui-design/`:

- `draft.md` — Initial UI design draft from ui-navigator skill
- `nano-banana-prompts.md` — Component styling prompts from ui-prompt skill

## Implementation Status

- [ ] Task 1: Scaffolding (in progress)
- [ ] Task 2: Skill installation
- [ ] Task 3: Interactive input
- [ ] Task 4: Output artifact formatting
- [ ] Task 5: System integration testing
