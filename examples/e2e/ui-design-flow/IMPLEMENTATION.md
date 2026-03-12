# UI Design Flow E2E — Implementation Architecture

## Overview

This document describes the architecture and implementation plan for the UI
Design Flow E2E example.

## Structure

```
examples/e2e/ui-design-flow/
├── main.py                  # Entrypoint: world setup, provider selection, agent loop
├── runtime.py               # Interactive input adapter (Task 3)
├── artifacts.py             # Path utilities and output management
├── README.md                # Usage guide
├── IMPLEMENTATION.md        # This file
└── ui-design/               # Output directory for artifacts
    ├── draft.md             # UI design from ui-navigator skill
    └── nano-banana-prompts.md # Component prompts from ui-prompt skill
```

## Design Phases (10 Tasks)

### Wave 1: Core Scaffolding & Skills (Tasks 1-5)

| Task | Focus | Status |
|------|-------|--------|
| 1 | File scaffolding (THIS TASK) | In Progress |
| 2 | Skill installation & loading | Pending |
| 3 | Interactive input + UserInputSystem | Pending |
| 4 | Artifact writing + @path rewriting | Pending |
| 5 | System integration + FakeProvider tests | Pending |

### Wave 2: Real LLM & Production (Tasks 6-10)

| Task | Focus | Status |
|------|-------|--------|
| 6 | OpenAI provider real-LLM tests | Pending |
| 7 | TDD slash expansion & @path security | Pending |
| 8 | Interactive CLI loop with DashScope | Pending |
| 9 | Artifact pipeline optimization | Pending |
| 10 | Full E2E documentation & cleanup | Pending |

## Key Patterns

### Provider Dual-Mode
See `main.py:42-63` — environment-gated provider selection

### Artifact Output
See `artifacts.py` — path utilities with traversal protection

### Async Input
See `runtime.py` — placeholder for Task 3 UserInputSystem integration

## Dependencies

- `ecs_agent` — Core ECS framework
- `ecs_agent.providers` — LLM provider protocol
- `ecs_agent.systems` — Built-in systems (Reasoning, ToolExecution, etc.)
- `ecs_agent.skills` — Markdown skill support (Task 2)

## Testing Strategy

**Task 1 (Scaffolding)**: Syntax validation via `py_compile`

**Task 5 (Integration)**: FakeProvider tests with mocked tool responses

**Task 6+**: Real-LLM tests gated by `@pytest.mark.skipif(not LLM_API_KEY)`

## Next Steps

See `.sisyphus/plans/ui-design-flow-e2e.md` for detailed task breakdown.
