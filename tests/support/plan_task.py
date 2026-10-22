"""Valid plan artifacts for controller scenarios that do not run a plan writer."""

from pathlib import Path


def write_plan(directory: Path, workflow_id: str) -> None:
    """Supply the plan-writer output required before review/finalization."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workflow_plan.md").write_text(
        f"""---
workflow_id: {workflow_id}
title: Test Workflow
description: A deterministic one-task plan
version: 1
status: finalized
created_at: "2026-10-22T00:00:00Z"
finalized_at: "2026-10-22T00:00:00Z"
---

## Tasks

### Task: task-001
```yaml
task_id: task-001
title: Confirm the plan
description: Confirm that the test plan is ready.
dependencies: []
acceptance_criteria:
  - A confirmation is recorded
execution_hints: []
```
""",
        encoding="utf-8",
    )
