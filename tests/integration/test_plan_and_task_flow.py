"""Integration tests for the plan-and-task E2E example command surface."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import World
from ecs_agent.systems import TerminalCleanupSystem
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
    build_scratchbook_prompt_config,
)
from examples.e2e.plan_and_task.commands import (
    Command,
    parse_command,
)
from examples.e2e.plan_and_task.controller import PlanController
from examples.e2e.plan_and_task.plan_schema import (
    PlanTask,
    WorkflowPlan,
    parse_plan,
    validate_plan,
)
from examples.e2e.plan_and_task.runtime import slug_from_description
from examples.e2e.plan_and_task.state_models import (
    ReviewVerdict,
    RuntimeState,
    SubagentRecord,
    TaskRecord,
)


_VALID_WORKFLOW_PLAN = """---
workflow_id: demo-workflow
title: Demo Workflow Plan
description: A demonstration workflow plan.
version: 1
status: finalized
created_at: "2026-04-14T00:00:00Z"
finalized_at: "2026-04-14T01:00:00Z"
---

# Demo Workflow Plan

Some narrative description here.

## Tasks

### Task: task-001
```yaml
task_id: task-001
title: Implement feature X
description: Build the X feature with tests.
dependencies: []
acceptance_criteria:
  - uv run pytest tests/test_x.py -v passes
  - No new mypy errors
execution_hints: Follow pattern in src/x.py
```

### Task: task-002
```yaml
task_id: task-002
title: Document feature X
description: Write docs for X.
dependencies:
  - task-001
acceptance_criteria:
  - README.md updated with X section
execution_hints: null
```
"""

_VALID_FINALIZED_TASK_PLAN = """---
workflow_id: test-workflow-001
title: Test Workflow
description: A test workflow
version: 1
status: finalized
created_at: "2026-01-01T00:00:00"
finalized_at: "2026-01-01T01:00:00"
---

## Tasks

### Task: task-001
```yaml
task_id: task-001
title: First Task
description: Do the first thing
dependencies: []
acceptance_criteria:
  - First thing is done
execution_hints:
  - Use the bash tool
```

### Task: task-002
```yaml
task_id: task-002
title: Second Task
description: Do the second thing
dependencies: [task-001]
acceptance_criteria:
  - Second thing is done
execution_hints: []
```
"""


def _make_runtime_state() -> RuntimeState:
    return RuntimeState(
        workflow_id="test-workflow-001",
        phase="PLAN_FINALIZED",
        status="ready",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        tasks=[],
    )


def _make_approved_verdicts() -> list[ReviewVerdict]:
    return [
        ReviewVerdict(
            phase="PLAN_ADVISOR_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        ),
        ReviewVerdict(
            phase="PLAN_QA_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        ),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "/plan:start Build a local runtime surface",
            Command(
                name="/plan:start",
                raw="/plan:start Build a local runtime surface",
                args=["Build", "a", "local", "runtime", "surface"],
            ),
        ),
        (
            "/plan:status",
            Command(name="/plan:status", raw="/plan:status", args=[]),
        ),
        (
            "/plan:finalize",
            Command(name="/plan:finalize", raw="/plan:finalize", args=[]),
        ),
        (
            "/task:start implement parser",
            Command(
                name="/task:start",
                raw="/task:start implement parser",
                args=["implement", "parser"],
            ),
        ),
        (
            "/task:status",
            Command(name="/task:status", raw="/task:status", args=[]),
        ),
        (
            "/task:resume phase-2",
            Command(name="/task:resume", raw="/task:resume phase-2", args=["phase-2"]),
        ),
        (
            "/task:replan blocked_on_review",
            Command(
                name="/task:replan",
                raw="/task:replan blocked_on_review",
                args=["blocked_on_review"],
            ),
        ),
        (
            "/task:abort",
            Command(name="/task:abort", raw="/task:abort", args=[]),
        ),
    ],
)
def test_parse_command_accepts_closed_supported_grammar(
    text: str, expected: Command
) -> None:
    assert parse_command(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "plan:start missing slash",
        "/plan",
        "/task",
        "/task:pause",
        "/plan:startx wrong prefix",
        "/task:status extra words still supported by name? no",
    ],
)
def test_parse_command_rejects_unsupported_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_command(text)


def test_parse_command_ignores_outer_whitespace() -> None:
    command = parse_command("  /plan:status   ")

    assert command == Command(name="/plan:status", raw="/plan:status", args=[])


def test_slug_from_description_returns_empty_on_blank() -> None:
    assert slug_from_description("") == ""
    assert slug_from_description("   ") == ""


def test_slug_from_description_explicit_id() -> None:
    assert slug_from_description("explicit-workflow") == "explicit-workflow"


def test_slug_from_description_ascii_slug() -> None:
    assert slug_from_description("Build a task manager") == "build-a-task-manager"


def test_scratchbook_adapter_writes_plan_under_scratchbook_root(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="wf-001")
    adapter.write_plan("# Plan\n")

    plan_file = adapter.plan_dir / "workflow_plan.md"
    assert plan_file.exists()
    assert plan_file.read_text(encoding="utf-8") == "# Plan\n"
    assert "scratchbook" in str(plan_file)
    assert ".artifacts" not in str(plan_file)


def test_scratchbook_adapter_write_and_read_state_roundtrip(tmp_path: Path) -> None:
    import datetime
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )
    from examples.e2e.plan_and_task.state_models import RuntimeState

    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="wf-rt")
    now = datetime.datetime.utcnow().isoformat()
    state = RuntimeState(
        workflow_id="wf-rt",
        phase="PLAN_INTERVIEW",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at=now,
        updated_at=now,
    )
    adapter.write_state(state)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    restored = adapter.read_state()
    assert restored.workflow_id == "wf-rt"
    assert restored.phase == "PLAN_INTERVIEW"


def test_scratchbook_adapter_write_review_verdict_creates_file(tmp_path: Path) -> None:
    import datetime
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )
    from examples.e2e.plan_and_task.state_models import ReviewVerdict

    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="wf-rv")
    verdict = ReviewVerdict(
        phase="PLAN_ADVISOR_REVIEW",
        verdict="approved",
        decided_at=datetime.datetime.utcnow().isoformat(),
    )
    path_str = adapter.write_review_verdict("PLAN_ADVISOR_REVIEW", verdict)
    assert path_str
    review_files = list(adapter.review_dir.iterdir())
    assert len(review_files) == 1


def test_scratchbook_adapter_append_event_and_memory(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="wf-ev")
    adapter.append_event({"type": "test_event", "value": 1})
    adapter.append_event({"type": "test_event", "value": 2})
    adapter.append_memory({"key": "fact"})

    events_file = adapter.state_dir / "events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    memory_file = adapter.memory_dir / "knowledge.jsonl"
    assert memory_file.exists()


def test_scratchbook_adapter_exposes_same_path_attributes(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="wf-paths")
    assert hasattr(adapter, "workflow_root")
    assert hasattr(adapter, "plan_dir")
    assert hasattr(adapter, "state_dir")
    assert hasattr(adapter, "memory_dir")
    assert hasattr(adapter, "evidence_dir")
    assert hasattr(adapter, "review_dir")
    assert hasattr(adapter, "workflow_id")


def test_artifacts_create_canonical_workflow_layout(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")

    root = tmp_path / "scratchbook" / "test-workflow-001"

    assert adapter.workflow_root == root
    assert (root / "plan").is_dir()
    assert (root / "state").is_dir()
    assert (root / "memory").is_dir()
    assert (root / "evidence").is_dir()
    assert (root / "review").is_dir()


def test_state_schema_round_trip_and_plan_artifact(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan("first plan\n")
    adapter.write_plan("second plan\n")

    state = RuntimeState(
        workflow_id="test-workflow-001",
        phase="TASK_RUNNING",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-1",
        completed_task_ids=["task-0"],
        retry_budget={"task-1": 1},
        review_verdicts=[
            ReviewVerdict(
                phase="PLAN_QA_REVIEW",
                verdict="approved",
                decided_at="2026-04-14T00:00:00Z",
                notes="ready",
            )
        ],
        active_subagents=[
            SubagentRecord(
                session_id="subagent-1",
                status="succeeded",
                task_id="task-0",
                started_at="2026-04-14T00:01:00Z",
                completed_at="2026-04-14T00:02:00Z",
            )
        ],
        memory_refs=["memory/knowledge.jsonl#1"],
        last_checkpoint="state/checkpoint-001.json",
        created_at="2026-04-14T00:00:00Z",
        updated_at="2026-04-14T00:03:00Z",
        tasks=[
            TaskRecord(
                task_id="task-1",
                title="Implement adapter",
                status="running",
                retry_count=1,
                last_error=None,
            )
        ],
    )

    adapter.write_state(state)
    adapter.append_event({"type": "task_started", "task_id": "task-1"})
    adapter.append_memory({"fact": "retry once"})
    adapter.write_review_verdict(
        "PLAN_QA_REVIEW",
        ReviewVerdict(
            phase="PLAN_QA_REVIEW",
            verdict="approved",
            decided_at="2026-04-14T00:04:00Z",
            notes="looks good",
        ),
    )

    reloaded = adapter.read_state()
    root = tmp_path / "scratchbook" / "test-workflow-001"

    assert reloaded == state
    assert (root / "plan" / "workflow_plan.md").read_text(
        encoding="utf-8"
    ) == "second plan\n"
    assert (root / "state" / "events.jsonl").read_text(encoding="utf-8") == (
        '{"type": "task_started", "task_id": "task-1"}\n'
    )
    assert (root / "memory" / "knowledge.jsonl").read_text(encoding="utf-8") == (
        '{"fact": "retry once"}\n'
    )
    assert (root / "review" / "plan_qa_review_verdict.json").is_file()
    assert not list(root.rglob("*.tmp"))


def test_recovery_files_mark_stale_subagents_and_requeue_tasks(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan("stable plan\n")
    state = RuntimeState(
        workflow_id="test-workflow-001",
        phase="TASK_RUNNING",
        status="recovering",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-2",
        completed_task_ids=["task-1"],
        retry_budget={"task-2": 0, "task-3": 2},
        review_verdicts=[],
        active_subagents=[
            SubagentRecord(
                session_id="subagent-queued",
                status="queued",
                task_id="task-2",
                started_at=None,
                completed_at=None,
            ),
            SubagentRecord(
                session_id="subagent-running",
                status="running",
                task_id="task-3",
                started_at="2026-04-14T00:10:00Z",
                completed_at=None,
            ),
            SubagentRecord(
                session_id="subagent-done",
                status="succeeded",
                task_id="task-4",
                started_at="2026-04-14T00:11:00Z",
                completed_at="2026-04-14T00:12:00Z",
            ),
        ],
        memory_refs=[],
        last_checkpoint=None,
        created_at="2026-04-14T00:00:00Z",
        updated_at="2026-04-14T00:12:00Z",
        tasks=[
            TaskRecord(
                task_id="task-2",
                title="Recover queued subagent",
                status="running",
                retry_count=0,
                last_error=None,
            ),
            TaskRecord(
                task_id="task-3",
                title="Recover running subagent",
                status="running",
                retry_count=2,
                last_error=None,
            ),
        ],
    )

    requeued = adapter.mark_stale_subagents(state)

    assert requeued == ["task-2", "task-3"]
    assert [record.status for record in state.active_subagents] == [
        "stale",
        "stale",
        "succeeded",
    ]
    assert state.retry_budget == {"task-2": 1, "task-3": 3}
    assert [task.retry_count for task in state.tasks] == [1, 3]


def test_corrupt_state_raises_explicit_value_error(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state_path = adapter.workflow_root / "state" / "runtime_state.json"
    state_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt runtime state JSON"):
        adapter.read_state()


def test_missing_plan_reference_raises_explicit_value_error(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = RuntimeState(
        workflow_id="test-workflow-001",
        phase="PLAN_FINALIZED",
        status="idle",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at="2026-04-14T00:00:00Z",
        updated_at="2026-04-14T00:00:00Z",
        tasks=[],
    )
    adapter.write_state(state)

    with pytest.raises(ValueError, match="Runtime state references missing plan file"):
        adapter.read_state()


@pytest.mark.asyncio
async def test_plan_task_terminal_cleanup_guard() -> None:
    """Test TerminalCleanupSystem only clears 'reasoning_complete'.

    Verifies that TerminalCleanupSystem with clear_reasons=("reasoning_complete",)
    does NOT remove a TerminalComponent with reason="user_abort" or other
    non-resumable terminal reasons. This is the GUARD behavior that prevents
    accidental clearing of critical terminal states.
    """
    world = World()
    entity = world.create_entity()

    # Attach a TerminalComponent with a non-resumable reason
    world.add_component(entity, TerminalComponent(reason="user_abort"))

    # Create and process TerminalCleanupSystem that only clears reasoning_complete
    system = TerminalCleanupSystem(priority=1, clear_reasons=("reasoning_complete",))
    await system.process(world)

    # Verify the user_abort TerminalComponent still exists — NOT removed
    terminal = world.get_component(entity, TerminalComponent)
    assert terminal is not None, (
        "TerminalComponent should not be removed for reason='user_abort'"
    )
    assert terminal.reason == "user_abort", (
        "TerminalComponent reason should remain unchanged"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set",
)
async def test_plan_task_cli_automation() -> None:
    input_sequence = b"exit\n"

    result = subprocess.run(
        ["uv", "run", "python", "examples/e2e/plan_and_task/main.py"],
        input=input_sequence,
        capture_output=True,
        timeout=30,
        env={**os.environ, "PLAN_TASK_INTERACTIVE": "1"},
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )

    output = result.stdout.decode("utf-8", errors="replace")
    assert "OpenAIProvider" in output, (
        f"Expected OpenAIProvider indication in output. Got:\n{output}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set",
)
async def test_cli_slash_command_plan_start_and_status() -> None:
    """Test CLI slash command execution: /plan:start and /plan:status.

    This test pipes slash commands to main.py and verifies:
    - Exit code is 0
    - Output contains PLAN_INTERVIEW phase
    - Output contains workflow_id
    - Runtime state file exists with correct phase
    """
    input_sequence = b"/plan:start Build demo\n/plan:status\nexit\n"

    env = {
        **os.environ,
        "PLAN_TASK_INTERACTIVE": "1",
        "PLAN_TASK_WORKFLOW_ID": "cli-slash-cmd-test",
    }
    result = subprocess.run(
        ["uv", "run", "python", "examples/e2e/plan_and_task/main.py"],
        input=input_sequence,
        capture_output=True,
        timeout=30,
        env=env,
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )

    output = result.stdout.decode("utf-8", errors="replace")
    assert "PLAN_INTERVIEW" in output, (
        f"Expected 'PLAN_INTERVIEW' phase in output. Got:\n{output}"
    )
    assert "workflow_id" in output, f"Expected 'workflow_id' in output. Got:\n{output}"


def test_plan_schema_parse_plan_extracts_strict_task_blocks() -> None:
    plan = parse_plan(_VALID_WORKFLOW_PLAN)

    assert isinstance(plan, WorkflowPlan)
    assert plan.workflow_id == "demo-workflow"
    assert plan.title == "Demo Workflow Plan"
    assert plan.description == "A demonstration workflow plan."
    assert plan.status == "finalized"
    assert plan.created_at == "2026-04-14T00:00:00Z"
    assert plan.finalized_at == "2026-04-14T01:00:00Z"
    assert len(plan.tasks) == 2

    first_task = plan.tasks[0]
    second_task = plan.tasks[1]

    assert first_task.task_id == "task-001"
    assert first_task.title == "Implement feature X"
    assert first_task.description == "Build the X feature with tests."
    assert first_task.dependencies == []
    assert first_task.acceptance_criteria == [
        "uv run pytest tests/test_x.py -v passes",
        "No new mypy errors",
    ]
    assert first_task.execution_hints == ["Follow pattern in src/x.py"]

    assert second_task.task_id == "task-002"
    assert second_task.dependencies == ["task-001"]
    assert second_task.acceptance_criteria == ["README.md updated with X section"]
    assert second_task.execution_hints == []


def test_plan_finalize_validate_plan_accepts_finalized_status() -> None:
    plan = parse_plan(_VALID_WORKFLOW_PLAN)

    validate_plan(plan)


def test_plan_interview_creates_draft(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")

    state = PlanController().handle_plan_start(adapter, "Build a demo")

    assert state.phase == "PLAN_INTERVIEW"
    assert (adapter.plan_dir / "draft.md").exists()
    assert (adapter.state_dir / "runtime_state.json").exists()

    loaded_state = adapter.read_state()
    assert loaded_state.phase == "PLAN_INTERVIEW"


def test_plan_start_does_not_write_workflow_plan(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-wf")
    controller = PlanController()
    controller.handle_plan_start(adapter, "test description")
    assert not (adapter.plan_dir / "workflow_plan.md").exists()


def test_plan_finalize_rejects_without_verdicts(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_INTERVIEW"
    state.status = "active"
    state.review_verdicts = []

    with pytest.raises(ValueError, match="finalize"):
        PlanController().handle_plan_finalize(state, adapter)


def test_plan_finalize_rejects_with_only_advisor_approved(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_INTERVIEW"
    state.status = "active"
    state.review_verdicts = [
        ReviewVerdict(
            phase="PLAN_ADVISOR_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        )
    ]

    with pytest.raises(ValueError, match="PLAN_QA_REVIEW"):
        PlanController().handle_plan_finalize(state, adapter)


def test_plan_finalize_succeeds_when_both_approved(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "PLAN_INTERVIEW"
    state.status = "active"
    state.review_verdicts = _make_approved_verdicts()

    updated_state = PlanController().handle_plan_finalize(state, adapter)
    plan_path = adapter.plan_dir / "workflow_plan.md"

    assert updated_state.phase == "TASK_READY"
    assert plan_path.exists()
    parsed_plan = parse_plan(plan_path.read_text(encoding="utf-8"))
    assert parsed_plan.workflow_id == "test-workflow-001"


def test_start_without_final_plan_rejects_reviewed_status() -> None:
    plan = parse_plan(
        _VALID_WORKFLOW_PLAN.replace("status: finalized", "status: reviewed")
    )

    with pytest.raises(ValueError, match="must be finalized"):
        validate_plan(plan)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# Demo\n\n## Tasks\n", "YAML frontmatter"),
        (
            _VALID_WORKFLOW_PLAN.replace("## Tasks", "## Steps"),
            "required ## Tasks section",
        ),
        (
            _VALID_WORKFLOW_PLAN.replace("acceptance_criteria:", "notes:"),
            "acceptance_criteria",
        ),
        (
            _VALID_WORKFLOW_PLAN.replace(
                "acceptance_criteria:\n  - uv run pytest tests/test_x.py -v passes\n  - No new mypy errors",
                "acceptance_criteria: []",
            ),
            "non-empty acceptance_criteria",
        ),
        (
            _VALID_WORKFLOW_PLAN.split("### Task: task-001", maxsplit=1)[0],
            "at least one task",
        ),
    ],
)
def test_malformed_plan_rejection_raises_value_error(
    content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_plan(content)


def test_task_queue_extracted_from_finalized_plan(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)

    task_exec = TaskExec(state=state)

    plan = task_exec.load_plan(adapter)
    queue = task_exec.build_todo_queue(plan)

    assert [task.task_id for task in queue] == ["task-001", "task-002"]
    assert [task.title for task in queue] == ["First Task", "Second Task"]
    assert queue[0].description == "Do the first thing"
    assert queue[0].dependencies == []
    assert queue[0].acceptance_criteria == ["First thing is done"]
    assert queue[0].execution_hints == ["Use the bash tool"]
    assert queue[0].status == "pending"


def test_task_start_rejects_draft_plan(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(
        _VALID_FINALIZED_TASK_PLAN.replace("status: finalized", "status: draft")
    )
    adapter.write_state(state)

    with pytest.raises(ValueError, match="finalized"):
        TaskExec(state=state).load_plan(adapter)


def test_task_start_rejects_plan_without_review_approval(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)

    with pytest.raises(ValueError, match="approved"):
        TaskExec(state=state).initialize_task_queue(state, adapter)


def test_start_without_final_plan_rejects_task_start(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    state.status = "active"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)

    with pytest.raises(ValueError, match="approved"):
        TaskExec(state=state).initialize_task_queue(state, adapter)


def test_task_exec_initialize_task_queue_raises_from_wrong_phase(
    tmp_path: Path,
) -> None:
    """initialize_task_queue must reject calls from non-task phases."""
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-phase-gate")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "Phase gate test")
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_state(state)
    assert state.phase == "PLAN_INTERVIEW"
    task_exec = TaskExec(state=state)
    with pytest.raises(ValueError, match="Cannot initialize task queue from phase"):
        task_exec.initialize_task_queue(state, adapter)


def test_missing_acceptance_criteria_blocks_task_start() -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    task = PlanTask(
        task_id="task-003",
        title="Broken Task",
        description="Missing acceptance criteria",
        dependencies=[],
        acceptance_criteria=[],
        execution_hints=[],
    )

    with pytest.raises(ValueError, match="acceptance_criteria"):
        TaskExec(state=_make_runtime_state()).normalize_task(task)


def test_task_queue_order_respects_dependencies(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.review_verdicts = _make_approved_verdicts()
    dependency_reversed_plan = _VALID_FINALIZED_TASK_PLAN.replace(
        """### Task: task-001
```yaml
task_id: task-001
title: First Task
description: Do the first thing
dependencies: []
acceptance_criteria:
  - First thing is done
execution_hints:
  - Use the bash tool
```

### Task: task-002
```yaml
task_id: task-002
title: Second Task
description: Do the second thing
dependencies: [task-001]
acceptance_criteria:
  - Second thing is done
execution_hints: []
```""",
        """### Task: task-002
```yaml
task_id: task-002
title: Second Task
description: Do the second thing
dependencies: [task-001]
acceptance_criteria:
  - Second thing is done
execution_hints: []
```

### Task: task-001
```yaml
task_id: task-001
title: First Task
description: Do the first thing
dependencies: []
acceptance_criteria:
  - First thing is done
execution_hints:
  - Use the bash tool
```""",
    )
    adapter.write_plan(dependency_reversed_plan)
    adapter.write_state(state)

    queue = TaskExec(state=state).build_todo_queue(
        TaskExec(state=state).load_plan(adapter)
    )

    assert [task.task_id for task in queue] == ["task-001", "task-002"]


def test_task_queue_persisted_to_state(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)

    updated_state = TaskExec(state=state).initialize_task_queue(state, adapter)
    persisted_state = adapter.read_state()
    task_queue_path = (
        tmp_path / "scratchbook" / "test-workflow-001" / "state" / "task_queue.json"
    )

    assert [task.task_id for task in updated_state.tasks] == ["task-001", "task-002"]
    assert updated_state.current_task_id == "task-001"
    assert updated_state.phase == "TASK_RUNNING"
    assert persisted_state.tasks == updated_state.tasks
    assert persisted_state.current_task_id == "task-001"
    assert task_queue_path.is_file()


def test_context_assembly_returns_bounded_packet(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.completed_task_ids = ["task-001"]
    state.tasks = [
        TaskRecord(
            task_id="task-001",
            title="Done task",
            status="completed",
            description="already finished",
        ),
        TaskRecord(
            task_id="task-002",
            title="Current task",
            status="pending",
            description="implement the thing",
            dependencies=["task-001"],
            acceptance_criteria=["it works"],
            execution_hints=["stay bounded"],
        ),
    ]

    packet = TaskExec(state=state).assemble_execution_context(
        state, adapter, state.tasks[1]
    )

    assert packet == {
        "task_id": "task-002",
        "title": "Current task",
        "description": "implement the thing",
        "acceptance_criteria": ["it works"],
        "execution_hints": ["stay bounded"],
        "workflow_id": "test-workflow-001",
        "dependencies_completed": ["task-001"],
        "memory_entries": [],
    }


def test_context_assembly_excludes_unrelated_tasks(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.completed_task_ids = ["task-001"]
    unrelated_task = TaskRecord(
        task_id="task-999",
        title="Unrelated task",
        status="pending",
        description="should not leak into context",
        acceptance_criteria=["no leak"],
        execution_hints=["ignore me"],
    )
    current_task = TaskRecord(
        task_id="task-002",
        title="Current task",
        status="pending",
        description="bounded packet only",
        dependencies=["task-001"],
        acceptance_criteria=["bounded context"],
        execution_hints=["exclude unrelated tasks"],
    )
    state.tasks = [unrelated_task, current_task]

    packet = TaskExec(state=state).assemble_execution_context(
        state, adapter, current_task
    )

    assert "tasks" not in packet
    assert "task-999" not in json.dumps(packet)
    assert "Unrelated task" not in json.dumps(packet)


def test_context_assembly_includes_memory_entries(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    task = TaskRecord(
        task_id="task-001",
        title="Current task",
        status="pending",
        acceptance_criteria=["use memory"],
    )
    for index in range(7):
        adapter.append_memory(
            {
                "task_id": f"memory-{index}",
                "summary": f"fact {index}",
                "evidence_refs": [f"evidence/ref-{index}.json"],
                "appended_at": f"2026-01-01T00:00:0{index}",
            }
        )

    packet = TaskExec(state=state).assemble_execution_context(state, adapter, task)

    assert len(packet["memory_entries"]) == 5
    assert [entry["task_id"] for entry in packet["memory_entries"]] == [
        "memory-2",
        "memory-3",
        "memory-4",
        "memory-5",
        "memory-6",
    ]


def test_delegation_record_subagent_dispatch_persists_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.current_task_id = "task-001"
    state.tasks = [TaskRecord(task_id="task-001", title="Run task", status="pending")]
    adapter.write_plan("# draft")
    adapter.write_state(state)
    monkeypatch.setattr(
        TaskExec,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    TaskExec(state=state).record_subagent_dispatch(
        state, adapter, "task-001", "ses-001"
    )

    persisted_state = adapter.read_state()
    assert persisted_state.active_subagents == [
        SubagentRecord(
            session_id="ses-001",
            status="running",
            task_id="task-001",
            started_at="2026-01-02T03:04:05",
            completed_at=None,
        )
    ]


def test_delegation_record_subagent_dispatch_updates_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.current_task_id = "task-001"
    state.tasks = [TaskRecord(task_id="task-001", title="Run task", status="pending")]
    monkeypatch.setattr(
        TaskExec,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    updated_state = TaskExec(state=state).record_subagent_dispatch(
        state, adapter, "task-001", "ses-001"
    )

    assert updated_state.tasks[0].status == "running"
    assert updated_state.current_task_id == "task-001"
    assert updated_state.active_subagents[0].session_id == "ses-001"


def test_memory_update_record_task_completion_appends_knowledge_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.tasks = [TaskRecord(task_id="task-001", title="Run task", status="running")]
    monkeypatch.setattr(
        TaskExec,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    TaskExec(state=state).record_task_completion(
        state,
        adapter,
        "task-001",
        ["evidence/task-001.json"],
    )

    knowledge_path = adapter.memory_dir / "knowledge.jsonl"
    entries = [
        json.loads(line)
        for line in knowledge_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert entries == [
        {
            "task_id": "task-001",
            "summary": "",
            "evidence_refs": ["evidence/task-001.json"],
            "appended_at": "2026-01-02T03:04:05",
        }
    ]


def test_memory_update_record_task_completion_updates_completed_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.tasks = [
        TaskRecord(task_id="task-001", title="First", status="running"),
        TaskRecord(
            task_id="task-002",
            title="Second",
            status="pending",
            dependencies=["task-001"],
        ),
    ]
    monkeypatch.setattr(
        TaskExec,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    updated_state = TaskExec(state=state).record_task_completion(
        state,
        adapter,
        "task-001",
        ["evidence/task-001.json"],
    )

    assert updated_state.completed_task_ids == ["task-001"]
    assert updated_state.tasks[0].status == "completed"
    assert updated_state.current_task_id == "task-002"
    assert updated_state.memory_refs == ["memory/knowledge.jsonl#1"]


def test_circuit_breaker_blocks_at_max_retries() -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    state = _make_runtime_state()
    state.retry_budget = {"task-001": 3}

    assert TaskExec(state=state).check_circuit_breaker(state, "task-001") is True


def test_circuit_breaker_allows_below_max_retries() -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    state = _make_runtime_state()
    state.retry_budget = {"task-001": 2}

    assert TaskExec(state=state).check_circuit_breaker(state, "task-001") is False


def test_retry_budget_exhausted_blocks_task() -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    state = _make_runtime_state()
    state.retry_budget = {"task-001": 3}

    assert TaskExec(state=state).check_retry_budget_exhausted(state, "task-001") is True


def test_stale_subagent_on_restart_increments_retry(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.retry_budget = {"task-001": 0}
    state.tasks = [TaskRecord(task_id="task-001", title="Recover me", status="running")]
    state.active_subagents = [
        SubagentRecord(
            session_id="ses-abc123",
            status="running",
            task_id="task-001",
            started_at="2026-01-01T00:00:00",
            completed_at=None,
        )
    ]
    adapter.write_plan("# draft")
    adapter.write_state(state)

    result = WorkflowStateMachine().handle_restart(state, adapter)

    assert result.phase == "TASK_BLOCKED"
    assert result.retry_budget["task-001"] == 1
    assert result.tasks[0].retry_count == 1
    assert result.tasks[0].status == "pending"
    assert result.current_task_id == "task-001"
    assert result.active_subagents[0].status == "stale"


def test_advisor_review_creates_verdict_artifact(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    adapter.write_plan("# draft")
    adapter.write_state(state)

    PlanController().handle_advisor_review(state, adapter, "approved")

    artifact = (
        tmp_path
        / "scratchbook"
        / "test-workflow-001"
        / "review"
        / "plan_advisor_review_verdict.json"
    )
    assert artifact.is_file()


def test_advisor_review_appends_to_state_review_verdicts(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    adapter.write_plan("# draft")
    adapter.write_state(state)

    updated = PlanController().handle_advisor_review(
        state, adapter, "approved", notes="LGTM"
    )

    assert len(updated.review_verdicts) == 1
    v = updated.review_verdicts[0]
    assert v.phase == "PLAN_ADVISOR_REVIEW"
    assert v.verdict == "approved"
    assert v.notes == "LGTM"


def test_qa_review_approved_allows_finalization(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    adapter.write_plan("# draft")
    adapter.write_state(state)
    ctrl = PlanController()
    ctrl.handle_advisor_review(state, adapter, "approved")
    ctrl.handle_qa_review(state, adapter, "approved")

    result = ctrl.handle_plan_finalize(state, adapter)

    assert result.phase == "TASK_READY"
    assert result.status == "ready"


def test_qa_review_revise_blocks_finalization(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    adapter.write_plan("# draft")
    adapter.write_state(state)
    ctrl = PlanController()
    ctrl.handle_advisor_review(state, adapter, "approved")
    ctrl.handle_qa_review(state, adapter, "revise", notes="needs more detail")

    with pytest.raises(ValueError, match="PLAN_QA_REVIEW"):
        ctrl.handle_plan_finalize(state, adapter)


def test_qa_review_blocked_blocks_finalization(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    adapter.write_plan("# draft")
    adapter.write_state(state)
    ctrl = PlanController()
    ctrl.handle_advisor_review(state, adapter, "approved")
    ctrl.handle_qa_review(state, adapter, "blocked", notes="out of scope")

    with pytest.raises(ValueError, match="PLAN_QA_REVIEW"):
        ctrl.handle_plan_finalize(state, adapter)


def test_review_verdicts_persisted_in_state(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    adapter.write_plan("# draft")
    adapter.write_state(state)
    ctrl = PlanController()
    ctrl.handle_advisor_review(state, adapter, "approved")
    ctrl.handle_qa_review(state, adapter, "approved")

    persisted = adapter.read_state()

    assert len(persisted.review_verdicts) == 2
    phases = {v.phase for v in persisted.review_verdicts}
    assert "PLAN_ADVISOR_REVIEW" in phases
    assert "PLAN_QA_REVIEW" in phases


def test_replan_governance_scope_change_forces_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    monkeypatch.setattr(
        PlanController,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    updated = PlanController().handle_task_replan(
        state,
        adapter,
        reason="dependency changed",
        scope_changed=True,
    )

    assert updated.phase == "PLAN_ADVISOR_REVIEW"
    assert updated.status == "needs_review"
    assert updated.current_task_id == "task-001"
    assert updated.last_checkpoint == "dependency changed"
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    persisted = adapter.read_state()
    assert persisted.phase == "PLAN_ADVISOR_REVIEW"
    assert persisted.status == "needs_review"


def test_replan_governance_no_scope_change_stays_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    monkeypatch.setattr(
        PlanController,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    updated = PlanController().handle_task_replan(
        state,
        adapter,
        reason="missing evidence artifact",
        scope_changed=False,
    )

    assert updated.phase == "TASK_RUNNING"
    assert updated.status == "active"
    assert updated.last_checkpoint == "missing evidence artifact"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_RUNNING"


def test_abort_transitions_to_terminal_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    monkeypatch.setattr(
        PlanController,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    updated = PlanController().handle_task_abort(
        state,
        adapter,
        reason="operator requested stop",
    )

    assert updated.phase == "TASK_ABORTED"
    assert updated.status == "aborted"
    assert updated.abort_reason == "operator requested stop"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_ABORTED"
    assert persisted.abort_reason == "operator requested stop"


def test_abort_cannot_be_resumed(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_ABORTED"
    state.status = "aborted"
    state.abort_reason = "terminal stop"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)

    with pytest.raises(ValueError, match="terminal"):
        PlanController().handle_task_resume(state, adapter)


def test_task_resume_from_blocked_returns_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.status = "blocked"
    state.current_task_id = "task-001"
    state.last_checkpoint = "waiting for /task:resume"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    monkeypatch.setattr(
        PlanController,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    updated = PlanController().handle_task_resume(state, adapter)

    assert updated.phase == "TASK_RUNNING"
    assert updated.status == "active"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_RUNNING"


def test_state_machine_valid_transition_from_idle() -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    state = _make_runtime_state()
    state.phase = "IDLE"
    sm = WorkflowStateMachine()

    result = sm.transition(state, "PLAN_INTERVIEW")

    assert result.phase == "PLAN_INTERVIEW"


def test_state_machine_illegal_transition_raises() -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    state = _make_runtime_state()
    state.phase = "IDLE"
    sm = WorkflowStateMachine()

    with pytest.raises(ValueError, match="Invalid transition"):
        sm.transition(state, "TASK_RUNNING")


def test_state_machine_terminal_states_cannot_transition() -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    sm = WorkflowStateMachine()
    for terminal_phase in ("TASK_COMPLETED", "TASK_ABORTED"):
        state = _make_runtime_state()
        state.phase = terminal_phase
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition(state, "PLAN_INTERVIEW")


def test_state_machine_is_terminal_for_completed_and_aborted() -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    sm = WorkflowStateMachine()
    assert sm.is_terminal("TASK_COMPLETED") is True
    assert sm.is_terminal("TASK_ABORTED") is True
    assert sm.is_terminal("TASK_RUNNING") is False
    assert sm.is_terminal("PLAN_INTERVIEW") is False


def test_state_machine_can_resume_non_terminal() -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    sm = WorkflowStateMachine()
    assert sm.can_resume("PLAN_INTERVIEW") is True
    assert sm.can_resume("TASK_RUNNING") is True
    assert sm.can_resume("TASK_BLOCKED") is True
    assert sm.can_resume("TASK_COMPLETED") is False
    assert sm.can_resume("TASK_ABORTED") is False
    assert sm.can_resume("IDLE") is False


def test_state_machine_requires_continuation_for_active_workflows() -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    sm = WorkflowStateMachine()
    for active_phase in (
        "PLAN_INTERVIEW",
        "TASK_RUNNING",
        "TASK_BLOCKED",
        "PLAN_FINALIZED",
    ):
        state = _make_runtime_state()
        state.phase = active_phase
        assert sm.requires_continuation(state) is True, (
            f"Expected True for {active_phase}"
        )

    for done_phase in ("TASK_COMPLETED", "TASK_ABORTED", "IDLE"):
        state = _make_runtime_state()
        state.phase = done_phase
        assert sm.requires_continuation(state) is False, (
            f"Expected False for {done_phase}"
        )


def test_handle_restart_marks_stale_subagents_and_updates_phase(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.active_subagents = [
        SubagentRecord(
            session_id="ses_abc123",
            status="running",
            task_id="task-001",
            started_at="2026-01-01T00:00:00",
            completed_at=None,
        )
    ]
    adapter.write_plan("# draft")
    adapter.write_state(state)

    sm = WorkflowStateMachine()
    result = sm.handle_restart(state, adapter)

    assert result.phase == "TASK_BLOCKED"
    assert result.active_subagents[0].status == "stale"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_BLOCKED"


def test_handle_restart_no_active_subagents_keeps_phase(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.state_machine import WorkflowStateMachine

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.active_subagents = []
    adapter.write_plan("# draft")
    adapter.write_state(state)

    sm = WorkflowStateMachine()
    result = sm.handle_restart(state, adapter)

    assert result.phase == "TASK_BLOCKED"


def test_readme_command_examples_match_supported_commands() -> None:
    """Each command documented in README.md is accepted by parse_command().

    This test validates that the README's 'Supported Commands' section stays
    in sync with the actual command parser so documentation never drifts.
    """
    from examples.e2e.plan_and_task.commands import parse_command

    # The 8 commands listed under '## Supported Commands' in README.md
    documented_commands = [
        "/plan:start",
        "/plan:status",
        "/plan:finalize",
        "/task:start",
        "/task:status",
        "/task:resume",
        "/task:replan",
        "/task:abort",
    ]
    for cmd_str in documented_commands:
        result = parse_command(cmd_str)
        assert result is not None, (
            f"README-documented command {cmd_str!r} was rejected by parse_command(). "
            "Update README or commands.py to stay in sync."
        )
        assert result.name == cmd_str.split()[0], (
            f"Unexpected parse result for {cmd_str!r}: got name={result.name!r}"
        )


def test_plan_start_state_has_open_questions_and_confirmed_requirements(
    tmp_path: Path,
) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    controller = PlanController()

    state = controller.handle_plan_start(adapter, "test description")

    assert state.open_questions == []
    assert state.confirmed_requirements == []
    persisted_payload = json.loads(
        (adapter.state_dir / "runtime_state.json").read_text(encoding="utf-8")
    )
    assert persisted_payload["open_questions"] == []
    assert persisted_payload["confirmed_requirements"] == []


def test_review_verdict_artifact_has_phase_and_verdict(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "test description")
    controller.handle_qa_review(state, adapter, "approved")

    controller.handle_advisor_review(state, adapter, "approved")

    verdict_payload = json.loads(
        (adapter.review_dir / "plan_advisor_review_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict_payload["phase"] == "PLAN_ADVISOR_REVIEW"
    assert verdict_payload["verdict"] == "approved"


def test_task_completion_writes_evidence_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.tasks = [TaskRecord(task_id="task-001", title="First Task", status="running")]
    monkeypatch.setattr(
        TaskExec,
        "_utcnow_isoformat",
        lambda self: "2026-01-02T03:04:05",
    )

    TaskExec(state=state).record_task_completion(
        state,
        adapter,
        "task-001",
        evidence_refs=["ref-1"],
        summary="done",
    )

    evidence_payload = json.loads(
        (adapter.evidence_dir / "task-task-001-result.json").read_text(encoding="utf-8")
    )
    assert evidence_payload["task_id"] == "task-001"
    assert evidence_payload["summary"] == "done"
    assert evidence_payload["evidence_refs"] == ["ref-1"]
    assert "completed_at" in evidence_payload


def test_main_world_registers_trigger_script_handlers(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, _ = _build_test_world(tmp_path)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None

    trigger_patterns = {t.pattern for t in config.triggers}
    expected_patterns = {
        "/plan:start",
        "/plan:status",
        "/plan:finalize",
        "/task:start",
        "/task:status",
        "/task:resume",
        "/task:replan",
        "/task:abort",
    }
    assert expected_patterns.issubset(trigger_patterns)

    for trigger in config.triggers:
        if trigger.pattern in expected_patterns:
            assert trigger.action == "script"
            assert trigger.content in config.script_handlers


@pytest.mark.asyncio
async def test_trigger_plan_start_handler_creates_state(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)
    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None

    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:start")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, "/plan:start Build demo")

    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "PLAN_INTERVIEW"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_trigger_plan_status_handler_returns_status_string(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = _build_test_world(tmp_path)
    controller = PlanController()
    runtime_state[0] = controller.handle_plan_start(adapter, "Test plan")

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None

    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:status"
    )
    handler = config.script_handlers[handler_key]
    result = await handler(world, agent_id, "/plan:status")

    assert result is not None
    assert "PLAN_INTERVIEW" in result


def test_runtime_setup_does_not_intercept_slash_commands(tmp_path: Path) -> None:
    import inspect

    from examples.e2e.plan_and_task.runtime import setup_interactive_input

    sig = inspect.signature(setup_interactive_input)
    assert "command_handler" not in sig.parameters


def _build_test_world(
    tmp_path: Path,
) -> tuple[World, object, ArtifactAdapter, list[RuntimeState | None]]:
    from ecs_agent.providers import FakeProvider
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="ready"))]
    )

    world, agent_id, adapter_ref, runtime_state = build_plan_task_world(
        provider=provider,
        model="fake-plan-task",
        base_dir=tmp_path,
    )
    test_adapter = PlanTaskScratchbookAdapter(
        base_dir=tmp_path, workflow_id="test-workflow-001"
    )
    adapter_ref[0] = test_adapter
    return world, agent_id, test_adapter, runtime_state


@pytest.mark.asyncio
async def test_main_world_setup_installs_subagent_infrastructure(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import (
        SubagentRegistryComponent,
        SubagentSessionTableComponent,
        ToolRegistryComponent,
    )
    from ecs_agent.systems.subagent import SubagentSystem

    world, agent_id, _, _ = _build_test_world(tmp_path)
    world.apply_pending_system_operations()

    registry = world.get_component(agent_id, SubagentRegistryComponent)
    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    session_table = world.get_component(agent_id, SubagentSessionTableComponent)

    assert registry is not None
    assert set(registry.subagents) >= {"advisor", "qa"}
    assert tool_registry is not None
    assert "record_advisor_verdict" not in tool_registry.tools
    assert "record_advisor_verdict" not in tool_registry.handlers
    assert "record_qa_verdict" not in tool_registry.tools
    assert "record_qa_verdict" not in tool_registry.handlers
    assert session_table is not None
    assert any(
        isinstance(entry.system, SubagentSystem) and entry.priority == -1
        for entry in world._systems._systems
    )


# ── DelegationCompletedEvent verdict tests ─────────────────────────────────────


def test_main_world_does_not_register_record_verdict_tools(tmp_path: Path) -> None:
    """record_advisor_verdict and record_qa_verdict tools must NOT be registered."""
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, _, _ = _build_test_world(tmp_path)
    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    assert tool_registry is not None
    assert "record_advisor_verdict" not in tool_registry.tools
    assert "record_qa_verdict" not in tool_registry.tools
    assert "record_advisor_verdict" not in tool_registry.handlers
    assert "record_qa_verdict" not in tool_registry.handlers


def test_extract_verdict_parses_all_three_values() -> None:
    from examples.e2e.plan_and_task.main import _extract_verdict_from_result

    assert _extract_verdict_from_result("The plan looks good. APPROVED.") == "approved"
    assert _extract_verdict_from_result("I recommend revise the scope.") == "revise"
    assert (
        _extract_verdict_from_result("This is BLOCKED by missing requirements.")
        == "blocked"
    )
    assert (
        _extract_verdict_from_result("Looks great overall, approved by reviewer.")
        == "approved"
    )


def test_extract_verdict_defaults_to_revise_on_no_match() -> None:
    from examples.e2e.plan_and_task.main import _extract_verdict_from_result

    result = _extract_verdict_from_result("The analysis is complete.")
    assert result == "revise"


@pytest.mark.asyncio
async def test_delegation_completed_event_records_advisor_verdict(
    tmp_path: Path,
) -> None:
    """Publishing DelegationCompletedEvent for 'advisor' updates runtime state."""
    from ecs_agent.types import DelegationCompletedEvent
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = _build_test_world(tmp_path)
    controller = PlanController()
    runtime_state[0] = controller.handle_plan_start(adapter, "Test workflow")

    await world.event_bus.publish(
        DelegationCompletedEvent(
            entity_id=agent_id,
            subagent_name="advisor",
            result="The plan looks solid. Approved.",
            success=True,
        )
    )

    assert runtime_state[0] is not None
    verdicts = runtime_state[0].review_verdicts
    assert len(verdicts) == 1
    assert verdicts[0].phase == "PLAN_ADVISOR_REVIEW"
    assert verdicts[0].verdict == "approved"


@pytest.mark.asyncio
async def test_delegation_completed_event_records_qa_verdict(tmp_path: Path) -> None:
    """Publishing DelegationCompletedEvent for 'qa' updates runtime state."""
    from ecs_agent.types import DelegationCompletedEvent
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = _build_test_world(tmp_path)
    controller = PlanController()
    runtime_state[0] = controller.handle_plan_start(adapter, "Test workflow")

    await world.event_bus.publish(
        DelegationCompletedEvent(
            entity_id=agent_id,
            subagent_name="qa",
            result="QA review complete. Revise the acceptance criteria.",
            success=True,
        )
    )

    assert runtime_state[0] is not None
    verdicts = runtime_state[0].review_verdicts
    assert len(verdicts) == 1
    assert verdicts[0].phase == "PLAN_QA_REVIEW"
    assert verdicts[0].verdict == "revise"


@pytest.mark.asyncio
async def test_delegation_completed_event_ignores_other_entity(tmp_path: Path) -> None:
    """Events for a different entity_id must not update state."""
    from ecs_agent.types import DelegationCompletedEvent, EntityId
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = _build_test_world(tmp_path)
    controller = PlanController()
    runtime_state[0] = controller.handle_plan_start(adapter, "Test workflow")

    other_entity = EntityId(agent_id + 999)
    await world.event_bus.publish(
        DelegationCompletedEvent(
            entity_id=other_entity,
            subagent_name="advisor",
            result="approved",
            success=True,
        )
    )

    assert runtime_state[0] is not None
    assert len(runtime_state[0].review_verdicts) == 0


def test_prompt_builders_return_non_empty_strings() -> None:
    from examples.e2e.plan_and_task.prompts import (
        build_advisor_prompt,
        build_draft_prompt,
        build_qa_prompt,
    )

    assert build_advisor_prompt("some draft").strip()
    assert build_qa_prompt("some draft", "approved").strip()
    assert build_draft_prompt("a description", []).strip()


# ── ScratchbookPromptConfig tests ──────────────────────────────────────────────


def test_build_scratchbook_prompt_config_returns_valid_config() -> None:
    from ecs_agent.scratchbook.prompt_definition import ScratchbookPromptConfig

    config = build_scratchbook_prompt_config("wf-001")
    assert isinstance(config, ScratchbookPromptConfig)
    assert config.scratchbook_root_path == "scratchbook/wf-001"
    assert len(config.artifacts) == 6
    artifact_ids = {a.artifact_type_id for a in config.artifacts}
    assert "draft_plan" in artifact_ids
    assert "workflow_plan" in artifact_ids
    assert "runtime_state" in artifact_ids
    assert "events_journal" in artifact_ids
    assert "knowledge_memory" in artifact_ids
    assert "review_verdict" in artifact_ids


def test_build_scratchbook_prompt_config_artifacts_have_valid_paths() -> None:
    config = build_scratchbook_prompt_config("my-workflow")
    for artifact in config.artifacts:
        assert artifact.path.startswith("scratchbook/my-workflow/")
        assert not artifact.path.startswith("/")


def test_main_world_does_not_add_scratchbook_prompt_config_at_init(
    tmp_path: Path,
) -> None:
    """ScratchbookPromptConfig is NOT added at world init.

    It is added lazily inside _handle_plan_start after the workflow_id is
    derived from the user's task description.
    """
    from ecs_agent.scratchbook.prompt_definition import ScratchbookPromptConfig

    world, agent_id, _, _ = _build_test_world(tmp_path)
    config = world.get_component(agent_id, ScratchbookPromptConfig)
    assert config is None, (
        "ScratchbookPromptConfig must NOT be present at world init; "
        "it is added lazily when /plan:start is called"
    )


def test_build_scratchbook_prompt_config_includes_draft_md() -> None:
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        build_scratchbook_prompt_config,
    )

    config = build_scratchbook_prompt_config("wf-test")
    artifact_ids = {a.artifact_type_id for a in config.artifacts}
    assert "draft_plan" in artifact_ids

    draft = next(a for a in config.artifacts if a.artifact_type_id == "draft_plan")
    assert draft.path == "scratchbook/wf-test/plan/draft.md"
    assert draft.readonly is False


def test_main_world_installs_builtin_tools(tmp_path: Path) -> None:
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, _, _ = _build_test_world(tmp_path)
    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    assert tool_registry is not None
    for expected_tool in ("read_file", "write_file", "edit_file", "bash", "glob"):
        assert expected_tool in tool_registry.tools, f"missing tool: {expected_tool}"
        assert expected_tool in tool_registry.handlers, (
            f"missing handler: {expected_tool}"
        )


# ---------------------------------------------------------------------------
# Issue fix tests: progressive draft editing and edit_file guidance
# ---------------------------------------------------------------------------


def test_plan_interview_system_prompt_instructs_edit_file_usage() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "edit_file" in prompt_lower, (
        "System prompt must instruct LLM to use edit_file for plan updates"
    )
    assert (
        "write_file" not in prompt_lower
        or "do not" in prompt_lower
        or "never" in prompt_lower
        or "avoid" in prompt_lower
    ), "System prompt should not encourage write_file for plan updates"


def test_plan_interview_system_prompt_instructs_read_before_edit() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "read_file" in prompt_lower or "read the current" in prompt_lower, (
        "System prompt must instruct LLM to read draft.md before editing"
    )


def test_plan_interview_system_prompt_defines_interview_flow() -> None:
    """The system prompt must define a structured interview → section-update flow.

    Verifies fix for: draft not being guided as progressive editing.
    """
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    # Must mention asking one question at a time
    assert (
        "one question" in PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
        or "per turn" in PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
        or "single question" in PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    ), "System prompt must guide LLM to ask one question at a time"
    # Must reference the draft.md file specifically
    assert "draft.md" in PLAN_INTERVIEW_SYSTEM_PROMPT, (
        "System prompt must reference draft.md as the file to edit progressively"
    )


def test_draft_template_has_structured_sections(tmp_path: Path) -> None:
    """The initial draft template must have structured fillable sections.

    Verifies fix for: draft has no clear sections to progressively fill in.
    """
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-draft-sections")
    controller = PlanController()

    state = controller.handle_plan_start(adapter, "Test description")
    draft_content = (adapter.plan_dir / "draft.md").read_text(encoding="utf-8")

    # Must have dedicated sections that the LLM can fill in via edit_file
    required_sections = [
        "## Open Questions",
        "## Confirmed Requirements",
        "## Scope",
    ]
    for section in required_sections:
        assert section in draft_content, f"Draft template missing section: {section!r}"
    _ = state  # used


def test_draft_template_has_placeholder_content(tmp_path: Path) -> None:
    """Draft sections should have placeholder content (not empty) so edit_file can target them."""
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-draft-placeholders")
    controller = PlanController()

    controller.handle_plan_start(adapter, "Some description")
    draft_content = (adapter.plan_dir / "draft.md").read_text(encoding="utf-8")

    # Each section should have placeholder text the LLM can replace
    assert (
        "(to be filled" in draft_content.lower()
        or "tbd" in draft_content.lower()
        or "<!-- " in draft_content
    ), "Draft sections should have placeholder text for edit_file targets"


def test_slug_from_description_english() -> None:
    from examples.e2e.plan_and_task.runtime import slug_from_description

    slug = slug_from_description("Build a writing assistant software")
    assert slug
    assert slug == slug.lower() or any("\u4e00" <= c <= "\u9fff" for c in slug)
    assert len(slug) <= 60
    assert " " not in slug


def test_slug_from_description_chinese() -> None:
    from examples.e2e.plan_and_task.runtime import slug_from_description

    slug = slug_from_description("开发辅助写作软件，支持长篇小说")
    assert slug
    assert len(slug) <= 60
    assert "\n" not in slug


def test_slug_from_description_empty_falls_back() -> None:
    from examples.e2e.plan_and_task.runtime import slug_from_description

    assert slug_from_description("") == ""
    assert slug_from_description("   ") == ""


def test_slug_from_description_special_chars_stripped() -> None:
    from examples.e2e.plan_and_task.runtime import slug_from_description

    slug = slug_from_description("Hello! @World? #Test...")
    assert slug
    assert "!" not in slug
    assert "@" not in slug
    assert "?" not in slug
    assert "#" not in slug


def test_plan_start_handler_sets_workflow_id_from_description(
    tmp_path: Path,
) -> None:
    from ecs_agent.providers.fake_provider import FakeProvider
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    provider = FakeProvider(responses=["ok"])
    world, agent_id, adapter_ref, runtime_state = build_plan_task_world(
        provider=provider,
        model="fake",
        base_dir=tmp_path,
    )

    controller = PlanController()
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="initial-id")
    state = controller.handle_plan_start(adapter, "Build a task management app")
    assert state.workflow_id == "initial-id"
    assert (adapter.plan_dir / "draft.md").exists()
    draft = (adapter.plan_dir / "draft.md").read_text(encoding="utf-8")
    assert "Build a task management app" in draft


def test_edit_file_schema_exposes_direct_params() -> None:
    from ecs_agent.tools.builtins.edit_tool import edit_file

    schema = edit_file._tool_schema  # type: ignore[attr-defined]
    props = schema.parameters.get("properties", {})
    assert "file_path" in props
    assert "op" in props
    assert "pos" in props
    assert "content" in props
    assert "edits_json" not in props
    required = schema.parameters.get("required", [])
    assert "file_path" in required
    assert "op" in required
    assert "pos" in required


async def test_edit_file_direct_params_replaces_content(tmp_path: Path) -> None:
    from ecs_agent.tools.builtins.edit_tool import edit_file, format_file_with_hashes

    target = tmp_path / "draft.md"
    target.write_text("## Scope\n(to be filled)\n\n## Next\n", encoding="utf-8")
    rendered = format_file_with_hashes(target.read_text(encoding="utf-8"))
    line2_hash = rendered.splitlines()[1].split("|")[0].split("#")[1]
    result = await edit_file._tool_handler(  # type: ignore[attr-defined]
        file_path="draft.md",
        workspace_root=str(tmp_path),
        op="replace",
        pos=f"2#{line2_hash}",
        content="In scope: everything",
    )
    assert "draft.md" in result
    assert "(to be filled)" not in target.read_text(encoding="utf-8")
    assert "In scope: everything" in target.read_text(encoding="utf-8")


async def test_edit_file_raises_when_old_str_not_found(tmp_path: Path) -> None:
    import pytest
    from ecs_agent.tools.builtins.edit_tool import edit_file

    (tmp_path / "file.md").write_text("hello world", encoding="utf-8")
    with pytest.raises(Exception):
        await edit_file._tool_handler(  # type: ignore[attr-defined]
            file_path="file.md",
            workspace_root=str(tmp_path),
            edits_json='[{"op": "replace", "pos": "99#aaaa", "lines": ["x"]}]',
        )


async def test_edit_file_raises_on_invalid_edits_json(tmp_path: Path) -> None:
    import pytest
    from ecs_agent.tools.builtins.edit_tool import edit_file

    (tmp_path / "file.md").write_text("foo\nfoo\n", encoding="utf-8")
    with pytest.raises(Exception):
        await edit_file._tool_handler(  # type: ignore[attr-defined]
            file_path="file.md",
            workspace_root=str(tmp_path),
            edits_json="not-valid-json",
        )


async def test_derive_workflow_id_uses_llm_slug() -> None:
    from ecs_agent.providers.fake_provider import FakeProvider
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import derive_workflow_id_from_llm

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="writing-assistant-tool")
            )
        ]
    )
    result = await derive_workflow_id_from_llm("辅助写作软件", provider)
    assert result == "writing-assistant-tool"


async def test_derive_workflow_id_normalizes_llm_output() -> None:
    from ecs_agent.providers.fake_provider import FakeProvider
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import derive_workflow_id_from_llm

    provider = FakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Writing Assistant Tool!")
            )
        ]
    )
    result = await derive_workflow_id_from_llm("Writing assistant", provider)
    assert result == "writing-assistant-tool"


async def test_derive_workflow_id_falls_back_on_empty_response() -> None:
    from ecs_agent.providers.fake_provider import FakeProvider
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import (
        derive_workflow_id_from_llm,
        slug_from_description,
    )

    provider = FakeProvider(
        responses=[CompletionResult(message=Message(role="assistant", content="   "))]
    )
    result = await derive_workflow_id_from_llm("build a task manager", provider)
    assert result == slug_from_description("build a task manager")
    assert result != ""


async def test_derive_workflow_id_falls_back_on_provider_error() -> None:
    from ecs_agent.providers.fake_provider import FakeProvider
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import (
        derive_workflow_id_from_llm,
        slug_from_description,
    )

    provider = FakeProvider(responses=[])
    result = await derive_workflow_id_from_llm("build a task manager", provider)
    assert result == slug_from_description("build a task manager")


def test_plan_interview_system_prompt_contains_revise_instruction() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "revise" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must mention the 'revise' verdict"
    )
    assert "edit_file" in PLAN_INTERVIEW_SYSTEM_PROMPT, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must instruct to call edit_file after revise"
    )
    assert "advisor" in prompt_lower, "Prompt must mention calling advisor again"


def test_plan_interview_system_prompt_contains_blocked_instruction() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    assert "blocked" in PLAN_INTERVIEW_SYSTEM_PROMPT.lower(), (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must mention the 'blocked' verdict"
    )


def test_plan_interview_system_prompt_gates_qa_on_advisor_approval() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "approved" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must gate QA call on advisor 'approved' verdict"
    )
    assert "do not" in prompt_lower or "only" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must make the QA gating condition explicit"
    )


def test_controller_advisor_revise_state_stays_in_advisor_review(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "revise test workflow")
    state = controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs more detail"
    )

    assert state.phase == "PLAN_ADVISOR_REVIEW", (
        f"Expected PLAN_ADVISOR_REVIEW after revise, got {state.phase}"
    )
    assert state.phase != "PLAN_QA_REVIEW"


def test_controller_advisor_revise_followed_by_approved_allows_qa(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-then-approve-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "revise then approve workflow")

    state = controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs scope"
    )
    assert state.phase == "PLAN_ADVISOR_REVIEW"

    state = controller.handle_advisor_review(state, adapter, "approved", notes="LGTM")
    assert state.phase == "PLAN_ADVISOR_REVIEW"

    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "PLAN_ADVISOR_REVIEW" not in missing, (
        f"Expected PLAN_ADVISOR_REVIEW to be approved in missing list: {missing}"
    )


def test_controller_advisor_multiple_verdicts_all_recorded(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="multi-verdict-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "multi-verdict workflow")

    state = controller.handle_advisor_review(state, adapter, "revise")
    state = controller.handle_advisor_review(state, adapter, "blocked")
    state = controller.handle_advisor_review(state, adapter, "approved")

    advisor_verdicts = [
        v for v in state.review_verdicts if v.phase == "PLAN_ADVISOR_REVIEW"
    ]
    assert len(advisor_verdicts) == 3, (
        f"Expected 3 advisor verdicts recorded, got {len(advisor_verdicts)}"
    )
    assert advisor_verdicts[0].verdict == "revise"
    assert advisor_verdicts[1].verdict == "blocked"
    assert advisor_verdicts[2].verdict == "approved"


def test_controller_missing_approved_reviews_uses_last_verdict(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="last-verdict-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "last verdict test workflow")

    state = controller.handle_advisor_review(state, adapter, "revise")
    state = controller.handle_advisor_review(state, adapter, "approved")

    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "PLAN_ADVISOR_REVIEW" not in missing, (
        "After revise→approved, PLAN_ADVISOR_REVIEW should be satisfied"
    )
    # Must instruct to update/edit draft in response
    assert "edit_file" in PLAN_INTERVIEW_SYSTEM_PROMPT, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must instruct to call edit_file after revise"
    )
    # Must instruct to re-call the advisor after updating
    assert "advisor" in prompt_lower, "Prompt must mention calling advisor again"


def test_plan_interview_system_prompt_contains_blocked_instruction() -> None:
    """System prompt must instruct the LLM what to do when advisor returns 'blocked'."""
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "blocked" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must mention the 'blocked' verdict"
    )


def test_plan_interview_system_prompt_gates_qa_on_advisor_approval() -> None:
    """System prompt must say QA is only called after advisor *approved*, not before."""
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    # The word "approved" must appear — the gating logic
    assert "approved" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must gate QA call on advisor 'approved' verdict"
    )
    # Must explicitly mention NOT calling QA before approval
    assert "do not" in prompt_lower or "only" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must make the QA gating condition explicit"
    )


def test_controller_advisor_revise_state_stays_in_advisor_review(
    tmp_path: Path,
) -> None:
    """After a 'revise' verdict, phase must remain PLAN_ADVISOR_REVIEW (not advance to QA)."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "revise test workflow")

    # First advisor call → state transitions to PLAN_ADVISOR_REVIEW
    state = controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs more detail"
    )

    assert state.phase == "PLAN_ADVISOR_REVIEW", (
        f"Expected PLAN_ADVISOR_REVIEW after revise, got {state.phase}"
    )
    # Phase must NOT advance to QA
    assert state.phase != "PLAN_QA_REVIEW"


def test_controller_advisor_revise_followed_by_approved_allows_qa(
    tmp_path: Path,
) -> None:
    """After revise then approved, the advisor verdict is approved and QA can be called."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-then-approve-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "revise then approve workflow")

    # Round 1: revise
    state = controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs scope"
    )
    assert state.phase == "PLAN_ADVISOR_REVIEW"

    # Round 2: approved (LLM revised draft and re-called advisor)
    state = controller.handle_advisor_review(state, adapter, "approved", notes="LGTM")
    assert state.phase == "PLAN_ADVISOR_REVIEW"

    # Now the latest advisor verdict is "approved" — _missing_approved_reviews should pass advisor
    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "PLAN_ADVISOR_REVIEW" not in missing, (
        f"Expected PLAN_ADVISOR_REVIEW to be approved in missing list: {missing}"
    )


def test_controller_advisor_multiple_verdicts_all_recorded(
    tmp_path: Path,
) -> None:
    """All advisor verdicts (revise, blocked, approved) must be appended, not overwritten."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="multi-verdict-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "multi-verdict workflow")

    state = controller.handle_advisor_review(state, adapter, "revise")
    state = controller.handle_advisor_review(state, adapter, "blocked")
    state = controller.handle_advisor_review(state, adapter, "approved")

    advisor_verdicts = [
        v for v in state.review_verdicts if v.phase == "PLAN_ADVISOR_REVIEW"
    ]
    assert len(advisor_verdicts) == 3, (
        f"Expected 3 advisor verdicts recorded, got {len(advisor_verdicts)}"
    )
    assert advisor_verdicts[0].verdict == "revise"
    assert advisor_verdicts[1].verdict == "blocked"
    assert advisor_verdicts[2].verdict == "approved"


def test_controller_missing_approved_reviews_uses_last_verdict(
    tmp_path: Path,
) -> None:
    """_missing_approved_reviews must check the LAST verdict per phase, not the first."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="last-verdict-test")
    controller = PlanController()
    state = controller.handle_plan_start(adapter, "last verdict test workflow")

    # revise followed by approved — only last (approved) should count
    state = controller.handle_advisor_review(state, adapter, "revise")
    state = controller.handle_advisor_review(state, adapter, "approved")

    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "PLAN_ADVISOR_REVIEW" not in missing, (
        "After revise→approved, PLAN_ADVISOR_REVIEW should be satisfied"
    )


# ── /plan:resume command tests ─────────────────────────────────────────────────


def test_parse_command_accepts_plan_resume_with_workflow_id() -> None:
    from examples.e2e.plan_and_task.commands import Command, parse_command

    cmd = parse_command("/plan:resume my-workflow-id")
    assert cmd == Command(
        name="/plan:resume",
        raw="/plan:resume my-workflow-id",
        args=["my-workflow-id"],
    )


def test_parse_command_plan_resume_without_args_parses_cleanly() -> None:
    from examples.e2e.plan_and_task.commands import parse_command

    # /plan:resume is in _COMMANDS_WITH_ARGS — missing args is still parseable;
    # the handler is responsible for returning the "missing arg" error.
    cmd = parse_command("/plan:resume")
    assert cmd.name == "/plan:resume"
    assert cmd.args == []


def test_main_world_registers_plan_resume_trigger(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, _ = _build_test_world(tmp_path)
    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None

    trigger_patterns = {t.pattern for t in config.triggers}
    assert "/plan:resume" in trigger_patterns

    resume_trigger = next(t for t in config.triggers if t.pattern == "/plan:resume")
    assert resume_trigger.action == "script"
    assert resume_trigger.match_mode == "prefix"
    assert resume_trigger.content in config.script_handlers


@pytest.mark.asyncio
async def test_plan_resume_handler_restores_state_from_disk(tmp_path: Path) -> None:
    """Handler must load persisted RuntimeState and set adapter_ref[0] / runtime_state[0]."""
    import datetime

    from ecs_agent.components import UserPromptConfigComponent
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)

    workflow_id = "resume-test-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    now = datetime.datetime.utcnow().isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="TASK_BLOCKED",
        status="blocked",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-001",
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at=now,
        updated_at=now,
    )
    # The adapter requires the active_plan_file to exist for task-execution phases
    (adapter.plan_dir / "workflow_plan.md").write_text("# Plan\n", encoding="utf-8")
    adapter.write_state(persisted)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, f"/plan:resume {workflow_id}")

    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == workflow_id
    assert runtime_state[0].phase == "TASK_BLOCKED"
    assert isinstance(result, str)
    assert workflow_id in result


@pytest.mark.asyncio
async def test_plan_resume_handler_missing_workflow_id_returns_error(
    tmp_path: Path,
) -> None:
    """Handler must return an error string when no workflow_id argument is given."""
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, "/plan:resume")

    assert result is not None
    assert "Error" in result
    assert runtime_state[0] is None  # must NOT have mutated state


@pytest.mark.asyncio
async def test_plan_resume_handler_unknown_workflow_id_returns_error(
    tmp_path: Path,
) -> None:
    """Handler must return an error string when the scratchbook directory does not exist."""
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, "/plan:resume nonexistent-workflow-xyz")

    assert result is not None
    assert "Error" in result
    assert runtime_state[0] is None  # must NOT have mutated state


@pytest.mark.asyncio
async def test_plan_resume_handler_marks_stale_subagents(tmp_path: Path) -> None:
    """In-flight subagents in the persisted state must be marked stale on resume."""
    import datetime

    from ecs_agent.components import UserPromptConfigComponent
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )
    from examples.e2e.plan_and_task.state_models import SubagentRecord

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)

    workflow_id = "stale-subagent-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    now = datetime.datetime.utcnow().isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="TASK_RUNNING",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-001",
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[
            SubagentRecord(
                session_id="ses_stale123",
                task_id="task-001",
                status="running",
                started_at=now,
                completed_at=None,
            )
        ],
        memory_refs=[],
        last_checkpoint=None,
        created_at=now,
        updated_at=now,
    )
    (adapter.plan_dir / "workflow_plan.md").write_text("# Plan\n", encoding="utf-8")
    adapter.write_state(persisted)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]

    await handler(world, agent_id, f"/plan:resume {workflow_id}")

    # After resume the active_subagents should all be marked stale
    assert runtime_state[0] is not None
    assert all(
        s.status == "stale" for s in runtime_state[0].active_subagents
    ), "All previously active subagents must be stale after resume"


@pytest.mark.asyncio
async def test_plan_resume_handler_updates_scratchbook_prompt_config(
    tmp_path: Path,
) -> None:
    """The ECS ScratchbookPromptConfig component must be updated to the resumed workflow_id."""
    import datetime

    from ecs_agent.components import UserPromptConfigComponent
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
        ScratchbookPromptConfig,
    )

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)

    workflow_id = "scratchbook-config-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    now = datetime.datetime.utcnow().isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="PLAN_INTERVIEW",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at=now,
        updated_at=now,
    )
    adapter.write_state(persisted)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]

    await handler(world, agent_id, f"/plan:resume {workflow_id}")

    scratchbook_config = world.get_component(agent_id, ScratchbookPromptConfig)
    assert scratchbook_config is not None
    assert workflow_id in scratchbook_config.scratchbook_root_path


# ---------------------------------------------------------------------------
# Planning-phase file validation bug fix tests
# ---------------------------------------------------------------------------


def test_read_state_planning_phase_does_not_require_workflow_plan(
    tmp_path: Path,
) -> None:
    """read_state() must succeed for planning phases even when workflow_plan.md is absent."""
    import datetime

    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    for phase in ("PLAN_INTERVIEW", "PLAN_ADVISOR_REVIEW", "PLAN_QA_REVIEW"):
        workflow_id = f"planning-phase-{phase.lower().replace('_', '-')}"
        adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
        # Write draft.md but NOT workflow_plan.md
        adapter.plan_dir.mkdir(parents=True, exist_ok=True)
        (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")

        now = datetime.datetime.utcnow().isoformat()
        state = RuntimeState(
            workflow_id=workflow_id,
            phase=phase,
            status="active",
            active_plan_file="plan/workflow_plan.md",
            current_task_id=None,
            completed_task_ids=[],
            retry_budget={},
            review_verdicts=[],
            active_subagents=[],
            memory_refs=[],
            last_checkpoint=None,
            created_at=now,
            updated_at=now,
        )
        adapter.write_state(state)

        # Must NOT raise even though workflow_plan.md is missing
        loaded = adapter.read_state()
        assert loaded.phase == phase, f"Expected phase {phase}, got {loaded.phase}"


def test_read_state_task_execution_phase_requires_active_plan_file(
    tmp_path: Path,
) -> None:
    """read_state() must raise ValueError in task execution phases when workflow_plan.md is absent."""
    import datetime

    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    workflow_id = "task-phase-missing-plan"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    # Do NOT write workflow_plan.md

    now = datetime.datetime.utcnow().isoformat()
    state = RuntimeState(
        workflow_id=workflow_id,
        phase="TASK_BLOCKED",
        status="blocked",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-001",
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at=now,
        updated_at=now,
    )
    adapter.write_state(state)

    with pytest.raises(ValueError, match="missing plan file"):
        adapter.read_state()


@pytest.mark.asyncio
async def test_plan_resume_handler_restores_planning_phase(tmp_path: Path) -> None:
    """Resuming a workflow in a planning phase (PLAN_ADVISOR_REVIEW) must succeed.

    Only draft.md exists; workflow_plan.md is absent. The handler must return a
    success message and update runtime_state to the persisted phase.
    """
    import datetime

    from ecs_agent.components import UserPromptConfigComponent
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    world, agent_id, _, runtime_state = _build_test_world(tmp_path)

    workflow_id = "resume-planning-phase-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    # Intentionally do NOT create workflow_plan.md

    now = datetime.datetime.utcnow().isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="PLAN_ADVISOR_REVIEW",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
        completed_task_ids=[],
        retry_budget={},
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at=now,
        updated_at=now,
    )
    adapter.write_state(persisted)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, f"/plan:resume {workflow_id}")

    assert result is not None
    assert "Error" not in result, f"Expected success but got: {result}"
    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "PLAN_ADVISOR_REVIEW"


def test_require_plan_artifact_skipped_for_planning_phases(tmp_path: Path) -> None:
    """_require_plan_artifact must not raise for planning phases even without workflow_plan.md."""
    import datetime

    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    controller = PlanController()

    for phase in ("PLAN_INTERVIEW", "PLAN_ADVISOR_REVIEW", "PLAN_QA_REVIEW"):
        workflow_id = f"require-artifact-{phase.lower().replace('_', '-')}"
        adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
        # Only draft.md — no workflow_plan.md
        adapter.plan_dir.mkdir(parents=True, exist_ok=True)
        (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")

        now = datetime.datetime.utcnow().isoformat()
        state = RuntimeState(
            workflow_id=workflow_id,
            phase=phase,
            status="active",
            active_plan_file="plan/workflow_plan.md",
            current_task_id=None,
            completed_task_ids=[],
            retry_budget={},
            review_verdicts=[],
            active_subagents=[],
            memory_refs=[],
            last_checkpoint=None,
            created_at=now,
            updated_at=now,
        )

        # Should NOT raise — planning phases don't require workflow_plan.md
        controller._require_plan_artifact(adapter, state)  # type: ignore[attr-defined]
