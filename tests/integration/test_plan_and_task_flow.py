"""Integration tests for the plan-and-task E2E example command surface."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from ecs_agent.components.definitions import TerminalComponent
from ecs_agent.core import World
from ecs_agent.phases import bind_phase_graph, force
from ecs_agent.systems import TerminalCleanupSystem
from ecs_agent.types import EntityId
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
    build_scratchbook_prompt_config,
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
            phase="DRAFT_ADVISOR_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        ),
        ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        ),
        ReviewVerdict(
            phase="PLAN_QA_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        ),
    ]


async def _bound_world_at(phase: str):
    """World with the plan-task graph bound and forced to `phase`."""
    from ecs_agent.phases import force as _force
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    world = World()
    eid = world.create_entity()
    await bind_phase_graph(world, eid, PLAN_TASK_PHASE_GRAPH, agent_key="main")
    if phase != "IDLE":
        await _force(world, eid, phase, reason="test setup")
    return world, eid


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
    now = datetime.datetime.now(datetime.UTC).isoformat()
    state = RuntimeState(
        workflow_id="wf-rt",
        phase="DRAFT_INTERVIEW",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
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
    assert restored.phase == "DRAFT_INTERVIEW"


def test_scratchbook_adapter_write_review_verdict_creates_file(tmp_path: Path) -> None:
    import datetime
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )
    from examples.e2e.plan_and_task.state_models import ReviewVerdict

    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="wf-rv")
    verdict = ReviewVerdict(
        phase="DRAFT_ADVISOR_REVIEW",
        verdict="approved",
        decided_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    path_str = adapter.write_review_verdict("DRAFT_ADVISOR_REVIEW", verdict)
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
        review_verdicts=[
            ReviewVerdict(
                phase="DRAFT_QA_REVIEW",
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
        "DRAFT_QA_REVIEW",
        ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
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
    assert (root / "review" / "draft_qa_review_verdict.json").is_file()
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
        timeout=120,
        env={**os.environ, "PLAN_TASK_INTERACTIVE": "1"},
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )

    output = result.stdout.decode("utf-8", errors="replace")
    assert (
        "Using model:" in output
        or "Using Anthropic Messages API with model:" in output
    ), (
        f"Expected model selection indication in output. Got:\n{output}"
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
        # Limit ticks so the test exits before spawning the full advisor/QA cycle.
        # The plan status (with DRAFT_INTERVIEW + workflow_id) is emitted in tick 1.
        "PLAN_TASK_MAX_AGENT_TICKS": "10",
    }
    result = subprocess.run(
        ["uv", "run", "python", "examples/e2e/plan_and_task/main.py"],
        input=input_sequence,
        capture_output=True,
        timeout=120,
        env=env,
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )

    output = result.stdout.decode("utf-8", errors="replace")
    assert "DRAFT_INTERVIEW" in output, (
        f"Expected 'DRAFT_INTERVIEW' phase in output. Got:\n{output}"
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


async def test_plan_interview_creates_draft(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    world, eid = await _bound_world_at("IDLE")

    state = await PlanController(world, eid).handle_plan_start(adapter, "Build a demo")

    assert state.phase == "DRAFT_INTERVIEW"
    assert (adapter.plan_dir / "draft.md").exists()
    assert (adapter.state_dir / "runtime_state.json").exists()

    loaded_state = adapter.read_state()
    assert loaded_state.phase == "DRAFT_INTERVIEW"


async def test_plan_start_does_not_write_workflow_plan(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-wf")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    await controller.handle_plan_start(adapter, "test description")
    assert not (adapter.plan_dir / "workflow_plan.md").exists()


async def test_plan_finalize_rejects_without_verdicts(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    state.review_verdicts = []
    world, eid = await _bound_world_at(state.phase)

    with pytest.raises(ValueError, match="finalize"):
        await PlanController(world, eid).handle_plan_finalize(state, adapter)


async def test_plan_finalize_rejects_with_only_advisor_approved(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    state.review_verdicts = [
        ReviewVerdict(
            phase="DRAFT_ADVISOR_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        )
    ]
    world, eid = await _bound_world_at(state.phase)

    with pytest.raises(ValueError, match="DRAFT_QA_REVIEW"):
        await PlanController(world, eid).handle_plan_finalize(state, adapter)


async def test_plan_finalize_preserves_written_plan_and_advances(tmp_path: Path) -> None:
    """/plan:finalize must NOT overwrite the plan_writer's workflow_plan.md.

    Regression: handle_plan_finalize used to rewrite the plan from a hardcoded
    single-task stub, clobbering the plan_writer's real multi-task plan. It must
    now only gate on approvals + confirm the plan exists, then walk to TASK_READY.
    """
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)  # the plan_writer's real 2-task plan
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    state.review_verdicts = _make_approved_verdicts()
    world, eid = await _bound_world_at(state.phase)

    updated_state = await PlanController(world, eid).handle_plan_finalize(state, adapter)
    plan_path = adapter.plan_dir / "workflow_plan.md"

    assert updated_state.phase == "TASK_READY"
    assert plan_path.exists()
    parsed_plan = parse_plan(plan_path.read_text(encoding="utf-8"))
    # Both plan_writer tasks survive finalize (not clobbered to a 1-task stub).
    assert [t.task_id for t in parsed_plan.tasks] == ["task-001", "task-002"]
    assert parsed_plan.workflow_id == "test-workflow-001"


async def test_plan_finalize_rejects_when_plan_missing(tmp_path: Path) -> None:
    """Finalize with approvals but no written plan is a clear error, not a stub."""
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    state.review_verdicts = _make_approved_verdicts()
    world, eid = await _bound_world_at(state.phase)

    with pytest.raises(ValueError, match="workflow_plan.md"):
        await PlanController(world, eid).handle_plan_finalize(state, adapter)


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


async def test_task_start_rejects_plan_without_review_approval(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)

    with pytest.raises(ValueError, match="approved"):
        await TaskExec(state=state, world=world, entity_id=eid).initialize_task_queue(
            state, adapter
        )


async def test_start_without_final_plan_rejects_task_start(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    state.status = "active"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)

    with pytest.raises(ValueError, match="approved"):
        await TaskExec(state=state, world=world, entity_id=eid).initialize_task_queue(
            state, adapter
        )


async def test_task_exec_initialize_task_queue_raises_from_wrong_phase(
    tmp_path: Path,
) -> None:
    """initialize_task_queue must reject calls from non-task phases."""
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-phase-gate")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "Phase gate test")
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_state(state)
    assert state.phase == "DRAFT_INTERVIEW"
    task_exec = TaskExec(state=state, world=world, entity_id=eid)
    with pytest.raises(ValueError, match="Cannot initialize task queue from phase"):
        await task_exec.initialize_task_queue(state, adapter)


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


async def test_task_queue_persisted_to_state(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)

    updated_state = await TaskExec(
        state=state, world=world, entity_id=eid
    ).initialize_task_queue(state, adapter)
    persisted_state = adapter.read_state()
    task_queue_path = (
        tmp_path / "scratchbook" / "test-workflow-001" / "state" / "task_queue.json"
    )

    assert [task.task_id for task in updated_state.tasks] == ["task-001", "task-002"]
    assert updated_state.current_task_id == "task-001"
    assert updated_state.phase == "TASK_RUNNING"
    assert persisted_state.tasks == updated_state.tasks
    assert persisted_state.current_task_id == "task-001"
    assert not task_queue_path.exists()


def test_context_assembly_returns_bounded_packet(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
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


async def test_memory_update_record_task_completion_appends_knowledge_jsonl(
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
    world, eid = await _bound_world_at(state.phase)

    await TaskExec(state=state, world=world, entity_id=eid).record_task_completion(
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


async def test_memory_update_record_task_completion_updates_completed_ids(
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
    world, eid = await _bound_world_at(state.phase)

    updated_state = await TaskExec(
        state=state, world=world, entity_id=eid
    ).record_task_completion(
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
    state.tasks = [
        TaskRecord(task_id="task-001", title="Retry me", status="pending", retry_count=3)
    ]

    assert TaskExec(state=state).check_circuit_breaker(state, "task-001") is True


def test_circuit_breaker_allows_below_max_retries() -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    state = _make_runtime_state()
    state.tasks = [
        TaskRecord(task_id="task-001", title="Retry me", status="pending", retry_count=2)
    ]

    assert TaskExec(state=state).check_circuit_breaker(state, "task-001") is False


def test_circuit_breaker_treats_unknown_task_as_zero_retries() -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    state = _make_runtime_state()

    assert TaskExec(state=state).check_circuit_breaker(state, "task-unknown") is False


async def _run_restart_flow(state: RuntimeState, adapter: ArtifactAdapter) -> RuntimeState:
    """Drive the restart/load flow through the production resume_workflow()."""
    from examples.e2e.plan_and_task.main import resume_workflow

    adapter.write_state(state)
    world = World()
    eid = world.create_entity()
    loaded, _adapter, _actions = await resume_workflow(
        world, eid, adapter.workflow_id, base_dir=adapter.base_dir
    )
    return loaded


async def test_stale_subagent_on_restart_increments_retry(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
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

    result = await _run_restart_flow(state, adapter)

    assert result.phase == "TASK_BLOCKED"
    assert result.tasks[0].retry_count == 1
    assert result.tasks[0].status == "pending"
    assert result.current_task_id == "task-001"
    assert result.active_subagents[0].status == "stale"


async def test_advisor_review_creates_verdict_artifact(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)

    await PlanController(world, eid).handle_advisor_review(state, adapter, "approved")

    artifact = (
        tmp_path
        / "scratchbook"
        / "test-workflow-001"
        / "review"
        / "draft_advisor_review_verdict.json"
    )
    assert artifact.is_file()


async def test_advisor_review_appends_to_state_review_verdicts(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)

    updated = await PlanController(world, eid).handle_advisor_review(
        state, adapter, "approved", notes="LGTM"
    )

    assert len(updated.review_verdicts) == 1
    v = updated.review_verdicts[0]
    assert v.phase == "DRAFT_ADVISOR_REVIEW"
    assert v.verdict == "approved"
    assert v.notes is None


async def test_qa_review_approved_allows_finalization(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)
    await ctrl.handle_advisor_review(state, adapter, "approved")
    await ctrl.handle_qa_review(state, adapter, "approved")
    await ctrl.handle_plan_qa_review(state, adapter, "approved")

    result = await ctrl.handle_plan_finalize(state, adapter)

    assert result.phase == "TASK_READY"
    assert result.status == "ready"


async def test_qa_review_revise_blocks_finalization(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)
    await ctrl.handle_advisor_review(state, adapter, "approved")
    await ctrl.handle_qa_review(state, adapter, "revise", notes="needs more detail")

    with pytest.raises(ValueError, match="DRAFT_QA_REVIEW"):
        await ctrl.handle_plan_finalize(state, adapter)


async def test_qa_review_blocked_blocks_finalization(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)
    await ctrl.handle_advisor_review(state, adapter, "approved")
    await ctrl.handle_qa_review(state, adapter, "blocked", notes="out of scope")

    with pytest.raises(ValueError, match="DRAFT_QA_REVIEW"):
        await ctrl.handle_plan_finalize(state, adapter)


async def test_review_verdicts_persisted_in_state(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_draft("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)
    await ctrl.handle_advisor_review(state, adapter, "approved")
    await ctrl.handle_qa_review(state, adapter, "approved")

    persisted = adapter.read_state()

    assert len(persisted.review_verdicts) == 2
    phases = {v.phase for v in persisted.review_verdicts}
    assert "DRAFT_ADVISOR_REVIEW" in phases
    assert "DRAFT_QA_REVIEW" in phases


async def test_replan_governance_scope_change_forces_review(
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
    world, eid = await _bound_world_at(state.phase)

    updated = await PlanController(world, eid).handle_task_replan(
        state,
        adapter,
        reason="dependency changed",
        scope_changed=True,
    )

    assert updated.phase == "DRAFT_ADVISOR_REVIEW"
    assert updated.status == "needs_review"
    assert updated.current_task_id == "task-001"
    assert updated.last_checkpoint == "dependency changed"
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    persisted = adapter.read_state()
    assert persisted.phase == "DRAFT_ADVISOR_REVIEW"
    assert persisted.status == "needs_review"


async def test_replan_governance_no_scope_change_stays_running(
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
    world, eid = await _bound_world_at(state.phase)

    updated = await PlanController(world, eid).handle_task_replan(
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


async def test_abort_transitions_to_terminal_aborted(
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
    world, eid = await _bound_world_at(state.phase)

    updated = await PlanController(world, eid).handle_task_abort(
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


async def test_abort_cannot_be_resumed(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_ABORTED"
    state.status = "aborted"
    state.abort_reason = "terminal stop"
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)

    with pytest.raises(ValueError, match="terminal"):
        await PlanController(world, eid).handle_task_resume(state, adapter)


async def test_task_resume_from_blocked_returns_running(
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
    world, eid = await _bound_world_at(state.phase)

    updated = await PlanController(world, eid).handle_task_resume(state, adapter)

    assert updated.phase == "TASK_RUNNING"
    assert updated.status == "active"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_RUNNING"


async def test_phase_advance_valid_transition_from_idle() -> None:
    from ecs_agent.phases import advance

    world, eid = await _bound_world_at("IDLE")

    component = await advance(world, eid, "DRAFT_INTERVIEW", reason="test")

    assert component.phase == "DRAFT_INTERVIEW"


async def test_phase_advance_illegal_transition_raises() -> None:
    from ecs_agent.phases import InvalidPhaseTransitionError, advance

    world, eid = await _bound_world_at("IDLE")

    with pytest.raises(InvalidPhaseTransitionError, match="invalid transition"):
        await advance(world, eid, "TASK_RUNNING", reason="test")


async def test_phase_advance_terminal_phases_cannot_transition() -> None:
    from ecs_agent.phases import InvalidPhaseTransitionError, advance

    for terminal_phase in ("TASK_COMPLETED", "TASK_ABORTED"):
        world, eid = await _bound_world_at(terminal_phase)
        with pytest.raises(InvalidPhaseTransitionError, match="terminal"):
            await advance(world, eid, "DRAFT_INTERVIEW", reason="test")


def test_phase_graph_is_terminal_for_completed_and_aborted() -> None:
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    phases = PLAN_TASK_PHASE_GRAPH.phases_by_id
    assert phases["TASK_COMPLETED"].terminal is True
    assert phases["TASK_ABORTED"].terminal is True
    assert phases["TASK_RUNNING"].terminal is False
    assert phases["DRAFT_INTERVIEW"].terminal is False


def test_phase_graph_can_resume_non_terminal() -> None:
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    phases = PLAN_TASK_PHASE_GRAPH.phases_by_id

    assert phases["DRAFT_INTERVIEW"].terminal is False
    assert phases["TASK_RUNNING"].terminal is False
    assert phases["TASK_BLOCKED"].terminal is False
    assert phases["TASK_COMPLETED"].terminal is True
    assert phases["TASK_ABORTED"].terminal is True
    # IDLE is non-terminal but is the graph's pre-workflow initial phase.
    assert phases["IDLE"].terminal is False
    assert PLAN_TASK_PHASE_GRAPH.initial == "IDLE"


def test_phase_graph_requires_continuation_for_active_workflows() -> None:
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    phases = PLAN_TASK_PHASE_GRAPH.phases_by_id

    for active_phase in (
        "DRAFT_INTERVIEW",
        "TASK_RUNNING",
        "TASK_BLOCKED",
        "PLAN_FINALIZED",
    ):
        assert phases[active_phase].terminal is False, (
            f"Expected non-terminal phase for {active_phase}"
        )

    for terminal_phase in ("TASK_COMPLETED", "TASK_ABORTED"):
        assert phases[terminal_phase].terminal is True, (
            f"Expected terminal phase for {terminal_phase}"
        )
    # IDLE needs no continuation because it is the initial, pre-workflow phase.
    assert PLAN_TASK_PHASE_GRAPH.initial == "IDLE"


async def test_restart_flow_marks_stale_subagents_and_updates_phase(
    tmp_path: Path,
) -> None:
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

    result = await _run_restart_flow(state, adapter)

    assert result.phase == "TASK_BLOCKED"
    assert result.active_subagents[0].status == "stale"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_BLOCKED"


async def test_restart_flow_no_active_subagents_keeps_phase(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.active_subagents = []
    adapter.write_plan("# draft")
    adapter.write_state(state)

    result = await _run_restart_flow(state, adapter)

    assert result.phase == "TASK_BLOCKED"


async def test_restart_flow_running_without_subagents_becomes_blocked(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.active_subagents = []
    adapter.write_plan("# draft")
    adapter.write_state(state)

    result = await _run_restart_flow(state, adapter)

    assert result.phase == "TASK_BLOCKED"
    assert result.status == "blocked"
    persisted = adapter.read_state()
    assert persisted.phase == "TASK_BLOCKED"
    assert persisted.status == "blocked"


async def test_readme_command_examples_match_supported_commands(tmp_path: Path) -> None:
    """The live slash-command triggers stay in sync with README's 'Supported Commands'.

    The vocabulary is derived from the built world's TriggerSpec patterns — not a
    parallel copy — so this guard can never drift from what the harness routes.
    """
    from ecs_agent.components import UserPromptConfigComponent

    # The 11 commands listed under '## Supported Commands' in README.md.
    documented_commands = {
        "/plan:start",
        "/plan:resume",
        "/plan:status",
        "/plan:finalize",
        "/plan:write",
        "/plan:qa_review",
        "/task:start",
        "/task:status",
        "/task:resume",
        "/task:replan",
        "/task:abort",
    }

    world, agent_id, _, _ = await _build_test_world(tmp_path)
    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None

    live_patterns = {t.pattern for t in config.triggers}
    assert live_patterns == documented_commands, (
        "README '## Supported Commands' drifted from the live TriggerSpec patterns. "
        f"only-in-README={documented_commands - live_patterns}, "
        f"only-in-code={live_patterns - documented_commands}"
    )
    for trigger in config.triggers:
        assert trigger.content in config.script_handlers


async def test_review_verdict_artifact_has_phase_and_verdict(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "test description")
    await controller.handle_advisor_review(state, adapter, "approved")

    await controller.handle_qa_review(state, adapter, "approved")

    verdict_payload = json.loads(
        (adapter.review_dir / "draft_advisor_review_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict_payload["phase"] == "DRAFT_ADVISOR_REVIEW"
    assert verdict_payload["verdict"] == "approved"


async def test_task_completion_writes_evidence_artifact(
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
    world, eid = await _bound_world_at(state.phase)

    await TaskExec(state=state, world=world, entity_id=eid).record_task_completion(
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


async def test_main_world_registers_trigger_script_handlers(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, _ = await _build_test_world(tmp_path)

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
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.types import Message

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)
    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None

    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:start")
    handler = config.script_handlers[handler_key]

    user_text = "/plan:start Build demo"
    conversation.messages.append(Message(role="user", content=user_text))

    result = await handler(world, agent_id, user_text)

    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "DRAFT_INTERVIEW"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_plan_start_trigger_is_not_reprocessed_for_same_user_message(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="/plan:start Build demo"))

    system = UserPromptNormalizationSystem()

    await system.process(world)
    first_state = runtime_state[0]
    await system.process(world)

    assert first_state is not None
    assert runtime_state[0] is first_state


@pytest.mark.asyncio
async def test_trigger_plan_status_handler_returns_status_string(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    controller = PlanController(world, agent_id)
    runtime_state[0] = await controller.handle_plan_start(adapter, "Test plan")

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None

    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:status"
    )
    handler = config.script_handlers[handler_key]
    result = await handler(world, agent_id, "/plan:status")

    assert result is not None
    assert "DRAFT_INTERVIEW" in result


def test_runtime_setup_does_not_intercept_slash_commands(tmp_path: Path) -> None:
    import inspect

    from examples.e2e.plan_and_task.runtime import setup_interactive_input

    sig = inspect.signature(setup_interactive_input)
    assert "command_handler" not in sig.parameters


async def _build_test_world(
    tmp_path: Path,
) -> tuple[World, object, ArtifactAdapter, list[RuntimeState | None]]:
    from ecs_agent.providers import FakeModel
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ready"))]
    )

    world, agent_id, adapter_ref, runtime_state = await build_plan_task_world(
        model=model,
        base_dir=tmp_path,
    )
    test_adapter = PlanTaskScratchbookAdapter(
        base_dir=tmp_path, workflow_id="test-workflow-001"
    )
    adapter_ref[0] = test_adapter
    return world, agent_id, test_adapter, runtime_state


def _running_state_with_approvals() -> RuntimeState:
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.review_verdicts = _make_approved_verdicts()
    return state


@pytest.mark.asyncio
async def test_task_replan_command_scope_changed_forces_rereview(tmp_path: Path) -> None:
    """`/task:replan --scope-changed <reason>` clears verdicts and re-reviews.

    Regression: the command used to hardcode same-scope, leaving the controller's
    scope-change re-review branch unreachable from the CLI.
    """
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    eid: EntityId = agent_id  # type: ignore[assignment]
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    runtime_state[0] = _running_state_with_approvals()
    await force(world, eid, "TASK_RUNNING", reason="setup")

    config = world.get_component(eid, UserPromptConfigComponent)
    assert config is not None
    handler = config.script_handlers["task_replan"]
    result = await handler(world, eid, "/task:replan --scope-changed the data model changed")

    assert result is not None and not result.startswith("Error")
    final = runtime_state[0]
    assert final is not None
    assert final.phase == "DRAFT_ADVISOR_REVIEW"
    assert final.review_verdicts == []


@pytest.mark.asyncio
async def test_task_replan_command_same_scope_keeps_approvals(tmp_path: Path) -> None:
    """Plain `/task:replan <reason>` stays same-scope: TASK_RUNNING, verdicts kept."""
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    eid: EntityId = agent_id  # type: ignore[assignment]
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    runtime_state[0] = _running_state_with_approvals()
    await force(world, eid, "TASK_RUNNING", reason="setup")

    config = world.get_component(eid, UserPromptConfigComponent)
    assert config is not None
    handler = config.script_handlers["task_replan"]
    result = await handler(world, eid, "/task:replan just retry the same plan")

    assert result is not None and not result.startswith("Error")
    final = runtime_state[0]
    assert final is not None
    assert final.phase == "TASK_RUNNING"
    assert len(final.review_verdicts) == 3


@pytest.mark.asyncio
async def test_complete_task_tool_advances_queue_to_completed(tmp_path: Path) -> None:
    """The complete_task tool records completion and advances the live queue.

    Regression: task execution used to have no way to record completion, so the
    queue never advanced past the first task (never reached TASK_COMPLETED).
    """
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    eid: EntityId = agent_id  # type: ignore[assignment]
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.review_verdicts = _make_approved_verdicts()
    state.tasks = [
        TaskRecord(task_id="task-001", title="t1", status="running"),
        TaskRecord(task_id="task-002", title="t2", status="pending"),
    ]
    runtime_state[0] = state
    await force(world, eid, "TASK_RUNNING", reason="setup")

    registry = world.get_component(eid, ToolRegistryComponent)
    assert registry is not None
    complete = registry.handlers["complete_task"]

    res1 = await complete(summary="did task 1, verified its criterion")
    assert '"next_task": "task-002"' in res1
    cur = runtime_state[0]
    assert cur is not None and cur.current_task_id == "task-002"
    assert cur.tasks[0].status == "completed"

    res2 = await complete(summary="did task 2, verified its criterion")
    assert '"workflow_done": true' in res2
    done = runtime_state[0]
    assert done is not None and done.phase == "TASK_COMPLETED"
    assert (adapter.evidence_dir / "task-task-001-result.json").exists()
    assert (adapter.evidence_dir / "task-task-002-result.json").exists()


@pytest.mark.asyncio
async def test_complete_task_tool_rejects_outside_task_running(tmp_path: Path) -> None:
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, _adapter, runtime_state = await _build_test_world(tmp_path)
    eid: EntityId = agent_id  # type: ignore[assignment]
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    runtime_state[0] = state
    await force(world, eid, "DRAFT_INTERVIEW", reason="setup")
    registry = world.get_component(eid, ToolRegistryComponent)
    assert registry is not None
    result = await registry.handlers["complete_task"](summary="x")
    assert result.startswith("Error") and "TASK_RUNNING" in result


@pytest.mark.asyncio
async def test_complete_task_tool_requires_summary(tmp_path: Path) -> None:
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, _adapter, runtime_state = await _build_test_world(tmp_path)
    eid: EntityId = agent_id  # type: ignore[assignment]
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.current_task_id = "task-001"
    state.tasks = [TaskRecord(task_id="task-001", title="t1", status="running")]
    runtime_state[0] = state
    await force(world, eid, "TASK_RUNNING", reason="setup")
    registry = world.get_component(eid, ToolRegistryComponent)
    assert registry is not None
    result = await registry.handlers["complete_task"](summary="   ")
    assert result.startswith("Error") and "summary" in result


@pytest.mark.asyncio
async def test_handle_write_plan_idempotent_from_write_plan(tmp_path: Path) -> None:
    """/plan:write from WRITE_PLAN is an idempotent success (re-trigger), not an error.

    Regression: approving DRAFT_QA auto-advances to WRITE_PLAN, so the old
    DRAFT_QA_REVIEW-only guard made /plan:write unreachable-to-success.
    """
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    world, eid = await _bound_world_at("WRITE_PLAN")
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_write_plan(state, adapter)  # must not raise

    assert result.phase == "WRITE_PLAN"
    assert ctrl.current_phase() == "WRITE_PLAN"


@pytest.mark.asyncio
async def test_status_command_system_shows_reply_and_skips_reasoning() -> None:
    """StatusCommandSystem completes a /plan:status turn without the model.

    It appends the rendered result as the assistant reply, marks the transient
    'status_shown' terminal (ReasoningSystem's skip signal), and publishes the
    end-of-turn ReasoningCompleteEvent so front ends re-arm input.
    """
    from ecs_agent.components import RenderedUserPromptComponent
    from ecs_agent.components.definitions import ConversationComponent
    from ecs_agent.types import Message, ReasoningCompleteEvent
    from examples.e2e.plan_and_task.status_command import (
        STATUS_SHOWN_REASON,
        StatusCommandSystem,
    )

    world = World()
    eid = world.create_entity()
    world.add_component(
        eid, ConversationComponent(messages=[Message(role="user", content="/plan:status")])
    )
    world.add_component(
        eid, RenderedUserPromptComponent(text='{"phase": "IDLE"}')
    )
    completed: list[ReasoningCompleteEvent] = []

    async def _capture(event: ReasoningCompleteEvent) -> None:
        completed.append(event)

    world.event_bus.subscribe(ReasoningCompleteEvent, _capture)

    await StatusCommandSystem(eid, ["/plan:status", "/task:status"]).process(world)

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assert conv.messages[-1].role == "assistant"
    assert conv.messages[-1].content == '{"phase": "IDLE"}'
    terminal = world.get_component(eid, TerminalComponent)
    assert terminal is not None and terminal.reason == STATUS_SHOWN_REASON
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_status_command_system_ignores_non_status_messages() -> None:
    """A non-status user message is left untouched for the normal reasoning path."""
    from ecs_agent.components import RenderedUserPromptComponent
    from ecs_agent.components.definitions import ConversationComponent
    from ecs_agent.types import Message
    from examples.e2e.plan_and_task.status_command import StatusCommandSystem

    world = World()
    eid = world.create_entity()
    world.add_component(
        eid, ConversationComponent(messages=[Message(role="user", content="hello there")])
    )
    world.add_component(eid, RenderedUserPromptComponent(text="hello there"))

    await StatusCommandSystem(eid, ["/plan:status", "/task:status"]).process(world)

    conv = world.get_component(eid, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 1  # nothing appended
    assert world.get_component(eid, TerminalComponent) is None


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

    world, agent_id, _, _ = await _build_test_world(tmp_path)
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


@pytest.mark.asyncio
async def test_plan_task_world_keeps_tool_results_inline_without_scratchbook_sink(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        PendingToolCallsComponent,
        ToolRegistryComponent,
        ToolResultsComponent,
    )
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.types import Message, ToolCall, ToolSchema
    from examples.e2e.plan_and_task.main import build_plan_task_world

    async def emit_report(topic: str) -> str:
        return f"report for {topic}"

    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)
    world.apply_pending_system_operations()

    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    assert tool_registry is not None
    tool_registry.tools["emit_report"] = ToolSchema(
        name="emit_report",
        description="Emit a test report.",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    )
    tool_registry.handlers["emit_report"] = emit_report

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="run report"))
    world.add_component(
        agent_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="plan-tool-1",
                    name="emit_report",
                    arguments={"topic": "scratchbook"},
                )
            ]
        ),
    )

    tool_systems = [
        entry.system
        for entry in world._systems._systems
        if isinstance(entry.system, ToolExecutionSystem)
    ]
    assert len(tool_systems) == 1

    await tool_systems[0].process(world)

    results = world.get_component(agent_id, ToolResultsComponent)
    assert results is not None
    result_ref = results.results["plan-tool-1"]
    assert result_ref == "report for scratchbook"
    assert not (tmp_path / "scratchbook" / "records" / "tool").exists()
    tool_messages = [message for message in conversation.messages if message.role == "tool"]
    assert [message.content for message in tool_messages] == [result_ref]


@pytest.mark.asyncio
async def test_plan_task_world_persists_tool_results_via_sink_when_enabled(
    tmp_path: Path,
) -> None:
    """ISSUE-3: with enable_tool_sink=True, the tool output is written to
    scratchbook/records/tool/<id> and only the record_path is kept inline, so it
    is not resent verbatim every turn."""
    from ecs_agent.components import (
        ConversationComponent,
        PendingToolCallsComponent,
        ToolRegistryComponent,
        ToolResultsComponent,
    )
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    from ecs_agent.types import Message, ToolCall, ToolSchema
    from examples.e2e.plan_and_task.main import build_plan_task_world

    async def emit_report(topic: str) -> str:
        return f"report for {topic}"

    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(
        model=model, base_dir=tmp_path, enable_tool_sink=True
    )
    world.apply_pending_system_operations()

    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    assert tool_registry is not None
    tool_registry.tools["emit_report"] = ToolSchema(
        name="emit_report",
        description="Emit a test report.",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    )
    tool_registry.handlers["emit_report"] = emit_report

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="run report"))
    world.add_component(
        agent_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(
                    id="plan-tool-1",
                    name="emit_report",
                    arguments={"topic": "scratchbook"},
                )
            ]
        ),
    )

    tool_systems = [
        entry.system
        for entry in world._systems._systems
        if isinstance(entry.system, ToolExecutionSystem)
    ]
    assert len(tool_systems) == 1

    await tool_systems[0].process(world)

    results = world.get_component(agent_id, ToolResultsComponent)
    assert results is not None
    result_ref = results.results["plan-tool-1"]

    # Inline history carries only the record_path reference, not the raw output.
    assert result_ref.startswith("scratchbook/records/tool/")
    assert result_ref != "report for scratchbook"
    tool_messages = [
        message for message in conversation.messages if message.role == "tool"
    ]
    assert [message.content for message in tool_messages] == [result_ref]

    # The full result is persisted on disk and still recoverable.
    record_file = tmp_path / result_ref
    assert record_file.exists()
    assert "report for scratchbook" in record_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_plan_task_world_installs_auto_compaction(tmp_path: Path) -> None:
    from ecs_agent.components.definitions import (
        CompactionConfigComponent,
        ConversationArchiveComponent,
    )
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.systems.compaction import CompactionSystem
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)
    world.apply_pending_system_operations()

    compaction = world.get_component(agent_id, CompactionConfigComponent)
    archive = world.get_component(agent_id, ConversationArchiveComponent)

    assert compaction is not None
    assert archive is not None
    assert any(
        isinstance(entry.system, CompactionSystem) and entry.priority == -30
        for entry in world._systems._systems
    )


async def test_plan_task_langfuse_disabled_by_default(tmp_path: Path) -> None:
    from ecs_agent.observability.sinks import RecordingTelemetrySink
    from examples.e2e.plan_and_task.main import install_plan_task_langfuse_observability

    world, _, _, _ = await _build_test_world(tmp_path)
    sink = RecordingTelemetrySink()

    handle = await install_plan_task_langfuse_observability(
        world,
        env={},
        sink=sink,
    )

    assert handle is None
    assert not hasattr(world, "_ecs_agent_observability_sink")


async def test_plan_task_langfuse_installs_when_enabled(tmp_path: Path) -> None:
    from ecs_agent.observability.sinks import RecordingTelemetrySink
    from examples.e2e.plan_and_task.main import install_plan_task_langfuse_observability

    world, _, _, _ = await _build_test_world(tmp_path)
    sink = RecordingTelemetrySink()

    handle = await install_plan_task_langfuse_observability(
        world,
        env={
            "PLAN_TASK_LANGFUSE": "1",
            "PLAN_TASK_LANGFUSE_SESSION_ID": "session-one",
        },
        sink=sink,
    )

    assert handle is not None
    plugin = handle.plugin("langfuse")
    assert plugin is not None
    assert plugin.telemetry_sink() is sink
    config = getattr(plugin, "config")
    assert config.environment == "plan-and-task"
    assert config.session_id == "session-one"
    assert config.tags == ["plan-and-task"]
    assert config.metadata == {"source": "examples/e2e/plan_and_task"}


def test_plan_task_readme_documents_langfuse_observability() -> None:
    readme = Path("examples/e2e/plan_and_task/README.md").read_text()

    assert "PLAN_TASK_LANGFUSE" in readme
    assert "LANGFUSE_PUBLIC_KEY" in readme
    assert "LANGFUSE_SECRET_KEY" in readme
    assert "LANGFUSE_HOST" in readme or "LANGFUSE_BASE_URL" in readme
    assert "PLAN_TASK_LANGFUSE_SESSION_ID" in readme
    assert "install_plan_task_langfuse_observability" in readme
    assert "flush" in readme
    assert "shutdown" in readme


@pytest.mark.asyncio
async def test_plan_task_langfuse_records_runner_trace_when_enabled(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent
    from ecs_agent.core import Runner
    from ecs_agent.observability.sinks import RecordingTelemetrySink
    from ecs_agent.types import Message
    from examples.e2e.plan_and_task.main import install_plan_task_langfuse_observability

    world, agent_id, _, _ = await _build_test_world(tmp_path)
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="Summarize plan status"))
    sink = RecordingTelemetrySink()
    handle = await install_plan_task_langfuse_observability(
        world,
        env={"PLAN_TASK_LANGFUSE": "true"},
        sink=sink,
    )

    assert handle is not None
    await Runner().run(world, max_ticks=1)
    await handle.flush()
    await handle.shutdown()

    assert any(record.kind == "trace" for record in sink.records)
    assert any(record.kind == "generation" for record in sink.records)


@pytest.mark.asyncio
async def test_plan_task_compaction_summarizes_before_prompt_render(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        CurrentCompactionSummaryComponent,
        RenderedSystemPromptComponent,
    )
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.systems.compaction import CompactionSystem
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Compacted summary")
            )
        ]
    )
    world, agent_id, _, _ = await build_plan_task_world(
        model=model,
        base_dir=tmp_path,
        compaction_threshold_tokens=1,
    )

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.extend(
        [
            Message(role="user", content="word " * 40),
            Message(role="assistant", content="reply " * 40),
        ]
    )

    await CompactionSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    summary = world.get_component(agent_id, CurrentCompactionSummaryComponent)
    rendered = world.get_component(agent_id, RenderedSystemPromptComponent)

    assert summary is not None
    assert summary.summary == "Compacted summary"
    assert rendered is not None
    assert "<chat_history_summary>Compacted summary</chat_history_summary>" in rendered.text


# ── DelegationCompletedEvent verdict tests ─────────────────────────────────────


async def test_main_world_does_not_register_record_verdict_tools(tmp_path: Path) -> None:
    """record_advisor_verdict and record_qa_verdict tools must NOT be registered."""
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, _, _ = await _build_test_world(tmp_path)
    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    assert tool_registry is not None
    assert "record_advisor_verdict" not in tool_registry.tools
    assert "record_qa_verdict" not in tool_registry.tools
    assert "record_advisor_verdict" not in tool_registry.handlers
    assert "record_qa_verdict" not in tool_registry.handlers


@pytest.mark.asyncio
async def test_plan_start_resets_stale_compaction_state(tmp_path: Path) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        ConversationArchiveComponent,
        CurrentCompactionSummaryComponent,
        RenderedSystemPromptComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.types import Message

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)
    world.add_component(
        agent_id,
        CurrentCompactionSummaryComponent(summary="stale-summary"),
    )
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    assert archive is not None
    archive.archived_summaries.append("stale-archive")
    world.add_component(agent_id, RenderedSystemPromptComponent(text="stale-rendered"))
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.extend(
        [
            Message(role="user", content="workflow-a question"),
            Message(role="assistant", content="workflow-a answer"),
        ]
    )

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:start")
    handler = config.script_handlers[handler_key]

    user_text = "/plan:start Build demo"
    conversation.messages.append(Message(role="user", content=user_text))

    result = await handler(world, agent_id, user_text)

    assert result is not None
    assert runtime_state[0] is not None
    assert world.get_component(agent_id, CurrentCompactionSummaryComponent) is None
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    assert archive is not None
    assert archive.archived_summaries == []
    assert world.get_component(agent_id, RenderedSystemPromptComponent) is None
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    assert [message.content for message in conversation.messages] == [user_text]


@pytest.mark.asyncio
async def test_plan_resume_resets_stale_compaction_state(tmp_path: Path) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        ConversationArchiveComponent,
        CurrentCompactionSummaryComponent,
        RenderedSystemPromptComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    state = _make_state_at_phase("TASK_BLOCKED")
    state.workflow_id = adapter.workflow_id
    state.status = "blocked"
    state.current_task_id = "task-001"
    (adapter.plan_dir / "workflow_plan.md").write_text("# Plan\n", encoding="utf-8")
    adapter.write_state(state)

    world.add_component(
        agent_id,
        CurrentCompactionSummaryComponent(summary="stale-summary"),
    )
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    assert archive is not None
    archive.archived_summaries.append("stale-archive")
    world.add_component(agent_id, RenderedSystemPromptComponent(text="stale-rendered"))
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.extend(
        [
            Message(role="user", content="workflow-a question"),
            Message(role="assistant", content="workflow-a answer"),
        ]
    )

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:resume")
    handler = config.script_handlers[handler_key]

    user_text = f"/plan:resume {state.workflow_id}"
    conversation.messages.append(Message(role="user", content=user_text))

    result = await handler(world, agent_id, user_text)

    assert result is not None
    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert world.get_component(agent_id, CurrentCompactionSummaryComponent) is None
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    assert archive is not None
    assert archive.archived_summaries == []
    assert world.get_component(agent_id, RenderedSystemPromptComponent) is None
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    assert [message.content for message in conversation.messages] == [user_text]


@pytest.mark.asyncio
async def test_task_start_auto_load_resets_stale_compaction_state(tmp_path: Path) -> None:
    from ecs_agent.components import (
        ConversationArchiveComponent,
        ConversationComponent,
        CurrentCompactionSummaryComponent,
    )
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_FINALIZED"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_state(state)

    world.add_component(
        agent_id,
        CurrentCompactionSummaryComponent(summary="stale-summary"),
    )
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    assert archive is not None
    archive.archived_summaries.append("stale-archive")

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:start {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert world.get_component(agent_id, CurrentCompactionSummaryComponent) is None
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    assert archive is not None
    assert archive.archived_summaries == []


@pytest.mark.asyncio
async def test_plan_start_reset_does_not_retrigger_on_second_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    import examples.e2e.plan_and_task.main as plan_task_main

    calls = {"count": 0}

    async def fake_derive_workflow_id(description: str, model: object) -> str:
        _ = (description, model)
        calls["count"] += 1
        return "build-demo"

    monkeypatch.setattr(
        plan_task_main,
        "derive_workflow_id_from_llm",
        fake_derive_workflow_id,
    )

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.extend(
        [
            Message(role="user", content="workflow-a question"),
            Message(role="assistant", content="workflow-a answer"),
            Message(role="user", content="/plan:start Build demo"),
        ]
    )

    system = UserPromptNormalizationSystem()
    await system.process(world)
    await system.process(world)

    assert calls["count"] == 1
    assert runtime_state[0] is not None
    assert [message.content for message in conversation.messages] == [
        "/plan:start Build demo"
    ]


@pytest.mark.asyncio
async def test_plan_resume_reset_does_not_retrigger_on_second_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    state = _make_state_at_phase("TASK_BLOCKED")
    state.workflow_id = adapter.workflow_id
    state.status = "blocked"
    state.current_task_id = "task-001"
    (adapter.plan_dir / "workflow_plan.md").write_text("# Plan\n", encoding="utf-8")
    adapter.write_state(state)

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.extend(
        [
            Message(role="user", content="workflow-a question"),
            Message(role="assistant", content="workflow-a answer"),
            Message(role="user", content=f"/plan:resume {state.workflow_id}"),
        ]
    )

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:resume")
    original_handler = config.script_handlers[handler_key]
    calls = {"count": 0}

    async def counted_handler(world_obj: World, entity_id: EntityId, user_text: str) -> str | None:
        calls["count"] += 1
        return await original_handler(world_obj, entity_id, user_text)

    config.script_handlers[handler_key] = counted_handler

    system = UserPromptNormalizationSystem()
    await system.process(world)
    await system.process(world)

    assert calls["count"] == 1
    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert [message.content for message in conversation.messages] == [
        f"/plan:resume {state.workflow_id}"
    ]


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


def test_extract_verdict_prefers_verdict_marker_line_over_earlier_prose() -> None:
    """A `VERDICT: <token>` line wins over verdict words appearing in prose.

    Reviewer checklists legitimately contain words like "blocked" in FAIL
    reasons; only the marker line is the machine-readable decision.
    """
    from examples.e2e.plan_and_task.main import _extract_verdict_from_result

    result = _extract_verdict_from_result(
        "2. RISKS — FAIL: progress is blocked by a missing mitigation.\n"
        "All other items PASS.\n"
        "VERDICT: revise"
    )
    assert result == "revise"


def test_extract_verdict_uses_last_marker_line() -> None:
    from examples.e2e.plan_and_task.main import _extract_verdict_from_result

    result = _extract_verdict_from_result(
        "Earlier draft said VERDICT: revise\n"
        "After re-checking the fixes:\n"
        "verdict: approved"
    )
    assert result == "approved"


def test_extract_verdict_marker_is_case_insensitive() -> None:
    from examples.e2e.plan_and_task.main import _extract_verdict_from_result

    assert _extract_verdict_from_result("Verdict: Blocked") == "blocked"


@pytest.mark.asyncio
async def test_delegation_completed_event_records_advisor_verdict(
    tmp_path: Path,
) -> None:
    """Publishing DelegationCompletedEvent for 'advisor' updates runtime state."""
    from ecs_agent.types import DelegationCompletedEvent
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    controller = PlanController(world, agent_id)
    runtime_state[0] = await controller.handle_plan_start(adapter, "Test workflow")

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
    assert verdicts[0].phase == "DRAFT_ADVISOR_REVIEW"
    assert verdicts[0].verdict == "approved"


@pytest.mark.asyncio
async def test_delegation_completed_event_records_qa_verdict(tmp_path: Path) -> None:
    """Publishing DelegationCompletedEvent for 'qa' updates runtime state."""
    from ecs_agent.types import DelegationCompletedEvent
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    controller = PlanController(world, agent_id)
    runtime_state[0] = await controller.handle_plan_start(adapter, "Test workflow")

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
    assert verdicts[0].phase == "DRAFT_QA_REVIEW"
    assert verdicts[0].verdict == "revise"


@pytest.mark.asyncio
async def test_delegation_completed_event_ignores_other_entity(tmp_path: Path) -> None:
    """Events for a different entity_id must not update state."""
    from ecs_agent.types import DelegationCompletedEvent, EntityId
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    controller = PlanController(world, agent_id)
    runtime_state[0] = await controller.handle_plan_start(adapter, "Test workflow")

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

    assert build_advisor_prompt("scratchbook/wf-001/plan/draft.md").strip()
    assert build_qa_prompt("scratchbook/wf-001/plan/draft.md", "approved").strip()
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


async def test_main_world_does_not_add_scratchbook_prompt_config_at_init(
    tmp_path: Path,
) -> None:
    """ScratchbookPromptConfig is NOT added at world init.

    It is added lazily inside _handle_plan_start after the workflow_id is
    derived from the user's task description.
    """
    from ecs_agent.scratchbook.prompt_definition import ScratchbookPromptConfig

    world, agent_id, _, _ = await _build_test_world(tmp_path)
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


async def test_main_world_installs_builtin_tools(tmp_path: Path) -> None:
    from ecs_agent.components import ToolRegistryComponent

    world, agent_id, _, _ = await _build_test_world(tmp_path)
    tool_registry = world.get_component(agent_id, ToolRegistryComponent)
    assert tool_registry is not None
    for expected_tool in ("read_file", "write_file", "edit_file", "bash", "glob"):
        assert expected_tool in tool_registry.tools, f"missing tool: {expected_tool}"
        assert expected_tool in tool_registry.handlers, (
            f"missing handler: {expected_tool}"
        )
    assert "explore" not in tool_registry.tools
    assert "explore" not in tool_registry.handlers


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


def test_plan_interview_system_prompt_is_proactive_not_interrogative() -> None:
    """Draft phase must proactively propose/recommend per section, not ask the user everything.

    Verifies fix for: draft agent being passive and interrogating the user for
    every section instead of exploring options and recommending choices.
    """
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    # Proactive stance: the agent recommends/proposes content, not merely questions.
    assert any(
        word in prompt_lower for word in ("propose", "recommend")
    ), "Draft prompt must instruct the agent to propose/recommend content per section"
    # Unconfirmed proposals must be tagged so advisor/QA can audit assumptions.
    assert "(proposed)" in prompt_lower, (
        "Draft prompt must tag unconfirmed proposals so reviewers can spot assumptions"
    )
    # Must NOT revert to the old passive one-question-at-a-time interrogation.
    assert "ask one question at a time" not in prompt_lower, (
        "Draft prompt must not instruct passive one-question-at-a-time interviewing"
    )
    # User still steers: a confirm / redirect loop must remain.
    assert "confirm" in prompt_lower and (
        "redirect" in prompt_lower or "tweak" in prompt_lower
    ), "Draft prompt must keep a user confirm/redirect loop"
    # Choices are presented via the structured ask_question tool, not free prose,
    # so the recommendation + alternatives are selectable by the user.
    assert "ask_question" in prompt_lower, (
        "Draft prompt must route the confirm/choose step through the ask_question tool"
    )


async def test_draft_template_has_structured_sections(tmp_path: Path) -> None:
    """The initial draft template must have structured fillable sections.

    Verifies fix for: draft has no clear sections to progressively fill in.
    """
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-draft-sections")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)

    state = await controller.handle_plan_start(adapter, "Test description")
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


async def test_draft_template_has_placeholder_content(tmp_path: Path) -> None:
    """Draft sections should have placeholder content (not empty) so edit_file can target them."""
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-draft-placeholders")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)

    await controller.handle_plan_start(adapter, "Some description")
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


async def test_plan_start_handler_sets_workflow_id_from_description(
    tmp_path: Path,
) -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    model = FakeModel(responses=["ok"])
    world, agent_id, adapter_ref, runtime_state = await build_plan_task_world(
        model=model,
        base_dir=tmp_path,
    )

    controller = PlanController(world, agent_id)
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id="initial-id")
    state = await controller.handle_plan_start(adapter, "Build a task management app")
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
    assert "end" in props
    assert "content" in props
    assert "old_text" not in props
    assert "new_text" not in props
    assert "replace_all" not in props
    assert "read_id" not in props
    assert "snapshot_id" not in props
    assert "edits_json" not in props
    required = schema.parameters.get("required", [])
    assert "file_path" in required
    assert "op" in required
    assert "pos" in required


async def test_edit_file_direct_params_replaces_content(tmp_path: Path) -> None:
    from ecs_agent.tools.builtins.edit_tool import edit_file
    from ecs_agent.tools.builtins.file_tools import read_file

    target = tmp_path / "draft.md"
    target.write_text("## Scope\n(to be filled)\n\n## Next\n", encoding="utf-8")
    assert await read_file("draft.md", str(tmp_path)) == "## Scope\n(to be filled)\n\n## Next"
    result = await edit_file._tool_handler(  # type: ignore[attr-defined]
        file_path="draft.md",
        workspace_root=str(tmp_path),
        op="replace",
        pos="2",
        content="In scope: everything",
    )
    assert "draft.md" in result
    assert "(to be filled)" not in target.read_text(encoding="utf-8")
    assert "In scope: everything" in target.read_text(encoding="utf-8")


async def test_edit_file_raises_when_line_not_found(tmp_path: Path) -> None:
    import pytest
    from ecs_agent.tools.builtins.edit_tool import edit_file
    from ecs_agent.tools.builtins.file_tools import read_file

    (tmp_path / "file.md").write_text("hello world", encoding="utf-8")
    await read_file("file.md", str(tmp_path))
    with pytest.raises(Exception):
        await edit_file._tool_handler(  # type: ignore[attr-defined]
            file_path="file.md",
            workspace_root=str(tmp_path),
            op="replace",
            pos="2",
            content="x",
        )


async def test_edit_file_requires_recent_read(tmp_path: Path) -> None:
    import pytest
    from ecs_agent.tools.builtins.edit_tool import edit_file

    (tmp_path / "file.md").write_text("foo\nfoo\n", encoding="utf-8")
    with pytest.raises(Exception):
        await edit_file._tool_handler(  # type: ignore[attr-defined]
            file_path="file.md",
            workspace_root=str(tmp_path),
            op="replace",
            pos="1",
            content="bar",
        )


async def test_derive_workflow_id_uses_llm_slug() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import derive_workflow_id_from_llm

    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="writing-assistant-tool")
            )
        ]
    )
    result = await derive_workflow_id_from_llm("辅助写作软件", model)
    assert result == "writing-assistant-tool"


async def test_derive_workflow_id_normalizes_llm_output() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import derive_workflow_id_from_llm

    model = FakeModel(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="Writing Assistant Tool!")
            )
        ]
    )
    result = await derive_workflow_id_from_llm("Writing assistant", model)
    assert result == "writing-assistant-tool"


async def test_derive_workflow_id_falls_back_on_empty_response() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.runtime import (
        derive_workflow_id_from_llm,
        slug_from_description,
    )

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="   "))]
    )
    result = await derive_workflow_id_from_llm("build a task manager", model)
    assert result == slug_from_description("build a task manager")
    assert result != ""


async def test_derive_workflow_id_falls_back_on_provider_error() -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.runtime import (
        derive_workflow_id_from_llm,
        slug_from_description,
    )

    model = FakeModel(responses=[])
    result = await derive_workflow_id_from_llm("build a task manager", model)
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


def test_plan_interview_system_prompt_contains_blocked_instruction_duplicate() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    assert "blocked" in PLAN_INTERVIEW_SYSTEM_PROMPT.lower(), (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must mention the 'blocked' verdict"
    )


def test_plan_interview_system_prompt_gates_qa_on_advisor_approval_duplicate() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT

    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "approved" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must gate QA call on advisor 'approved' verdict"
    )
    assert "do not" in prompt_lower or "only" in prompt_lower, (
        "PLAN_INTERVIEW_SYSTEM_PROMPT must make the QA gating condition explicit"
    )


async def test_controller_advisor_revise_state_stays_in_advisor_review_duplicate(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "revise test workflow")
    state = await controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs more detail"
    )

    assert state.phase == "DRAFT_ADVISOR_REVIEW", (
        f"Expected PLAN_ADVISOR_REVIEW after revise, got {state.phase}"
    )
    assert state.phase != "DRAFT_QA_REVIEW"


async def test_controller_advisor_revise_followed_by_approved_allows_qa_duplicate(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-then-approve-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "revise then approve workflow")

    state = await controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs scope"
    )
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    state = await controller.handle_advisor_review(state, adapter, "approved", notes="LGTM")
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "DRAFT_ADVISOR_REVIEW" not in missing, (
        f"Expected PLAN_ADVISOR_REVIEW to be approved in missing list: {missing}"
    )


async def test_controller_advisor_multiple_verdicts_upsert_keeps_latest(
    tmp_path: Path,
) -> None:
    """Same-phase verdicts are upserted: only the latest (approved) is kept per phase."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="multi-verdict-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "multi-verdict workflow")

    state = await controller.handle_advisor_review(state, adapter, "revise")
    state = await controller.handle_advisor_review(state, adapter, "blocked")
    state = await controller.handle_advisor_review(state, adapter, "approved")

    advisor_verdicts = [
        v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"
    ]
    assert len(advisor_verdicts) == 1, (
        f"Expected 1 advisor verdict (upsert keeps latest), got {len(advisor_verdicts)}"
    )
    assert advisor_verdicts[0].verdict == "approved"


async def test_controller_missing_approved_reviews_uses_last_verdict_duplicate(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.prompts import PLAN_INTERVIEW_SYSTEM_PROMPT
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="last-verdict-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "last verdict test workflow")

    state = await controller.handle_advisor_review(state, adapter, "revise")
    state = await controller.handle_advisor_review(state, adapter, "approved")

    missing = controller._missing_approved_reviews(state.review_verdicts)
    prompt_lower = PLAN_INTERVIEW_SYSTEM_PROMPT.lower()
    assert "DRAFT_ADVISOR_REVIEW" not in missing, (
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


async def test_controller_advisor_revise_state_stays_in_advisor_review(
    tmp_path: Path,
) -> None:
    """After a 'revise' verdict, phase must remain PLAN_ADVISOR_REVIEW (not advance to QA)."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "revise test workflow")

    # First advisor call → state transitions to PLAN_ADVISOR_REVIEW
    state = await controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs more detail"
    )

    assert state.phase == "DRAFT_ADVISOR_REVIEW", (
        f"Expected PLAN_ADVISOR_REVIEW after revise, got {state.phase}"
    )
    # Phase must NOT advance to QA
    assert state.phase != "DRAFT_QA_REVIEW"


async def test_controller_advisor_revise_followed_by_approved_allows_qa(
    tmp_path: Path,
) -> None:
    """After revise then approved, the advisor verdict is approved and QA can be called."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="revise-then-approve-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "revise then approve workflow")

    # Round 1: revise
    state = await controller.handle_advisor_review(
        state, adapter, "revise", notes="Needs scope"
    )
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    # Round 2: approved (LLM revised draft and re-called advisor)
    state = await controller.handle_advisor_review(state, adapter, "approved", notes="LGTM")
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    # Now the latest advisor verdict is "approved" — _missing_approved_reviews should pass advisor
    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "DRAFT_ADVISOR_REVIEW" not in missing, (
        f"Expected PLAN_ADVISOR_REVIEW to be approved in missing list: {missing}"
    )


async def test_controller_advisor_multiple_verdicts_upsert_keeps_latest_2(
    tmp_path: Path,
) -> None:
    """All advisor verdict calls upsert: final state has only 1 entry per phase (the latest)."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="multi-verdict-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "multi-verdict workflow")

    state = await controller.handle_advisor_review(state, adapter, "revise")
    state = await controller.handle_advisor_review(state, adapter, "blocked")
    state = await controller.handle_advisor_review(state, adapter, "approved")

    advisor_verdicts = [
        v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"
    ]
    assert len(advisor_verdicts) == 1, (
        f"Expected 1 advisor verdict (upsert keeps latest), got {len(advisor_verdicts)}"
    )
    assert advisor_verdicts[0].verdict == "approved"


async def test_controller_qa_verdict_blocked_while_advisor_revise_does_not_advance(
    tmp_path: Path,
) -> None:
    """A QA verdict recorded while the advisor review is still on 'revise' must
    not vault the workflow from DRAFT_ADVISOR_REVIEW into DRAFT_QA_REVIEW.

    Regression: the 'enter the review phase to record its verdict' advance fired
    unconditionally, so recording the next reviewer's verdict progressed the
    phase past an un-approved advisor gate — a forward hop the state machine's
    approval gates forbid.
    """
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="advisor-revise-then-qa")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "advisor revise then qa")

    state = await controller.handle_advisor_review(state, adapter, "revise", notes="no")
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    # QA gets consulted before the advisor approved — the phase must hold and no
    # QA verdict may be recorded.
    with pytest.raises(ValueError, match="not approved"):
        await controller.handle_qa_review(state, adapter, "revise", notes="also no")
    assert controller.current_phase() == "DRAFT_ADVISOR_REVIEW"
    assert all(v.phase != "DRAFT_QA_REVIEW" for v in state.review_verdicts)

    # Once the advisor approves, QA may proceed and a 'revise' stays in QA.
    state = await controller.handle_advisor_review(state, adapter, "approved")
    state = await controller.handle_qa_review(state, adapter, "revise", notes="qa")
    assert state.phase == "DRAFT_QA_REVIEW"


async def test_controller_missing_approved_reviews_uses_last_verdict(
    tmp_path: Path,
) -> None:
    """_missing_approved_reviews must check the LAST verdict per phase, not the first."""
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="last-verdict-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "last verdict test workflow")

    # revise followed by approved — only last (approved) should count
    state = await controller.handle_advisor_review(state, adapter, "revise")
    state = await controller.handle_advisor_review(state, adapter, "approved")

    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "DRAFT_ADVISOR_REVIEW" not in missing, (
        "After revise→approved, PLAN_ADVISOR_REVIEW should be satisfied"
    )


# ── /plan:resume command tests ─────────────────────────────────────────────────


async def test_main_world_registers_plan_resume_trigger(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent

    world, agent_id, _, _ = await _build_test_world(tmp_path)
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

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)

    workflow_id = "resume-test-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="TASK_BLOCKED",
        status="blocked",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-001",
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

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)

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

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)

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

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)

    workflow_id = "stale-subagent-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="TASK_RUNNING",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-001",
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
    assert runtime_state[0].phase == "TASK_BLOCKED"
    assert runtime_state[0].status == "blocked"

    restored = adapter.read_state()
    assert restored.phase == "TASK_BLOCKED"
    assert restored.status == "blocked"
    assert restored.active_subagents[0].status == "stale"


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

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)

    workflow_id = "scratchbook-config-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    now = datetime.datetime.now(datetime.UTC).isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="DRAFT_INTERVIEW",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
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

    for phase in ("DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "WRITE_PLAN", "PLAN_QA_REVIEW"):
        workflow_id = f"planning-phase-{phase.lower().replace('_', '-')}"
        adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
        # Write draft.md but NOT workflow_plan.md
        adapter.plan_dir.mkdir(parents=True, exist_ok=True)
        (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")

        now = datetime.datetime.now(datetime.UTC).isoformat()
        state = RuntimeState(
            workflow_id=workflow_id,
            phase=phase,
            status="active",
            active_plan_file="plan/workflow_plan.md",
            current_task_id=None,
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

    now = datetime.datetime.now(datetime.UTC).isoformat()
    state = RuntimeState(
        workflow_id=workflow_id,
        phase="TASK_BLOCKED",
        status="blocked",
        active_plan_file="plan/workflow_plan.md",
        current_task_id="task-001",
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

    world, agent_id, _, runtime_state = await _build_test_world(tmp_path)

    workflow_id = "resume-planning-phase-workflow"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    # Intentionally do NOT create workflow_plan.md

    now = datetime.datetime.now(datetime.UTC).isoformat()
    persisted = RuntimeState(
        workflow_id=workflow_id,
        phase="DRAFT_ADVISOR_REVIEW",
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
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
    assert runtime_state[0].phase == "DRAFT_ADVISOR_REVIEW"


async def test_require_plan_artifact_skipped_for_planning_phases(tmp_path: Path) -> None:
    """_require_plan_artifact must not raise for planning phases even without workflow_plan.md."""
    import datetime

    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)

    for phase in ("DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "WRITE_PLAN", "PLAN_QA_REVIEW"):
        workflow_id = f"require-artifact-{phase.lower().replace('_', '-')}"
        adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
        # Only draft.md — no workflow_plan.md
        adapter.plan_dir.mkdir(parents=True, exist_ok=True)
        (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")

        now = datetime.datetime.now(datetime.UTC).isoformat()
        state = RuntimeState(
            workflow_id=workflow_id,
            phase=phase,
            status="active",
            active_plan_file="plan/workflow_plan.md",
            current_task_id=None,
            review_verdicts=[],
            active_subagents=[],
            memory_refs=[],
            last_checkpoint=None,
            created_at=now,
            updated_at=now,
        )

        # Should NOT raise — planning phases don't require workflow_plan.md.
        # The guard reads PhaseComponent, so move the runtime phase there.
        await force(world, eid, phase, reason="test")
        controller._require_plan_artifact(adapter, state)  # type: ignore[attr-defined]


async def test_handle_write_plan_transitions_to_write_plan(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_QA_REVIEW"
    state.review_verdicts = [
        ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        )
    ]
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_write_plan(state, adapter)

    assert result.phase == "WRITE_PLAN"


async def test_handle_write_plan_rejects_unapproved_qa(tmp_path: Path) -> None:
    """/plan:write mirrors the QA gate: it must refuse while QA is on 'revise'."""
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_QA_REVIEW"
    state.review_verdicts = [
        ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
            verdict="revise",
            decided_at="2026-01-01T00:00:00",
        )
    ]
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    with pytest.raises(ValueError, match="not approved"):
        await ctrl.handle_write_plan(state, adapter)
    # The rejected call must not move the workflow forward.
    assert ctrl.current_phase() == "DRAFT_QA_REVIEW"


async def test_handle_write_plan_rejects_wrong_phase(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    with pytest.raises(ValueError, match="DRAFT_QA_REVIEW"):
        await ctrl.handle_write_plan(state, adapter)


async def test_handle_write_plan_persists_state(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_QA_REVIEW"
    state.review_verdicts = [
        ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        )
    ]
    adapter.write_state(state)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    await ctrl.handle_write_plan(state, adapter)

    reloaded = adapter.read_state()
    assert reloaded.phase == "WRITE_PLAN"


async def test_handle_plan_qa_review_revise_stays_in_write_plan(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_plan_qa_review(state, adapter, "revise", notes="needs more detail")

    assert any(v.phase == "PLAN_QA_REVIEW" and v.verdict == "revise" for v in result.review_verdicts)


async def test_handle_plan_qa_review_creates_verdict_artifact(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    await ctrl.handle_plan_qa_review(state, adapter, "approved")

    assert (adapter.review_dir / "plan_qa_review_verdict.json").is_file()


async def test_handle_plan_qa_review_invalid_verdict_raises(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    with pytest.raises(ValueError, match="Invalid verdict"):
        await ctrl.handle_plan_qa_review(state, adapter, "maybe")


async def test_full_plan_flow_all_three_reviews(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    await ctrl.handle_advisor_review(state, adapter, "approved")
    await ctrl.handle_qa_review(state, adapter, "approved")
    await ctrl.handle_plan_qa_review(state, adapter, "approved")

    result = await ctrl.handle_plan_finalize(state, adapter)

    assert result.phase == "TASK_READY"
    assert result.status == "ready"


async def test_finalize_blocked_without_plan_qa_review(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "active"
    adapter.write_plan("# draft")
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    await ctrl.handle_advisor_review(state, adapter, "approved")
    await ctrl.handle_qa_review(state, adapter, "approved")

    with pytest.raises(ValueError, match="PLAN_QA_REVIEW"):
        await ctrl.handle_plan_finalize(state, adapter)


@pytest.mark.asyncio
async def test_plan_write_command_transitions_phase(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, adapter_ref, runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "write-plan-cmd-test"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter_ref[0] = adapter

    state = _make_runtime_state()
    state.workflow_id = workflow_id
    state.phase = "DRAFT_QA_REVIEW"
    state.review_verdicts = [
        ReviewVerdict(
            phase="DRAFT_QA_REVIEW",
            verdict="approved",
            decided_at="2026-01-01T00:00:00",
        )
    ]
    adapter.write_state(state)
    runtime_state[0] = state
    await force(world, agent_id, state.phase, reason="test setup")

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:write")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, "/plan:write")

    assert result is not None
    assert "Error" not in result
    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "WRITE_PLAN"


@pytest.mark.asyncio
async def test_plan_qa_review_command_approved(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, adapter_ref, runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "plan-qa-cmd-test"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter_ref[0] = adapter

    state = _make_runtime_state()
    state.workflow_id = workflow_id
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    runtime_state[0] = state
    await force(world, agent_id, state.phase, reason="test setup")

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:qa_review")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, "/plan:qa_review approved")

    assert result is not None
    assert "Error" not in result
    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "PLAN_FINALIZED"


@pytest.mark.asyncio
async def test_plan_qa_review_command_invalid_verdict(tmp_path: Path) -> None:
    from ecs_agent.components import UserPromptConfigComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, adapter_ref, runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "plan-qa-cmd-bad-verdict"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter_ref[0] = adapter

    state = _make_runtime_state()
    state.workflow_id = workflow_id
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    runtime_state[0] = state
    await force(world, agent_id, state.phase, reason="test setup")

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:qa_review")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, "/plan:qa_review maybe")

    assert result is not None
    assert "Error" in result


# ---------------------------------------------------------------------------
# handle_write_plan_completed tests
# ---------------------------------------------------------------------------

async def test_handle_write_plan_completed_transitions_to_plan_qa_review(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_write_plan_completed(state, adapter)

    assert result.phase == "PLAN_QA_REVIEW"


async def test_handle_write_plan_completed_persists_state(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    adapter.plan_dir.mkdir(parents=True, exist_ok=True)
    (adapter.plan_dir / "draft.md").write_text("# Draft\n", encoding="utf-8")
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    await ctrl.handle_write_plan_completed(state, adapter)

    reloaded = adapter.read_state()
    assert reloaded.phase == "PLAN_QA_REVIEW"


async def test_handle_write_plan_completed_rejects_wrong_phase(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_QA_REVIEW"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    with pytest.raises(ValueError, match="WRITE_PLAN"):
        await ctrl.handle_write_plan_completed(state, adapter)


async def test_plan_writer_subagent_registered(tmp_path: Path) -> None:
    from ecs_agent.components.definitions import SubagentRegistryComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    registry = world.get_component(agent_id, SubagentRegistryComponent)
    assert registry is not None
    assert "plan_writer" in registry.subagents
    cfg = registry.subagents["plan_writer"]
    assert "writing-plans" in cfg.skills


async def test_writing_plans_skill_registered_in_catalog(tmp_path: Path) -> None:
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.skills import catalog as _catalog
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(responses=["ok"])
    await build_plan_task_world(model=model, base_dir=tmp_path)

    descriptor = _catalog.lookup("writing-plans")
    assert descriptor is not None
    assert descriptor.name == "writing-plans"


async def test_web_search_tool_installed_when_brave_key_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ecs_agent.components.definitions import ToolRegistryComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world

    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)

    registry = world.get_component(agent_id, ToolRegistryComponent)
    assert registry is not None
    assert "web_search" in registry.tools


async def test_web_search_tool_absent_without_brave_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ecs_agent.components.definitions import ToolRegistryComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world

    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)

    registry = world.get_component(agent_id, ToolRegistryComponent)
    assert registry is not None
    assert "web_search" not in registry.tools
    # The always-on built-in webfetch stays available regardless of the key.
    assert "webfetch" in registry.tools


async def test_gh_skill_installed_and_listed_when_gh_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from ecs_agent.components.definitions import ToolRegistryComponent
    from ecs_agent.prompts.provider import InventoryPlaceholderProvider
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world

    # Force the gh binary to look installed so the test is deterministic in CI.
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None
    )
    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)

    # Progressive disclosure: the skill is listed by name/description...
    skills = InventoryPlaceholderProvider().resolve_placeholders(world, agent_id)
    assert "gh:" in skills["_installed_skills"]

    # ...and its full instructions load on demand via load_skill_details.
    registry = world.get_component(agent_id, ToolRegistryComponent)
    assert registry is not None
    details = await registry.handlers["load_skill_details"](skill_name="gh")
    assert details.startswith("Skill: gh")
    assert "gh api" in details  # a marker from the skill body


async def test_gh_skill_absent_when_gh_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from ecs_agent.prompts.provider import InventoryPlaceholderProvider
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world

    monkeypatch.setattr(shutil, "which", lambda name: None)
    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)

    skills = InventoryPlaceholderProvider().resolve_placeholders(world, agent_id)
    assert "gh:" not in skills["_installed_skills"]


# ---------------------------------------------------------------------------
# BillingSubscriber tests
# ---------------------------------------------------------------------------

def test_billing_subscriber_accumulates_token_counts() -> None:
    import asyncio
    from ecs_agent.accounting.models import LLMInvocationEvent, UsageRecord
    from ecs_agent.types import EntityId
    from examples.e2e.plan_and_task.billing import BillingSubscriber

    sub = BillingSubscriber()

    async def _run() -> None:
        usage = UsageRecord(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            cached_input_tokens=10,
        )
        event = LLMInvocationEvent(
            entity_id=EntityId(1),
            provider_id="test",
            model="m",
            usage=usage,
        )
        await sub._handle_llm_invocation(event)
        await sub._handle_llm_invocation(event)

    asyncio.run(_run())

    assert sub._invocation_count == 2
    assert sub._total_prompt_tokens == 200
    assert sub._total_completion_tokens == 80
    assert sub._total_tokens == 280
    assert sub._total_cached_input_tokens == 20


def test_billing_subscriber_handles_zero_usage() -> None:
    import asyncio
    from ecs_agent.accounting.models import LLMInvocationEvent, UsageRecord
    from ecs_agent.types import EntityId
    from examples.e2e.plan_and_task.billing import BillingSubscriber

    sub = BillingSubscriber()

    async def _run() -> None:
        usage = UsageRecord()  # all None
        event = LLMInvocationEvent(
            entity_id=EntityId(1),
            provider_id="test",
            model="m",
            usage=usage,
        )
        await sub._handle_llm_invocation(event)

    asyncio.run(_run())

    assert sub._invocation_count == 1
    assert sub._total_prompt_tokens == 0
    assert sub._total_completion_tokens == 0
    assert sub._total_tokens == 0


def test_billing_subscriber_subscribe_wires_event_bus(tmp_path: Path) -> None:
    from ecs_agent.core.event_bus import EventBus
    from ecs_agent.accounting.models import LLMInvocationEvent
    from examples.e2e.plan_and_task.billing import BillingSubscriber

    bus = EventBus()
    sub = BillingSubscriber()
    sub.subscribe(bus)

    assert LLMInvocationEvent in bus._handlers  # type: ignore[attr-defined]


async def test_billing_subscriber_wired_in_build_plan_task_world(tmp_path: Path) -> None:
    """build_plan_task_world + wiring billing subscriber does not raise."""
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.accounting.models import LLMInvocationEvent
    from examples.e2e.plan_and_task.billing import BillingSubscriber
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(responses=["ok"])
    world, _, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)

    sub = BillingSubscriber()
    sub.subscribe(world.event_bus)

    assert LLMInvocationEvent in world.event_bus._handlers  # type: ignore[attr-defined]


async def test_accounting_subscriber_wired_in_main(tmp_path: Path) -> None:
    """AccountingSubscriber can be subscribed to the world event_bus without error."""
    from ecs_agent.providers.fake_model import FakeModel
    from ecs_agent.accounting import AccountingSubscriber
    from ecs_agent.accounting.models import LLMInvocationEvent
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(responses=["ok"])
    world, _, _, _ = await build_plan_task_world(model=model, base_dir=tmp_path)

    acc = AccountingSubscriber()
    acc.subscribe(world.event_bus)

    assert LLMInvocationEvent in world.event_bus._handlers  # type: ignore[attr-defined]


async def test_qa_review_approved_auto_transitions_to_write_plan(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_QA_REVIEW"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_qa_review(state, adapter, "approved")

    assert result.phase == "WRITE_PLAN"


async def test_qa_review_revise_stays_in_draft_qa_review(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "DRAFT_QA_REVIEW"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_qa_review(state, adapter, "revise")

    assert result.phase == "DRAFT_QA_REVIEW"


async def test_plan_qa_review_approved_auto_transitions_to_plan_finalized(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "PLAN_QA_REVIEW"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_plan_qa_review(state, adapter, "approved")

    assert result.phase == "PLAN_FINALIZED"


async def test_plan_qa_review_revise_stays_in_plan_qa_review(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "PLAN_QA_REVIEW"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_plan_qa_review(state, adapter, "revise")

    assert result.phase == "PLAN_QA_REVIEW"


async def test_handle_write_plan_completed_transitions_write_plan_to_plan_qa_review(tmp_path: Path) -> None:
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "WRITE_PLAN"
    adapter.write_state(state)
    world, eid = await _bound_world_at(state.phase)
    ctrl = PlanController(world, eid)

    result = await ctrl.handle_write_plan_completed(state, adapter)

    assert result.phase == "PLAN_QA_REVIEW"


def test_provider_config_enable_store_defaults_false() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig

    config = ProviderConfig(
        provider_id="test",
        base_url="https://example.com",
        api_key="sk-test",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )

    assert config.enable_store is False


def test_provider_config_enable_store_can_be_set_true() -> None:
    from ecs_agent.providers.config import ApiFormat, ProviderConfig

    config = ProviderConfig(
        provider_id="test",
        base_url="https://example.com",
        api_key="sk-test",
        api_format=ApiFormat.OPENAI_RESPONSES,
        enable_store=True,
    )

    assert config.enable_store is True


def test_plan_controller_utcnow_isoformat_is_timezone_aware() -> None:
    import datetime

    world = World()
    controller = PlanController(world, world.create_entity())

    value = controller._utcnow_isoformat()  # type: ignore[attr-defined]
    parsed = datetime.datetime.fromisoformat(value)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == datetime.timedelta(0)


def test_task_exec_utcnow_isoformat_is_timezone_aware() -> None:
    import datetime
    from examples.e2e.plan_and_task.task_exec import TaskExec

    executor = TaskExec(state=_make_runtime_state())

    value = executor._utcnow_isoformat()  # type: ignore[attr-defined]
    parsed = datetime.datetime.fromisoformat(value)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == datetime.timedelta(0)


def _make_state_at_phase(phase: str) -> RuntimeState:
    return RuntimeState(
        workflow_id="test-wf",
        phase=phase,
        status="active",
        active_plan_file="plan/workflow_plan.md",
        current_task_id=None,
        review_verdicts=[],
        active_subagents=[],
        memory_refs=[],
        last_checkpoint=None,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        tasks=[],
    )


def _make_verdict(phase: str, verdict: str) -> ReviewVerdict:
    return ReviewVerdict(phase=phase, verdict=verdict, decided_at="2026-01-01T00:00:00")


async def _resume_at(
    tmp_path: Path, phase: str, verdict: str | None = None
) -> tuple[RuntimeState, list["ResumeAction"]]:
    """Persist a state at `phase` (optionally with a verdict) and restore it."""
    from examples.e2e.plan_and_task.main import resume_workflow

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-wf")
    state = _make_state_at_phase(phase)
    if verdict is not None:
        state.review_verdicts.append(_make_verdict(phase, verdict))
    adapter.write_draft("# Draft\n")
    adapter.write_plan("# Plan\n")
    adapter.write_state(state)
    world = World()
    eid = world.create_entity()
    loaded, _adapter, actions = await resume_workflow(
        world, eid, "test-wf", base_dir=tmp_path
    )
    return loaded, actions


async def test_resume_draft_qa_approved_triggers_plan_writer(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import ResumeAction

    loaded, actions = await _resume_at(tmp_path, "DRAFT_QA_REVIEW", "approved")

    assert ResumeAction.TRIGGER_PLAN_WRITER in actions
    assert loaded.phase == "WRITE_PLAN"


async def test_resume_write_plan_triggers_plan_writer(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import ResumeAction

    loaded, actions = await _resume_at(tmp_path, "WRITE_PLAN")

    assert ResumeAction.TRIGGER_PLAN_WRITER in actions
    assert loaded.phase == "WRITE_PLAN"


async def test_resume_plan_qa_approved_advances_to_finalized(tmp_path: Path) -> None:
    loaded, actions = await _resume_at(tmp_path, "PLAN_QA_REVIEW", "approved")

    assert actions == []
    assert loaded.phase == "PLAN_FINALIZED"


async def test_resume_draft_qa_revise_returns_no_triggers(tmp_path: Path) -> None:
    loaded, actions = await _resume_at(tmp_path, "DRAFT_QA_REVIEW", "revise")

    assert actions == []
    assert loaded.phase == "DRAFT_QA_REVIEW"


async def test_resume_draft_interview_returns_no_triggers(tmp_path: Path) -> None:
    loaded, actions = await _resume_at(tmp_path, "DRAFT_INTERVIEW")

    assert actions == []
    assert loaded.phase == "DRAFT_INTERVIEW"


async def test_resume_plan_qa_revise_returns_no_triggers(tmp_path: Path) -> None:
    loaded, actions = await _resume_at(tmp_path, "PLAN_QA_REVIEW", "revise")

    assert actions == []
    assert loaded.phase == "PLAN_QA_REVIEW"


@pytest.mark.asyncio
async def test_plan_resume_draft_qa_approved_injects_write_plan_message(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        PhaseComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, _adapter_ref, _runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "resume-draft-qa-approved"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter.write_draft("draft content")
    state = _make_state_at_phase("DRAFT_QA_REVIEW")
    state.workflow_id = workflow_id
    state.review_verdicts.append(_make_verdict("DRAFT_QA_REVIEW", "approved"))
    adapter.write_state(state)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:resume")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, f"/plan:resume {workflow_id}")

    assert result is not None
    assert "Error" not in result
    assert _runtime_state[0] is not None
    assert _runtime_state[0].phase == "WRITE_PLAN"
    phase_component = world.get_component(agent_id, PhaseComponent)
    assert phase_component is not None
    assert phase_component.phase == "WRITE_PLAN"
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    user_messages = [m for m in conv.messages if m.role == "user"]
    assert any("draft" in m.content.lower() or "plan" in m.content.lower() for m in user_messages)


@pytest.mark.asyncio
async def test_plan_resume_write_plan_phase_injects_write_plan_message(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, _adapter_ref, _runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "resume-write-plan"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter.write_draft("draft content")
    state = _make_state_at_phase("WRITE_PLAN")
    state.workflow_id = workflow_id
    adapter.write_state(state)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:resume")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, f"/plan:resume {workflow_id}")

    assert result is not None
    assert "Error" not in result
    assert _runtime_state[0] is not None
    assert _runtime_state[0].phase == "WRITE_PLAN"
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    user_messages = [m for m in conv.messages if m.role == "user"]
    assert any("draft" in m.content.lower() or "plan" in m.content.lower() for m in user_messages)


@pytest.mark.asyncio
async def test_plan_resume_plan_qa_approved_advances_to_finalized(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        PhaseComponent,
        UserPromptConfigComponent,
    )
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, _adapter_ref, _runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "resume-plan-qa-approved"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter.write_draft("draft content")
    state = _make_state_at_phase("PLAN_QA_REVIEW")
    state.workflow_id = workflow_id
    state.review_verdicts.append(_make_verdict("PLAN_QA_REVIEW", "approved"))
    adapter.write_state(state)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:resume")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, f"/plan:resume {workflow_id}")

    assert result is not None
    assert "Error" not in result
    assert _runtime_state[0] is not None
    assert _runtime_state[0].phase == "PLAN_FINALIZED"
    phase_component = world.get_component(agent_id, PhaseComponent)
    assert phase_component is not None
    assert phase_component.phase == "PLAN_FINALIZED"
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert [m.content for m in conv.messages if m.role == "user"] == [
        f"/plan:resume {workflow_id}"
    ]


@pytest.mark.asyncio
async def test_plan_resume_draft_interview_no_message_injected(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.scratchbook_adapter import PlanTaskScratchbookAdapter

    model = FakeModel(responses=["ok"])
    world, agent_id, _adapter_ref, _runtime_state = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    workflow_id = "resume-draft-interview"
    adapter = PlanTaskScratchbookAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    adapter.write_draft("draft content")
    state = _make_state_at_phase("DRAFT_INTERVIEW")
    state.workflow_id = workflow_id
    adapter.write_state(state)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(t.content for t in config.triggers if t.pattern == "/plan:resume")
    handler = config.script_handlers[handler_key]

    result = await handler(world, agent_id, f"/plan:resume {workflow_id}")

    assert result is not None
    assert "Error" not in result
    assert _runtime_state[0] is not None
    assert _runtime_state[0].phase == "DRAFT_INTERVIEW"
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert [m.content for m in conv.messages if m.role == "user"] == [
        f"/plan:resume {workflow_id}"
    ]


def test_upsert_verdict_same_phase_keeps_latest(tmp_path: Path) -> None:
    state = _make_runtime_state()
    v1 = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="revise", decided_at="2026-01-01T00:00:00")
    v2 = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="2026-01-01T00:01:00")
    state.upsert_verdict(v1)
    state.upsert_verdict(v2)
    advisor_verdicts = [v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"]
    assert len(advisor_verdicts) == 1
    assert advisor_verdicts[0].verdict == "approved"


def test_upsert_verdict_approved_clears_notes(tmp_path: Path) -> None:
    state = _make_runtime_state()
    v = ReviewVerdict(
        phase="DRAFT_ADVISOR_REVIEW",
        verdict="approved",
        decided_at="2026-01-01T00:00:00",
        notes="looks good",
    )
    state.upsert_verdict(v)
    assert state.review_verdicts[0].notes is None


def test_upsert_verdict_non_approved_keeps_notes(tmp_path: Path) -> None:
    state = _make_runtime_state()
    v = ReviewVerdict(
        phase="DRAFT_ADVISOR_REVIEW",
        verdict="revise",
        decided_at="2026-01-01T00:00:00",
        notes="needs work",
    )
    state.upsert_verdict(v)
    assert state.review_verdicts[0].notes == "needs work"


def test_upsert_verdict_different_phases_both_kept(tmp_path: Path) -> None:
    state = _make_runtime_state()
    v1 = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="2026-01-01T00:00:00")
    v2 = ReviewVerdict(phase="DRAFT_QA_REVIEW", verdict="approved", decided_at="2026-01-01T00:01:00")
    state.upsert_verdict(v1)
    state.upsert_verdict(v2)
    assert len(state.review_verdicts) == 2


def test_upsert_verdict_approved_removes_prior_non_approved_for_same_phase() -> None:
    state = _make_runtime_state()
    v_revise = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="revise", decided_at="2026-01-01T00:00:00", notes="needs work")
    v_blocked = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="blocked", decided_at="2026-01-01T00:00:30", notes="critical issue")
    v_approved = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="2026-01-01T00:01:00")
    state.upsert_verdict(v_revise)
    state.upsert_verdict(v_blocked)
    state.upsert_verdict(v_approved)
    phase_verdicts = [v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"]
    assert len(phase_verdicts) == 1
    assert phase_verdicts[0].verdict == "approved"


def test_upsert_verdict_approved_is_sticky_cannot_be_overwritten() -> None:
    """Once a phase has an approved verdict, subsequent upserts for that phase are ignored."""
    state = _make_runtime_state()
    v_approved = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="2026-01-01T00:00:00")
    v_revise = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="revise", decided_at="2026-01-01T00:01:00", notes="needs work")
    state.upsert_verdict(v_approved)
    state.upsert_verdict(v_revise)
    advisor_verdicts = [v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"]
    assert len(advisor_verdicts) == 1
    assert advisor_verdicts[0].verdict == "approved"
    assert advisor_verdicts[0].notes is None


def test_upsert_verdict_approved_sticky_across_all_phases() -> None:
    """Stickiness applies to every review phase, not just DRAFT_ADVISOR_REVIEW."""
    state = _make_runtime_state()
    for phase in ("DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW", "PLAN_QA_REVIEW"):
        v_ok = ReviewVerdict(phase=phase, verdict="approved", decided_at="2026-01-01T00:00:00")
        v_bad = ReviewVerdict(phase=phase, verdict="blocked", decided_at="2026-01-01T00:01:00", notes="blocked")
        state.upsert_verdict(v_ok)
        state.upsert_verdict(v_bad)
    for v in state.review_verdicts:
        assert v.verdict == "approved", f"Phase {v.phase} should remain approved, got {v.verdict!r}"


def test_review_verdict_has_no_plan_version_field() -> None:
    v = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="2026-01-01T00:00:00")
    assert not hasattr(v, "plan_version")


def test_review_verdict_has_no_citation_fields() -> None:
    v = ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="2026-01-01T00:00:00")
    assert not hasattr(v, "citations")
    assert not hasattr(v, "evidence_refs")


async def test_handle_advisor_review_sets_status_active_after_transition(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="status-lifecycle-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "status lifecycle workflow")
    assert state.status == "active"

    updated = await controller.handle_advisor_review(state, adapter, "revise")
    assert updated.status == "active"


async def test_handle_qa_review_approved_sets_status_active(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="status-qa-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "qa status workflow")
    state = await controller.handle_advisor_review(state, adapter, "approved")

    updated = await controller.handle_qa_review(state, adapter, "approved")
    assert updated.status == "active"


async def test_persisted_snapshot_derives_status(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.phase_sync import save_state

    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "complete"
    world, eid = await _bound_world_at(state.phase)
    controller = PlanController(world, eid)
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=state.workflow_id)

    # A gate phase persisted with an empty ledger is awaiting its review.
    await controller._advance("DRAFT_ADVISOR_REVIEW", reason="test")
    save_state(world, eid, state, adapter)
    assert state.status == "needs_review"

    state.review_verdicts = [
        ReviewVerdict(phase="DRAFT_ADVISOR_REVIEW", verdict="revise", decided_at="t")
    ]
    await controller._advance("DRAFT_INTERVIEW", reason="test")
    save_state(world, eid, state, adapter)
    assert state.status == "active"


async def test_controller_transition_sets_complete_then_active(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter as ArtifactAdapter,
    )

    statuses: list[str] = []

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="complete-active-test")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "complete active workflow")
    statuses.append(state.status)

    state = await controller.handle_advisor_review(state, adapter, "approved")
    statuses.append(state.status)

    state = await controller.handle_qa_review(state, adapter, "approved")
    statuses.append(state.status)

    assert "active" in statuses, f"Expected 'active' status at some point, got {statuses}"


async def test_advisor_qa_subagents_inherit_readonly_tools_only(tmp_path: Path) -> None:
    from ecs_agent.components import SubagentRegistryComponent
    from ecs_agent.providers import FakeModel
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    world, agent_id, _, _ = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )
    registry = world.get_component(agent_id, SubagentRegistryComponent)
    assert registry is not None, "SubagentRegistryComponent not found"

    for name in ("advisor", "qa", "plan_qa"):
        cfg = registry.subagents[name]
        assert set(cfg.inheritance_policy.inherit_tools) == {
            "read_file",
            "glob",
        }, f"{name} inherit_tools should be {{'read_file','glob'}}, got {cfg.inheritance_policy.inherit_tools}"
        assert (
            cfg.inheritance_policy.inherit_permissions is True
        ), f"{name} inherit_permissions should be True"

    plan_writer_cfg = registry.subagents["plan_writer"]
    assert (
        "write_file" in plan_writer_cfg.inheritance_policy.inherit_tools
    ), "plan_writer should have write_file in inherit_tools"


def test_advisor_prompt_contains_read_file_path_not_content() -> None:
    from examples.e2e.plan_and_task.prompts import build_advisor_prompt

    path = "scratchbook/wf-001/plan/draft.md"
    prompt = build_advisor_prompt(path)
    assert path in prompt, "prompt must contain the draft path"
    assert "read_file" in prompt, "prompt must mention read_file tool"
    assert "## Draft Content" not in prompt


def test_qa_prompt_contains_read_file_path_not_content() -> None:
    from examples.e2e.plan_and_task.prompts import build_qa_prompt

    path = "scratchbook/wf-001/plan/draft.md"
    prompt = build_qa_prompt(path, "approved")
    assert path in prompt, "prompt must contain the draft path"
    assert "read_file" in prompt, "prompt must mention read_file tool"
    assert "## Draft Content" not in prompt


def test_write_plan_prompt_contains_read_file_path_not_content() -> None:
    from examples.e2e.plan_and_task.prompts import build_write_plan_prompt

    path = "scratchbook/wf-001/plan/draft.md"
    plan_path = "scratchbook/wf-001/plan/workflow_plan.md"
    prompt = build_write_plan_prompt(path, plan_path)
    assert path in prompt, "prompt must contain the draft path"
    assert "read_file" in prompt, "prompt must mention read_file tool"
    assert "## Draft Content" not in prompt


def test_plan_qa_prompt_contains_read_file_path_not_content() -> None:
    from examples.e2e.plan_and_task.prompts import build_plan_qa_prompt

    path = "scratchbook/wf-001/plan/workflow_plan.md"
    prompt = build_plan_qa_prompt(path)
    assert path in prompt, "prompt must contain the plan path"
    assert "read_file" in prompt, "prompt must mention read_file tool"
    assert "## Plan Content" not in prompt


async def test_plan_qa_subagent_registered_with_plan_qa_system_prompt(tmp_path: Path) -> None:
    from ecs_agent.components import SubagentRegistryComponent
    from ecs_agent.providers import FakeModel
    from ecs_agent.types import CompletionResult, Message
    from examples.e2e.plan_and_task.main import build_plan_task_world
    from examples.e2e.plan_and_task.prompts import PLAN_QA_REVIEW_SYSTEM_PROMPT, QA_SYSTEM_PROMPT

    model = FakeModel(
        responses=[CompletionResult(message=Message(role="assistant", content="ok"))]
    )
    world, agent_id, _, _ = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )
    registry = world.get_component(agent_id, SubagentRegistryComponent)
    assert registry is not None, "SubagentRegistryComponent not found"
    assert "plan_qa" in registry.subagents, "plan_qa subagent must be registered"
    assert "qa" in registry.subagents, "qa subagent must still be registered"
    assert (
        registry.subagents["plan_qa"].system_prompt == PLAN_QA_REVIEW_SYSTEM_PROMPT
    ), "plan_qa must use PLAN_QA_REVIEW_SYSTEM_PROMPT"
    assert (
        registry.subagents["qa"].system_prompt == QA_SYSTEM_PROMPT
    ), "qa must use QA_SYSTEM_PROMPT"


def test_draft_interview_prompt_uses_separate_qa_categories() -> None:
    from examples.e2e.plan_and_task.prompts import DRAFT_INTERVIEW_SYSTEM_PROMPT

    assert (
        'category="plan_qa"' in DRAFT_INTERVIEW_SYSTEM_PROMPT
    ), "plan QA phase must call subagent with category='plan_qa'"
    assert (
        'category="qa"' in DRAFT_INTERVIEW_SYSTEM_PROMPT
    ), "draft QA phase must still call subagent with category='qa'"


def test_plan_main_agent_system_prompt_exists() -> None:
    from examples.e2e.plan_and_task.prompts import PLAN_MAIN_AGENT_SYSTEM_PROMPT

    assert isinstance(PLAN_MAIN_AGENT_SYSTEM_PROMPT, str)
    assert PLAN_MAIN_AGENT_SYSTEM_PROMPT.strip()


def test_draft_interview_system_prompt_is_alias() -> None:
    from examples.e2e.plan_and_task.prompts import (
        DRAFT_INTERVIEW_SYSTEM_PROMPT,
        PLAN_MAIN_AGENT_SYSTEM_PROMPT,
    )

    assert DRAFT_INTERVIEW_SYSTEM_PROMPT is PLAN_MAIN_AGENT_SYSTEM_PROMPT


def test_task_main_agent_system_prompt_exists() -> None:
    from examples.e2e.plan_and_task.prompts import (
        PLAN_MAIN_AGENT_SYSTEM_PROMPT,
        TASK_MAIN_AGENT_SYSTEM_PROMPT,
    )

    assert isinstance(TASK_MAIN_AGENT_SYSTEM_PROMPT, str)
    assert TASK_MAIN_AGENT_SYSTEM_PROMPT.strip()
    assert TASK_MAIN_AGENT_SYSTEM_PROMPT != PLAN_MAIN_AGENT_SYSTEM_PROMPT


def test_plan_and_task_prompts_embed_scratchbook_context() -> None:
    from examples.e2e.plan_and_task.prompts import (
        PLAN_MAIN_AGENT_SYSTEM_PROMPT,
        TASK_MAIN_AGENT_SYSTEM_PROMPT,
    )

    for prompt in (PLAN_MAIN_AGENT_SYSTEM_PROMPT, TASK_MAIN_AGENT_SYSTEM_PROMPT):
        assert "${_scratchbook_overview}" in prompt
        assert "${_scratchbook_artifacts}" in prompt


def test_user_facing_prompts_reply_in_user_language() -> None:
    """PLAN / TASK / IDLE agents (which see the user) must reply in the user's language."""
    from examples.e2e.plan_and_task.prompts import (
        IDLE_MAIN_AGENT_SYSTEM_PROMPT,
        PLAN_MAIN_AGENT_SYSTEM_PROMPT,
        TASK_MAIN_AGENT_SYSTEM_PROMPT,
    )

    for prompt in (
        PLAN_MAIN_AGENT_SYSTEM_PROMPT,
        TASK_MAIN_AGENT_SYSTEM_PROMPT,
        IDLE_MAIN_AGENT_SYSTEM_PROMPT,
    ):
        assert "## Language" in prompt, "user-facing prompt must carry a Language directive"
        assert "same language the user writes in" in prompt, (
            "user-facing prompt must instruct replying in the user's input language"
        )


def test_reviewer_and_writer_prompts_match_artifact_language() -> None:
    """Reviewer / writer subagents never see the user, so they match the artifact's language."""
    from examples.e2e.plan_and_task.prompts import (
        ADVISOR_SYSTEM_PROMPT,
        PLAN_QA_REVIEW_SYSTEM_PROMPT,
        QA_SYSTEM_PROMPT,
        WRITE_PLAN_SYSTEM_PROMPT,
    )

    for prompt in (
        ADVISOR_SYSTEM_PROMPT,
        QA_SYSTEM_PROMPT,
        PLAN_QA_REVIEW_SYSTEM_PROMPT,
        WRITE_PLAN_SYSTEM_PROMPT,
    ):
        assert "## Language" in prompt, "reviewer/writer prompt must carry a Language directive"
        assert "You do not see the user" in prompt, (
            "reviewer/writer prompt must key language off the artifact, not the user"
        )
        assert "read_file" in prompt


def test_language_directives_preserve_machine_tokens() -> None:
    """The Language directives must protect machine-parsed / schema tokens and stay template-safe."""
    from examples.e2e.plan_and_task import prompts as p

    # The verbatim-English guardrail names the load-bearing tokens.
    main = p.PLAN_MAIN_AGENT_SYSTEM_PROMPT
    assert "VERDICT:" in main
    for token in ("TASK_RUNNING", "plan_writer", "/plan:start", "finalized"):
        assert token in main, f"main Language directive must protect {token!r}"

    # Reviewers must still be told to emit the exact machine-parsed verdict line.
    assert "VERDICT: approved | revise | blocked" in p.ADVISOR_SYSTEM_PROMPT

    # Directive constants must contain no $ / { } so they stay inert inside the
    # f-string prompts and the later string.Template ${...} substitution pass.
    for const in (
        p._LANGUAGE_MAIN_DIRECTIVE,
        p._LANGUAGE_REVIEWER_DIRECTIVE,
        p._LANGUAGE_PLAN_WRITER_DIRECTIVE,
        p._LANGUAGE_IDLE_DIRECTIVE,
    ):
        assert not (set("${}") & set(const)), "Language directive must not break template rendering"


async def test_build_plan_task_world_uses_plan_main_agent_system_prompt(tmp_path: Path) -> None:
    from ecs_agent.prompts.contracts import SystemPromptConfigSpec
    from ecs_agent.providers.fake_model import FakeModel
    from examples.e2e.plan_and_task.main import build_plan_task_world

    model = FakeModel(responses=["ok"])
    world, agent_id, _, _ = await build_plan_task_world(
        model=model, base_dir=tmp_path
    )

    spec = world.get_component(agent_id, SystemPromptConfigSpec)
    assert spec is not None
    assert spec.template_source.inline == "${_phase_prompt}"


async def test_phase_graph_bound_in_world(tmp_path: Path) -> None:
    """build_plan_task_world binds the phase graph."""
    from ecs_agent.components import PhaseComponent

    world, agent_id, _, _ = await _build_test_world(tmp_path)

    component = world.get_component(agent_id, PhaseComponent)
    assert component is not None
    assert component.phase == "IDLE"
    assert component.agent_key == "main"


@pytest.mark.asyncio
async def test_task_start_swaps_system_prompt(tmp_path: Path) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        PhaseComponent,
        RenderedSystemPromptComponent,
    )
    from ecs_agent.prompts.contracts import SystemPromptConfigSpec
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_state(state)
    runtime_state[0] = state
    await force(world, agent_id, state.phase, reason="test setup")
    world.add_component(agent_id, build_scratchbook_prompt_config(state.workflow_id))
    world.add_component(agent_id, RenderedSystemPromptComponent(text="stale"))

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="/task:start"))

    await UserPromptNormalizationSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    spec = world.get_component(agent_id, SystemPromptConfigSpec)
    assert spec is not None
    assert spec.template_source.inline == "${_phase_prompt}"
    phase_component = world.get_component(agent_id, PhaseComponent)
    assert phase_component is not None
    assert phase_component.phase == "TASK_RUNNING"
    rendered = world.get_component(agent_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "task execution main agent" in rendered.text
    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "TASK_RUNNING"
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "/task:start"

@pytest.mark.asyncio
async def test_task_resume_swaps_system_prompt(tmp_path: Path) -> None:
    from ecs_agent.components import (
        ConversationComponent,
        PhaseComponent,
        RenderedSystemPromptComponent,
    )
    from ecs_agent.prompts.contracts import SystemPromptConfigSpec
    from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.status = "blocked"
    state.current_task_id = "task-001"
    state.tasks = [
        TaskRecord(
            task_id="task-001",
            title="First Task",
            status="blocked",
            retry_count=1,
            last_error="waiting on input",
        )
    ]
    adapter.write_state(state)
    runtime_state[0] = state
    await force(world, agent_id, state.phase, reason="test setup")
    world.add_component(agent_id, build_scratchbook_prompt_config(state.workflow_id))
    world.add_component(agent_id, RenderedSystemPromptComponent(text="stale"))

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="/task:resume"))

    await UserPromptNormalizationSystem().process(world)
    await SystemPromptRenderSystem().process(world)

    spec = world.get_component(agent_id, SystemPromptConfigSpec)
    assert spec is not None
    assert spec.template_source.inline == "${_phase_prompt}"
    phase_component = world.get_component(agent_id, PhaseComponent)
    assert phase_component is not None
    assert phase_component.phase == "TASK_RUNNING"
    rendered = world.get_component(agent_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "task execution main agent" in rendered.text
    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "TASK_RUNNING"
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "/task:resume"


@pytest.mark.asyncio
async def test_task_resume_renders_task_prompt_before_reasoning_same_tick(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent
    from ecs_agent.core import Runner
    from ecs_agent.types import CompletionResult, Message, ToolSchema
    from examples.e2e.plan_and_task.main import build_plan_task_world

    class CapturingModel:
        model_id = "capturing"

        def __init__(self) -> None:
            self.seen_messages: list[list[Message]] = []

        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolSchema] | None = None,
            stream: bool = False,
            response_format: dict[str, object] | None = None,
        ) -> CompletionResult:
            del tools, response_format
            if stream:
                raise ValueError("streaming is not used in this regression test")
            self.seen_messages.append(list(messages))
            return CompletionResult(message=Message(role="assistant", content="ready"))

    model = CapturingModel()
    world, agent_id, adapter_ref, runtime_state = await build_plan_task_world(
        model=model,
        base_dir=tmp_path,
    )
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.status = "blocked"
    state.current_task_id = "task-001"
    state.tasks = [
        TaskRecord(
            task_id="task-001",
            title="First Task",
            status="blocked",
            retry_count=1,
            last_error="waiting on input",
        )
    ]
    adapter.write_state(state)
    adapter_ref[0] = adapter
    runtime_state[0] = state
    await force(world, agent_id, state.phase, reason="test setup")
    world.add_component(agent_id, build_scratchbook_prompt_config(state.workflow_id))

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="/task:resume"))

    await Runner().run(world, max_ticks=1)

    assert model.seen_messages
    system_messages = [
        message.content for message in model.seen_messages[0] if message.role == "system"
    ]
    assert system_messages
    assert "task execution main agent" in system_messages[0]
    assert "${_scratchbook_" not in system_messages[0]


@pytest.mark.asyncio
async def test_task_start_auto_loads_state_from_workflow_id(tmp_path: Path) -> None:
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_FINALIZED"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_state(state)

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:start {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert runtime_state[0].phase == "TASK_RUNNING"


@pytest.mark.asyncio
async def test_task_resume_auto_loads_blocked_state_from_workflow_id(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.status = "blocked"
    state.current_task_id = "task-001"
    state.last_checkpoint = "waiting on external input"
    state.tasks = [
        TaskRecord(
            task_id="task-001",
            title="First Task",
            status="blocked",
            retry_count=1,
            last_error="waiting on external input",
        )
    ]
    adapter.write_state(state)

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:resume {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert runtime_state[0].phase == "TASK_RUNNING"


@pytest.mark.asyncio
async def test_task_resume_auto_loads_running_state_from_workflow_id(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent, RenderedUserPromptComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    state.current_task_id = "task-001"
    state.last_checkpoint = "interrupted while running"
    state.tasks = [
        TaskRecord(
            task_id="task-001",
            title="First Task",
            status="running",
            retry_count=0,
            last_error=None,
        )
    ]
    adapter.write_state(state)

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:resume {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert runtime_state[0].phase == "TASK_RUNNING"
    assert runtime_state[0].status == "active"
    rendered = world.get_component(agent_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert rendered.text.startswith("Task resumed:")


@pytest.mark.asyncio
async def test_task_resume_without_state_and_no_workflow_id_returns_error(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent, RenderedUserPromptComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, _adapter, runtime_state = await _build_test_world(tmp_path)
    assert runtime_state[0] is None

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="/task:resume"))

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is None
    rendered = world.get_component(agent_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert "workflow_id" in rendered.text.lower() or "Error" in rendered.text


@pytest.mark.asyncio
async def test_task_start_without_state_and_no_workflow_id_returns_error(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import ConversationComponent, RenderedUserPromptComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, _adapter, runtime_state = await _build_test_world(tmp_path)
    assert runtime_state[0] is None

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content="/task:start"))

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is None
    rendered = world.get_component(agent_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert "workflow_id" in rendered.text.lower() or "Error" in rendered.text


# ---------------------------------------------------------------------------
# Phase graph definition (Stage-2 migration)
# ---------------------------------------------------------------------------


def test_phase_graph_terminal_and_resume_policy() -> None:
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    phases = PLAN_TASK_PHASE_GRAPH.phases_by_id
    assert phases["TASK_COMPLETED"].terminal and phases["TASK_ABORTED"].terminal
    assert phases["TASK_RUNNING"].on_resume == "TASK_BLOCKED"
    assert PLAN_TASK_PHASE_GRAPH.initial == "IDLE"


def test_phase_graph_approval_gates_encode_current_routing() -> None:
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    phases = PLAN_TASK_PHASE_GRAPH.phases_by_id
    advisor = phases["DRAFT_ADVISOR_REVIEW"].approval
    assert advisor is not None
    assert advisor.verdicts == {"approved": None, "revise": None, "blocked": None}
    qa = phases["DRAFT_QA_REVIEW"].approval
    assert qa is not None
    assert qa.verdicts == {"approved": "WRITE_PLAN", "revise": None, "blocked": None}
    plan_qa = phases["PLAN_QA_REVIEW"].approval
    assert plan_qa is not None
    assert plan_qa.verdicts == {"approved": "PLAN_FINALIZED", "revise": None, "blocked": None}


def test_phase_graph_prompt_bindings_match_legacy_profiles() -> None:
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH
    from examples.e2e.plan_and_task.prompts import (
        IDLE_MAIN_AGENT_SYSTEM_PROMPT,
        PLAN_MAIN_AGENT_SYSTEM_PROMPT,
        TASK_MAIN_AGENT_SYSTEM_PROMPT,
    )

    phases = PLAN_TASK_PHASE_GRAPH.phases_by_id
    assert phases["IDLE"].prompts["main"] == IDLE_MAIN_AGENT_SYSTEM_PROMPT
    for pid in ("DRAFT_INTERVIEW", "DRAFT_ADVISOR_REVIEW", "DRAFT_QA_REVIEW",
                "WRITE_PLAN", "PLAN_QA_REVIEW", "PLAN_FINALIZED", "TASK_READY"):
        assert phases[pid].prompts["main"] == PLAN_MAIN_AGENT_SYSTEM_PROMPT, pid
    for pid in ("TASK_RUNNING", "TASK_BLOCKED", "TASK_REPLAN",
                "TASK_COMPLETED", "TASK_ABORTED"):
        assert phases[pid].prompts["main"] == TASK_MAIN_AGENT_SYSTEM_PROMPT, pid


async def test_task_exec_transition_persists_running_status(tmp_path: Path) -> None:
    """Entering TASK_RUNNING through TaskExec persists status="active"."""
    from examples.e2e.plan_and_task.phase_sync import save_state
    from examples.e2e.plan_and_task.task_exec import TaskExec

    state = _make_runtime_state()
    state.phase = "PLAN_FINALIZED"
    state.status = "ready"
    world, eid = await _bound_world_at("PLAN_FINALIZED")
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=state.workflow_id)
    task_exec = TaskExec(state=state, world=world, entity_id=eid)
    updated = await task_exec._transition_to_running(state)
    save_state(world, eid, updated, adapter)
    assert updated.phase == "TASK_RUNNING"
    assert updated.status == "active"


# ── Review hardening: sticky-before-write + reachability (P1-2, P1-3) ──────────


async def test_late_verdict_after_sticky_approval_is_fully_ignored(
    tmp_path: Path,
) -> None:
    """Probe P1-2: a late verdict after sticky approval leaves no trace anywhere.

    State keeps the approved verdict, the persisted review artifact stays
    "approved", the PhaseApprovalsComponent ledger gains no "blocked" record,
    and the phase does not change.
    """
    from ecs_agent.components import PhaseApprovalsComponent

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="sticky-late-verdict")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "sticky late verdict workflow")

    state = await controller.handle_advisor_review(state, adapter, "approved", notes="LGTM")
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    state = await controller.handle_advisor_review(
        state, adapter, "blocked", notes="stale subagent completion"
    )

    advisor_verdicts = [
        v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"
    ]
    assert len(advisor_verdicts) == 1
    assert advisor_verdicts[0].verdict == "approved"
    artifact_payload = json.loads(
        (adapter.review_dir / "draft_advisor_review_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact_payload["verdict"] == "approved"
    ledger = world.get_component(eid, PhaseApprovalsComponent)
    assert ledger is not None
    assert [record["verdict"] for record in ledger.records] == ["approved"]
    assert state.phase == "DRAFT_ADVISOR_REVIEW"


async def test_out_of_phase_verdict_is_rejected_without_writes(
    tmp_path: Path,
) -> None:
    """Probe P1-3: a verdict for a non-current, non-adjacent phase is rejected.

    handle_plan_qa_review("approved") while in DRAFT_INTERVIEW raises
    ValueError and writes nothing: no review artifact, no state verdict,
    no phase change.
    """
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="out-of-phase-verdict")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)
    state = await controller.handle_plan_start(adapter, "out of phase verdict workflow")
    assert state.phase == "DRAFT_INTERVIEW"

    with pytest.raises(ValueError, match="Cannot record PLAN_QA_REVIEW verdict"):
        await controller.handle_plan_qa_review(state, adapter, "approved")

    assert not (adapter.review_dir / "plan_qa_review_verdict.json").exists()
    assert state.review_verdicts == []
    assert state.phase == "DRAFT_INTERVIEW"


async def test_replan_scope_change_advisor_review_still_accepted(
    tmp_path: Path,
) -> None:
    """Legit adjacency guard: TASK_REPLAN -> DRAFT_ADVISOR_REVIEW stays accepted.

    The scope-changed replan path records an advisor verdict from TASK_REPLAN
    (adjacent in the graph) and enters DRAFT_ADVISOR_REVIEW.
    """
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_REPLAN"
    state.status = "active"
    world, eid = await _bound_world_at(state.phase)
    controller = PlanController(world, eid)

    result = await controller.handle_advisor_review(
        state, adapter, "revise", notes="scope changed"
    )

    assert result.phase == "DRAFT_ADVISOR_REVIEW"
    assert any(
        v.phase == "DRAFT_ADVISOR_REVIEW" and v.verdict == "revise"
        for v in result.review_verdicts
    )


async def test_task_start_auto_load_replays_approved_plan_qa_gate(
    tmp_path: Path,
) -> None:
    """Probe P1-1: /task:start auto-load replays an approved PLAN_QA_REVIEW gate.

    Persisted state at PLAN_QA_REVIEW with an approved PLAN_QA_REVIEW verdict
    reconciles to PLAN_FINALIZED on auto-load, then task init proceeds
    PLAN_FINALIZED -> TASK_READY -> TASK_RUNNING with the queue initialized.
    """
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_draft("# Draft\n\nApproved draft content.\n")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_QA_REVIEW"
    state.status = "needs_review"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_state(state)

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:start {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].workflow_id == state.workflow_id
    assert runtime_state[0].phase == "TASK_RUNNING"
    assert runtime_state[0].current_task_id == "task-001"
    assert len(runtime_state[0].tasks) == 2


async def test_task_start_auto_load_unapproved_plan_qa_stays_blocked(
    tmp_path: Path,
) -> None:
    """Auto-load at PLAN_QA_REVIEW with a "revise" verdict must not advance.

    The gate maps "revise" to no target, so reconcile is a no-op and
    /task:start surfaces the existing "Cannot initialize task queue" error
    with the phase unchanged.
    """
    from ecs_agent.components import (
        ConversationComponent,
        RenderedUserPromptComponent,
    )
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_draft("# Draft\n\nDraft pending plan revision.\n")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_QA_REVIEW"
    state.status = "needs_review"
    state.review_verdicts = [
        ReviewVerdict(
            phase="PLAN_QA_REVIEW",
            verdict="revise",
            decided_at="2026-01-01T00:00:00",
        )
    ]
    adapter.write_state(state)

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:start {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "PLAN_QA_REVIEW"
    rendered = world.get_component(agent_id, RenderedUserPromptComponent)
    assert rendered is not None
    assert "Cannot initialize task queue" in rendered.text


# ── Persisted graph structure hash for restore drift detection (P2-1) ──────────


async def test_plan_start_state_carries_current_graph_hash_and_round_trips(
    tmp_path: Path,
) -> None:
    """A fresh workflow is stamped with the graph's structure hash.

    handle_plan_start writes the hash into RuntimeState and it survives the
    adapter write/read round trip.
    """
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="graph-hash-fresh")
    world, eid = await _bound_world_at("IDLE")
    controller = PlanController(world, eid)

    state = await controller.handle_plan_start(adapter, "graph hash workflow")

    assert state.graph_hash == PLAN_TASK_PHASE_GRAPH.structure_hash
    loaded = adapter.read_state()
    assert loaded.graph_hash == PLAN_TASK_PHASE_GRAPH.structure_hash


async def test_load_without_graph_hash_key_is_backward_compatible(
    tmp_path: Path,
) -> None:
    """Persisted JSON written before the graph_hash field still loads.

    The key is absent from the payload; the load flow succeeds and leaves the
    state stamped with the current structure hash.
    """
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_draft("# Draft\n\nLegacy state without graph_hash.\n")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_QA_REVIEW"
    state.status = "needs_review"
    adapter.write_state(state)

    state_path = adapter.state_dir / "runtime_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload.pop("graph_hash", None)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert "graph_hash" not in json.loads(state_path.read_text(encoding="utf-8"))

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:start {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "PLAN_QA_REVIEW"
    assert runtime_state[0].graph_hash == PLAN_TASK_PHASE_GRAPH.structure_hash
    persisted = adapter.read_state()
    assert persisted.graph_hash == PLAN_TASK_PHASE_GRAPH.structure_hash


async def test_load_with_stale_graph_hash_updates_state_to_current_hash(
    tmp_path: Path,
) -> None:
    """A stale persisted hash with a surviving phase loads successfully.

    bind_phase_graph detects the structural drift (logs
    phase_graph_structure_changed) and the load flow re-stamps the state with
    the current structure hash.
    """
    from ecs_agent.components import ConversationComponent
    from ecs_agent.systems.user_prompt_normalization_system import (
        UserPromptNormalizationSystem,
    )
    from ecs_agent.types import Message
    from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    adapter.write_draft("# Draft\n\nState persisted under an older graph.\n")
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    state = _make_runtime_state()
    state.phase = "PLAN_QA_REVIEW"
    state.status = "needs_review"
    state.graph_hash = "stale-structure-hash"
    adapter.write_state(state)
    stale_payload = json.loads(
        (adapter.state_dir / "runtime_state.json").read_text(encoding="utf-8")
    )
    assert stale_payload["graph_hash"] == "stale-structure-hash"

    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(
        Message(role="user", content=f"/task:start {state.workflow_id}")
    )

    await UserPromptNormalizationSystem().process(world)

    assert runtime_state[0] is not None
    assert runtime_state[0].phase == "PLAN_QA_REVIEW"
    assert runtime_state[0].graph_hash == PLAN_TASK_PHASE_GRAPH.structure_hash
    persisted = adapter.read_state()
    assert persisted.graph_hash == PLAN_TASK_PHASE_GRAPH.structure_hash


# --- Gate-derived review vocabulary (state-simplification Task 1) -----------


def test_required_review_phases_and_verdicts_derived_from_gates() -> None:
    from examples.e2e.plan_and_task.phase_graph import (
        REQUIRED_REVIEW_PHASES,
        REVIEW_VERDICTS,
    )

    assert REQUIRED_REVIEW_PHASES == (
        "DRAFT_ADVISOR_REVIEW",
        "DRAFT_QA_REVIEW",
        "PLAN_QA_REVIEW",
    )
    assert REVIEW_VERDICTS == ("approved", "revise", "blocked")


def test_gate_derivations_track_added_gates() -> None:
    from ecs_agent.phases import ApprovalGate, PhaseSpec, build_graph
    from examples.e2e.plan_and_task.phase_graph import (
        derive_required_review_phases,
        derive_review_verdicts,
    )

    variant = build_graph(
        "variant",
        initial="A",
        phases=[
            PhaseSpec(phase_id="A", prompts={"main": "p"}, to=("B",)),
            PhaseSpec(
                phase_id="B",
                prompts={"main": "p"},
                to=("C",),
                approval=ApprovalGate(verdicts={"ship": "C", "hold": None}),
            ),
            PhaseSpec(phase_id="C", prompts={"main": "p"}, terminal=True),
        ],
    )
    assert derive_required_review_phases(variant) == ("B",)
    assert derive_review_verdicts(variant) == ("ship", "hold")


def test_missing_approvals_fold_matches_gate_set() -> None:
    from examples.e2e.plan_and_task.phase_graph import REQUIRED_REVIEW_PHASES
    from examples.e2e.plan_and_task.state_models import missing_approvals

    verdicts = [
        ReviewVerdict(
            phase="DRAFT_ADVISOR_REVIEW", verdict="approved", decided_at="t"
        ),
        ReviewVerdict(phase="DRAFT_QA_REVIEW", verdict="revise", decided_at="t"),
    ]
    assert missing_approvals(verdicts) == ["DRAFT_QA_REVIEW", "PLAN_QA_REVIEW"]
    assert missing_approvals([]) == list(REQUIRED_REVIEW_PHASES)


# --- Graph-derived finalize walk (state-simplification Task 2) --------------


def test_finalize_hops_derived_from_graph_match_legacy_routing() -> None:
    from examples.e2e.plan_and_task.controller import _FINALIZE_HOPS

    assert _FINALIZE_HOPS == {
        "DRAFT_INTERVIEW": "DRAFT_ADVISOR_REVIEW",
        "DRAFT_ADVISOR_REVIEW": "DRAFT_QA_REVIEW",
        "DRAFT_QA_REVIEW": "WRITE_PLAN",
        "WRITE_PLAN": "PLAN_QA_REVIEW",
        "PLAN_QA_REVIEW": "PLAN_FINALIZED",
        "PLAN_FINALIZED": "TASK_READY",
    }


def test_finalize_walk_rejects_gate_off_the_happy_path() -> None:
    from ecs_agent.phases import ApprovalGate, PhaseSpec, build_graph
    from examples.e2e.plan_and_task.controller import _derive_finalize_hops

    variant = build_graph(
        "variant",
        initial="START",
        phases=[
            PhaseSpec(phase_id="START", prompts={"main": "p"}, to=("A", "B")),
            PhaseSpec(
                phase_id="A",
                prompts={"main": "p"},
                to=("TASK_READY",),
                approval=ApprovalGate(verdicts={"approved": "TASK_READY"}),
            ),
            PhaseSpec(
                phase_id="B",
                prompts={"main": "p"},
                to=("TASK_READY",),
                approval=ApprovalGate(verdicts={"approved": "TASK_READY"}),
            ),
            PhaseSpec(phase_id="TASK_READY", prompts={"main": "p"}, terminal=True),
        ],
    )
    with pytest.raises(AssertionError, match="off the finalize walk"):
        _derive_finalize_hops(variant)


# --- Derived status (state-simplification Task 4) ---------------------------


def test_derive_status_is_a_pure_function_of_phase_and_domain_fields() -> None:
    from examples.e2e.plan_and_task.phase_sync import derive_status

    assert derive_status("IDLE", abort_reason=None, review_verdicts=[]) == "active"
    assert (
        derive_status("TASK_COMPLETED", abort_reason=None, review_verdicts=[])
        == "completed"
    )
    assert (
        derive_status("TASK_ABORTED", abort_reason="user stop", review_verdicts=[])
        == "aborted"
    )
    assert derive_status("TASK_READY", abort_reason=None, review_verdicts=[]) == "ready"
    assert (
        derive_status("TASK_BLOCKED", abort_reason=None, review_verdicts=[])
        == "blocked"
    )
    assert (
        derive_status("DRAFT_QA_REVIEW", abort_reason=None, review_verdicts=[])
        == "needs_review"
    )
    verdicts = [
        ReviewVerdict(phase="DRAFT_QA_REVIEW", verdict="revise", decided_at="t")
    ]
    assert (
        derive_status("DRAFT_QA_REVIEW", abort_reason=None, review_verdicts=verdicts)
        == "active"
    )


async def _resume_and_read_status(
    tmp_path: Path, state: "RuntimeState"
) -> tuple[str, str]:
    """Drive /plan:resume through the production handler; return (live, persisted) status."""
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.types import Message

    world, agent_id, adapter, runtime_state = await _build_test_world(tmp_path)
    state.workflow_id = adapter.workflow_id
    (adapter.plan_dir / "workflow_plan.md").write_text("# Plan\n", encoding="utf-8")
    adapter.write_draft("# Draft\n")
    adapter.write_state(state)

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]
    conversation = world.get_component(agent_id, ConversationComponent)
    assert conversation is not None
    user_text = f"/plan:resume {state.workflow_id}"
    conversation.messages.append(Message(role="user", content=user_text))

    result = await handler(world, agent_id, user_text)
    assert result is not None
    loaded = runtime_state[0]
    assert loaded is not None
    return loaded.status, adapter.read_state().status


async def test_resume_preserves_ready_status(tmp_path: Path) -> None:
    state = _make_state_at_phase("TASK_READY")
    state.status = "ready"
    live, persisted = await _resume_and_read_status(tmp_path, state)
    assert live == "ready"
    assert persisted == "ready"


async def test_resume_preserves_aborted_status_with_abort_reason(
    tmp_path: Path,
) -> None:
    state = _make_state_at_phase("TASK_ABORTED")
    state.status = "aborted"
    state.abort_reason = "user requested stop"
    live, persisted = await _resume_and_read_status(tmp_path, state)
    assert live == "aborted"
    assert persisted == "aborted"


async def test_resume_preserves_needs_review_status(tmp_path: Path) -> None:
    # Scope-changed replan lands in DRAFT_ADVISOR_REVIEW with a cleared ledger.
    state = _make_state_at_phase("DRAFT_ADVISOR_REVIEW")
    state.status = "needs_review"
    live, persisted = await _resume_and_read_status(tmp_path, state)
    assert live == "needs_review"
    assert persisted == "needs_review"


async def test_second_resume_of_blocked_workflow_stays_blocked(
    tmp_path: Path,
) -> None:
    # A persisted TASK_BLOCKED state is NOT demoted again on re-bind, so the
    # old demotion override never fired and the status was stomped to active.
    state = _make_state_at_phase("TASK_BLOCKED")
    state.status = "blocked"
    live, persisted = await _resume_and_read_status(tmp_path, state)
    assert live == "blocked"
    assert persisted == "blocked"


# --- Persist-time snapshot (state-simplification Task 5) --------------------


async def test_save_state_snapshots_phase_and_derives_status(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.phase_sync import save_state

    state = _make_runtime_state()
    state.phase = "DRAFT_INTERVIEW"
    state.status = "complete"
    world, eid = await _bound_world_at("DRAFT_INTERVIEW")
    controller = PlanController(world, eid)
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=state.workflow_id)
    adapter.write_draft("# Draft\n")

    # advance alone leaves the persisted mirror untouched...
    await controller._advance("DRAFT_ADVISOR_REVIEW", reason="test")
    assert state.phase == "DRAFT_INTERVIEW"

    # ...save_state stamps phase + derived status at persist time.
    save_state(world, eid, state, adapter)
    assert state.phase == "DRAFT_ADVISOR_REVIEW"
    assert state.status == "needs_review"
    assert adapter.read_state().status == "needs_review"


# --- Single restore entrypoint (state-simplification Task 6) ----------------


async def test_resume_workflow_is_the_single_restore_entrypoint(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.main import resume_workflow

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_RUNNING"
    state.status = "active"
    adapter.write_plan("# plan")
    adapter.write_state(state)

    world = World()
    eid = world.create_entity()
    loaded, loaded_adapter, actions = await resume_workflow(
        world, eid, adapter.workflow_id, base_dir=tmp_path
    )

    # on_resume demotion + persist-time snapshot + gate replay, in one path.
    assert loaded.phase == "TASK_BLOCKED"
    assert loaded.status == "blocked"
    assert actions == []
    assert loaded_adapter.workflow_id == adapter.workflow_id
    assert adapter.read_state().phase == "TASK_BLOCKED"


async def test_resume_and_task_start_share_one_restore_path(tmp_path: Path) -> None:
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.types import Message

    async def _drive(command_pattern: str, tmp: Path) -> RuntimeState:
        world, agent_id, adapter, runtime_state = await _build_test_world(tmp)
        state = _make_state_at_phase("PLAN_QA_REVIEW")
        state.workflow_id = adapter.workflow_id
        state.review_verdicts = [
            ReviewVerdict(phase="PLAN_QA_REVIEW", verdict="approved", decided_at="t")
        ]
        (adapter.plan_dir / "workflow_plan.md").write_text("# Plan\n", encoding="utf-8")
        adapter.write_draft("# Draft\n")
        adapter.write_state(state)
        config = world.get_component(agent_id, UserPromptConfigComponent)
        assert config is not None
        handler_key = next(
            t.content for t in config.triggers if t.pattern == command_pattern
        )
        handler = config.script_handlers[handler_key]
        conv = world.get_component(agent_id, ConversationComponent)
        assert conv is not None
        user_text = f"{command_pattern} {state.workflow_id}"
        conv.messages.append(Message(role="user", content=user_text))
        await handler(world, agent_id, user_text)
        loaded = runtime_state[0]
        assert loaded is not None
        return loaded

    via_resume = await _drive("/plan:resume", tmp_path / "a")
    via_task_start = await _drive("/task:start", tmp_path / "b")

    # Same persisted input, same post-restore state through either command.
    assert via_resume.phase == via_task_start.phase == "PLAN_FINALIZED"
    assert via_resume.status == via_task_start.status
    assert [v.verdict for v in via_resume.review_verdicts] == [
        v.verdict for v in via_task_start.review_verdicts
    ]


# --- Completed-task carry-forward on re-init (state-simplification Task 8) --


async def test_reinit_carries_completed_tasks_forward(tmp_path: Path) -> None:
    """/task:start on a blocked workflow must not re-run completed tasks."""
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.tasks = [
        TaskRecord(task_id="task-001", title="Prior run", status="completed")
    ]
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at("TASK_BLOCKED")

    updated = await TaskExec(
        state=state, world=world, entity_id=eid
    ).initialize_task_queue(state, adapter)

    statuses = {t.task_id: t.status for t in updated.tasks}
    assert statuses["task-001"] == "completed"
    assert statuses["task-002"] == "pending"
    assert updated.current_task_id == "task-002"
    assert updated.phase == "TASK_RUNNING"


async def test_reinit_ignores_orphaned_completed_ids(tmp_path: Path) -> None:
    """Completed ids from a replanned-away task must not break queue building."""
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.tasks = [
        TaskRecord(
            task_id="task-removed-by-replan", title="Old task", status="completed"
        )
    ]
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at("TASK_BLOCKED")

    updated = await TaskExec(
        state=state, world=world, entity_id=eid
    ).initialize_task_queue(state, adapter)

    assert all(t.status == "pending" for t in updated.tasks)
    assert updated.current_task_id == "task-001"


async def test_reinit_with_all_tasks_completed_finishes_workflow(
    tmp_path: Path,
) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_BLOCKED"
    state.tasks = [
        TaskRecord(task_id="task-001", title="Prior run", status="completed"),
        TaskRecord(task_id="task-002", title="Prior run", status="completed"),
    ]
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at("TASK_BLOCKED")

    updated = await TaskExec(
        state=state, world=world, entity_id=eid
    ).initialize_task_queue(state, adapter)

    assert updated.current_task_id is None
    assert updated.phase == "TASK_COMPLETED"
    assert updated.status == "completed"


async def test_fresh_init_still_starts_at_first_task(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.task_exec import TaskExec

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-workflow-001")
    state = _make_runtime_state()
    state.phase = "TASK_READY"
    state.review_verdicts = _make_approved_verdicts()
    adapter.write_plan(_VALID_FINALIZED_TASK_PLAN)
    adapter.write_state(state)
    world, eid = await _bound_world_at("TASK_READY")

    updated = await TaskExec(
        state=state, world=world, entity_id=eid
    ).initialize_task_queue(state, adapter)

    assert all(t.status == "pending" for t in updated.tasks)
    assert updated.current_task_id == "task-001"
    assert updated.phase == "TASK_RUNNING"


# --- Derived completed_task_ids (state-simplification Task 9) ---------------


def test_completed_task_ids_is_derived_from_task_status() -> None:
    state = _make_runtime_state()
    state.tasks = [
        TaskRecord(task_id="task-001", title="Done", status="completed"),
        TaskRecord(task_id="task-002", title="Pending", status="pending"),
    ]
    assert state.completed_task_ids == ["task-001"]
    payload = state.to_dict()
    assert payload["completed_task_ids"] == ["task-001"]


def test_from_dict_migrates_legacy_completed_ids_onto_tasks() -> None:
    # A pre-carry-forward runtime_state.json could hold the completion truth
    # only in the ledger while tasks[] had been rebuilt all-pending.
    state = _make_runtime_state()
    state.tasks = [TaskRecord(task_id="task-001", title="Stomped", status="pending")]
    payload = state.to_dict()
    payload["completed_task_ids"] = ["task-001"]
    loaded = RuntimeState.from_dict(payload)
    assert loaded.tasks[0].status == "completed"
    assert loaded.completed_task_ids == ["task-001"]


def test_from_dict_drops_legacy_review_verdict_citation_keys() -> None:
    # State files written before citations/evidence_refs were removed still load.
    state = _make_runtime_state()
    payload = state.to_dict()
    payload["review_verdicts"] = [
        {
            "phase": "DRAFT_ADVISOR_REVIEW",
            "verdict": "approved",
            "decided_at": "2026-01-01T00:00:00",
            "notes": None,
            "citations": ["legacy-cite"],
            "evidence_refs": ["legacy-ref"],
        }
    ]
    loaded = RuntimeState.from_dict(payload)
    assert loaded.review_verdicts[0].verdict == "approved"
    assert not hasattr(loaded.review_verdicts[0], "citations")


# --- PhaseChangedEvent journal (framework-extensions Task B) ----------------


def _read_journal(adapter: ArtifactAdapter) -> list[dict[str, object]]:
    path = adapter.state_dir / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _resume_via_handler(world, agent_id, workflow_id: str) -> None:
    from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
    from ecs_agent.types import Message

    config = world.get_component(agent_id, UserPromptConfigComponent)
    assert config is not None
    handler_key = next(
        t.content for t in config.triggers if t.pattern == "/plan:resume"
    )
    handler = config.script_handlers[handler_key]
    conv = world.get_component(agent_id, ConversationComponent)
    assert conv is not None
    user_text = f"/plan:resume {workflow_id}"
    conv.messages.append(Message(role="user", content=user_text))
    result = await handler(world, agent_id, user_text)
    assert result is not None


def _persist_workflow(base_dir: Path, workflow_id: str, phase: str) -> ArtifactAdapter:
    adapter = ArtifactAdapter(base_dir=base_dir, workflow_id=workflow_id)
    state = _make_state_at_phase(phase)
    state.workflow_id = workflow_id
    adapter.write_draft("# Draft\n")
    adapter.write_plan("# Plan\n")
    adapter.write_state(state)
    return adapter


async def test_restore_demotion_is_journaled_as_phase_transition(
    tmp_path: Path,
) -> None:
    world, agent_id, _adapter, _runtime_state = await _build_test_world(tmp_path)
    wf = _persist_workflow(tmp_path, "journal-wf", "TASK_RUNNING")

    await _resume_via_handler(world, agent_id, "journal-wf")

    entries = _read_journal(wf)
    demotions = [
        e
        for e in entries
        if e["type"] == "phase_transition" and e["reason"] == "on_resume"
    ]
    assert len(demotions) == 1
    assert demotions[0]["from"] == "TASK_RUNNING"
    assert demotions[0]["to"] == "TASK_BLOCKED"
    assert demotions[0]["forced"] is True
    assert demotions[0]["workflow_id"] == "journal-wf"


async def test_workflow_switch_journals_restore_to_the_new_workflow(
    tmp_path: Path,
) -> None:
    world, agent_id, _adapter, _runtime_state = await _build_test_world(tmp_path)
    wf_a = _persist_workflow(tmp_path, "journal-wf-a", "TASK_RUNNING")
    wf_b = _persist_workflow(tmp_path, "journal-wf-b", "TASK_RUNNING")

    await _resume_via_handler(world, agent_id, "journal-wf-a")
    await _resume_via_handler(world, agent_id, "journal-wf-b")

    a_entries = _read_journal(wf_a)
    b_entries = _read_journal(wf_b)
    assert all(e["workflow_id"] == "journal-wf-a" for e in a_entries)
    assert any(
        e["reason"] == "on_resume" and e["workflow_id"] == "journal-wf-b"
        for e in b_entries
    )


async def test_task_lifecycle_transitions_are_journaled(tmp_path: Path) -> None:
    from examples.e2e.plan_and_task.controller import PlanController

    world, agent_id, _adapter, runtime_state = await _build_test_world(tmp_path)
    wf = _persist_workflow(tmp_path, "journal-wf-life", "TASK_BLOCKED")

    await _resume_via_handler(world, agent_id, "journal-wf-life")
    loaded = runtime_state[0]
    assert loaded is not None
    controller = PlanController(world, agent_id)
    from examples.e2e.plan_and_task.scratchbook_adapter import (
        PlanTaskScratchbookAdapter,
    )

    live_adapter = PlanTaskScratchbookAdapter(
        base_dir=tmp_path, workflow_id="journal-wf-life"
    )
    await controller.handle_task_resume(loaded, live_adapter)
    await controller.handle_task_abort(loaded, live_adapter, "operator stop")

    entries = _read_journal(wf)
    reasons = [e["reason"] for e in entries if e["type"] == "phase_transition"]
    assert "task:resume" in reasons
    assert any(str(r).startswith("task:abort:") for r in reasons)
    # The three hand-built event types are gone from the journal vocabulary.
    assert all(
        e["type"] == "phase_transition"
        for e in entries
        if e["type"] in {"task_resumed", "task_aborted", "task_replan_requested"}
        or e["type"] == "phase_transition"
    )


# --- Faithful approvals ledger (framework-extensions Task C) ----------------


async def test_restore_rehydrates_the_approvals_ledger(tmp_path: Path) -> None:
    from ecs_agent.phases import latest_verdicts
    from examples.e2e.plan_and_task.main import resume_workflow

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="ledger-wf")
    state = _make_state_at_phase("DRAFT_QA_REVIEW")
    state.workflow_id = "ledger-wf"
    state.review_verdicts.append(_make_verdict("DRAFT_ADVISOR_REVIEW", "approved"))
    adapter.write_draft("# Draft\n")
    adapter.write_state(state)

    world = World()
    eid = world.create_entity()
    await resume_workflow(world, eid, "ledger-wf", base_dir=tmp_path)

    assert latest_verdicts(world, eid) == {"DRAFT_ADVISOR_REVIEW": "approved"}


async def test_plan_start_resets_the_approvals_ledger(tmp_path: Path) -> None:
    from ecs_agent.phases import latest_verdicts, record_approval
    from examples.e2e.plan_and_task.controller import PlanController

    world, eid = await _bound_world_at("DRAFT_ADVISOR_REVIEW")
    await record_approval(world, eid, "approved")
    assert latest_verdicts(world, eid) == {"DRAFT_ADVISOR_REVIEW": "approved"}

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="fresh-wf")
    controller = PlanController(world, eid)
    await controller.handle_plan_start(adapter, "brand new workflow")

    assert latest_verdicts(world, eid) == {}


async def test_scope_replan_resets_the_approvals_ledger(tmp_path: Path) -> None:
    from ecs_agent.components import PhaseApprovalsComponent
    from ecs_agent.phases import latest_verdicts
    from examples.e2e.plan_and_task.controller import PlanController

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="test-wf")
    state = _make_state_at_phase("TASK_RUNNING")
    adapter.write_plan("# Plan\n")
    world, eid = await _bound_world_at("TASK_RUNNING")
    world.add_component(
        eid,
        PhaseApprovalsComponent(
            records=[
                {
                    "phase": "DRAFT_QA_REVIEW",
                    "verdict": "approved",
                    "notes": None,
                    "decided_at": "t",
                }
            ]
        ),
    )
    controller = PlanController(world, eid)

    await controller.handle_task_replan(state, adapter, "scope grew", scope_changed=True)

    assert latest_verdicts(world, eid) == {}


async def test_live_review_after_rehydration_appends_latest(tmp_path: Path) -> None:
    from ecs_agent.components import PhaseApprovalsComponent
    from ecs_agent.phases import latest_verdicts
    from examples.e2e.plan_and_task.controller import PlanController
    from examples.e2e.plan_and_task.main import resume_workflow

    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id="ledger-wf-2")
    state = _make_state_at_phase("DRAFT_QA_REVIEW")
    state.workflow_id = "ledger-wf-2"
    state.review_verdicts.append(_make_verdict("DRAFT_QA_REVIEW", "revise"))
    adapter.write_draft("# Draft\n")
    adapter.write_state(state)

    world = World()
    eid = world.create_entity()
    loaded, live_adapter, _actions = await resume_workflow(
        world, eid, "ledger-wf-2", base_dir=tmp_path
    )
    controller = PlanController(world, eid)
    await controller.handle_qa_review(loaded, live_adapter, "approved", notes="great")

    # The fold returns the newest verdict; the rehydrated record is retained.
    assert latest_verdicts(world, eid)["DRAFT_QA_REVIEW"] == "approved"
    ledger = world.get_component(eid, PhaseApprovalsComponent)
    assert ledger is not None
    assert [r["verdict"] for r in ledger.records] == ["revise", "approved"]
    # Notes alignment: sticky clearing applies to the ledger copy too.
    assert ledger.records[-1]["notes"] is None
