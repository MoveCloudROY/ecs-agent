"""Workflow plan parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import yaml

from ecs_agent.logging import get_logger

logger = get_logger(__name__)

PlanStatus = Literal["draft", "reviewed", "finalized"]
_ALLOWED_STATUSES = {"draft", "reviewed", "finalized"}
_FRONTMATTER_DELIMITER = "---"
_TASKS_HEADING = "## Tasks"
_TASK_HEADING_PREFIX = "### Task:"
_YAML_FENCE = "```yaml"
_FENCE_CLOSE = "```"


@dataclass(slots=True)
class PlanTask:
    """Represents a single executable task within a workflow plan."""

    task_id: str
    title: str
    description: str
    dependencies: list[str]
    acceptance_criteria: list[str]
    execution_hints: list[str]


@dataclass(slots=True)
class WorkflowPlan:
    """Represents a complete, parsed workflow plan with metadata and tasks."""

    workflow_id: str
    title: str
    description: str
    status: PlanStatus
    tasks: list[PlanTask]
    created_at: str
    finalized_at: str | None


def parse_plan(content: str) -> WorkflowPlan:
    frontmatter, body_lines = _extract_frontmatter(content)
    tasks = _parse_tasks(body_lines)
    if not tasks:
        raise ValueError("Workflow plan must contain at least one task.")

    status = _require_string(frontmatter, "status")
    if status not in _ALLOWED_STATUSES:
        raise ValueError(f"Invalid workflow plan status: {status}")
    plan_status = cast(PlanStatus, status)

    plan = WorkflowPlan(
        workflow_id=_require_string(frontmatter, "workflow_id"),
        title=_require_string(frontmatter, "title"),
        description=_require_string(frontmatter, "description"),
        status=plan_status,
        tasks=tasks,
        created_at=_require_string(frontmatter, "created_at"),
        finalized_at=_optional_string(frontmatter, "finalized_at"),
    )
    logger.info(
        "workflow_plan_parsed",
        workflow_id=plan.workflow_id,
        task_count=len(plan.tasks),
    )
    return plan


def validate_plan(plan: WorkflowPlan) -> None:
    if plan.status != "finalized":
        logger.warning(
            "plan_task_plan_not_finalized",
            workflow_id=plan.workflow_id,
            status=plan.status,
        )
        raise ValueError("Workflow plan must be finalized before execution.")


def _extract_frontmatter(content: str) -> tuple[dict[str, Any], list[str]]:
    lines = [line.rstrip("\r") for line in content.splitlines()]
    first_non_empty_idx = _find_first_non_empty_line(lines)
    if (
        first_non_empty_idx is None
        or lines[first_non_empty_idx].strip() != _FRONTMATTER_DELIMITER
    ):
        raise ValueError("Workflow plan must begin with YAML frontmatter.")

    closing_idx: int | None = None
    for idx in range(first_non_empty_idx + 1, len(lines)):
        if lines[idx].strip() == _FRONTMATTER_DELIMITER:
            closing_idx = idx
            break

    if closing_idx is None:
        raise ValueError("Workflow plan frontmatter is missing a closing delimiter.")

    frontmatter_text = "\n".join(lines[first_non_empty_idx + 1 : closing_idx])
    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        logger.error("plan_schema_yaml_parse_error", exception=str(exc))
        raise ValueError(f"Workflow plan frontmatter is malformed: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Workflow plan frontmatter must be a YAML mapping.")

    return parsed, lines[closing_idx + 1 :]


def _parse_tasks(body_lines: list[str]) -> list[PlanTask]:
    tasks_heading_index = _find_tasks_heading(body_lines)
    if tasks_heading_index is None:
        raise ValueError("Workflow plan is missing the required ## Tasks section.")

    tasks_lines = body_lines[tasks_heading_index + 1 :]
    tasks: list[PlanTask] = []
    index = 0
    while index < len(tasks_lines):
        line = tasks_lines[index].strip()
        if not line:
            index += 1
            continue
        if not line.startswith(_TASK_HEADING_PREFIX):
            raise ValueError(
                "Workflow plan tasks must use '### Task: <task_id>' headings."
            )

        heading_task_id = line.removeprefix(_TASK_HEADING_PREFIX).strip()
        if not heading_task_id:
            raise ValueError("Workflow plan task heading must include a task_id.")

        index += 1
        while index < len(tasks_lines) and not tasks_lines[index].strip():
            index += 1

        if index >= len(tasks_lines) or tasks_lines[index].strip() != _YAML_FENCE:
            raise ValueError(
                f"Workflow plan task '{heading_task_id}' must be followed by a ```yaml block."
            )

        index += 1
        block_lines: list[str] = []
        while index < len(tasks_lines) and tasks_lines[index].strip() != _FENCE_CLOSE:
            block_lines.append(tasks_lines[index])
            index += 1

        if index >= len(tasks_lines):
            raise ValueError(
                f"Workflow plan task '{heading_task_id}' YAML block is missing a closing fence."
            )

        task = _parse_task_block(heading_task_id, "\n".join(block_lines))
        tasks.append(task)
        index += 1

    return tasks


def _parse_task_block(heading_task_id: str, task_yaml: str) -> PlanTask:
    try:
        parsed = yaml.safe_load(task_yaml)
    except yaml.YAMLError as exc:
        logger.error("plan_schema_yaml_parse_error", exception=str(exc))
        raise ValueError(
            f"Workflow plan task '{heading_task_id}' YAML is malformed: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Workflow plan task '{heading_task_id}' must be a YAML mapping."
        )

    task_id = _require_string(parsed, "task_id")
    if task_id != heading_task_id:
        raise ValueError(
            f"Workflow plan task heading '{heading_task_id}' does not match YAML task_id '{task_id}'."
        )

    acceptance_criteria = _require_string_list(parsed, "acceptance_criteria")
    if not acceptance_criteria:
        raise ValueError(
            f"Workflow plan task '{heading_task_id}' must define non-empty acceptance_criteria."
        )

    return PlanTask(
        task_id=task_id,
        title=_require_string(parsed, "title"),
        description=_require_string(parsed, "description"),
        dependencies=_require_string_list(parsed, "dependencies"),
        acceptance_criteria=acceptance_criteria,
        execution_hints=_optional_string_list(parsed, "execution_hints"),
    )


def _find_first_non_empty_line(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if line.strip():
            return idx
    return None


def _find_tasks_heading(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if line.strip() == _TASKS_HEADING:
            return idx
    return None


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Workflow plan field '{key}' must be a non-empty string.")
    return value


def _optional_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(
            f"Workflow plan field '{key}' must be a string, list of strings, or null."
        )

    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Workflow plan field '{key}' must be a list of strings.")
        items.append(item)
    return items


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Workflow plan field '{key}' must be a string or null.")
    return value


def _require_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Workflow plan field '{key}' must be a list of strings.")

    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Workflow plan field '{key}' must be a list of strings.")
        items.append(item)
    return items
