"""GitHub Actions workflow contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
RELEASE_NOTES = REPO_ROOT / "docs" / "releases" / "v0.1.0.md"


def _read_workflow(path: Path) -> dict[str, Any]:
    assert path.exists(), f"Expected workflow file to exist: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _on_section(workflow: dict[str, Any]) -> Any:
    return workflow.get("on", workflow.get(True))


def _job_steps(workflow: dict[str, Any], job_name: str) -> list[dict[str, Any]]:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), "Workflow must define jobs"
    job = jobs.get(job_name)
    assert isinstance(job, dict), f"Workflow missing job: {job_name}"
    steps = job.get("steps")
    assert isinstance(steps, list), f"Job {job_name} must define steps"
    return [step for step in steps if isinstance(step, dict)]


def _joined_run_commands(steps: list[dict[str, Any]]) -> str:
    return "\n".join(step.get("run", "") for step in steps if isinstance(step.get("run"), str))


def test_release_notes_contract_exists() -> None:
    assert RELEASE_NOTES.exists(), (
        "Expected a versioned release-notes file at docs/releases/v0.1.0.md"
    )
    content = RELEASE_NOTES.read_text(encoding="utf-8").strip()
    assert content, "Release notes file must not be empty"
    assert "0.1.0" in content or "v0.1.0" in content


def test_ci_workflow_runs_pytest_on_push_and_pull_request() -> None:
    workflow = _read_workflow(CI_WORKFLOW)
    on_section = _on_section(workflow)
    assert isinstance(on_section, dict), "CI workflow must define triggers"
    assert "push" in on_section
    assert "pull_request" in on_section

    steps = _job_steps(workflow, "test")
    commands = _joined_run_commands(steps)
    assert "uv sync --group dev" in commands
    assert "uv run pytest" in commands
    assert "mypy" not in commands


def test_release_workflow_runs_on_version_tags() -> None:
    workflow = _read_workflow(RELEASE_WORKFLOW)
    on_section = _on_section(workflow)
    assert isinstance(on_section, dict), "Release workflow must define triggers"

    push = on_section.get("push")
    assert isinstance(push, dict), "Release workflow must trigger on push tags"
    tags = push.get("tags")
    assert isinstance(tags, list)
    assert "v*" in tags


def test_release_workflow_validates_version_before_release() -> None:
    workflow = _read_workflow(RELEASE_WORKFLOW)
    steps = _job_steps(workflow, "validate-version")
    commands = _joined_run_commands(steps)

    assert "pyproject.toml" in commands
    assert "src/ecs_agent/__init__.py" in commands
    assert "GITHUB_REF_NAME" in commands or "github.ref_name" in commands


def test_release_workflow_runs_standard_and_full_live_tests() -> None:
    workflow = _read_workflow(RELEASE_WORKFLOW)

    test_commands = _joined_run_commands(_job_steps(workflow, "test"))
    live_commands = _joined_run_commands(_job_steps(workflow, "live-tests"))

    assert "uv sync --group dev" in test_commands
    assert "uv run pytest" in test_commands
    assert "tests/live" in live_commands


def test_release_workflow_validates_manual_release_notes() -> None:
    workflow = _read_workflow(RELEASE_WORKFLOW)
    commands = _joined_run_commands(_job_steps(workflow, "build-and-release"))

    assert "docs/releases/" in commands
    assert ".md" in commands


def test_release_workflow_builds_and_publishes() -> None:
    workflow = _read_workflow(RELEASE_WORKFLOW)

    build_commands = _joined_run_commands(_job_steps(workflow, "build-and-release"))
    assert "python -m build" in build_commands or "uv build" in build_commands

    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict)
    pypi_job = jobs.get("publish-pypi")
    assert isinstance(pypi_job, dict), "Release workflow must define publish-pypi job"
    steps = pypi_job.get("steps")
    assert isinstance(steps, list)
    joined = "\n".join(str(step) for step in steps)
    assert "pypi" in joined.lower()
