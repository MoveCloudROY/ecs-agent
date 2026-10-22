"""Explicit live integration coverage migrated from mixed/offline directories."""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest
from tests.support.environment import example_environment

from tests.support.models import LiveModelFactory


_CLI_DRIVER = """
import asyncio
import builtins
import sys
from pathlib import Path
from examples.e2e.plan_and_task import main

read_stdin = builtins.input

def scripted_input(prompt=""):
    # Keep the REPL commands on stdin, while answering tool-driven interviews
    # from a separate deterministic test policy.
    if prompt.startswith(("Your answer:", "Choose ")):
        return (
            "Pause this interview now and leave the draft incomplete. "
            "Reply in ordinary text to acknowledge the pause, without further tools or questions. "
            "I will check /plan:status in my next turn; do not start reviews or implementation."
        )
    return read_stdin(prompt)

builtins.input = scripted_input
main._WORKFLOW_BASE_DIR = Path(sys.argv[1])
asyncio.run(main.main())
"""


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.workflow
@pytest.mark.asyncio
async def test_plan_task_cli_automation(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    input_sequence = b"exit\n"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CLI_DRIVER,
            str(tmp_path),
        ],
        input=input_sequence,
        capture_output=True,
        timeout=live_model_factory.profile.timeouts.case_seconds,
        env={
            **example_environment(live_model_factory.settings),
            "PLAN_TASK_INTERACTIVE": "1",
        },
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )

    output = result.stdout.decode("utf-8", errors="replace")
    assert (
        "Using model:" in output or "Using Anthropic Messages API with model:" in output
    ), f"Expected model selection indication in output. Got:\n{output}"


@pytest.mark.live(
    formats=["openai_chat_completions", "openai_responses"],
    capabilities=["text", "tools"],
)
@pytest.mark.workflow
@pytest.mark.asyncio
async def test_cli_slash_command_plan_start_and_status(
    tmp_path: Path, *, live_model_factory: LiveModelFactory
) -> None:
    """Test CLI slash command execution: /plan:start and /plan:status.

    This test pipes slash commands to main.py and verifies:
    - Exit code is 0
    - Both slash commands run, and status reports DRAFT_INTERVIEW
    - Output contains workflow_id
    - Runtime state file exists with correct phase
    """
    input_sequence = (
        b"/plan:start Build a tiny local CLI demo. Ask one initial question; "
        b"I may pause the interview after answering.\n\n/plan:status\n\nexit\n"
    )

    env = {
        **example_environment(live_model_factory.settings),
        "PLAN_TASK_INTERACTIVE": "1",
        # Bound the interview; its scripted answer pauses before the review cycle.
        "PLAN_TASK_MAX_AGENT_TICKS": "10",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CLI_DRIVER,
            str(tmp_path),
        ],
        input=input_sequence,
        capture_output=True,
        timeout=live_model_factory.profile.timeouts.case_seconds,
        env=env,
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}. "
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )

    output = result.stdout.decode("utf-8", errors="replace")
    assert "user: /plan:status" in output, output[-4000:]
    assert "DRAFT_INTERVIEW" in output, (
        f"Expected 'DRAFT_INTERVIEW' phase in output. Got:\n{output}"
    )
    assert "workflow_id" in output, f"Expected 'workflow_id' in output. Got:\n{output}"
    state_paths = list(tmp_path.glob("scratchbook/*/state/runtime_state.json"))
    assert len(state_paths) == 1
    state = json.loads(state_paths[0].read_text())
    assert state["phase"] == "DRAFT_INTERVIEW"
    assert state["workflow_id"] in output
