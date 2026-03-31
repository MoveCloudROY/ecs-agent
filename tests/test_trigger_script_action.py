from __future__ import annotations

from typing import get_args

from ecs_agent.prompts.contracts import TriggerSpec


def test_trigger_spec_accepts_script_action() -> None:
    assert "script" in get_args(TriggerSpec.__annotations__["action"])

    spec = TriggerSpec(
        pattern="@run",
        match_mode="keyword",
        action="script",
        content="my_handler",
    )
    assert spec.action == "script"
    assert spec.content == "my_handler"
