from __future__ import annotations

from typing import get_args

from ecs_agent.components.definitions import UserPromptConfigComponent
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


def test_user_prompt_config_has_script_handlers_field() -> None:
    comp = UserPromptConfigComponent()
    assert hasattr(comp, "script_handlers")
    assert isinstance(comp.script_handlers, dict)
    assert len(comp.script_handlers) == 0


def test_user_prompt_config_accepts_script_handler() -> None:
    from ecs_agent.prompts.contracts import TriggerSpec

    async def my_handler(
        world: object, entity_id: object, user_text: str
    ) -> str | None:
        return "rewritten"

    trigger = TriggerSpec(
        pattern="@run",
        match_mode="keyword",
        action="script",
        content="my_handler",
    )
    comp = UserPromptConfigComponent(
        triggers=[trigger],
        script_handlers={"my_handler": my_handler},
    )
    assert "my_handler" in comp.script_handlers


def test_user_prompt_config_positional_backward_compat() -> None:
    triggers = []
    comp = UserPromptConfigComponent(triggers, True, 4096)
    assert comp.enable_context_pool is True
    assert comp.context_pool_max_chars == 4096
    assert comp.script_handlers == {}
