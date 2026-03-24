from ecs_agent.components import UserPromptConfigComponent
from ecs_agent.prompts.contracts import PromptTemplate, TriggerSpec
from ecs_agent.prompts.keyword_injection import inject_triggers
from ecs_agent.prompts.registry import PromptRegistry


def test_user_prompt_config_exposes_triggers_contract() -> None:
    fields = UserPromptConfigComponent.__dataclass_fields__
    assert "triggers" in fields


def test_inject_triggers_prioritizes_higher_priority_trigger_spec() -> None:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="low-tpl", content="LOW_PRIORITY_TEMPLATE")
    )
    registry.register(
        PromptTemplate(template_id="high-tpl", content="HIGH_PRIORITY_TEMPLATE")
    )

    trigger_specs = [
        TriggerSpec(
            pattern="@code",
            match_mode="keyword",
            action="skill",
            content="low-tpl",
            priority=5,
        ),
        TriggerSpec(
            pattern="@debug",
            match_mode="keyword",
            action="skill",
            content="high-tpl",
            priority=10,
        ),
    ]

    user_text = "Please handle @code @debug path"
    result = inject_triggers(
        user_text,
        registry,
        trigger_specs=trigger_specs,
    )

    assert result.startswith("[PROMPT_INJECT:@debug]")
    assert "HIGH_PRIORITY_TEMPLATE" in result
    assert "LOW_PRIORITY_TEMPLATE" not in result
    assert result.endswith(user_text)


def test_inject_triggers_tiebreaks_with_spec_order() -> None:
    registry = PromptRegistry()
    registry.register(PromptTemplate(template_id="tpl-a", content="TEMPLATE_A"))
    registry.register(PromptTemplate(template_id="tpl-b", content="TEMPLATE_B"))

    trigger_specs = [
        TriggerSpec(
            pattern="@code",
            match_mode="keyword",
            action="skill",
            content="tpl-a",
            priority=10,
        ),
        TriggerSpec(
            pattern="@code",
            match_mode="keyword",
            action="skill",
            content="tpl-b",
            priority=10,
        ),
    ]

    result = inject_triggers(
        "@code status",
        registry,
        trigger_specs=trigger_specs,
    )

    assert result.startswith("[PROMPT_INJECT:@code]")
    assert "TEMPLATE_A" in result
    assert "TEMPLATE_B" not in result
