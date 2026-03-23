from ecs_agent.components import UserPromptConfigComponent
from ecs_agent.prompts.contracts import PromptTemplate, TriggerSpec
from ecs_agent.prompts.keyword_injection import inject_keywords, inject_triggers
from ecs_agent.prompts.registry import PromptRegistry


def test_inject_keywords_first_keyword_match() -> None:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="code-tpl", content="CODE_TEMPLATE_CONTENT")
    )
    registry.register_keyword("@code", "code-tpl")
    registry.register(
        PromptTemplate(template_id="debug-tpl", content="DEBUG_TEMPLATE_CONTENT")
    )
    registry.register_keyword("@debug", "debug-tpl")

    user_text = "Help me with @code and @debug"
    result = inject_keywords(user_text, registry)

    assert result.startswith("[PROMPT_INJECT:@code]")
    assert "CODE_TEMPLATE_CONTENT" in result
    assert "DEBUG_TEMPLATE_CONTENT" not in result
    assert result.endswith(user_text)


def test_inject_keywords_no_match() -> None:
    registry = PromptRegistry()

    user_text = "Just a normal message"
    result = inject_keywords(user_text, registry)

    assert result == user_text


def test_inject_keywords_duplicate_prevention() -> None:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="code-tpl", content="CODE_TEMPLATE_CONTENT")
    )
    registry.register_keyword("@code", "code-tpl")

    user_text = "[PROMPT_INJECT:@code]\nCODE_TEMPLATE_CONTENT\n\nHelp me with @code"
    result = inject_keywords(user_text, registry)

    assert result == user_text


def test_user_prompt_config_exposes_triggers_contract() -> None:
    fields = UserPromptConfigComponent.__dataclass_fields__
    assert "triggers" in fields


def test_inject_triggers_prioritizes_higher_priority_trigger_spec() -> None:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="keyword-tpl", content="KEYWORD_TEMPLATE")
    )
    registry.register(PromptTemplate(template_id="event-tpl", content="EVENT_TEMPLATE"))

    trigger_specs = [
        TriggerSpec(
            pattern="@code",
            match_mode="keyword",
            action="skill",
            content="keyword-tpl",
            priority=5,
        ),
        TriggerSpec(
            pattern="event:tool_error",
            match_mode="keyword",
            action="skill",
            content="event-tpl",
            priority=10,
        ),
    ]

    user_text = "Please handle @code path"
    result = inject_triggers(
        user_text,
        registry,
        trigger_specs=trigger_specs,
        active_events={"tool_error"},
    )

    assert result.startswith("[PROMPT_INJECT:event:tool_error]")
    assert "EVENT_TEMPLATE" in result
    assert "KEYWORD_TEMPLATE" not in result
    assert result.endswith(user_text)


def test_inject_triggers_tiebreaks_with_spec_order() -> None:
    registry = PromptRegistry()
    registry.register(PromptTemplate(template_id="event-a", content="EVENT_A"))
    registry.register(PromptTemplate(template_id="event-b", content="EVENT_B"))

    trigger_specs = [
        TriggerSpec(
            pattern="event:build_complete",
            match_mode="keyword",
            action="skill",
            content="event-a",
            priority=10,
        ),
        TriggerSpec(
            pattern="event:build_complete",
            match_mode="keyword",
            action="skill",
            content="event-b",
            priority=10,
        ),
    ]

    result = inject_triggers(
        "status",
        registry,
        trigger_specs=trigger_specs,
        active_events={"build_complete"},
    )

    assert result.startswith("[PROMPT_INJECT:event:build_complete]")
    assert "EVENT_A" in result
    assert "EVENT_B" not in result
