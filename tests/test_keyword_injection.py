import ecs_agent.prompts.keyword_injection as keyword_injection_module

from ecs_agent.components import PromptConfigComponent
from ecs_agent.prompts.contracts import PromptTemplate, PromptTriggerSpec
from ecs_agent.prompts.registry import PromptRegistry
from ecs_agent.prompts.keyword_injection import inject_keywords, inject_triggers


def test_inject_keywords_first_match():
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
    # Should only match the first one: @code
    result = inject_keywords(user_text, registry)

    expected_marker = "[PROMPT_INJECT:@code]"
    assert result.startswith(expected_marker)
    assert "CODE_TEMPLATE_CONTENT" in result
    assert "DEBUG_TEMPLATE_CONTENT" not in result
    assert result.endswith(user_text)


def test_inject_keywords_no_match():
    registry = PromptRegistry()
    user_text = "Just a normal message"
    result = inject_keywords(user_text, registry)
    assert result == user_text


def test_inject_keywords_duplicate_prevention():
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="code-tpl", content="CODE_TEMPLATE_CONTENT")
    )
    registry.register_keyword("@code", "code-tpl")

    user_text = "[PROMPT_INJECT:@code]\nCODE_TEMPLATE_CONTENT\n\nHelp me with @code"
    result = inject_keywords(user_text, registry)

    # Should not inject again
    assert result == user_text


def test_inject_keywords_order():
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="code-tpl", content="CODE_TEMPLATE_CONTENT")
    )
    registry.register_keyword("@code", "code-tpl")

    user_text = "Help me with @code"
    result = inject_keywords(user_text, registry)

    lines = result.split("\n")
    assert lines[0] == "[PROMPT_INJECT:@code]"
    assert lines[1] == "CODE_TEMPLATE_CONTENT"
    # There might be some spacing
    assert user_text in result


def test_prompt_config_exposes_trigger_templates_contract():
    fields = PromptConfigComponent.__dataclass_fields__
    assert "trigger_templates" in fields


def test_keyword_injection_exposes_trigger_entrypoint():
    assert hasattr(keyword_injection_module, "inject_triggers")


def test_inject_triggers_selects_highest_priority_before_registration_order():
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(template_id="keyword-tpl", content="KEYWORD_TEMPLATE_CONTENT")
    )
    registry.register(
        PromptTemplate(template_id="event-tpl", content="EVENT_TEMPLATE_CONTENT")
    )
    trigger_specs = [
        PromptTriggerSpec(
            kind="keyword",
            trigger="@code",
            template_id="keyword-tpl",
            priority=5,
            registration_order=0,
        ),
        PromptTriggerSpec(
            kind="event",
            trigger="tool_error",
            template_id="event-tpl",
            priority=10,
            registration_order=1,
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
    assert "EVENT_TEMPLATE_CONTENT" in result
    assert "KEYWORD_TEMPLATE_CONTENT" not in result
    assert result.endswith(user_text)


def test_inject_triggers_tiebreaks_on_registration_order_for_first_match():
    registry = PromptRegistry()
    registry.register(PromptTemplate(template_id="event-a", content="EVENT_A"))
    registry.register(PromptTemplate(template_id="event-b", content="EVENT_B"))
    trigger_specs = [
        PromptTriggerSpec(
            kind="event",
            trigger="build_complete",
            template_id="event-b",
            priority=10,
            registration_order=1,
        ),
        PromptTriggerSpec(
            kind="event",
            trigger="build_complete",
            template_id="event-a",
            priority=10,
            registration_order=0,
        ),
    ]

    user_text = "status"
    result = inject_triggers(
        user_text,
        registry,
        trigger_specs=trigger_specs,
        active_events={"build_complete"},
    )

    assert result.startswith("[PROMPT_INJECT:event:build_complete]")
    assert "EVENT_A" in result
    assert "EVENT_B" not in result


def test_inject_triggers_keyword_and_event_kind_contracts_are_supported():
    registry = PromptRegistry()
    registry.register(PromptTemplate(template_id="event-tpl", content="EVENT_CONTENT"))
    trigger_specs = [
        PromptTriggerSpec(
            kind="event",
            trigger="tool_success",
            template_id="event-tpl",
            priority=0,
            registration_order=0,
        )
    ]

    result = inject_triggers(
        "plain user text",
        registry,
        trigger_specs=trigger_specs,
        active_events={"tool_success"},
    )

    assert result.startswith("[PROMPT_INJECT:event:tool_success]")
