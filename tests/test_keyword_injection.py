from ecs_agent.prompts.contracts import PromptTemplate
from ecs_agent.prompts.registry import PromptRegistry
from ecs_agent.prompts.keyword_injection import inject_keywords


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
