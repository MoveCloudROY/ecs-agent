"""Tests for PromptRegistry — centralized template store with keyword mapping."""

import pytest

from ecs_agent.prompts import PromptRegistry, PromptTemplate


def _make_template(template_id: str, content: str = "Hello {name}") -> PromptTemplate:
    """Helper: build a minimal PromptTemplate."""
    return PromptTemplate(
        template_id=template_id,
        content=content,
        description=f"Template {template_id}",
    )


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_register_and_get_returns_correct_template(self) -> None:
        registry = PromptRegistry()
        tmpl = _make_template("coding-assistant")
        registry.register(tmpl)
        result = registry.get("coding-assistant")
        assert result is tmpl

    def test_get_unknown_template_raises_value_error_with_context(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(ValueError, match="unknown-id"):
            registry.get("unknown-id")

    def test_register_duplicate_raises_value_error(self) -> None:
        registry = PromptRegistry()
        tmpl = _make_template("dup")
        registry.register(tmpl)
        tmpl2 = _make_template("dup")
        with pytest.raises(ValueError, match="dup"):
            registry.register(tmpl2)

    def test_register_keyword_maps_to_template(self) -> None:
        registry = PromptRegistry()
        tmpl = _make_template("code")
        registry.register(tmpl)
        registry.register_keyword("@code", "code")
        resolved = registry.resolve_keyword("@code")
        assert resolved is tmpl

    def test_register_keyword_unknown_template_raises_value_error(self) -> None:
        registry = PromptRegistry()
        with pytest.raises(ValueError, match="nonexistent"):
            registry.register_keyword("@kw", "nonexistent")

    def test_resolve_keyword_returns_template(self) -> None:
        registry = PromptRegistry()
        tmpl = _make_template("writer")
        registry.register(tmpl)
        registry.register_keyword("@write", "writer")
        assert registry.resolve_keyword("@write") is tmpl

    def test_resolve_keyword_unknown_returns_none(self) -> None:
        registry = PromptRegistry()
        assert registry.resolve_keyword("@missing") is None

    def test_list_templates_stable_insertion_order(self) -> None:
        registry = PromptRegistry()
        ids = ["alpha", "beta", "gamma", "delta"]
        for tid in ids:
            registry.register(_make_template(tid))
        result_ids = [t.template_id for t in registry.list_templates()]
        assert result_ids == ids

    def test_list_templates_stable_order_across_multiple_calls(self) -> None:
        registry = PromptRegistry()
        for tid in ["z", "a", "m"]:
            registry.register(_make_template(tid))
        first_call = [t.template_id for t in registry.list_templates()]
        second_call = [t.template_id for t in registry.list_templates()]
        assert first_call == second_call == ["z", "a", "m"]

    def test_list_keywords_stable_order(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template("t1"))
        registry.register(_make_template("t2"))
        registry.register_keyword("@kw1", "t1")
        registry.register_keyword("@kw2", "t2")
        registry.register_keyword("@kw3", "t1")
        assert registry.list_keywords() == ["@kw1", "@kw2", "@kw3"]

    def test_list_keywords_stable_order_across_multiple_calls(self) -> None:
        registry = PromptRegistry()
        registry.register(_make_template("t1"))
        registry.register_keyword("@b", "t1")
        registry.register_keyword("@a", "t1")
        first = registry.list_keywords()
        second = registry.list_keywords()
        assert first == second == ["@b", "@a"]

    def test_register_keyword_after_register_template(self) -> None:
        registry = PromptRegistry()
        tmpl = _make_template("assistant")
        registry.register(tmpl)
        # Keyword can only be registered after template is present
        registry.register_keyword("@assistant", "assistant")
        assert registry.resolve_keyword("@assistant") is tmpl

    def test_list_templates_empty_registry_returns_empty_list(self) -> None:
        registry = PromptRegistry()
        assert registry.list_templates() == []

    def test_list_keywords_empty_returns_empty_list(self) -> None:
        registry = PromptRegistry()
        assert registry.list_keywords() == []

    def test_register_multiple_keywords_for_same_template(self) -> None:
        registry = PromptRegistry()
        tmpl = _make_template("poly")
        registry.register(tmpl)
        registry.register_keyword("@p1", "poly")
        registry.register_keyword("@p2", "poly")
        assert registry.resolve_keyword("@p1") is tmpl
        assert registry.resolve_keyword("@p2") is tmpl

    def test_get_returns_exact_template_object(self) -> None:
        registry = PromptRegistry()
        tmpl = PromptTemplate(
            template_id="rich",
            content="content here",
            description="desc",
            metadata={"key": "val"},
        )
        registry.register(tmpl)
        got = registry.get("rich")
        assert got.template_id == "rich"
        assert got.content == "content here"
        assert got.description == "desc"
        assert got.metadata == {"key": "val"}
