"""Tests for LLMProvider Protocol."""

import pytest
from typing import Protocol, get_type_hints
from ecs_agent.providers import LLMProvider
from ecs_agent.types import Message, CompletionResult, ToolSchema


def test_llmprovider_is_protocol() -> None:
    """LLMProvider should be a Protocol."""
    assert isinstance(LLMProvider, type)
    # Check it's protocol-like by inspecting _is_protocol
    assert hasattr(LLMProvider, "_is_protocol")


def test_llmprovider_has_complete_method() -> None:
    """LLMProvider should have a complete method."""
    assert hasattr(LLMProvider, "complete")


def test_llmprovider_complete_signature() -> None:
    """complete method should have correct signature."""
    # Get the method
    complete_method = getattr(LLMProvider, "complete")
    assert complete_method is not None

    # Check it's callable
    assert callable(complete_method)

    # Get type hints
    hints = get_type_hints(complete_method)

    # Should have 'messages' parameter
    assert "messages" in hints

    # Should have 'tools' parameter
    assert "tools" in hints

    # Should have 'return' annotation
    assert "return" in hints
    assert hints["return"] == CompletionResult


def test_llmprovider_complete_is_async() -> None:
    """complete method should be async."""
    import inspect

    complete_method = getattr(LLMProvider, "complete")
    assert inspect.iscoroutinefunction(complete_method)
