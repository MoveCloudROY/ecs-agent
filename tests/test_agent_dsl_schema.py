"""Tests for AgentSpec schema validation — placeholders and triggers fields."""

from __future__ import annotations
import pytest
from ecs_agent.dsl.schema import validate_agent_spec


def test_validate_agent_spec_accepts_placeholders_field() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are ${role}.",
        "placeholders": [{"name": "role", "value": "a helpful assistant"}],
    }
    spec = validate_agent_spec(data)
    assert spec.placeholders == [{"name": "role", "value": "a helpful assistant"}]


def test_validate_agent_spec_accepts_triggers_field() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are an assistant.",
        "triggers": [
            {
                "pattern": "@help",
                "match_mode": "keyword",
                "action": "inject",
                "content": "Show help.",
                "priority": 0,
            }
        ],
    }
    spec = validate_agent_spec(data)
    assert spec.triggers == [
        {
            "pattern": "@help",
            "match_mode": "keyword",
            "action": "inject",
            "content": "Show help.",
            "priority": 0,
        }
    ]


def test_validate_agent_spec_rejects_placeholder_without_name() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are ${role}.",
        "placeholders": [{"value": "a helpful assistant"}],
    }
    with pytest.raises((ValueError, TypeError)):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_rejects_placeholder_without_value() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are ${role}.",
        "placeholders": [{"name": "role"}],
    }
    with pytest.raises((ValueError, TypeError)):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_rejects_placeholder_reserved_name() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are ${_role}.",
        "placeholders": [{"name": "_role", "value": "assistant"}],
    }
    with pytest.raises(ValueError, match="reserved"):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_rejects_trigger_missing_pattern() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are an assistant.",
        "triggers": [{"match_mode": "keyword", "action": "inject", "content": "help"}],
    }
    with pytest.raises((ValueError, TypeError)):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_rejects_trigger_invalid_match_mode() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are an assistant.",
        "triggers": [
            {
                "pattern": "@help",
                "match_mode": "invalid",
                "action": "inject",
                "content": "help",
            }
        ],
    }
    with pytest.raises(ValueError, match="match_mode"):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_rejects_trigger_invalid_action() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are an assistant.",
        "triggers": [
            {
                "pattern": "@help",
                "match_mode": "keyword",
                "action": "bad_action",
                "content": "help",
            }
        ],
    }
    with pytest.raises(ValueError, match="action"):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_trigger_default_priority_zero() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are an assistant.",
        "triggers": [
            {
                "pattern": "@help",
                "match_mode": "keyword",
                "action": "inject",
                "content": "help",
            }
        ],
    }
    spec = validate_agent_spec(data)
    assert spec.triggers[0]["priority"] == 0


def test_validate_agent_spec_rejects_unknown_fields_still_works() -> None:
    data = {
        "mode": "primary",
        "model": "gpt-4",
        "prompt": "You are an assistant.",
        "unknown_field": "oops",
    }
    with pytest.raises(ValueError, match="Unknown field"):
        validate_agent_spec(data, source_name="test")


def test_validate_agent_spec_empty_placeholders_ok() -> None:
    data = {"mode": "primary", "model": "gpt-4", "prompt": "Hi.", "placeholders": []}
    spec = validate_agent_spec(data)
    assert spec.placeholders == []


def test_validate_agent_spec_empty_triggers_ok() -> None:
    data = {"mode": "primary", "model": "gpt-4", "prompt": "Hi.", "triggers": []}
    spec = validate_agent_spec(data)
    assert spec.triggers == []
