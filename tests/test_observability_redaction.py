from __future__ import annotations

from dataclasses import dataclass

import pytest

from ecs_agent.observability.redaction import SecretRedactor, sanitize_payload


@dataclass(slots=True)
class ToolResult:
    name: str
    payload: dict[str, object]


def test_raw_non_secret_prompt_is_preserved() -> None:
    redactor = SecretRedactor(extra_secret_values=["synthetic-extra-secret-value"])
    payload = {
        "input": {
            "messages": [
                {"role": "user", "content": "write a haiku about telemetry"}
            ],
        },
        "output": ("raw", "non-secret", "text"),
        "metadata": {"tags": {"safe", "observability"}},
    }

    sanitized, report = sanitize_payload(payload, redactor)

    assert sanitized["input"] == {
        "messages": [{"role": "user", "content": "write a haiku about telemetry"}],
    }
    assert sanitized["output"] == ["raw", "non-secret", "text"]
    assert sorted(sanitized["metadata"]["tags"]) == ["observability", "safe"]
    assert report.total_redactions == 0
    assert report.counts_by_rule == {}
    assert report.to_payload() == {
        "total_redactions": 0,
        "counts_by_rule": {},
    }


def test_secret_values_are_redacted_recursively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    langfuse_secret = "synthetic-langfuse-secret-value"
    llm_key = "synthetic-llm-key-value"
    model_name = "synthetic-model-name-value"
    caller_secret = "synthetic-caller-secret-value"
    nested_token = "synthetic-nested-token-value"
    ignored_short_value = "short"
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", langfuse_secret)
    monkeypatch.setenv("LLM_API_KEY", llm_key)
    monkeypatch.setenv("LLM_MODEL", model_name)
    monkeypatch.setenv("OPENAI_API_KEY", ignored_short_value)

    payload = {
        "headers": {
            "Authorization": "Bearer synthetic-header-token-value",
            "X-API-Key": "synthetic-header-api-key-value",
            "cookie": "session=synthetic-cookie-value",
        },
        "api_key": "synthetic-field-api-key-value",
        "tool": ToolResult(
            name="lookup",
            payload={
                "args": {
                    "safe_query": "weather in Paris",
                    "nested": [{"token": nested_token}],
                    "configured": langfuse_secret,
                    "extra": caller_secret,
                    "short": ignored_short_value,
                },
            },
        ),
        "model_output": [
            "assistant response",
            llm_key,
            f"using {model_name} for generation",
        ],
        "tuple_value": (caller_secret,),
        "set_value": frozenset({langfuse_secret}),
    }

    redactor = SecretRedactor(extra_secret_values=[caller_secret, ignored_short_value])
    sanitized, report = sanitize_payload(payload, redactor)

    assert sanitized["headers"] == {
        "Authorization": "[REDACTED:key:authorization]",
        "X-API-Key": "[REDACTED:key:x-api-key]",
        "cookie": "[REDACTED:key:cookie]",
    }
    assert sanitized["api_key"] == "[REDACTED:key:api_key]"
    assert sanitized["tool"]["payload"]["args"]["safe_query"] == "weather in Paris"
    assert sanitized["tool"]["payload"]["args"]["nested"] == [
        {"token": "[REDACTED:key:token]"}
    ]
    assert sanitized["tool"]["payload"]["args"]["configured"] == (
        "[REDACTED:value:LANGFUSE_SECRET_KEY]"
    )
    assert sanitized["tool"]["payload"]["args"]["extra"] == (
        "[REDACTED:value:extra_secret]"
    )
    assert sanitized["tool"]["payload"]["args"]["short"] == ignored_short_value
    assert sanitized["model_output"] == [
        "assistant response",
        "[REDACTED:value:LLM_API_KEY]",
        "using [REDACTED:value:LLM_MODEL] for generation",
    ]
    assert sanitized["tuple_value"] == ["[REDACTED:value:extra_secret]"]
    assert sanitized["set_value"] == ["[REDACTED:value:LANGFUSE_SECRET_KEY]"]

    sanitized_text = str(sanitized)
    report_text = str(report)
    for secret in [
        langfuse_secret,
        llm_key,
        model_name,
        caller_secret,
        nested_token,
        "synthetic-header-token-value",
        "synthetic-header-api-key-value",
        "synthetic-cookie-value",
        "synthetic-field-api-key-value",
    ]:
        assert secret not in sanitized_text
        assert secret not in report_text

    assert ignored_short_value in sanitized_text
    assert ignored_short_value not in report_text
    assert report.total_redactions == 11
    assert report.counts_by_rule == {
        "key:authorization": 1,
        "key:x-api-key": 1,
        "key:cookie": 1,
        "key:api_key": 1,
        "key:token": 1,
        "value:LANGFUSE_SECRET_KEY": 2,
        "value:extra_secret": 2,
        "value:LLM_API_KEY": 1,
        "value:LLM_MODEL": 1,
    }
    report_payload = report.to_payload()
    assert set(report_payload) == {"total_redactions", "counts_by_rule"}
    assert report_payload["counts_by_rule"] == report.counts_by_rule
    assert "headers" not in report_text
    assert "Authorization" not in report_text
    assert "tool" not in report_text
    assert "safe_query" not in report_text
    assert "configured" not in report_text
    assert "model_output" not in report_text


def test_key_name_redaction_is_case_insensitive() -> None:
    payload = {
        "Secret": "synthetic-secret-field-value",
        "PASSWORD": "synthetic-password-field-value",
        "credential": "synthetic-credential-field-value",
        "Set-Cookie": "synthetic-set-cookie-field-value",
    }

    sanitized, report = sanitize_payload(payload, SecretRedactor())

    assert sanitized == {
        "Secret": "[REDACTED:key:secret]",
        "PASSWORD": "[REDACTED:key:password]",
        "credential": "[REDACTED:key:credential]",
        "Set-Cookie": "[REDACTED:key:set-cookie]",
    }
    assert report.counts_by_rule == {
        "key:secret": 1,
        "key:password": 1,
        "key:credential": 1,
        "key:set-cookie": 1,
    }


def test_env_and_extra_secret_values_shorter_than_eight_chars_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "seven77")
    redactor = SecretRedactor(extra_secret_values=["tiny"])

    sanitized, report = sanitize_payload(
        {"message": "seven77", "other": "tiny"},
        redactor,
    )

    assert sanitized == {"message": "seven77", "other": "tiny"}
    assert report.total_redactions == 0


def test_secret_like_camel_case_keys_are_redacted() -> None:
    payload = {
        "authToken": "synthetic-auth-token-value",
        "sessionToken": "synthetic-session-token-value",
        "secretValue": "synthetic-secret-value",
        "apiKey": "synthetic-api-key-value",
        "authorizationHeader": "synthetic-authorization-header-value",
        "passwordValue": "synthetic-password-value",
        "credentialId": "synthetic-credential-value",
        "cookieHeader": "synthetic-cookie-value",
        "xApiKey": "synthetic-x-api-key-value",
    }

    sanitized, report = sanitize_payload(payload, SecretRedactor())

    assert sanitized == {
        "authToken": "[REDACTED:key:token]",
        "sessionToken": "[REDACTED:key:token]",
        "secretValue": "[REDACTED:key:secret]",
        "apiKey": "[REDACTED:key:api_key]",
        "authorizationHeader": "[REDACTED:key:authorization]",
        "passwordValue": "[REDACTED:key:password]",
        "credentialId": "[REDACTED:key:credential]",
        "cookieHeader": "[REDACTED:key:cookie]",
        "xApiKey": "[REDACTED:key:x-api-key]",
    }
    assert report.counts_by_rule == {
        "key:token": 2,
        "key:secret": 1,
        "key:api_key": 1,
        "key:authorization": 1,
        "key:password": 1,
        "key:credential": 1,
        "key:cookie": 1,
        "key:x-api-key": 1,
    }


def test_langfuse_usage_counter_fields_are_preserved() -> None:
    payload = {
        "usage_details": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cached_input_tokens": 4,
            "cache_creation_tokens": 5,
            "cache_read_tokens": 6,
        }
    }

    sanitized, report = sanitize_payload(payload, SecretRedactor())

    assert sanitized == payload
    assert report.total_redactions == 0
