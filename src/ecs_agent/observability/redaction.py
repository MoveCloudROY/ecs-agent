"""Mandatory secret redaction for telemetry payloads."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ecs_agent.observability.schema import JsonSafe, json_safe


SECRET_KEY_NAMES: tuple[str, ...] = (
    "x-api-key",
    "set-cookie",
    "api_key",
    "secret",
    "token",
    "authorization",
    "password",
    "credential",
    "cookie",
)
SECRET_ENV_NAMES: tuple[str, ...] = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
MIN_SECRET_VALUE_LENGTH = 8


@dataclass(frozen=True, slots=True)
class RedactionRule:
    """Named redaction rule with a stable replacement marker."""

    name: str
    replacement: str


@dataclass(slots=True)
class RedactionReport:
    """Secret-safe redaction summary with counts and rule names only."""

    total_redactions: int = 0
    counts_by_rule: dict[str, int] = field(default_factory=dict)

    def record(self, rule: RedactionRule) -> None:
        """Record one redaction without storing the redacted value."""
        self.total_redactions += 1
        self.counts_by_rule[rule.name] = self.counts_by_rule.get(rule.name, 0) + 1

    def to_payload(self) -> dict[str, JsonSafe]:
        """Serialize the report into JSON-safe data without secret values."""
        return {
            "total_redactions": self.total_redactions,
            "counts_by_rule": json_safe(self.counts_by_rule),
        }


class SecretRedactor:
    """Recursively sanitize secret-like telemetry fields and values."""

    def __init__(
        self,
        *,
        extra_secret_values: Iterable[str] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        source_env = os.environ if env is None else env
        self._key_rules = tuple(
            RedactionRule(name=f"key:{key_name}", replacement=f"[REDACTED:key:{key_name}]")
            for key_name in SECRET_KEY_NAMES
        )
        self._value_rules = self._build_value_rules(source_env, extra_secret_values)

    def sanitize(self, payload: Any) -> tuple[JsonSafe, RedactionReport]:
        """Return a sanitized JSON-safe payload and redaction report."""
        report = RedactionReport()
        return self._sanitize_value(json_safe(payload), report), report

    def _build_value_rules(
        self,
        env: Mapping[str, str],
        extra_secret_values: Iterable[str],
    ) -> tuple[tuple[str, RedactionRule], ...]:
        rules: list[tuple[str, RedactionRule]] = []
        seen_values: set[str] = set()
        for env_name in SECRET_ENV_NAMES:
            value = env.get(env_name)
            if value is None or len(value) < MIN_SECRET_VALUE_LENGTH:
                continue
            if value in seen_values:
                continue
            seen_values.add(value)
            rules.append(
                (
                    value,
                    RedactionRule(
                        name=f"value:{env_name}",
                        replacement=f"[REDACTED:value:{env_name}]",
                    ),
                )
            )
        for value in extra_secret_values:
            if len(value) < MIN_SECRET_VALUE_LENGTH:
                continue
            if value in seen_values:
                continue
            seen_values.add(value)
            rules.append(
                (
                    value,
                    RedactionRule(
                        name="value:extra_secret",
                        replacement="[REDACTED:value:extra_secret]",
                    ),
                )
            )
        return tuple(rules)

    def _sanitize_value(
        self,
        value: JsonSafe,
        report: RedactionReport,
    ) -> JsonSafe:
        if isinstance(value, dict):
            sanitized: dict[str, JsonSafe] = {}
            for key, item in value.items():
                key_rule = self._matching_key_rule(key)
                if key_rule is not None:
                    report.record(key_rule)
                    sanitized[key] = key_rule.replacement
                else:
                    sanitized[key] = self._sanitize_value(item, report)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_value(item, report) for item in value]
        if isinstance(value, str):
            return self._sanitize_string(value, report)
        return value

    def _matching_key_rule(self, key: str) -> RedactionRule | None:
        key_lower = key.lower()
        normalized_key = key_lower.replace("_", "-")
        key_segments = self._key_segments(key)
        for rule in self._key_rules:
            key_name = rule.name.removeprefix("key:")
            normalized_rule = key_name.replace("_", "-")
            if normalized_key == normalized_rule:
                return rule
            if normalized_rule == "x-api-key" and {"x", "api", "key"}.issubset(key_segments):
                return rule
            if normalized_rule == "set-cookie" and {"set", "cookie"}.issubset(key_segments):
                return rule
            if key_name in {"authorization", "cookie", "credential", "password", "secret", "token"}:
                if key_name in key_segments:
                    return rule
            if key_name == "api_key" and "api" in key_segments and "key" in key_segments:
                return rule
        return None

    def _key_segments(self, key: str) -> set[str]:
        camel_spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
        acronym_spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_spaced)
        return {
            segment.lower()
            for segment in re.split(r"[_\-.\s]+", acronym_spaced)
            if segment
        }

    def _sanitize_string(
        self,
        value: str,
        report: RedactionReport,
    ) -> str:
        sanitized = value
        for secret_value, rule in self._value_rules:
            occurrences = sanitized.count(secret_value)
            if occurrences == 0:
                continue
            sanitized = sanitized.replace(secret_value, rule.replacement)
            for _ in range(occurrences):
                report.record(rule)
        return sanitized


def sanitize_payload(
    payload: Any,
    redactor: SecretRedactor,
) -> tuple[JsonSafe, RedactionReport]:
    """Sanitize telemetry payload before it reaches any sink or adapter."""
    return redactor.sanitize(payload)


__all__ = [
    "RedactionReport",
    "RedactionRule",
    "SecretRedactor",
    "sanitize_payload",
]
