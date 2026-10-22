"""Parse atomic live profiles without reading process state or opening clients."""

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
import tomllib
from urllib.parse import urlsplit

from ecs_agent.providers.config import ApiFormat

FORMATS = frozenset(
    {
        ApiFormat.OPENAI_CHAT_COMPLETIONS,
        ApiFormat.OPENAI_RESPONSES,
        ApiFormat.ANTHROPIC_MESSAGES,
    }
)
CAPABILITIES = frozenset(
    {
        "text",
        "tools",
        "streaming",
        "vision",
        "cache",
        "response_chaining",
        "parallel_tools",
        "multi_model",
    }
)
ALIASES = {
    "chat": "openai_chat_completions",
    "openai": "openai_chat_completions",
    "responses": "openai_responses",
    "anthropic": "anthropic_messages",
}
LEGACY_SWITCHES = (
    "ECS_P1_LIVE",
    "ECS_P2_LIVE",
    "ECS_AUTHORING_LIVE",
    "RUN_LANGFUSE_LIVE_TESTS",
)
LEGACY_CONFIG = (
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_FORMAT",
    "LLM_RESPONSES_BASE_URL",
    "LLM_MODEL_ALTERNATIVE",
    "LLM_LIVE_TIMEOUT",
    "ALIYUN_LIVE_CHAT_BASE_URL",
    "ALIYUN_LIVE_RESPONSES_BASE_URL",
    "ALIYUN_LIVE_MODEL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_LIVE_BASE_URL",
    "ANTHROPIC_LIVE_MODEL",
    "LLM_CACHE_WRITE_LAG_SECONDS",
    "LLM_CACHE_POLL_INTERVAL_SECONDS",
)


@dataclass(frozen=True, slots=True)
class Timeouts:
    """Independent request, streaming-read and whole-case budgets, in seconds."""

    request_seconds: float = 60
    stream_read_seconds: float = 60
    case_seconds: float = 180


@dataclass(frozen=True, slots=True)
class Profile:
    """An indivisible connection and its explicitly declared capabilities."""

    name: str
    base_url: str
    model: str
    api_format: ApiFormat
    api_key_env: str
    capabilities: frozenset[str]
    timeouts: Timeouts = field(default_factory=Timeouts)

    def key(self, environment: Mapping[str, str]) -> str:
        """Resolve a credential only when this profile is actually required."""
        value = environment.get(self.api_key_env, "").strip()
        if not value:
            raise ValueError(
                f"Profile {self.name}: set credential variable {self.api_key_env}"
            )
        return value


@dataclass(frozen=True, slots=True)
class LiveSettings:
    """Immutable configuration, with credentials held outside diagnostic repr."""

    profiles: Mapping[str, Profile]
    roles: Mapping[str, str]
    environment: Mapping[str, str] = field(repr=False)
    source: str = ""
    notices: tuple[str, ...] = ()
    image_url: str = field(default="", repr=False)
    langfuse: Mapping[str, str] = field(default_factory=dict, repr=False)
    cache_write_lag_seconds: float = 150
    cache_poll_interval_seconds: float = 30
    cache_case_seconds: float = 1200

    def profile(self, role: str = "primary") -> Profile:
        """Resolve a role without silently falling back to another connection."""
        name = self.roles.get(role)
        if name is None:
            raise ValueError(f"Configure roles.{role} for the selected scenario")
        return self.profiles[name]

    def service(self, name: str) -> dict[str, str]:
        """Resolve required service credentials without mixing them with LLM keys."""
        if name != "langfuse":
            raise ValueError(f"Unknown live service: {name}")
        result = {"host": self.langfuse.get("host", "")}
        for field_name in ("public_key", "secret_key"):
            env_name = self.langfuse.get(f"{field_name}_env", "")
            result[field_name] = self.environment.get(env_name, "")
        if not all(result.values()):
            raise ValueError(
                "Configure services.langfuse host/public_key_env/secret_key_env and referenced credentials"
            )
        return result

    def secrets(self) -> tuple[str, ...]:
        """Values to redact from reports, including credentials for secondary roles."""
        names = {p.api_key_env for p in self.profiles.values()}
        names.update(v for k, v in self.langfuse.items() if k.endswith("_env"))
        return tuple(self.environment[n] for n in names if self.environment.get(n))


def safe_endpoint(url: str) -> str:
    """Display only the endpoint origin and path, never URL credentials/query."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or ''}{':' + str(parts.port) if parts.port else ''}{parts.path}"


def redact(text: str, secrets: tuple[str, ...]) -> str:
    """Remove known secrets and common authorization tokens from diagnostics."""
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?:sk-|pk-)[A-Za-z0-9_-]{12,}", "[REDACTED]", text)
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", text)
    return text


def _table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table")
    return value


def _fields(data: dict[str, Any], allowed: set[str], label: str) -> None:
    if set(data) - allowed:
        raise ValueError(f"{label}: unknown fields {sorted(set(data) - allowed)}")


def _string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _seconds(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _timeouts(data: Any, default: Timeouts) -> Timeouts:
    data = _table(data, "timeouts")
    _fields(
        data, {"request_seconds", "stream_read_seconds", "case_seconds"}, "timeouts"
    )
    return replace(
        default, **{k: _seconds(v, f"timeouts.{k}") for k, v in data.items()}
    )


def _url(value: str, label: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"https", "http"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            f"{label} must be an HTTP(S) base URL without credentials, query or fragment"
        )
    _ = parts.port
    return value.rstrip("/")


def _env_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"{label} must name an environment variable")
    return value


def parse_settings(
    data: dict[str, Any],
    environment: Mapping[str, str],
    *,
    source: str,
    profile_name: str | None = None,
    notices: tuple[str, ...] = (),
) -> LiveSettings:
    """Validate a TOML-shaped document; do not require unused credentials."""
    _fields(
        data,
        {"profiles", "roles", "timeouts", "assets", "services", "cache"},
        "live config",
    )
    defaults = _timeouts(data.get("timeouts", {}), Timeouts())
    profiles = {}
    for name, raw in _table(data.get("profiles"), "profiles").items():
        raw = _table(raw, f"profiles.{name}")
        _fields(
            raw,
            {
                "base_url",
                "model",
                "api_format",
                "api_key_env",
                "capabilities",
                "timeouts",
            },
            f"profiles.{name}",
        )
        try:
            api_format = ApiFormat(_string(raw, "api_format", name))
        except ValueError:
            raise ValueError(
                f"Profile {name}: api_format must be a canonical LLM protocol name"
            ) from None
        if api_format not in FORMATS:
            raise ValueError(f"Profile {name}: unsupported LLM protocol")
        caps = raw.get("capabilities", ["text"])
        if not isinstance(caps, list) or not all(
            isinstance(c, str) and c in CAPABILITIES for c in caps
        ):
            raise ValueError(
                f"Profile {name}: capabilities must be drawn from {sorted(CAPABILITIES)}"
            )
        profiles[name] = Profile(
            name,
            _url(_string(raw, "base_url", name), name),
            _string(raw, "model", name),
            api_format,
            _env_name(_string(raw, "api_key_env", name), name),
            frozenset(caps),
            _timeouts(raw.get("timeouts", {}), defaults),
        )
    roles = _table(data.get("roles", {}), "roles").copy()
    if profile_name:
        roles["primary"] = profile_name
    if not roles.get("primary"):
        raise ValueError("Select --llm-profile NAME or configure roles.primary")
    for role, name in roles.items():
        if not isinstance(name, str) or name not in profiles:
            raise ValueError(f"Role {role} references an unknown profile")
    assets = _table(data.get("assets", {}), "assets")
    _fields(assets, {"image_url", "image_url_env"}, "assets")
    if "image_url" in assets and "image_url_env" in assets:
        raise ValueError("Choose assets.image_url or image_url_env, not both")
    image_url = ""
    if "image_url" in assets:
        image_url = _string(assets, "image_url", "assets")
    elif "image_url_env" in assets:
        image_url = environment.get(
            _env_name(_string(assets, "image_url_env", "assets"), "assets"), ""
        )
    services = _table(data.get("services", {}), "services")
    _fields(services, {"langfuse"}, "services")
    langfuse = _table(services.get("langfuse", {}), "services.langfuse")
    _fields(langfuse, {"host", "public_key_env", "secret_key_env"}, "services.langfuse")
    if langfuse:
        langfuse = {
            k: _string(langfuse, k, "services.langfuse")
            for k in ("host", "public_key_env", "secret_key_env")
        }
        langfuse["host"] = _url(langfuse["host"], "services.langfuse.host")
        for k in ("public_key_env", "secret_key_env"):
            _env_name(langfuse[k], k)
    cache = _table(data.get("cache", {}), "cache")
    _fields(
        cache, {"write_lag_seconds", "poll_interval_seconds", "case_seconds"}, "cache"
    )
    return LiveSettings(
        MappingProxyType(profiles),
        MappingProxyType(roles),
        MappingProxyType(dict(environment)),
        source,
        notices,
        image_url,
        MappingProxyType(langfuse),
        _seconds(cache.get("write_lag_seconds", 150), "cache.write_lag_seconds"),
        _seconds(cache.get("poll_interval_seconds", 30), "cache.poll_interval_seconds"),
        _seconds(cache.get("case_seconds", 1200), "cache.case_seconds"),
    )


def _legacy(environment: Mapping[str, str]) -> dict[str, Any]:
    def one(*names: str) -> str:
        values = {
            environment[n].strip() for n in names if environment.get(n, "").strip()
        }
        if len(values) > 1:
            raise ValueError(f"Conflicting legacy variables: {', '.join(names)}")
        return next(iter(values), "")

    raw_format = one("LLM_API_FORMAT").lower()
    if not raw_format:
        raise ValueError(
            "Configure --live-config PATH, or provide explicit LLM_API_FORMAT/LLM_BASE_URL/LLM_MODEL and LLM_API_KEY"
        )
    fmt = ALIASES.get(raw_format, raw_format)
    api_format = ApiFormat(fmt)
    if api_format is ApiFormat.ANTHROPIC_MESSAGES:
        base = one("LLM_BASE_URL", "ANTHROPIC_BASE_URL", "ANTHROPIC_LIVE_BASE_URL")
        model = one("LLM_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_LIVE_MODEL")
        one("LLM_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_LIVE_API_KEY")
        key_env = next(
            (
                n
                for n in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_LIVE_API_KEY")
                if environment.get(n)
            ),
            "LLM_API_KEY",
        )
    else:
        specific = (
            ("LLM_RESPONSES_BASE_URL", "ALIYUN_LIVE_RESPONSES_BASE_URL")
            if api_format is ApiFormat.OPENAI_RESPONSES
            else ("ALIYUN_LIVE_CHAT_BASE_URL",)
        )
        base = one("LLM_BASE_URL", *specific)
        model = one("LLM_MODEL", "ALIYUN_LIVE_MODEL")
        key_env = "LLM_API_KEY"
    primary: dict[str, Any] = dict(
        base_url=base,
        model=model,
        api_format=fmt,
        api_key_env=key_env,
        capabilities=["text", "tools", "streaming"],
    )
    data: dict[str, Any] = {
        "profiles": {"legacy": primary},
        "roles": {"primary": "legacy"},
        "assets": {"image_url_env": "IMAGE_URL"},
    }
    if environment.get("LLM_MODEL_ALTERNATIVE"):
        data["profiles"]["legacy-secondary"] = {
            **primary,
            "model": environment["LLM_MODEL_ALTERNATIVE"],
        }
        data["roles"]["secondary"] = "legacy-secondary"
        primary["capabilities"].append("multi_model")
    host = one("LANGFUSE_HOST", "LANGFUSE_BASE_URL")
    if host:
        data["services"] = {
            "langfuse": dict(
                host=host,
                public_key_env="LANGFUSE_PUBLIC_KEY",
                secret_key_env="LANGFUSE_SECRET_KEY",
            )
        }
    if environment.get("LLM_LIVE_TIMEOUT", "").strip():
        data["timeouts"] = {"request_seconds": float(environment["LLM_LIVE_TIMEOUT"])}
    for old, new in [
        ("LLM_CACHE_WRITE_LAG_SECONDS", "write_lag_seconds"),
        ("LLM_CACHE_POLL_INTERVAL_SECONDS", "poll_interval_seconds"),
    ]:
        if environment.get(old, "").strip():
            data.setdefault("cache", {})[new] = float(environment[old])
    return data


def load_settings(
    path: Path | None,
    environment: Mapping[str, str],
    *,
    profile_name: str | None = None,
    local_path: Path = Path("tests/live/profiles.local.toml"),
) -> LiveSettings:
    """Read one explicit/local file or centrally adapt legacy environment input."""
    chosen = path if path is not None else (local_path if local_path.exists() else None)
    if chosen is not None:
        with chosen.open("rb") as stream:
            data = tomllib.load(stream)
        old = [n for n in (*LEGACY_CONFIG, *LEGACY_SWITCHES) if environment.get(n)]
        notices = (
            ("Ignored legacy overrides/switches: " + ", ".join(old),) if old else ()
        )
        return parse_settings(
            data,
            environment,
            source=str(chosen),
            profile_name=profile_name,
            notices=notices,
        )
    used = ", ".join(
        n for n in (*LEGACY_CONFIG, *LEGACY_SWITCHES) if environment.get(n)
    )
    return parse_settings(
        _legacy(environment),
        environment,
        source="legacy environment",
        profile_name=profile_name,
        notices=(
            f"Deprecated legacy configuration: {used}; migrate to a profile file. Old switches do not enable live.",
        ),
    )
