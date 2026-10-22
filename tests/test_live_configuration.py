"""Configuration contracts independent of credentials and network access."""

import ast
from pathlib import Path

import pytest

from ecs_agent.providers.config import ApiFormat
from tests.support.config import load_settings, parse_settings, redact, safe_endpoint
from tests.support.environment import example_environment, offline_environment
from tests.support.models import LiveModelFactory


def document() -> dict:
    return {
        "profiles": {
            "primary": {
                "base_url": "https://gateway.example/v1",
                "model": "test-model",
                "api_format": "openai_responses",
                "api_key_env": "TEST_SECRET",
                "capabilities": ["text", "tools", "streaming"],
            }
        },
        "roles": {"primary": "primary"},
    }


def test_profile_is_atomic_and_credentials_are_lazy() -> None:
    data = document()
    data["profiles"]["unused"] = {
        **data["profiles"]["primary"],
        "api_key_env": "UNSET_SECRET",
    }
    settings = parse_settings(data, {"TEST_SECRET": "a-sensitive-value"}, source="test")
    assert settings.profile().key(settings.environment) == "a-sensitive-value"
    assert "a-sensitive-value" not in repr(settings)
    assert settings.profile().api_format is ApiFormat.OPENAI_RESPONSES
    with pytest.raises(TypeError):
        settings.roles["primary"] = "unused"
    with pytest.raises(ValueError, match="roles.secondary"):
        settings.profile("secondary")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", ""),
        ("base_url", "ftp://host"),
        ("base_url", "https://user:password@host"),
        ("api_format", "responses"),
        ("api_format", "openai_files"),
        ("api_key_env", "secret with spaces"),
        ("model", ""),
        ("capabilities", ["typo"]),
        ("api_key", "no-inline-secrets"),
    ],
)
def test_invalid_profile_fields_fail_without_network(field: str, value: object) -> None:
    data = document()
    data["profiles"]["primary"][field] = value
    with pytest.raises(ValueError):
        parse_settings(data, {}, source="test")


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True, "30"])
def test_timeouts_require_finite_positive_numbers(seconds: object) -> None:
    data = document()
    data["timeouts"] = {"request_seconds": seconds}
    with pytest.raises(ValueError, match="finite positive"):
        parse_settings(data, {}, source="test")


def test_profile_overrides_timeouts_and_selection_not_connection_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text("""[timeouts]
request_seconds = 45
[profiles.chosen]
base_url = "https://chosen.example"
model = "chosen-model"
api_format = "openai_responses"
api_key_env = "TEST_SECRET"
[profiles.chosen.timeouts]
stream_read_seconds = 25
""")
    settings = load_settings(
        path, {"LLM_BASE_URL": "https://ignored.example"}, profile_name="chosen"
    )
    assert settings.profile().base_url == "https://chosen.example"
    assert settings.profile().timeouts.request_seconds == 45
    assert settings.profile().timeouts.stream_read_seconds == 25
    assert "LLM_BASE_URL" in settings.notices[0]
    with pytest.raises(ValueError, match="unknown profile"):
        load_settings(path, {}, profile_name="missing")
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "absent.toml", {"LLM_API_KEY": "irrelevant"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("chat", ApiFormat.OPENAI_CHAT_COMPLETIONS),
        ("openai", ApiFormat.OPENAI_CHAT_COMPLETIONS),
        ("responses", ApiFormat.OPENAI_RESPONSES),
        ("anthropic", ApiFormat.ANTHROPIC_MESSAGES),
        ("  ANTHROPIC_MESSAGES ", ApiFormat.ANTHROPIC_MESSAGES),
    ],
)
def test_legacy_protocol_aliases_are_centralized(
    tmp_path: Path, raw: str, expected: ApiFormat
) -> None:
    settings = load_settings(
        None,
        {
            "LLM_API_FORMAT": raw,
            "LLM_BASE_URL": "https://legacy.example",
            "LLM_MODEL": "legacy-model",
        },
        local_path=tmp_path / "missing",
    )
    assert settings.profile().api_format is expected
    assert "Deprecated" in settings.notices[0]


def test_legacy_conflicts_fail_instead_of_choosing_a_vendor(tmp_path: Path) -> None:
    env = {
        "LLM_API_FORMAT": "responses",
        "LLM_BASE_URL": "https://one.example",
        "LLM_MODEL": "model",
        "ALIYUN_LIVE_RESPONSES_BASE_URL": "https://two.example",
    }
    with pytest.raises(ValueError, match="Conflicting legacy variables"):
        load_settings(None, env, local_path=tmp_path / "missing")
    del env["ALIYUN_LIVE_RESPONSES_BASE_URL"]
    del env["LLM_BASE_URL"]
    with pytest.raises(ValueError, match="base_url"):
        load_settings(None, env, local_path=tmp_path / "missing")


def test_legacy_requires_explicit_protocol_and_rejects_unknown(tmp_path: Path) -> None:
    for protocol in ("", "gemini"):
        with pytest.raises(ValueError):
            load_settings(
                None, {"LLM_API_FORMAT": protocol}, local_path=tmp_path / "missing"
            )


def test_secondary_and_service_keys_never_enter_child_offline_environment() -> None:
    data = document()
    data["profiles"]["other"] = {
        **data["profiles"]["primary"],
        "api_key_env": "SECOND_SECRET",
    }
    data["roles"]["secondary"] = "other"
    data["services"] = {
        "langfuse": {
            "host": "https://traces.example",
            "public_key_env": "PUBLIC_SECRET",
            "secret_key_env": "PRIVATE_SECRET",
        }
    }
    env = dict(
        TEST_SECRET="primary-value",
        SECOND_SECRET="other-value",
        PUBLIC_SECRET="public-value",
        PRIVATE_SECRET="private-value",
        PATH="/bin",
        PYTEST_ADDOPTS="--live",
        ECS_P1_LIVE="1",
    )
    settings = parse_settings(data, env, source="test")
    child = offline_environment(env, settings)
    assert child == {"PATH": "/bin"}
    assert env["TEST_SECRET"] == "primary-value"
    example = example_environment(settings)
    assert example["LLM_API_KEY"] == "primary-value"
    assert example["LLM_API_FORMAT"] == "openai_responses"
    assert "SECOND_SECRET" not in example
    assert example["LLM_ENABLE_STORE"] == "false"
    assert example["PLAN_TASK_LANGFUSE"] == "0"
    assert settings.service("langfuse")["secret_key"] == "private-value"
    assert (
        redact("primary-value other-value private-value", settings.secrets())
        == "[REDACTED] [REDACTED] [REDACTED]"
    )
    assert (
        safe_endpoint("https://user:password@host.example/path?token=secret#secret")
        == "https://host.example/path"
    )


async def test_factory_maps_timeouts_and_closes_direct_and_factory_clients() -> None:
    data = document()
    data["timeouts"] = {"request_seconds": 17, "stream_read_seconds": 23}
    settings = parse_settings(data, {"TEST_SECRET": "dummy"}, source="test")
    factory = LiveModelFactory(settings)
    model = factory.openai()
    assert model._provider_config.timeout == 17
    assert model._stream_read_timeout == 23
    assert factory.track(model) is model
    assert not model._client.is_closed
    await factory.aclose()
    assert model._client.is_closed
    await factory.aclose()
    with pytest.raises(ValueError, match="connection fields"):
        factory.create(base_url="https://other.example")


async def test_factory_closes_remaining_clients_if_one_close_raises() -> None:
    settings = parse_settings(document(), {}, source="test")
    factory = LiveModelFactory(settings)
    closed = []

    class Client:
        async def aclose(self) -> None:
            closed.append(self)
            if len(closed) == 1:
                raise RuntimeError("close failed")

    first, second = Client(), Client()
    factory.track(first)
    factory.track(second)
    with pytest.raises(RuntimeError, match="close failed"):
        await factory.aclose()
    assert closed == [second, first]


def test_live_scenarios_do_not_read_environment_or_skip_failed_requests() -> None:
    for path in (Path(__file__).parent / "live").glob("test_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in {
                    "os.getenv",
                    "os.environ.get",
                    "pytest.skip",
                    "pytest.importorskip",
                }, path
            if isinstance(node, ast.Subscript):
                assert ast.unparse(node.value) != "os.environ", path
