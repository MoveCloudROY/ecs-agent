"""Exercise pytest selection, preflight and reporting in isolated processes."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def suite(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> pytest.Pytester:
    root = Path(__file__).resolve().parents[1]
    pytester.makeconftest(
        f'import sys\nsys.path.insert(0, {str(root)!r})\npytest_plugins = ["tests.support.pytest_plugin"]\n'
    )
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    (pytester.path / "profiles.toml").write_text("""[profiles.test]
base_url = "https://never-contact.example"
model = "dummy"
api_format = "openai_responses"
api_key_env = "CONFIG_TEST_KEY"
capabilities = ["text"]
[roles]
primary = "test"
""")
    monkeypatch.setenv("CONFIG_TEST_KEY", "credential-not-for-network-123")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    return pytester


def args() -> tuple[str, ...]:
    return ("--live", "--live-config=profiles.toml", "-q")


def test_default_offline_even_with_credentials_and_explicit_live_path(
    suite: pytest.Pytester,
) -> None:
    suite.makepyfile("""import pytest
@pytest.mark.live
def test_live():
    raise AssertionError("must not execute")
def test_offline():
    assert True
""")
    result = suite.runpytest_subprocess("-q")
    result.assert_outcomes(passed=1, deselected=1)
    assert result.ret == 0


def test_preflight_missing_key_runs_no_tests(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONFIG_TEST_KEY")
    suite.makepyfile("""import pytest
from pathlib import Path
@pytest.mark.live
def test_live():
    Path("ran").touch()
""")
    result = suite.runpytest_subprocess(*args())
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert not (suite.path / "ran").exists()
    assert "CONFIG_TEST_KEY" in result.stderr.str()


def test_user_selection_precedes_service_validation(suite: pytest.Pytester) -> None:
    suite.makepyfile("""import pytest
@pytest.mark.live
def test_text():
    assert True
@pytest.mark.live(services=["langfuse"])
def test_traces():
    assert False
""")
    suite.runpytest_subprocess(*args(), "-k", "text").assert_outcomes(
        passed=1, deselected=1
    )


def test_inapplicable_capability_and_service_failure_have_different_results(
    suite: pytest.Pytester,
) -> None:
    suite.makepyfile("""import pytest
@pytest.mark.live(capabilities=["vision"])
def test_image():
    assert False
@pytest.mark.live
def test_timeout():
    raise TimeoutError("remote timed out")
""")
    result = suite.runpytest_subprocess(*args())
    result.assert_outcomes(failed=1, skipped=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_no_applicable_live_cases_is_error(suite: pytest.Pytester) -> None:
    suite.makepyfile("""import pytest
@pytest.mark.live(formats=["anthropic_messages"])
def test_inapplicable():
    assert False
""")
    assert suite.runpytest_subprocess(*args()).ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize(
    "outcome", ['pytest.skip("unexpected skip")', 'pytest.xfail("unexpected xfail")']
)
def test_all_skipped_or_xfailed_live_is_not_success(
    suite: pytest.Pytester, outcome: str
) -> None:
    suite.makepyfile(
        f"import pytest\n@pytest.mark.live\ndef test_case():\n    {outcome}\n"
    )
    assert suite.runpytest_subprocess(*args()).ret == pytest.ExitCode.TESTS_FAILED


def test_required_cases_detect_deselection_and_skip(suite: pytest.Pytester) -> None:
    suite.makepyfile(
        test_cases="""import pytest
@pytest.mark.live
def test_one():
    assert True
@pytest.mark.live
def test_two():
    pytest.skip("not acceptable for release")
"""
    )
    path = suite.path / "required.json"
    path.write_text(json.dumps(["test_cases.py::test_one", "test_cases.py::test_two"]))
    result = suite.runpytest_subprocess(
        *args(), "--live-required=required.json", "-k", "one"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result = suite.runpytest_subprocess(*args(), "--live-required=required.json")
    result.assert_outcomes(passed=1, skipped=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_case_timeout_cleans_up_factory_and_redacts_failure(
    suite: pytest.Pytester,
) -> None:
    path = suite.path / "profiles.toml"
    path.write_text("[timeouts]\ncase_seconds = 0.03\n" + path.read_text())
    suite.makepyfile("""import asyncio
from pathlib import Path
import pytest
@pytest.mark.live
async def test_timeout(live_model_factory):
    class Client:
        async def aclose(self):
            assert Path("cancelled").exists()
            Path("closed").touch()
    async def background():
        try:
            await asyncio.sleep(30)
        finally:
            Path("cancelled").touch()
    live_model_factory.track(Client())
    asyncio.create_task(background())
    await asyncio.sleep(1)
@pytest.mark.live
def test_secret(live_api_key):
    print(live_api_key)
    raise AssertionError(live_api_key)
""")
    result = suite.runpytest_subprocess(
        *args(), "--showlocals", "--live-report=report.json"
    )
    result.assert_outcomes(failed=2)
    assert (suite.path / "closed").exists()
    assert (
        "credential-not-for-network-123"
        not in result.stdout.str() + result.stderr.str()
    )
    assert "[REDACTED]" in result.stdout.str()
    assert len(json.loads((suite.path / "report.json").read_text())["cases"]) == 2


def test_live_marker_not_directory_controls_execution(suite: pytest.Pytester) -> None:
    suite.makepyfile(
        test_elsewhere="""import pytest
@pytest.mark.live
def test_selected(live_profile):
    assert live_profile.model == "dummy"
"""
    )
    suite.runpytest_subprocess(*args()).assert_outcomes(passed=1)
