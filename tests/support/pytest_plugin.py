"""Explicit live selection, preflight, resource ownership and coverage gates."""

import asyncio
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import wraps
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pytest
import pytest_asyncio

from tests.support.config import (
    CAPABILITIES,
    FORMATS,
    LiveSettings,
    Profile,
    load_settings,
    redact,
    safe_endpoint,
)
from tests.support.models import LiveModelFactory


@dataclass(slots=True)
class LiveRun:
    """Session-local state, never shared between nested pytest sessions."""

    settings: LiveSettings | None = None
    selected: dict[str, tuple[str, str]] = field(default_factory=dict)
    required: set[str] = field(default_factory=set)
    outcomes: dict[str, str] = field(default_factory=dict)
    deselected: int = 0
    problems: list[str] = field(default_factory=list)


STATE = pytest.StashKey[LiveRun]()
SCENARIOS = (
    "smoke",
    "authoring",
    "vision",
    "cache",
    "observability",
    "workflow",
    "tools",
    "subagent",
    "compaction",
    "session",
    "prompts",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("live service tests")
    group.addoption(
        "--live", action="store_true", help="Explicitly enable real service tests"
    )
    group.addoption(
        "--live-config", type=Path, help="TOML file containing atomic model profiles"
    )
    group.addoption("--llm-profile", help="Select the primary model profile")
    group.addoption(
        "--live-required",
        type=Path,
        help="JSON list of exact node IDs that must all pass",
    )
    group.addoption(
        "--live-report", type=Path, help="Write a redacted JSON live summary"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[STATE] = LiveRun()
    config.addinivalue_line(
        "markers",
        "live(formats, capabilities, role, roles, services, dependencies): real service requirements",
    )
    for name in SCENARIOS:
        config.addinivalue_line("markers", f"{name}: live {name} scenario")


def requirements(item: pytest.Item) -> dict[str, Any]:
    """Merge module defaults and test requirements; protocol parameters win."""
    result: dict[str, Any] = {
        "capabilities": ["text"],
        "role": "primary",
        "roles": [],
        "services": [],
        "dependencies": [],
    }
    for mark in reversed(list(item.iter_markers("live"))):
        unknown = set(mark.kwargs) - {
            "formats",
            "capabilities",
            "role",
            "roles",
            "services",
            "dependencies",
        }
        if unknown:
            raise ValueError(
                f"{item.nodeid}: unknown live requirements {sorted(unknown)}"
            )
        result.update(mark.kwargs)
    callspec = getattr(item, "callspec", None)
    if callspec and "api_format" in callspec.params:
        result["formats"] = [callspec.params["api_format"]]
    if not set(result["capabilities"]) <= CAPABILITIES:
        raise ValueError(f"{item.nodeid}: unknown capability")
    if "formats" in result and not set(result["formats"]) <= FORMATS:
        raise ValueError(f"{item.nodeid}: unknown protocol requirement")
    return result


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    state = config.stash[STATE]
    live = [item for item in items if item.get_closest_marker("live")]
    if not config.getoption("--live"):
        if config.getoption("--live-required"):
            raise pytest.UsageError("--live-required requires --live")
        if live:
            items[:] = [item for item in items if item not in live]
            config.hook.pytest_deselected(items=live)
        return
    try:
        required_path = config.getoption("--live-required")
        if required_path:
            required = json.loads(required_path.read_text())
            if (
                not isinstance(required, list)
                or not required
                or not all(isinstance(n, str) for n in required)
                or len(set(required)) != len(required)
            ):
                raise ValueError(
                    "--live-required must contain a non-empty unique JSON list of node IDs"
                )
            state.required = set(required)
            missing = state.required - {item.nodeid for item in live}
            if missing:
                raise ValueError(f"Required live cases not selected: {sorted(missing)}")
        if not live:
            raise ValueError("--live selected no live cases; check paths, -m and -k")
        state.settings = load_settings(
            config.getoption("--live-config"),
            os.environ,
            profile_name=config.getoption("--llm-profile"),
            local_path=config.rootpath / "tests/live/profiles.local.toml",
        )
        settings = state.settings
        applicable = 0
        for item in live:
            req = requirements(item)
            profile = settings.profile(req["role"])
            scenario = next(
                (n for n in SCENARIOS if item.get_closest_marker(n)), "live"
            )
            state.selected[item.nodeid] = (profile.name, scenario)
            missing_caps = set(req["capabilities"]) - profile.capabilities
            wrong_format = "formats" in req and profile.api_format not in req["formats"]
            if wrong_format or missing_caps:
                reason = f"not applicable: profile {profile.name}; protocol={profile.api_format.value}; missing capabilities={sorted(missing_caps)}"
                if item.nodeid in state.required:
                    raise ValueError(f"Required case {item.nodeid}: {reason}")
                item.add_marker(pytest.mark.skip(reason=reason))
                continue
            applicable += 1
            _bound_case(
                item,
                settings.cache_case_seconds
                if item.get_closest_marker("cache")
                else profile.timeouts.case_seconds,
            )
            for role in [req["role"], *req["roles"]]:
                p = settings.profile(role)
                if not config.option.collectonly:
                    p.key(settings.environment)
            if "vision" in req["capabilities"] and not settings.image_url:
                raise ValueError(
                    "Configure assets.image_url or assets.image_url_env for vision tests"
                )
            for service in req["services"]:
                if not config.option.collectonly:
                    settings.service(service)
            for dependency in req["dependencies"]:
                present = (
                    shutil.which("tmux")
                    if dependency == "tmux"
                    else importlib.util.find_spec(dependency)
                )
                if not present:
                    raise ValueError(
                        f"{item.nodeid}: install required dependency {dependency}"
                    )
        if applicable == 0:
            raise ValueError("No applicable live cases for the selected profile")
    except (ValueError, OSError) as exc:
        secrets = state.settings.secrets() if state.settings else ()
        raise pytest.UsageError(redact(str(exc), secrets)) from None


def pytest_deselected(items: list[pytest.Item]) -> None:
    if items:
        items[0].config.stash[STATE].deselected += sum(
            bool(i.get_closest_marker("live")) for i in items
        )


@pytest.fixture(scope="session")
def live_settings(pytestconfig: pytest.Config) -> LiveSettings:
    settings = pytestconfig.stash[STATE].settings
    if settings is None:
        raise RuntimeError("Live fixtures require --live and a selected live marker")
    return settings


@pytest.fixture
def live_profile(
    request: pytest.FixtureRequest, live_settings: LiveSettings
) -> Profile:
    return live_settings.profile(requirements(request.node)["role"])


@pytest_asyncio.fixture
async def live_model_factory(
    request: pytest.FixtureRequest, live_settings: LiveSettings
) -> AsyncIterator[LiveModelFactory]:
    factory = LiveModelFactory(live_settings, requirements(request.node)["role"])
    initial_tasks = asyncio.all_tasks()
    try:
        yield factory
    finally:
        # Function-scoped live tests own tasks started after this fixture. Keep
        # pre-existing fixture tasks and the teardown task itself untouched.
        pending = asyncio.all_tasks() - initial_tasks - {asyncio.current_task()}
        try:
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*pending, return_exceptions=True)
            errors = [
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            ]
            if errors:
                raise BaseExceptionGroup("Live background task cleanup failed", errors)
        finally:
            await factory.aclose()


@pytest.fixture
def live_api_key(live_profile: Profile, live_settings: LiveSettings) -> str:
    return live_profile.key(live_settings.environment)


@pytest.fixture
def live_image_url(live_settings: LiveSettings) -> str:
    return live_settings.image_url


@pytest.fixture
def live_image(live_image_url: str) -> str:
    return live_image_url


@pytest.fixture
def live_model(live_model_factory: LiveModelFactory) -> Any:
    return live_model_factory.create()


def _bound_case(item: pytest.Item, seconds: float) -> None:
    """Wrap the coroutine before pytest-asyncio synchronizes it at runtime."""
    if not isinstance(item, pytest.Function) or not asyncio.iscoroutinefunction(
        item.obj
    ):
        return
    target = item.obj

    @wraps(target)
    async def bounded(**kwargs: Any) -> None:
        async with asyncio.timeout(seconds):
            await target(**kwargs)

    item.obj = bounded


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    outcome = yield
    report = outcome.get_result()
    state = item.config.stash[STATE]
    if item.nodeid not in state.selected:
        return
    assert state.settings is not None
    secrets = state.settings.secrets()
    if report.longrepr:
        if isinstance(report.longrepr, tuple):
            path, line, reason = report.longrepr
            report.longrepr = (path, line, redact(reason, secrets))
        else:
            report.longrepr = redact(str(report.longrepr), secrets)
    report.sections = [(name, redact(body, secrets)) for name, body in report.sections]
    if report.failed:
        state.outcomes[item.nodeid] = "failed"
    elif report.skipped:
        state.outcomes[item.nodeid] = (
            "xfail"
            if hasattr(report, "wasxfail")
            else "not_applicable"
            if "not applicable:" in str(report.longrepr)
            else "skipped"
        )
    elif report.when == "call" and state.outcomes.get(item.nodeid) != "failed":
        state.outcomes[item.nodeid] = (
            "xpass" if hasattr(report, "wasxfail") else "passed"
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = session.config.stash[STATE]
    if (
        not session.config.getoption("--live")
        or session.config.option.collectonly
        or not state.selected
    ):
        return
    missing = sorted(n for n in state.required if state.outcomes.get(n) != "passed")
    if missing:
        state.problems.append(f"Required cases did not pass: {missing}")
    if not any(v in {"passed", "failed"} for v in state.outcomes.values()):
        state.problems.append(
            "No live case executed successfully or failed: all skipped/not applicable"
        )
    if state.problems and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    report_path = session.config.getoption("--live-report")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "cases": {
                        n: {
                            "profile": p,
                            "scenario": s,
                            "outcome": state.outcomes.get(n, "not_run"),
                        }
                        for n, (p, s) in state.selected.items()
                    },
                    "deselected": state.deselected,
                    "problems": state.problems,
                },
                indent=2,
            )
            + "\n"
        )


def pytest_report_collectionfinish(config: pytest.Config) -> list[str]:
    """Show the effective non-secret configuration before any scenario runs."""
    state = config.stash[STATE]
    if state.settings is None:
        return []
    settings = state.settings
    lines = []
    for name in sorted({p for p, _ in state.selected.values()}):
        p = settings.profiles[name]
        lines.append(
            f"Live profile {name}: {p.model} / {p.api_format.value} / {safe_endpoint(p.base_url)}; {p.timeouts}; source={settings.source}"
        )
    return lines + list(settings.notices)


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    state = config.stash[STATE]
    if not config.getoption("--live"):
        terminalreporter.write_line(
            f"Live disabled: {state.deselected} real-service cases deselected (use --live)."
        )
        return
    terminalreporter.section("live coverage")
    if state.settings:
        settings = state.settings
        used = {p for p, _ in state.selected.values()}
        for name in sorted(used):
            p = settings.profiles[name]
            terminalreporter.write_line(
                f"{name}: {p.model}; {p.api_format.value}; {safe_endpoint(p.base_url)}; {p.timeouts}; source={settings.source}"
            )
        for notice in settings.notices:
            terminalreporter.write_line(notice)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for nodeid, group in state.selected.items():
        counts[group][state.outcomes.get(nodeid, "not_run")] += 1
    for (profile, scenario), totals in sorted(counts.items()):
        terminalreporter.write_line(f"{profile}/{scenario}: {dict(totals)}")
    terminalreporter.write_line(f"Live deselected by selection: {state.deselected}")
    for problem in state.problems:
        terminalreporter.write_line(problem, red=True)
