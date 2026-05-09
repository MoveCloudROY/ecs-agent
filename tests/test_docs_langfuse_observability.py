"""Documentation safety tests for Langfuse observability.

These tests ensure that Langfuse documentation exists in both README.md
and docs/features/langfuse.md, contains required information, and does
not leak secrets or provide unsafe examples.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
README_PATH = PROJECT_ROOT / "README.md"
LANGFUSE_DOCS_PATH = PROJECT_ROOT / "docs" / "features" / "langfuse.md"


def get_readme_content() -> str:
    """Helper to get README content and fail if Langfuse section is missing."""
    if not README_PATH.exists():
        pytest.fail(f"Missing README file: {README_PATH}")
    content = README_PATH.read_text(encoding="utf-8")
    if "Langfuse" not in content:
        pytest.fail("README is missing Langfuse documentation section.")
    return content


def get_langfuse_docs_content() -> str:
    """Helper to get dedicated Langfuse docs content and fail if file is missing."""
    if not LANGFUSE_DOCS_PATH.exists():
        pytest.fail(f"Missing dedicated Langfuse documentation file: {LANGFUSE_DOCS_PATH}")
    return LANGFUSE_DOCS_PATH.read_text(encoding="utf-8")


def test_langfuse_docs_describe_optional_extra_and_install_api() -> None:
    """README and docs must mention ecs-agent[langfuse] and install_langfuse_observability."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "ecs-agent[langfuse]" in content, f"{source} missing installation instructions for ecs-agent[langfuse]"
        assert "install_langfuse_observability" in content, f"{source} missing mention of install_langfuse_observability"


def test_langfuse_docs_mention_configuration() -> None:
    """README and docs must mention Langfuse configuration environment variables."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "LANGFUSE_PUBLIC_KEY" in content, f"{source} missing LANGFUSE_PUBLIC_KEY"
        assert "LANGFUSE_SECRET_KEY" in content, f"{source} missing LANGFUSE_SECRET_KEY"
        has_host = "LANGFUSE_HOST" in content or "LANGFUSE_BASE_URL" in content
        assert has_host, f"{source} missing LANGFUSE_HOST or LANGFUSE_BASE_URL alias policy"


def test_langfuse_docs_describe_export_timeout_controls() -> None:
    """Docs must explain how to tune self-hosted Langfuse export timeouts."""
    docs = get_langfuse_docs_content()

    required_phrases = [
        "LangfuseConfig(timeout=",
        "LANGFUSE_TIMEOUT",
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
        "read timeout",
        "flush_interval",
    ]
    for phrase in required_phrases:
        assert phrase in docs, f"docs/features/langfuse.md missing timeout guidance: {phrase}"


def test_langfuse_docs_describe_session_attribute_propagation() -> None:
    """Docs must explain Langfuse Sessions need trace-level session propagation."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "session_id" in content, f"{source} missing session_id guidance"
        assert "trace-level" in content, f"{source} missing trace-level session guidance"
        assert "metadata" in content, f"{source} missing metadata-only session warning"

    assert "propagate_attributes" in docs
    assert "start_as_current_observation" in docs


def test_langfuse_docs_mention_skip_behavior() -> None:
    """README and docs must mention live-test skip behavior."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "skip" in content.lower(), f"{source} must mention skip behavior for live tests"
        assert "live" in content.lower(), f"{source} must mention live tests"


def test_langfuse_docs_describe_generation_nested_tool_observations() -> None:
    """Docs must explain that tool/cache work stays attached to the calling generation."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    expected_phrases = [
        "generation",
        "tool",
        "requested",
    ]

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        lowered = content.lower()
        for phrase in expected_phrases:
            assert phrase in lowered, f"{source} missing nested tool/generation guidance: {phrase}"

    assert "tool calls nest under the generation that requested them" in docs.lower()


def test_langfuse_docs_mention_credential_rotation() -> None:
    """README and docs must mention credential rotation."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "rotation" in content.lower() or "rotate" in content.lower(), f"{source} must mention credential rotation"


def test_langfuse_docs_describe_capture_controls_and_private_otel_opt_in() -> None:
    """Docs must keep Langfuse safety controls visible where behavior is advertised."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()
    plan_task_readme = (PROJECT_ROOT / "examples" / "e2e" / "plan_and_task" / "README.md").read_text(
        encoding="utf-8"
    )

    for content, source in [
        (readme, "README"),
        (docs, "docs/features/langfuse.md"),
        (plan_task_readme, "examples/e2e/plan_and_task/README.md"),
    ]:
        assert "capture_input=False" in content, f"{source} missing capture_input opt-out guidance"
        assert "capture_output=False" in content, f"{source} missing capture_output opt-out guidance"
        assert "enable_private_v4_historical_otel" in content, f"{source} missing private OTel opt-in guidance"


def test_langfuse_docs_explain_trace_and_observation_roots() -> None:
    """Dedicated docs must distinguish Langfuse trace containers from root observations."""
    docs = get_langfuse_docs_content()

    required_phrases = [
        "Session > Trace > Observation",
        "trace container",
        "root observation",
        "root_observation_id",
        "parent_observation_id",
        "Span / Generation / Event",
    ]
    for phrase in required_phrases:
        assert phrase in docs, f"docs/features/langfuse.md missing trace-root concept: {phrase}"


def test_langfuse_docs_do_not_include_secret_values() -> None:
    """Docs must not contain secret values or assignment examples for secret env vars."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    secret_patterns = [
        r"sk-lf-[a-zA-Z0-9]+",
        r"pk-lf-[a-zA-Z0-9]+",
        r"sk-[a-zA-Z0-9]{20,}",
    ]

    unsafe_assignments = [
        "LANGFUSE_SECRET_KEY=",
        "LANGFUSE_PUBLIC_KEY=",
        "LANGFUSE_HOST=",
        "LANGFUSE_BASE_URL=",
    ]

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        for assignment in unsafe_assignments:
            assert assignment not in content, f"{source} contains unsafe {assignment} assignment example"

        for pattern in secret_patterns:
            match = re.search(pattern, content)
            assert not match, f"{source} contains value matching secret pattern: {pattern}"


def test_langfuse_docs_include_live_test_nodes() -> None:
    """Docs must include specific live test node commands."""
    docs = get_langfuse_docs_content()
    required_nodes = [
        "test_live_langfuse_openai_chat_agent_run",
        "test_live_langfuse_openai_responses_agent_run",
        "test_live_langfuse_anthropic_messages_agent_run",
    ]
    for node in required_nodes:
        assert node in docs, f"docs/features/langfuse.md missing live test node: {node}"


def test_langfuse_docs_do_not_include_concrete_urls() -> None:
    """Docs must not include concrete host URLs."""
    docs = get_langfuse_docs_content()
    forbidden_urls = [
        "https://cloud.langfuse.com",
        "https://api.langfuse.com",
    ]
    for url in forbidden_urls:
        assert url not in docs, f"docs/features/langfuse.md contains forbidden concrete URL: {url}"
