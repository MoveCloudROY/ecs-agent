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


def test_langfuse_docs_mention_skip_behavior() -> None:
    """README and docs must mention live-test skip behavior."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "skip" in content.lower(), f"{source} must mention skip behavior for live tests"
        assert "live" in content.lower(), f"{source} must mention live tests"


def test_langfuse_docs_mention_credential_rotation() -> None:
    """README and docs must mention credential rotation."""
    readme = get_readme_content()
    docs = get_langfuse_docs_content()

    for content, source in [(readme, "README"), (docs, "docs/features/langfuse.md")]:
        assert "rotation" in content.lower() or "rotate" in content.lower(), f"{source} must mention credential rotation"


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
