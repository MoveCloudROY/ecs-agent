"""Characterization tests pinning subagent behaviors exposed by the refactor.

These tests capture the *current* observable contract of pieces that move during
the subagent package refactor (docs/plans/2026-07-02-subagent-architecture-refactor.md)
so extractions can be verified behavior-preserving. They are permanent regression
guards, not scaffolding.

Coverage gap this fills: the existing suite exercises the background-result envelope
only indirectly through ``subagent_result(read_method="summary")``. The pure parser
``_parse_background_result_envelope`` — relocated to ``result_envelope.py`` in Task 2 —
had no direct unit test.
"""

from __future__ import annotations

from ecs_agent.systems.subagent import SubagentSystem


def _parse(result: str) -> tuple[str, str] | None:
    # Indirection point: currently a method on SubagentSystem. Task 2 relocates this
    # to a module function in result_envelope.py (re-exported); update this helper's
    # target then. The asserted contract below must not change.
    return SubagentSystem()._parse_background_result_envelope(result)


def test_envelope_parses_well_formed_summary_and_full_result() -> None:
    result = (
        "<subagent_background_result>\n"
        "<summary>brief summary</summary>\n"
        "<full_result>the complete result</full_result>\n"
        "</subagent_background_result>"
    )
    assert _parse(result) == ("brief summary", "the complete result")


def test_envelope_tolerates_surrounding_whitespace() -> None:
    result = (
        "   \n<subagent_background_result>"
        "<summary>s</summary><full_result>f</full_result>"
        "</subagent_background_result>\n  "
    )
    assert _parse(result) == ("s", "f")


def test_envelope_preserves_inner_whitespace_of_sections() -> None:
    result = (
        "<subagent_background_result>"
        "<summary> spaced summary </summary>"
        "<full_result> spaced full </full_result>"
        "</subagent_background_result>"
    )
    assert _parse(result) == (" spaced summary ", " spaced full ")


def test_envelope_returns_none_when_not_wrapped() -> None:
    assert _parse("just a plain result string") is None


def test_envelope_returns_none_when_wrapper_end_missing() -> None:
    result = "<subagent_background_result><summary>s</summary><full_result>f</full_result>"
    assert _parse(result) is None


def test_envelope_returns_none_when_summary_missing() -> None:
    result = (
        "<subagent_background_result>"
        "<full_result>f</full_result>"
        "</subagent_background_result>"
    )
    assert _parse(result) is None


def test_envelope_returns_none_when_full_result_missing() -> None:
    result = (
        "<subagent_background_result>"
        "<summary>s</summary>"
        "</subagent_background_result>"
    )
    assert _parse(result) is None


def test_envelope_returns_none_when_summary_not_first() -> None:
    # Contract: the <summary> block must begin the body (summary_start == 0).
    result = (
        "<subagent_background_result>"
        "prefix<summary>s</summary><full_result>f</full_result>"
        "</subagent_background_result>"
    )
    assert _parse(result) is None


def test_envelope_returns_none_when_content_trails_full_result() -> None:
    # Contract: the full-result section must end the body exactly.
    result = (
        "<subagent_background_result>"
        "<summary>s</summary><full_result>f</full_result>trailing"
        "</subagent_background_result>"
    )
    assert _parse(result) is None
