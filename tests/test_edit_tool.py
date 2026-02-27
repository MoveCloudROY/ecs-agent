from __future__ import annotations

import re

import pytest

from ecs_agent.tools.builtins.edit_tool import (
    EditOperation,
    apply_edits,
    compute_line_hash,
    format_file_with_hashes,
    normalize_line,
    parse_edit_instruction,
    validate_hash,
)


def test_compute_line_hash_deterministic() -> None:
    hash_a = compute_line_hash(1, "value")
    hash_b = compute_line_hash(1, "value")

    assert hash_a == hash_b


def test_compute_line_hash_changes_for_different_content() -> None:
    hash_a = compute_line_hash(2, "first")
    hash_b = compute_line_hash(2, "second")

    assert hash_a != hash_b


def test_compute_line_hash_returns_4_hex_chars() -> None:
    value = compute_line_hash(9, "abc")

    assert re.fullmatch(r"[0-9a-f]{4}", value) is not None


def test_normalize_line_strips_trailing_but_preserves_leading() -> None:
    assert normalize_line("  keep-leading   \t") == "  keep-leading"


def test_format_file_with_hashes() -> None:
    content = "alpha\nbeta"
    expected_first_hash = compute_line_hash(1, "alpha")
    expected_second_hash = compute_line_hash(2, "beta")

    assert format_file_with_hashes(content) == (
        f"1#{expected_first_hash}|alpha\n2#{expected_second_hash}|beta"
    )


def test_format_file_with_hashes_starts_line_numbers_at_1() -> None:
    formatted = format_file_with_hashes("x")

    assert formatted.startswith("1#")


def test_parse_edit_instruction() -> None:
    assert parse_edit_instruction("5#a3f2") == (5, "a3f2")


def test_validate_hash_true_and_false() -> None:
    content = "target"
    expected = compute_line_hash(4, content)

    assert validate_hash(4, content, expected)
    assert not validate_hash(4, content, "ffff")


def test_apply_edits_single_line_replace() -> None:
    original = "a\nb\nc"
    target_hash = compute_line_hash(2, "b")
    edits = [EditOperation(op="replace", pos=f"2#{target_hash}", lines=["B"])]

    assert apply_edits(original, edits) == "a\nB\nc"


def test_apply_edits_range_replace() -> None:
    original = "\n".join([f"line-{index}" for index in range(1, 11)])
    start_hash = compute_line_hash(3, "line-3")
    end_hash = compute_line_hash(5, "line-5")
    edits = [
        EditOperation(
            op="replace",
            pos=f"3#{start_hash}",
            end=f"5#{end_hash}",
            lines=["new-3", "new-4"],
        )
    ]

    updated = apply_edits(original, edits)
    assert updated.splitlines() == [
        "line-1",
        "line-2",
        "new-3",
        "new-4",
        "line-6",
        "line-7",
        "line-8",
        "line-9",
        "line-10",
    ]


def test_apply_edits_append() -> None:
    original = "top\nmid\nbot"
    mid_hash = compute_line_hash(2, "mid")
    edits = [EditOperation(op="append", pos=f"2#{mid_hash}", lines=["after-mid"])]

    assert apply_edits(original, edits) == "top\nmid\nafter-mid\nbot"


def test_apply_edits_prepend() -> None:
    original = "top\nmid\nbot"
    mid_hash = compute_line_hash(2, "mid")
    edits = [EditOperation(op="prepend", pos=f"2#{mid_hash}", lines=["before-mid"])]

    assert apply_edits(original, edits) == "top\nbefore-mid\nmid\nbot"


def test_apply_edits_hash_mismatch() -> None:
    original = "a\nb\nc"
    wrong_hash = "ffff"

    with pytest.raises(ValueError, match=r"expected ffff, got [0-9a-f]{4}"):
        apply_edits(
            original,
            [EditOperation(op="replace", pos=f"2#{wrong_hash}", lines=["B"])],
        )

    assert original == "a\nb\nc"


def test_apply_edits_multiple_operations_applied_bottom_up() -> None:
    original = "one\ntwo\nthree\nfour"
    line_1_hash = compute_line_hash(1, "one")
    line_3_hash = compute_line_hash(3, "three")
    edits = [
        EditOperation(op="prepend", pos=f"1#{line_1_hash}", lines=["zero"]),
        EditOperation(op="replace", pos=f"3#{line_3_hash}", lines=["THREE"]),
    ]

    assert apply_edits(original, edits) == "zero\none\ntwo\nTHREE\nfour"


def test_format_file_with_hashes_empty_content_returns_empty_string() -> None:
    assert format_file_with_hashes("") == ""


def test_apply_edits_single_line_file_last_line_edit() -> None:
    original = "solo"
    only_hash = compute_line_hash(1, "solo")

    assert (
        apply_edits(
            original,
            [EditOperation(op="replace", pos=f"1#{only_hash}", lines=["done"])],
        )
        == "done"
    )


def test_apply_edits_last_line_append() -> None:
    original = "a\nb"
    last_hash = compute_line_hash(2, "b")

    assert (
        apply_edits(
            original, [EditOperation(op="append", pos=f"2#{last_hash}", lines=["c"])]
        )
        == "a\nb\nc"
    )


def test_apply_edits_empty_file_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        apply_edits(
            "",
            [EditOperation(op="replace", pos="1#abcd", lines=["value"])],
        )
