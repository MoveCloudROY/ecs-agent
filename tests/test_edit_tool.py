from __future__ import annotations

import re

import pytest

from ecs_agent.tools.builtins.edit_tool import (
    EditOperation,
    apply_edits,
    compute_line_hash,
    normalize_line,
    parse_edit_instruction,
    validate_hash,
)


def _parse_bounded_integer(
    value: int | str,
    *,
    minimum: int,
    invalid_message: str,
    range_message: str,
) -> int:
    try:
        from ecs_agent.tools.builtins._numeric import parse_bounded_integer
    except ModuleNotFoundError:
        pytest.fail("shared numeric parser is missing")

    return parse_bounded_integer(
        value,
        minimum=minimum,
        invalid_message=invalid_message,
        range_message=range_message,
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


def test_parse_bounded_integer_accepts_int_and_decimal_string() -> None:
    assert _parse_bounded_integer(4, minimum=1, invalid_message="bad", range_message="low") == 4
    assert _parse_bounded_integer("5", minimum=1, invalid_message="bad", range_message="low") == 5


@pytest.mark.parametrize("value", [True, False, "abc", "-1"])
def test_parse_bounded_integer_rejects_non_decimal_input(value: int | str) -> None:
    with pytest.raises(ValueError, match="bad"):
        _parse_bounded_integer(
            value,
            minimum=1,
            invalid_message="bad",
            range_message="low",
        )


def test_parse_bounded_integer_rejects_values_below_minimum() -> None:
    with pytest.raises(ValueError, match="low"):
        _parse_bounded_integer(
            0,
            minimum=1,
            invalid_message="bad",
            range_message="low",
        )


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


def test_apply_edits_line_shift_invalidation() -> None:
    original = "line1\nline2\nline3"
    line_1_hash = compute_line_hash(1, "line1")
    stale_line_2_hash = compute_line_hash(2, "line2")

    shifted = apply_edits(
        original,
        [EditOperation(op="prepend", pos=f"1#{line_1_hash}", lines=["new-line0"])],
    )

    with pytest.raises(ValueError, match="Hash mismatch"):
        apply_edits(
            shifted,
            [
                EditOperation(
                    op="replace",
                    pos=f"2#{stale_line_2_hash}",
                    lines=["updated-line2"],
                )
            ],
        )


def test_apply_edits_crlf_content() -> None:
    original = "alpha\r\nbeta\r\ngamma"
    assert compute_line_hash(1, "alpha\r") == compute_line_hash(1, "alpha")

    alpha_hash = compute_line_hash(1, "alpha")
    updated = apply_edits(
        original,
        [EditOperation(op="replace", pos=f"1#{alpha_hash}", lines=["ALPHA"])],
    )

    assert updated == "ALPHA\nbeta\ngamma"


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
