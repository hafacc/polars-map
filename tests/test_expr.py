"""Tests for MapExpr namespace."""

from __future__ import annotations

import polars as pl
import pytest
from polars.exceptions import InvalidOperationError

from polars_map import Map

from .conftest import emap

_ENTRIES = pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))


def test_keys(map_frame: pl.DataFrame) -> None:
    """Verify keys extracts all keys per row."""
    [ser] = map_frame.select(emap(pl.col("map")).keys())  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [["a", "b", "c"], ["x"], None, []]


def test_values(map_frame: pl.DataFrame) -> None:
    """Verify values extracts all values per row."""
    [ser] = map_frame.select(emap(pl.col("map")).values())  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [[1, 2, 3], [10], None, []]


def test_len(map_frame: pl.DataFrame) -> None:
    """Verify len returns entry count per row."""
    [ser] = map_frame.select(emap(pl.col("map")).len())  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [3, 1, None, 0]


def test_get(map_frame: pl.DataFrame) -> None:
    """Verify get returns value for existing key, None for missing/null/empty."""
    [ser] = map_frame.select(emap(pl.col("map")).get("a"))  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [1, None, None, None]


def test_getitem(map_frame: pl.DataFrame) -> None:
    """Verify __getitem__ forwards to get for a single key."""
    [ser] = map_frame.select(emap(pl.col("map"))["a"])  # pyright: ignore[reportUnknownMemberType]
    assert ser.to_list() == [1, None, None, None]


def test_get_multi(map_frame: pl.DataFrame) -> None:
    """Verify get with multiple keys returns multiple columns."""
    [a_ser, z_ser] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).get("a", "z"),
    )
    assert a_ser.to_list() == [1, None, None, None]
    assert z_ser.to_list() == [None, None, None, None]


def test_contains_key(map_frame: pl.DataFrame) -> None:
    """Verify contains_key for present and absent keys."""
    [a_ser, z_ser] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).contains_key("a").alias("a"),
        emap(pl.col("map")).contains_key("z").alias("z"),
    )
    assert a_ser.to_list() == [True, False, None, False]
    assert z_ser.to_list() == [False, False, None, False]


def test_filter(map_frame: pl.DataFrame) -> None:
    """Verify filter on struct field keeps matching entries and retains Map dtype."""
    [ser] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).filter(pl.element().struct["value"] > 1)
    )
    assert isinstance(ser.dtype, Map)
    vals = ser.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b", "c"]


def test_filter_keys(map_frame: pl.DataFrame) -> None:
    """Verify filter_keys keeps entries with matching keys."""
    [ser] = map_frame.select(emap(pl.col("map")).filter_keys(pl.element() > "a"))  # pyright: ignore[reportUnknownMemberType]
    vals = ser.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b", "c"]


def test_filter_values(map_frame: pl.DataFrame) -> None:
    """Verify filter_values keeps entries with matching values."""
    [ser] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).filter_values(pl.element() >= 2)  # noqa: PLR2004
    )
    vals = ser.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b", "c"]


def test_eval_values(map_frame: pl.DataFrame) -> None:
    """Verify eval_values transforms values and preserves Map type."""
    [ser] = map_frame.select(emap(pl.col("map")).eval_values(pl.element() * 2))  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(ser.dtype, Map)
    vals = ser.ext.storage().to_list()[0]
    val_dict = {e["key"]: e["value"] for e in vals}
    assert val_dict == {"a": 2, "b": 4, "c": 6}


def test_eval_keys(map_frame: pl.DataFrame) -> None:
    """Verify eval_keys transforms keys and preserves Map type."""
    [ser] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).eval_keys(pl.element().str.to_uppercase())
    )
    assert isinstance(ser.dtype, Map)
    vals = ser.ext.storage().to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["A", "B", "C"]


def test_entries_strips_extension() -> None:
    """Verify entries strips Map type back to plain List."""
    ser = pl.Series(
        "map", [[{"key": "a", "value": 1}]], dtype=Map(pl.String(), pl.Int64())
    )
    frame = pl.DataFrame([ser])
    [result] = frame.select(emap(pl.col("map")).entries())  # pyright: ignore[reportUnknownMemberType]
    assert not isinstance(result.dtype, Map)
    assert isinstance(result.dtype, pl.List)


def test_from_entries() -> None:
    """Verify from_entries wraps plain list-of-structs as Map."""
    ser = pl.Series(
        "map",
        [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
        dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
    )
    frame = pl.DataFrame([ser])
    [result] = frame.select(emap(pl.col("map")).from_entries())  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(result.dtype, Map)


def test_eval_keys_dedup_collapsing() -> None:
    """Verify eval_keys collapses colliding keys to first position, last value."""
    ser = pl.Series(
        "map",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "A", "value": 3},
            ]
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame([ser])
    [result] = frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).eval_keys(pl.element().str.to_uppercase())
    )
    assert result.ext.storage().to_list()[0] == [
        {"key": "A", "value": 3},
        {"key": "B", "value": 2},
    ]


def test_eval_dedup_collapsing_keys() -> None:
    """Verify eval collapses colliding keys to first position, last value."""
    ser = pl.Series(
        "map",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "A", "value": 3},
            ]
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame([ser])
    [result] = frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).eval(
            pl.element().struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
                key=pl.element().struct["key"].str.to_uppercase()
            )
        )
    )
    assert result.ext.storage().to_list()[0] == [
        {"key": "A", "value": 3},
        {"key": "B", "value": 2},
    ]


def test_from_entries_dedup_first_position_last_value() -> None:
    """Verify a repeated key keeps its first position and its last value."""
    ser = pl.Series(
        "map",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "a", "value": 3},
            ]
        ],
        dtype=_ENTRIES,
    )
    frame = pl.DataFrame([ser])
    [result] = frame.select(emap(pl.col("map")).from_entries())  # pyright: ignore[reportUnknownMemberType]
    assert result.ext.storage().to_list()[0] == [
        {"key": "a", "value": 3},
        {"key": "b", "value": 2},
    ]


def test_from_entries_reversed_fields() -> None:
    """Verify entries ordered value, key are accepted and canonicalized."""
    ser = pl.Series(
        "map",
        [[{"value": 1, "key": "a"}]],
        dtype=pl.List(pl.Struct({"value": pl.Int64, "key": pl.String})),
    )
    frame = pl.DataFrame([ser])
    [result] = frame.select(emap(pl.col("map")).from_entries())  # pyright: ignore[reportUnknownMemberType]
    assert result.dtype == Map(pl.String(), pl.Int64())
    assert result.ext.storage().to_list()[0] == [{"key": "a", "value": 1}]


def test_from_entries_null_key() -> None:
    """Verify a null key in a live row is rejected."""
    ser = pl.Series("map", [[{"key": None, "value": 1}]], dtype=_ENTRIES)
    frame = pl.DataFrame([ser])
    with pytest.raises(InvalidOperationError, match="Map keys cannot be null"):
        frame.select(emap(pl.col("map")).from_entries())  # pyright: ignore[reportUnknownMemberType]


def test_from_entries_null_entry() -> None:
    """Verify a null entry in a live row is rejected."""
    ser = pl.Series("map", [[{"key": "a", "value": 1}, None]], dtype=_ENTRIES)
    frame = pl.DataFrame([ser])
    with pytest.raises(InvalidOperationError, match="Map entries cannot be null"):
        frame.select(emap(pl.col("map")).from_entries())  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    "ser",
    [
        pl.Series("map", [[{"k": "a", "v": 1}]]),
        pl.Series("map", [[{"key": "a", "value": 1, "extra": True}]]),
        pl.Series("map", [[{"key": "a"}]]),
        pl.Series("map", [[1, 2]]),
    ],
    ids=["wrong_names", "extra_field", "missing_field", "not_a_struct"],
)
def test_from_entries_bad_entry_shape(ser: pl.Series) -> None:
    """Verify entries that aren't exactly key and value structs are rejected.

    A bad dtype is caught while Polars resolves the output type of the wrapped
    function, which replaces the message, so only the error type is asserted.
    """
    frame = pl.DataFrame([ser])
    with pytest.raises(InvalidOperationError):
        frame.select(emap(pl.col("map")).from_entries())  # pyright: ignore[reportUnknownMemberType]


def test_eval_rejects_renamed_fields(map_frame: pl.DataFrame) -> None:
    """Verify an expression that renames the entry fields is rejected."""
    with pytest.raises(InvalidOperationError):
        map_frame.select(  # pyright: ignore[reportUnknownMemberType]
            emap(pl.col("map")).eval(pl.element().struct.rename_fields(["k", "v"]))
        )


def test_canonicalizing_ops_keep_null_and_empty_rows(map_frame: pl.DataFrame) -> None:
    """Verify null rows stay null and empty rows stay empty through every op."""
    [entries, evaled, keyed, valued, merged] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(emap(pl.col("map")).entries()).from_entries().alias("entries"),
        emap(pl.col("map")).eval(pl.element()).alias("evaled"),
        emap(pl.col("map")).eval_keys(pl.element()).alias("keyed"),
        emap(pl.col("map")).eval_values(pl.element()).alias("valued"),
        emap(pl.col("map")).merge(pl.col("map")).alias("merged"),
    )
    for ser in (entries, evaled, keyed, valued, merged):
        lengths = ser.ext.storage().list.len().to_list()
        assert lengths == [3, 1, None, 0]


def test_merge_no_overlap() -> None:
    """Verify merge combines non-overlapping maps and retains Map dtype."""
    left = pl.Series(
        "l", [[{"key": "a", "value": 1}]], dtype=Map(pl.String(), pl.Int64())
    )
    right = pl.Series(
        "r", [[{"key": "b", "value": 2}]], dtype=Map(pl.String(), pl.Int64())
    )
    frame = pl.DataFrame([left, right])
    [result] = frame.select(emap(pl.col("l")).merge(pl.col("r")))  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(result.dtype, Map)
    vals = result.to_list()[0]
    keys = sorted(e["key"] for e in vals)
    assert keys == ["a", "b"]


def test_merge_overlap_right_wins() -> None:
    """Verify merge takes the right-side value at the left-side position."""
    left = pl.Series(
        "l",
        [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    right = pl.Series(
        "r",
        [[{"key": "c", "value": 3}, {"key": "a", "value": 99}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame([left, right])
    [result] = frame.select(emap(pl.col("l")).merge(pl.col("r")))  # pyright: ignore[reportUnknownMemberType]
    assert result.ext.storage().to_list()[0] == [
        {"key": "a", "value": 99},
        {"key": "b", "value": 2},
        {"key": "c", "value": 3},
    ]


def test_intersection() -> None:
    """Verify intersection keeps entries with shared keys and retains Map dtype."""
    left = pl.Series(
        "l",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "c", "value": 3},
            ]
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )
    right = pl.Series(
        "r",
        [[{"key": "a", "value": 10}, {"key": "c", "value": 30}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame([left, right])
    [result] = frame.select(emap(pl.col("l")).intersection(pl.col("r")))  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(result.dtype, Map)
    vals = result.to_list()[0]
    keys = sorted(e["key"] for e in vals)
    assert keys == ["a", "c"]
    val_dict = {e["key"]: e["value"] for e in vals}
    assert val_dict["a"] == 1
    assert val_dict["c"] == 3  # noqa: PLR2004


def test_difference() -> None:
    """Verify difference keeps entries with keys not in other and retains Map dtype."""
    left = pl.Series(
        "l",
        [
            [
                {"key": "a", "value": 1},
                {"key": "b", "value": 2},
                {"key": "c", "value": 3},
            ]
        ],
        dtype=Map(pl.String(), pl.Int64()),
    )
    right = pl.Series(
        "r",
        [[{"key": "a", "value": 10}, {"key": "c", "value": 30}]],
        dtype=Map(pl.String(), pl.Int64()),
    )
    frame = pl.DataFrame([left, right])
    [result] = frame.select(emap(pl.col("l")).difference(pl.col("r")))  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(result.dtype, Map)
    vals = result.to_list()[0]
    keys = [e["key"] for e in vals]
    assert keys == ["b"]
    assert vals[0]["value"] == 2  # noqa: PLR2004


def test_eval(map_frame: pl.DataFrame) -> None:
    """Verify eval transforms entries and preserves Map type."""
    [ser] = map_frame.select(  # pyright: ignore[reportUnknownMemberType]
        emap(pl.col("map")).eval(
            pl.element().struct.with_fields(  # pyright: ignore[reportUnknownMemberType]
                value=pl.element().struct["value"] * 2
            )
        )
    )
    assert isinstance(ser.dtype, Map)
    vals = ser.ext.storage().to_list()[0]
    val_dict = {e["key"]: e["value"] for e in vals}
    assert val_dict == {"a": 2, "b": 4, "c": 6}
