"""Shared helpers for Map operations."""

from __future__ import annotations

import functools

import polars as pl
from polars.exceptions import InvalidOperationError

from ._dtype import Map


def canonical() -> pl.Expr:
    """Interior ``list.eval`` expr rebuilding entries into canonical form."""
    key = pl.element().struct["key"]
    value = pl.element().struct["value"]
    return pl.struct(  # pyright: ignore[reportUnknownMemberType]
        key,
        value.last().over(key),  # pyright: ignore[reportUnknownMemberType]
    ).filter(key.is_first_distinct())


def tag(ser: pl.Series) -> pl.Series:
    """Relabel canonical entries as a Map extension type."""
    [key, value] = ser.dtype.inner.fields  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
    return ser.ext.to(Map(key.dtype, value.dtype))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


def canonicalize(ser: pl.Series, *, parallel: bool = False) -> pl.Series:
    """Rebuild raw entries into canonical form and tag them as a Map.

    Replicates ``list.to_map`` from Polars 2: exact ``key``/``value`` fields, no null
    entries or keys, duplicate keys keep their first position and last value.
    """
    dtype = ser.dtype
    if not (
        isinstance(dtype, pl.List)
        and isinstance(dtype.inner, pl.Struct)
        and {field.name for field in dtype.inner.fields} == {"key", "value"}
    ):
        raise InvalidOperationError(
            "Map entries must be a Struct with exactly two fields named "
            "`key` and `value`"
        )
    elif ser.list.eval(pl.element().is_null()).list.any().any():
        raise InvalidOperationError("Map entries cannot be null")
    elif ser.list.eval(pl.element().struct["key"].is_null()).list.any().any():
        raise InvalidOperationError("Map keys cannot be null")
    else:
        return tag(ser.list.eval(canonical(), parallel=parallel))


def infer_map(expr: pl.Expr) -> pl.Expr:
    """Relabel a canonical entries expr as a Map, inferring the key and value types."""
    return expr.map_batches(tag, is_elementwise=True)


def canonicalize_expr(expr: pl.Expr, *, parallel: bool = False) -> pl.Expr:
    """Rebuild a raw entries expr into canonical form and tag it as a Map."""
    return expr.map_batches(
        functools.partial(canonicalize, parallel=parallel), is_elementwise=True
    )


def expr_eval(expr: pl.Expr, evaled: pl.Expr) -> pl.Expr:
    """Evaluate one expression in the context of another.

    Wraps *expr* in a single-element array, evaluates *evaled* on it
    (where ``pl.element()`` is rebound to the value), then unwraps.
    """
    return (
        pl.concat_arr(expr)  # pyright: ignore[reportUnknownMemberType]
        .arr.eval(evaled)
        .arr.first()
    )
