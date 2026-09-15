"""Polars plugin providing Map operations on List(Struct({key, value})) columns."""

import warnings

from polars_map._conv import (
    from_arrow,  # pyright: ignore[reportUnknownVariableType]
    from_arrow_array,  # pyright: ignore[reportUnknownVariableType]
    scan_arrow,  # pyright: ignore[reportUnknownVariableType]
    to_arrow,  # pyright: ignore[reportUnknownVariableType]
    to_arrow_array,  # pyright: ignore[reportUnknownVariableType]
)
from polars_map._dtype import Map
from polars_map._expr import MapExpr
from polars_map._series import MapSeries

warnings.warn(
    "polars-map is deprecated: Polars 2.0 ships a native pl.Map dtype "
    "(pl.Map(key, value), .map.entries(), .list.to_map()) and reserves the .map "
    "namespace, so this package only supports polars<2. See the README for "
    "migration notes.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = (
    "Map",
    "MapExpr",
    "MapSeries",
    "from_arrow",
    "from_arrow_array",
    "scan_arrow",
    "to_arrow",
    "to_arrow_array",
)
