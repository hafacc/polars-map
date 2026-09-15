"""Map extension data type for Polars."""

from __future__ import annotations

import polars as pl
from polars._typing import PolarsDataType, PythonDataType
from polars.datatypes import parse_into_dtype


def _ensure_instance(dtype: PolarsDataType) -> pl.DataType:
    return dtype() if isinstance(dtype, type) else dtype


class Map(pl.BaseExtension):
    """Map extension type backed by List(Struct({key, value})).

    Usage as a dtype for Series construction::

        dtype = Map(pl.String(), pl.Int64())
        dtype = Map(pl.String, pl.Int64)
        dtype = Map(str, int)
    """

    def __init__(
        self,
        key: PolarsDataType | PythonDataType,
        value: PolarsDataType | PythonDataType,
    ) -> None:
        storage = pl.List(
            pl.Struct({"key": parse_into_dtype(key), "value": parse_into_dtype(value)})
        )
        super().__init__("polars_map.map", storage)

    @property
    def key(self) -> pl.DataType:
        """Key data type."""
        [key, _] = self.ext_storage().inner.fields  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        return _ensure_instance(key.dtype)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    @property
    def value(self) -> pl.DataType:
        """Value data type."""
        [_, value] = self.ext_storage().inner.fields  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]
        return _ensure_instance(value.dtype)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    def _string_repr(self) -> str:
        return f"map[{self.key._string_repr()}, {self.value._string_repr()}]"  # pyright: ignore[reportUnknownMemberType]

    def __repr__(self) -> str:
        """Return the canonical representation, e.g. ``Map(String, Int64)``."""
        return f"Map({self.key!r}, {self.value!r})"


pl.register_extension_type("polars_map.map", Map)
