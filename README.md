# polars-map

[![build](https://github.com/hafaio/polars-map/actions/workflows/build.yml/badge.svg)](https://github.com/hafaio/polars-map/actions/workflows/build.yml)
[![pypi](https://img.shields.io/pypi/v/polars-map)](https://pypi.org/project/polars-map/)

Polars plugin providing a `Map` extension type stored as `List(Struct({key, value}))`.

> **Deprecated** — Polars 2.0 ships a native `pl.Map` and reserves the `.map` namespace,
> so this package is pinned to `polars<2` and emits a `DeprecationWarning` on import.
> Its semantics match the native type; see
> [Migrating to native `pl.Map`](#migrating-to-native-plmap).

The type-preserving methods (`filter`, `filter_keys`, `filter_values`, `merge`,
`intersection`, `difference`) require a `Map` input so they can keep its dtype instead of
inferring it; the accessors also accept the raw `List(Struct)`.

## Installation

```bash
pip install polars-map
```

## Supported operations (`.map.*`)

| Category   | Methods                                                    |
| ---------- | ---------------------------------------------------------- |
| Accessors  | `entries`, `keys`, `values`, `len`, `get`, `contains_key`  |
| Filtering  | `filter`, `filter_keys`, `filter_values`                   |
| Transform  | `eval`, `eval_keys`, `eval_values`                         |
| Set ops    | `merge`, `intersection`, `difference`                      |
| Conversion | `from_entries`                                             |
| Iteration  | `__iter__`, `to_list` (Series only)                        |

## Arrow conversion

| Function                  | Description                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `from_arrow(table)`       | Arrow Table/RecordBatch to Polars DataFrame, preserving `map<>` as `Map` |
| `from_arrow_array(array)` | Arrow Array to Polars Series, preserving `map<>` as `Map`                |
| `to_arrow(frame)`         | Polars DataFrame to Arrow Table, converting `Map` back to `map<>`        |
| `to_arrow_array(series)`  | Polars Series to Arrow Array, converting `Map` back to `map<>`           |
| `scan_arrow(source)`      | Lazy scan from an Arrow source with `Map` preservation                   |

## Usage

```python
import polars as pl
import pyarrow as pa
from polars_map import Map, from_arrow, to_arrow, scan_arrow

ser = pl.Series(
    "m",
    [
        [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
        [{"key": "x", "value": 10}],
    ],
    dtype=Map(pl.String(), pl.Int64()),
)
df = pl.DataFrame([ser])

# accessors
df.select(pl.col("m").map.keys())  # [["a", "b"], ["x"]]
df.select(pl.col("m").map.values())  # [[1, 2], [10]]
df.select(pl.col("m").map.len())  # [2, 1]

# lookup
df.select(pl.col("m").map.get("a"))  # [1, None]
df.select(pl.col("m").map.contains_key("a"))  # [True, False]

# filtering
df.select(pl.col("m").map.filter(pl.element().struct["value"] > 1))
df.select(pl.col("m").map.filter_keys(pl.element() > "a"))
df.select(pl.col("m").map.filter_values(pl.element() >= 2))

# transform keys or values
df.select(pl.col("m").map.eval_keys(pl.element().str.to_uppercase()))
df.select(pl.col("m").map.eval_values(pl.element() * 2))

# merge: right value, left position
left = pl.Series(
    "l",
    [[{"key": "a", "value": 1}, {"key": "b", "value": 2}]],
    dtype=Map(pl.String(), pl.Int64()),
)
right = pl.Series(
    "r",
    [[{"key": "a", "value": 99}, {"key": "c", "value": 3}]],
    dtype=Map(pl.String(), pl.Int64()),
)
pair = pl.DataFrame([left, right])
pair.select(pl.col("l").map.merge(pl.col("r")))
# [{"a": 99, "b": 2, "c": 3}]

# set operations
pair.select(pl.col("l").map.intersection(pl.col("r")))  # keys in both
pair.select(pl.col("l").map.difference(pl.col("r")))  # keys only in left

# strip Map -> List(Struct)
df.select(pl.col("m").map.entries())

# from_entries is the inverse
entries = pl.Series(
    "e",
    [[{"key": "a", "value": 1}, {"key": "b", "value": 2}, {"key": "a", "value": 3}]],
    dtype=pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
)
pl.DataFrame([entries]).select(pl.col("e").map.from_entries())  # {"a": 3, "b": 2}

# Series iteration yields dicts
for d in ser.map:
    print(d)  # {"a": 1, "b": 2}, {"x": 10}

# arrow roundtrip
table = pa.table({"m": pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))})
df = from_arrow(table)  # Map(String, Int64) dtype preserved
table2 = to_arrow(df)

# lazy scanning from an arrow source
lf = scan_arrow(lambda: [table])
result = lf.collect()
```

## Caveats

- **Extension types** — used to wrap the underlying `List(Struct)` storage with a semantic `Map` dtype, are not yet stabilized and may change across Polars releases.
- **`pl.dtype_of`** — used to efficiently cast to the extension type after _some_ operations is also unstable.
- **GIL** - is required to automatically wrap an expression as the extension type, and so operations which could change the underlying key or value types will briefly lock the GIL to do the cast. This may also prevent the polars engine from reasoning about the type.
- **Large offsets** — Arrow's `map<>` type uses only 32-bit offsets, so exporting a Polars map backed by a `LargeList` whose offsets don't fit in a `u32` will error. Arrow has no large-offset map type.

## Migrating to native `pl.Map`

`Map(pl.String(), pl.Int64())` becomes `pl.Map(pl.String, pl.Int64)`, `.map.from_entries()`
becomes `.list.to_map()`, and the Arrow helpers become `pl.from_arrow` / `.to_arrow()`.

The remaining methods have no native counterpart; write them as `.map.entries()`, a
`list.eval`, and `.list.to_map()` when a map is rebuilt.

```python
m.map.entries().list.eval(pl.element().struct["key"])  # keys
pl.concat_list(l.map.entries(), r.map.entries()).list.to_map()  # merge
```
