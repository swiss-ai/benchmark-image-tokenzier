"""Streaming helpers for image-map Parquet row groups."""

from __future__ import annotations

from typing import Iterable

import pyarrow as pa

DEFAULT_IMAGE_MAP_BATCH_SIZE = 4096


def iter_image_map_row_group_batches(
    parquet_file,
    row_group_idx: int,
    columns: list[str],
    *,
    batch_size: int | None = None,
):
    """Yield ``(row_base, batch)`` without materializing a full row group."""
    row_base = 0
    for batch in parquet_file.iter_batches(
        row_groups=[row_group_idx],
        columns=columns,
        batch_size=batch_size or DEFAULT_IMAGE_MAP_BATCH_SIZE,
    ):
        yield row_base, batch
        row_base += batch.num_rows


def read_image_map_row_group_rows(
    parquet_file,
    row_group_idx: int,
    columns: list[str],
    row_indices: Iterable[int],
    *,
    batch_size: int | None = None,
) -> tuple[pa.Table, dict[int, int]]:
    """Read selected row offsets from one row group via streamed batches."""
    wanted = sorted(set(int(row_idx) for row_idx in row_indices))
    if not wanted:
        return pa.table({col: [] for col in columns}), {}

    tables: list[pa.Table] = []
    row_positions: dict[int, int] = {}
    wanted_pos = 0
    out_pos = 0
    for row_base, batch in iter_image_map_row_group_batches(
        parquet_file,
        row_group_idx,
        columns,
        batch_size=batch_size,
    ):
        batch_end = row_base + batch.num_rows
        local_rows: list[int] = []
        original_rows: list[int] = []
        while wanted_pos < len(wanted) and wanted[wanted_pos] < batch_end:
            row_idx = wanted[wanted_pos]
            if row_idx >= row_base:
                local_rows.append(row_idx - row_base)
                original_rows.append(row_idx)
            wanted_pos += 1

        if local_rows:
            table = pa.Table.from_batches([batch])
            selected = table.take(pa.array(local_rows, type=pa.int32()))
            tables.append(selected)
            for row_idx in original_rows:
                row_positions[row_idx] = out_pos
                out_pos += 1

        if wanted_pos >= len(wanted):
            break

    if not tables:
        return pa.table({col: [] for col in columns}), {}
    return pa.concat_tables(tables, promote_options="none"), row_positions
