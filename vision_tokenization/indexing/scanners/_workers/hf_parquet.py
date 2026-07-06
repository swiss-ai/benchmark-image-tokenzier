"""Single-shard HF Parquet scan helpers."""

from typing import Optional, Tuple

import pyarrow as pa

from vision_tokenization.indexing.scanners._workers.hf_common import (
    build_hf_output_columns,
    build_hf_output_table,
    scan_hf_batch_columns,
)


def scan_single_hf_parquet_shard(
    shard_path: str,
    image_column: str = "image",
    image_list_column: Optional[str] = None,
    contaminated_rows: frozenset[int] = frozenset(),
) -> Tuple[pa.Table, int, int, int, Optional[str]]:
    """Scan one HF Parquet shard and return manifest columns."""
    is_multi = image_list_column is not None
    column = image_list_column if is_multi else image_column

    import pyarrow.parquet as pq

    out = build_hf_output_columns(is_multi)
    failed_dims = 0
    contaminated_skipped = 0
    source_rows = 0
    parquet_file = pq.ParquetFile(shard_path)
    if column not in parquet_file.schema_arrow.names:
        return build_hf_output_table(out, is_multi), 0, 0, 0, f"missing column {column!r}"

    for row_group_idx in range(parquet_file.metadata.num_row_groups):
        batch = parquet_file.read_row_group(row_group_idx, columns=[column])
        out, source_rows, failed_dims, batch_skipped = scan_hf_batch_columns(
            out,
            batch.column(column),
            row_group_idx,
            source_rows,
            failed_dims,
            is_multi=is_multi,
            contaminated_rows=contaminated_rows,
        )
        contaminated_skipped += batch_skipped

    return build_hf_output_table(out, is_multi), source_rows, failed_dims, contaminated_skipped, None
