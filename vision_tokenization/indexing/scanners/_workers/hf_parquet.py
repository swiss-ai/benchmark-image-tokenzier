"""Single-shard HF Parquet scan helpers."""

from typing import Optional, Tuple

import pyarrow as pa

from vision_tokenization.indexing.scanners._workers.hf_common import (
    build_hf_output_columns,
    build_hf_output_table,
    scan_hf_batch_columns,
    scan_hf_image_map_batch_columns,
)
from vision_tokenization.utils.image_map_parquet import iter_image_map_row_group_batches


def scan_single_hf_parquet_shard(
    shard_path: str,
    image_column: str = "image",
    image_list_column: Optional[str] = None,
    image_map_column: Optional[str] = None,
    message_column: Optional[str] = None,
    contaminated_rows: frozenset[int] = frozenset(),
    compute_media_sha256: bool = False,
) -> Tuple[pa.Table, int, int, int, int, int, Optional[str]]:
    """Scan one HF Parquet shard and return manifest columns."""
    is_map = image_map_column is not None
    is_multi = image_list_column is not None or is_map
    column = image_list_column if is_multi else image_column

    import pyarrow.parquet as pq

    out = build_hf_output_columns(is_multi, compute_media_sha256=compute_media_sha256)
    failed_dims = 0
    failed_messages = 0
    failed_image_maps = 0
    contaminated_skipped = 0
    source_rows = 0
    parquet_file = pq.ParquetFile(shard_path)
    if is_map:
        if message_column is None:
            return (
                build_hf_output_table(out, is_multi, compute_media_sha256=compute_media_sha256),
                0,
                0,
                0,
                0,
                0,
                "missing message_column",
            )
        missing = [
            col
            for col in (image_map_column, message_column)
            if col not in parquet_file.schema_arrow.names
        ]
        if missing:
            return (
                build_hf_output_table(out, is_multi, compute_media_sha256=compute_media_sha256),
                0,
                0,
                0,
                0,
                0,
                f"missing column(s) {missing!r}",
            )

        columns = [image_map_column, message_column]
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            for row_base, batch in iter_image_map_row_group_batches(
                parquet_file,
                row_group_idx,
                columns,
            ):
                (
                    out,
                    source_rows,
                    failed_dims,
                    failed_messages,
                    failed_image_maps,
                    batch_skipped,
                ) = scan_hf_image_map_batch_columns(
                    out,
                    batch.column(image_map_column),
                    batch.column(message_column),
                    row_group_idx,
                    source_rows,
                    failed_dims,
                    failed_messages,
                    failed_image_maps,
                    row_base=row_base,
                    contaminated_rows=contaminated_rows,
                    compute_media_sha256=compute_media_sha256,
                )
                contaminated_skipped += batch_skipped

        return (
            build_hf_output_table(out, is_multi, compute_media_sha256=compute_media_sha256),
            source_rows,
            failed_dims,
            failed_messages,
            failed_image_maps,
            contaminated_skipped,
            None,
        )

    if column not in parquet_file.schema_arrow.names:
        return (
            build_hf_output_table(out, is_multi, compute_media_sha256=compute_media_sha256),
            0,
            0,
            0,
            0,
            0,
            f"missing column {column!r}",
        )

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
            compute_media_sha256=compute_media_sha256,
        )
        contaminated_skipped += batch_skipped

    return (
        build_hf_output_table(out, is_multi, compute_media_sha256=compute_media_sha256),
        source_rows,
        failed_dims,
        0,
        0,
        contaminated_skipped,
        None,
    )
