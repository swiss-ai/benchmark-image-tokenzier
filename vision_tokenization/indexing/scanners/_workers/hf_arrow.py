"""Single-shard HF Arrow scan helpers."""

from typing import Optional, Tuple

import pyarrow as pa

from vision_tokenization.indexing.scanners._workers.hf_common import (
    build_hf_output_columns,
    build_hf_output_table,
    scan_hf_batch_columns,
)


def _iter_arrow_batches(shard_path: str):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(shard_path, "r") as source:
        try:
            reader = ipc.open_file(source)
            for batch_idx in range(reader.num_record_batches):
                yield batch_idx, reader.get_batch(batch_idx)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_stream(source)
            for batch_idx, batch in enumerate(reader):
                yield batch_idx, batch


def scan_single_hf_arrow_shard(
    shard_path: str,
    image_column: str = "image",
    image_list_column: Optional[str] = None,
    contaminated_rows: frozenset[int] = frozenset(),
) -> Tuple[pa.Table, int, int, int, int, int, Optional[str]]:
    """Scan one HF Arrow shard and return manifest columns."""
    is_multi = image_list_column is not None
    column = image_list_column if is_multi else image_column

    out = build_hf_output_columns(is_multi)
    failed_dims = 0
    contaminated_skipped = 0
    source_rows = 0

    for chunk_index, batch in _iter_arrow_batches(shard_path):
        if column not in batch.schema.names:
            return (
                build_hf_output_table(out, is_multi),
                0,
                0,
                0,
                0,
                0,
                f"missing column {column!r}",
            )
        out, source_rows, failed_dims, batch_skipped = scan_hf_batch_columns(
            out,
            batch.column(column),
            chunk_index,
            source_rows,
            failed_dims,
            is_multi=is_multi,
            contaminated_rows=contaminated_rows,
        )
        contaminated_skipped += batch_skipped

    return (
        build_hf_output_table(out, is_multi),
        source_rows,
        failed_dims,
        0,
        0,
        contaminated_skipped,
        None,
    )
