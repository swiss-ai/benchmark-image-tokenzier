"""Dataset contamination ID parsing helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

_INNOVATOR_VL_PAT = re.compile(r"SFT_(\d{6})_(\d{6})")


@dataclass(frozen=True)
class ContaminationIndex:
    """Per-source row ids to skip during scanning or manifest filtering."""

    format: str
    path: str
    by_source: dict[str, frozenset[int]]

    @property
    def total_ids(self) -> int:
        return sum(len(rows) for rows in self.by_source.values())

    def rows_for_path(self, path: Union[str, Path]) -> frozenset[int]:
        return self.by_source.get(Path(path).stem, frozenset())


def load_contamination_index(
    path: Union[str, Path],
    *,
    format: str = "innovator_vl",
    allow_empty: bool = False,
) -> ContaminationIndex:
    """Parse a contamination ID file into source-basename -> row-id buckets."""
    path = Path(path)
    text = path.read_text()
    by_source: dict[str, set[int]] = defaultdict(set)

    if format == "innovator_vl":
        for token in re.split(r"[,\s]+", text):
            if not token:
                continue
            match = _INNOVATOR_VL_PAT.fullmatch(token.strip())
            if match:
                by_source[f"SFT_{match.group(1)}"].add(int(match.group(2)))
    else:
        raise ValueError(f"Unknown contamination format: {format!r}")

    frozen = {source: frozenset(rows) for source, rows in by_source.items()}
    index = ContaminationIndex(format=format, path=str(path), by_source=frozen)
    if not allow_empty and index.total_ids == 0:
        raise ValueError(
            f"No contamination ids parsed from {path} with format={format!r}"
        )
    return index


def chunk_offsets(shard_path: Union[str, Path]) -> list[int]:
    """Cumulative row offsets per chunk for a sharded HF source.

    Parquet shards use row-group sizes; arrow (IPC) shards use batch sizes.
    Maps a ``(shard, chunk, row_in_chunk)`` manifest position to a global row
    index during contamination filtering / decontaminated rebuild.

    Imports are deferred so contamination.py stays light on the scanner path.
    """
    import pyarrow.parquet as pq
    from vision_tokenization.indexing.scanners._workers.hf_arrow import _iter_arrow_batches

    source_path = Path(shard_path)
    if source_path.suffix == ".parquet":
        metadata = pq.ParquetFile(str(source_path)).metadata
        lengths = [metadata.row_group(idx).num_rows for idx in range(metadata.num_row_groups)]
    elif source_path.suffix == ".arrow":
        lengths = [batch.num_rows for _, batch in _iter_arrow_batches(str(source_path))]
    else:
        raise ValueError(f"Unsupported HF source shard: {shard_path}")

    offsets = [0]
    for length in lengths[:-1]:
        offsets.append(offsets[-1] + int(length))
    return offsets
