"""Contamination ID parsing, shard matching, and manifest-row resolution.

Covers the full "apply an exclusion list" lifecycle: parsing an ID file into a
:class:`ContaminationIndex`, matching ids to shards/tars, and resolving an HF
manifest row (``shard_path``/``chunk_index``/``row_in_chunk``) to its shard-local
source-row index so callers can test it against the index. ``chunk_offsets`` does
light shard I/O for that resolution; everything else is pure in-memory work.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

_INNOVATOR_VL_PAT = re.compile(r"SFT_(\d{6})_(\d{6})")


@dataclass(frozen=True)
class ContaminationIndex:
    """Per-source ids to skip during scanning or manifest filtering.

    Bucket values are row indices (``int``, HF formats) or sample_key strings
    (``str``, the WebDataset ``wds_key`` format).
    """

    format: str
    path: str
    by_source: dict[str, frozenset]

    @property
    def total_ids(self) -> int:
        return sum(len(rows) for rows in self.by_source.values())

    def candidate_keys(self, path: Union[str, Path]) -> list[str]:
        """``by_source`` keys that could match this shard/tar.

        ``stem_row`` and ``wds_key`` yield the filename stem (extension dropped) and
        every longer path suffix (``stem``, ``dir/stem``, ``dir2/dir/stem`` …), so
        ``name:R`` matches by stem while ``subset/name:R`` disambiguates the same
        stem across subsets. ``innovator_vl`` yields just the bare stem
        (e.g. ``SFT_000099``).
        """
        p = Path(path)
        if self.format in ("stem_row", "wds_key"):
            parts = [part for part in p.parts if part not in ("/", p.anchor, "")]
            if not parts:
                return []
            parts[-1] = Path(parts[-1]).stem
            return ["/".join(parts[i:]) for i in range(len(parts))]
        if self.format == "innovator_vl":
            return [p.stem]
        raise ValueError(f"candidate_keys: unhandled contamination format {self.format!r}")

    def rows_for_path(self, path: Union[str, Path]) -> frozenset:
        """Source ids to skip for this shard/tar — row ints (HF) or sample_key strings (WDS)."""
        rows: set = set()
        for key in self.candidate_keys(path):
            hit = self.by_source.get(key)
            if hit:
                rows |= hit
        return frozenset(rows)

    def resolve(self, shard_paths: "list[Union[str, Path]]") -> dict[str, list[str]]:
        """Map each source key to the shard paths it matches among ``shard_paths``."""
        matches: dict[str, list[str]] = {key: [] for key in self.by_source}
        for sp in shard_paths:
            for key in self.candidate_keys(sp):
                if key in matches:
                    matches[key].append(str(sp))
        return matches


def load_contamination_index(
    path: Union[str, Path],
    *,
    format: str = "innovator_vl",
    allow_empty: bool = False,
) -> ContaminationIndex:
    """Parse a contamination ID file into source -> row-id buckets.

    Supported formats:

    - ``innovator_vl``: tokens ``SFT_<shard6>_<row6>``, bucketed by file stem
      ``SFT_<shard6>``. Shards must be named ``SFT_<shard6>.{parquet,arrow}``.
    - ``stem_row``: general HF format ``<stem>:<row>`` where ``<stem>`` is a shard
      filename stem (the ``.parquet``/``.arrow`` extension is dropped if present,
      e.g. ``train-00000-of-00010``), optionally path-prefixed
      (e.g. ``configA/train-00000-of-00010``) to disambiguate the same stem across
      subsets, and ``<row>`` is the 0-based row index within that shard file.
      ``:`` separates file from row (split on the last ``:``). Matched against each
      shard's stem, then progressively longer path suffixes.
    - ``wds_key``: WebDataset format ``<tar>:<sample_key>`` where ``<tar>`` is the
      tar filename stem (``.tar`` dropped if present), optionally path-prefixed to
      disambiguate the same stem across subdirs, and ``<sample_key>`` is the
      tar-local WebDataset ``__key__``. Split on the last ``:``; bucket values are
      sample_key strings. Matched like ``stem_row`` (stem then longer path
      suffixes). There is no bare-key form — sample_key is tar-local.

    Tokens are separated by commas or whitespace; malformed tokens are ignored.
    """
    path = Path(path)
    text = path.read_text()
    by_source: dict[str, set] = defaultdict(set)

    if format == "innovator_vl":
        for token in re.split(r"[,\s]+", text):
            if not token:
                continue
            match = _INNOVATOR_VL_PAT.fullmatch(token.strip())
            if match:
                by_source[f"SFT_{match.group(1)}"].add(int(match.group(2)))
    elif format == "stem_row":
        for token in re.split(r"[,\s]+", text):
            token = token.strip()
            if not token:
                continue
            file_part, sep, row_part = token.rpartition(":")
            if not (sep and file_part and row_part.isdigit()):
                continue
            fp = Path(file_part)
            if fp.suffix in (".parquet", ".arrow"):
                file_part = str(fp.with_suffix(""))
            by_source[file_part].add(int(row_part))
    elif format == "wds_key":
        for token in re.split(r"[,\s]+", text):
            token = token.strip()
            if not token:
                continue
            tar_part, sep, key_part = token.rpartition(":")
            if not (sep and tar_part and key_part):
                continue
            tp = Path(tar_part)
            if tp.suffix == ".tar":
                tar_part = str(tp.with_suffix(""))
            by_source[tar_part].add(key_part)
    else:
        raise ValueError(f"Unknown contamination format: {format!r}")

    frozen = {source: frozenset(rows) for source, rows in by_source.items()}
    index = ContaminationIndex(format=format, path=str(path), by_source=frozen)
    if not allow_empty and index.total_ids == 0:
        raise ValueError(f"No contamination ids parsed from {path} with format={format!r}")
    return index


def assert_unique_resolution(
    index: ContaminationIndex,
    shard_paths: "list[Union[str, Path]]",
    *,
    source: str = "input pattern",
) -> list[str]:
    """Ensure each contamination id key resolves to exactly one shard.

    Raises ``ValueError`` if any key matches more than one shard among
    ``shard_paths`` (ambiguous -> would over-exclude every matching file). Returns
    the list of keys that matched no shard, for the caller to warn about (a list
    may legitimately reference files outside a given run).
    """
    matches = index.resolve(shard_paths)
    # Dedup paths so a shard discovered more than once can't look "ambiguous".
    ambiguous = {key: sorted(set(paths)) for key, paths in matches.items() if len(set(paths)) > 1}
    if ambiguous:
        preview = "; ".join(f"{key!r} -> {len(paths)} shards" for key, paths in list(ambiguous.items())[:5])
        raise ValueError(
            f"{len(ambiguous)} contamination id key(s) match multiple shards in the {source}; "
            f"each id must resolve to exactly one file. Path-qualify them "
            f"(e.g. 'config/split/stem:row'). Examples: {preview}"
        )
    return [key for key, paths in matches.items() if not paths]


def chunk_offsets(shard_path: Union[str, Path]) -> list[int]:
    """Cumulative row offsets per chunk for a sharded HF source.

    Parquet shards use row-group sizes; arrow (IPC) shards use batch sizes.
    Maps a ``(shard, chunk, row_in_chunk)`` manifest position to a global row
    index during contamination filtering / decontaminated rebuild.

    Imports are deferred so contamination.py stays light on the scanner path.
    """
    import pyarrow.parquet as pq

    from vision_tokenization.indexing.scanners._workers.hf_arrow import iter_arrow_batches

    source_path = Path(shard_path)
    if source_path.suffix == ".parquet":
        metadata = pq.ParquetFile(str(source_path)).metadata
        lengths = [metadata.row_group(idx).num_rows for idx in range(metadata.num_row_groups)]
    elif source_path.suffix == ".arrow":
        lengths = [batch.num_rows for _, batch in iter_arrow_batches(str(source_path))]
    else:
        raise ValueError(f"Unsupported HF source shard: {shard_path}")

    offsets = [0]
    for length in lengths[:-1]:
        offsets.append(offsets[-1] + int(length))
    return offsets


class HfSourceRowResolver:
    """Resolve whether an HF manifest row is contaminated, caching per-shard offsets.

    Maps a manifest ``(shard_path, chunk_index, row_in_chunk)`` position to its
    shard-local source-row index (``chunk_offsets[chunk_index] + row_in_chunk``) and
    tests it against ``index``. Each shard's offsets are read once and cached, so a
    caller can stream a large manifest batch by batch. Shared by
    ``decontaminate_manifest.py`` and ``rebuild_decontaminated.py`` so the (fiddly)
    position arithmetic lives in exactly one place.
    """

    def __init__(self, index: ContaminationIndex):
        self._index = index
        self._cache: dict[str, tuple[frozenset, Optional[list[int]]]] = {}

    def _rows_and_offsets(self, shard_path: str) -> tuple[frozenset, Optional[list[int]]]:
        cached = self._cache.get(shard_path)
        if cached is None:
            rows = self._index.rows_for_path(shard_path)
            offsets = chunk_offsets(shard_path) if rows else None
            cached = (rows, offsets)
            self._cache[shard_path] = cached
        return cached

    def contaminated_source_row(self, shard_path: str, chunk_index: int, row_in_chunk: int) -> Optional[int]:
        """Shard-local source-row index if this manifest row is contaminated, else ``None``."""
        rows, offsets = self._rows_and_offsets(shard_path)
        if not rows or offsets is None:
            return None
        source_row = int(offsets[int(chunk_index)]) + int(row_in_chunk)
        return source_row if source_row in rows else None


def load_and_validate(
    contamination_ids_path: Union[str, Path],
    *,
    format: str,
    shard_paths: "list[Union[str, Path]]",
    source: str = "input pattern",
) -> tuple[ContaminationIndex, list[str]]:
    """Load a contamination index and assert each id resolves to at most one shard.

    Returns ``(index, unmatched_keys)`` where ``unmatched_keys`` are ids that matched
    no shard/tar (the caller warns about them). Raises ``ValueError`` on ambiguous ids
    (an id matching multiple shards) — see :func:`assert_unique_resolution`. Callers do
    their own count logging so the storage-specific wording stays with the scanner.
    """
    index = load_contamination_index(contamination_ids_path, format=format)
    unmatched = assert_unique_resolution(index, shard_paths, source=source)
    return index, unmatched


def contamination_metadata(index: ContaminationIndex) -> dict:
    """Sidecar metadata describing an applied contamination index (for scan manifests)."""
    return {
        "contamination_ids_path": index.path,
        "contamination_format": index.format,
        "contamination_ids": index.total_ids,
        "contamination_source_shards": len(index.by_source),
    }
