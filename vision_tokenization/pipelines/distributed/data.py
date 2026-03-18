"""Image loading (random-access) and optional augmentation for the distributed pipeline.

Two loader classes:
- ``WDSImageLoader``: Random-access via TarRandomAccessReader (byte offsets from manifest).
- ``HFImageLoader``: Reads from HF Arrow/Parquet shard files, preferring
  physical manifest coordinates when available.

Both support loading associated text for SFT / image-text-pair modes.

``ImageAugmenter``: Optional CPU-only PIL transforms applied after load, before tokenization.
"""

import json
import logging
import threading
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from PIL import Image

from vision_tokenization.indexing.manifest import load_hf_manifest, load_wds_manifest
from vision_tokenization.indexing.reader import TarRandomAccessReader
from vision_tokenization.indexing.scanner_hf import _discover_shards as _discover_hf_shards

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WDS image loader
# ---------------------------------------------------------------------------


class WDSImageLoader:
    """Load images (and optional text) from WebDataset tar files via random access.

    The manifest provides ``(tar_path, offset_data, file_size)`` per sample.
    For SFT / image-text-pair modes, text is read from ``.json`` sidecar files
    stored adjacent to images in the tar.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        text_field: Optional[str] = None,
        max_open_files: int = 64,
    ):
        self.manifest = load_wds_manifest(manifest_path)
        self.text_field = text_field

        self._tar_paths = self.manifest.column("tar_path")
        self._offsets = self.manifest.column("offset_data").to_numpy()
        self._sizes = self.manifest.column("file_size").to_numpy()

        self._has_text_columns = "offset_text" in self.manifest.column_names
        if text_field is not None and not self._has_text_columns:
            raise ValueError(
                "WDS manifest does not contain text sidecar columns "
                "(offset_text, text_file_size, text_ext). "
                "Re-create the manifest with text_extensions enabled in scan_wds_dataset()."
            )
        if self._has_text_columns:
            self._text_offsets = self.manifest.column("offset_text").to_numpy()
            self._text_sizes = self.manifest.column("text_file_size").to_numpy()
            self._text_exts = self.manifest.column("text_ext")

        self._reader = TarRandomAccessReader(max_open_files=max_open_files)

    def load_batch(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Load a batch of images (and optionally text) by manifest index."""
        refs = [
            (self._tar_paths[int(i)].as_py(), int(self._offsets[i]), int(self._sizes[i]))
            for i in sample_indices
        ]
        images = self._reader.read_batch(refs)

        texts = None
        if self.text_field is not None:
            if group_slices is not None:
                texts = self._load_texts_grouped(sample_indices, group_slices)
            else:
                texts = self._load_texts(sample_indices)

        return images, texts

    def _load_texts(self, sample_indices: np.ndarray) -> List[Optional[Any]]:
        texts: List[Optional[Any]] = []
        for i in sample_indices:
            i = int(i)
            offset = int(self._text_offsets[i])
            if offset < 0:
                texts.append(None)
                continue

            tar_path = self._tar_paths[i].as_py()
            size = int(self._text_sizes[i])
            ext = self._text_exts[i].as_py()

            try:
                raw = self._reader.read_bytes(tar_path, offset, size)
                if ext == "json":
                    parsed = json.loads(raw)
                    if self.text_field and isinstance(parsed, dict):
                        texts.append(parsed.get(self.text_field))
                    else:
                        texts.append(parsed)
                else:
                    texts.append(raw.decode("utf-8"))
            except Exception:
                logger.warning(
                    f"Failed to read text sidecar at offset {offset} in {tar_path}",
                    exc_info=True,
                )
                texts.append(None)

        return texts

    def _load_single_text(self, manifest_idx: int) -> Optional[Any]:
        offset = int(self._text_offsets[manifest_idx])
        if offset < 0:
            return None
        tar_path = self._tar_paths[manifest_idx].as_py()
        size = int(self._text_sizes[manifest_idx])
        ext = self._text_exts[manifest_idx].as_py()
        try:
            raw = self._reader.read_bytes(tar_path, offset, size)
            if ext == "json":
                parsed = json.loads(raw)
                if self.text_field and isinstance(parsed, dict):
                    return parsed.get(self.text_field)
                return parsed
            return raw.decode("utf-8")
        except Exception:
            logger.warning(
                f"Failed to read text sidecar at offset {offset} in {tar_path}",
                exc_info=True,
            )
            return None

    def _load_texts_grouped(
        self,
        sample_indices: np.ndarray,
        group_slices: np.ndarray,
    ) -> List[Optional[Any]]:
        return [
            self._load_single_text(int(sample_indices[int(start)]))
            for start, _end in group_slices
        ]

    def close(self):
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# HF image loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HFShardInfo:
    path: str
    suffix: str
    num_rows: int
    start: int
    chunk_starts: Tuple[int, ...]
    is_stream: bool = False


class HFImageLoader:
    """Load images (and optional text) from HuggingFace Arrow/Parquet shards."""

    _PHYSICAL_COLUMNS = ("shard_path", "chunk_index", "row_in_chunk")

    def __init__(
        self,
        input_pattern: Union[str, Path],
        image_column: str = "image",
        text_column: Optional[str] = None,
        max_cached_chunks: int = 32,
        manifest_path: Optional[Union[str, Path]] = None,
        image_list_column: Optional[str] = None,
    ):
        self.input_pattern = str(input_pattern)
        self.image_column = image_column
        self.text_column = text_column
        self._image_list_column = image_list_column

        self._shards: List[_HFShardInfo] = []
        self._shard_starts: List[int] = []
        self._total_rows = 0
        self._shard_refs_by_path: Dict[str, _HFShardInfo] = {}
        self._shard_refs_lock = threading.Lock()

        self._is_multi = False
        self._uses_physical_manifest = False
        self._manifest_shard_paths = None
        self._manifest_chunk_indices = None
        self._manifest_row_in_chunk = None
        self._image_indices = None

        if manifest_path is not None:
            manifest_schema = pq.read_schema(str(manifest_path))
            has_physical = all(name in manifest_schema.names for name in self._PHYSICAL_COLUMNS)
            if has_physical:
                self._load_physical_manifest(manifest_path, manifest_schema)
            else:
                self._build_shard_index()
                if "image_index" in manifest_schema.names:
                    if not image_list_column:
                        raise ValueError(
                            "Multi-image manifest (has image_index column) requires "
                            "image_list_column to be set."
                        )
                    manifest = load_hf_manifest(
                        manifest_path,
                        columns=["sample_index", "image_index"],
                    )
                    self._is_multi = True
                    self._sample_indices = manifest.column("sample_index").to_numpy()
                    self._image_indices = manifest.column("image_index").to_numpy()
                    logger.info(
                        f"HFImageLoader: multi-image mode enabled "
                        f"({len(self._sample_indices):,} manifest rows, "
                        f"image_list_column={image_list_column!r})"
                    )
        else:
            self._build_shard_index()

        self._chunk_cache: OrderedDict[Tuple[str, int, Tuple[str, ...]], pa.Table] = OrderedDict()
        self._max_cached_chunks = max(1, int(max_cached_chunks))
        self._chunk_cache_lock = threading.Lock()
        self._parquet_file_cache: OrderedDict[str, pq.ParquetFile] = OrderedDict()
        self._max_open_parquet = max(1, min(self._max_cached_chunks, 16))
        self._parquet_cache_lock = threading.Lock()

    def _load_physical_manifest(self, manifest_path: Union[str, Path], schema: pa.Schema) -> None:
        columns = list(self._PHYSICAL_COLUMNS)
        has_image_index = "image_index" in schema.names
        if has_image_index:
            if not self._image_list_column:
                raise ValueError(
                    "Multi-image manifest (has image_index column) requires "
                    "image_list_column to be set."
                )
            columns.append("image_index")
            self._is_multi = True

        manifest = load_hf_manifest(manifest_path, columns=columns)
        self._manifest_shard_paths = manifest.column("shard_path")
        self._manifest_chunk_indices = manifest.column("chunk_index").to_numpy().astype(np.int32)
        self._manifest_row_in_chunk = manifest.column("row_in_chunk").to_numpy().astype(np.int32)
        self._total_rows = len(self._manifest_chunk_indices)
        self._uses_physical_manifest = True

        if has_image_index:
            self._image_indices = manifest.column("image_index").to_numpy().astype(np.int16)
            logger.info(
                f"HFImageLoader: multi-image mode enabled "
                f"({self._total_rows:,} manifest rows, "
                f"image_list_column={self._image_list_column!r})"
            )
        else:
            logger.info(
                f"HFImageLoader: physical manifest mode enabled "
                f"({self._total_rows:,} manifest rows)"
            )

    @staticmethod
    def _decode_image(img_data) -> Image.Image:
        """Decode a single image from an Arrow/Parquet cell value."""
        if isinstance(img_data, dict) and img_data.get("bytes") is not None:
            img = Image.open(BytesIO(img_data["bytes"]))
        elif isinstance(img_data, dict) and img_data.get("path") is not None:
            img = Image.open(img_data["path"])
        elif isinstance(img_data, bytes):
            img = Image.open(BytesIO(img_data))
        else:
            return img_data
        img.load()
        return img

    def _describe_arrow_shard(self, path: Path) -> Tuple[int, Tuple[int, ...], bool]:
        """Return row count, record-batch starts, and IPC format flag."""
        with pa.memory_map(str(path), "r") as source:
            try:
                reader = ipc.open_file(source)
                chunk_starts = []
                total_rows = 0
                for batch_idx in range(reader.num_record_batches):
                    chunk_starts.append(total_rows)
                    total_rows += reader.get_batch(batch_idx).num_rows
                return total_rows, tuple(chunk_starts), False
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_stream(source)
                chunk_starts = []
                total_rows = 0
                for batch in reader:
                    chunk_starts.append(total_rows)
                    total_rows += batch.num_rows
                return total_rows, tuple(chunk_starts), True

    @staticmethod
    def _detect_arrow_stream(path: Path) -> bool:
        """Return True if the Arrow IPC shard uses stream format."""
        with pa.memory_map(str(path), "r") as source:
            try:
                ipc.open_file(source)
                return False
            except pa.ArrowInvalid:
                source.seek(0)
                ipc.open_stream(source)
                return True

    @staticmethod
    def _describe_parquet_shard(path: Path) -> Tuple[int, Tuple[int, ...]]:
        """Return row count and row-group starts for a parquet shard."""
        parquet_file = pq.ParquetFile(str(path))
        chunk_starts = []
        total_rows = 0
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            chunk_starts.append(total_rows)
            total_rows += parquet_file.metadata.row_group(row_group_idx).num_rows
        return total_rows, tuple(chunk_starts)

    def _build_shard_index(self):
        """Resolve shards from input_pattern and build cumulative row offsets."""
        shard_paths = _discover_hf_shards(self.input_pattern)
        if not shard_paths:
            raise FileNotFoundError(
                f"No arrow/parquet shard files found matching: {self.input_pattern}"
            )

        cum = 0
        num_arrow = 0
        num_parquet = 0
        for shard_path in shard_paths:
            path = Path(shard_path)
            if path.suffix == ".arrow":
                num_rows, chunk_starts, is_stream = self._describe_arrow_shard(path)
                num_arrow += 1
                shard = _HFShardInfo(
                    path=str(path),
                    suffix=path.suffix,
                    num_rows=num_rows,
                    start=cum,
                    chunk_starts=chunk_starts,
                    is_stream=is_stream,
                )
            else:
                num_rows, chunk_starts = self._describe_parquet_shard(path)
                num_parquet += 1
                shard = _HFShardInfo(
                    path=str(path),
                    suffix=path.suffix,
                    num_rows=num_rows,
                    start=cum,
                    chunk_starts=chunk_starts,
                    is_stream=False,
                )

            if shard.num_rows == 0:
                logger.warning(f"Skipping empty HF shard: {shard.path}")
                continue

            self._shards.append(shard)
            self._shard_starts.append(cum)
            self._shard_refs_by_path[shard.path] = shard
            cum += shard.num_rows

        self._total_rows = cum
        logger.info(
            f"HFImageLoader: {len(self._shards)} shards "
            f"({num_arrow} arrow, {num_parquet} parquet), "
            f"{self._total_rows:,} total rows"
        )

    def _locate(self, global_row: int) -> Tuple[int, int]:
        """Map global row index to (shard_index, local_row)."""
        if global_row < 0 or global_row >= self._total_rows:
            raise IndexError(f"global_row {global_row} out of range (total {self._total_rows})")
        shard_idx = bisect_right(self._shard_starts, global_row) - 1
        shard = self._shards[shard_idx]
        return shard_idx, global_row - shard.start

    @staticmethod
    def _locate_chunk(shard: _HFShardInfo, local_row: int) -> Tuple[int, int]:
        """Map a row within a shard to (chunk_index, row_within_chunk)."""
        chunk_idx = bisect_right(shard.chunk_starts, local_row) - 1
        return chunk_idx, local_row - shard.chunk_starts[chunk_idx]

    def _get_shard_ref(self, shard_path: str) -> _HFShardInfo:
        with self._shard_refs_lock:
            cached = self._shard_refs_by_path.get(shard_path)
            if cached is not None:
                return cached

        path = Path(shard_path)
        suffix = path.suffix
        if suffix == ".arrow":
            is_stream = self._detect_arrow_stream(path)
        elif suffix == ".parquet":
            is_stream = False
        else:
            raise ValueError(f"Unsupported HF shard format: {shard_path}")

        shard = _HFShardInfo(
            path=str(path),
            suffix=suffix,
            num_rows=0,
            start=0,
            chunk_starts=(),
            is_stream=is_stream,
        )
        with self._shard_refs_lock:
            cached = self._shard_refs_by_path.get(shard_path)
            if cached is not None:
                return cached
            self._shard_refs_by_path[shard_path] = shard
            return shard

    def _get_parquet_file(self, shard_path: str) -> pq.ParquetFile:
        with self._parquet_cache_lock:
            cached = self._parquet_file_cache.get(shard_path)
            if cached is not None:
                self._parquet_file_cache.move_to_end(shard_path)
                return cached

        parquet_file = pq.ParquetFile(shard_path)

        with self._parquet_cache_lock:
            cached = self._parquet_file_cache.get(shard_path)
            if cached is not None:
                self._parquet_file_cache.move_to_end(shard_path)
                return cached

            while len(self._parquet_file_cache) >= self._max_open_parquet:
                self._parquet_file_cache.popitem(last=False)

            self._parquet_file_cache[shard_path] = parquet_file
            return parquet_file

    def _get_cached_chunk(
        self,
        shard_path: str,
        chunk_idx: int,
        columns_key: Tuple[str, ...],
    ) -> Optional[pa.Table]:
        cache_key = (shard_path, chunk_idx, columns_key)
        with self._chunk_cache_lock:
            cached = self._chunk_cache.get(cache_key)
            if cached is not None:
                self._chunk_cache.move_to_end(cache_key)
                return cached
            return None

    def _cache_chunk(
        self,
        shard_path: str,
        chunk_idx: int,
        columns_key: Tuple[str, ...],
        table: pa.Table,
    ) -> pa.Table:
        cache_key = (shard_path, chunk_idx, columns_key)
        with self._chunk_cache_lock:
            cached = self._chunk_cache.get(cache_key)
            if cached is not None:
                self._chunk_cache.move_to_end(cache_key)
                return cached

            while len(self._chunk_cache) >= self._max_cached_chunks:
                self._chunk_cache.popitem(last=False)
            self._chunk_cache[cache_key] = table
            return table

    def _ensure_chunks(
        self,
        shard: _HFShardInfo,
        chunk_indices: List[int],
        columns: List[str],
    ) -> None:
        """Ensure requested chunks are loaded into the cache."""
        columns_key = tuple(columns)
        missing = [
            chunk_idx
            for chunk_idx in sorted(set(chunk_indices))
            if self._get_cached_chunk(shard.path, chunk_idx, columns_key) is None
        ]
        if not missing:
            return

        if shard.suffix == ".parquet":
            parquet_file = self._get_parquet_file(shard.path)
            for chunk_idx in missing:
                table = parquet_file.read_row_group(chunk_idx, columns=columns)
                self._cache_chunk(shard.path, chunk_idx, columns_key, table)
            return

        with pa.memory_map(shard.path, "r") as source:
            if not shard.is_stream:
                reader = ipc.open_file(source)
                for chunk_idx in missing:
                    batch = reader.get_batch(chunk_idx)
                    table = pa.Table.from_batches([batch]).select(columns)
                    self._cache_chunk(shard.path, chunk_idx, columns_key, table)
                return

            source.seek(0)
            reader = ipc.open_stream(source)
            next_missing = iter(missing)
            target_idx = next(next_missing, None)
            for chunk_idx, batch in enumerate(reader):
                if target_idx is None:
                    break
                if chunk_idx < target_idx:
                    continue
                if chunk_idx == target_idx:
                    table = pa.Table.from_batches([batch]).select(columns)
                    self._cache_chunk(shard.path, chunk_idx, columns_key, table)
                    target_idx = next(next_missing, None)

    def _chunk_table(
        self,
        shard: _HFShardInfo,
        chunk_idx: int,
        columns: List[str],
    ) -> pa.Table:
        columns_key = tuple(columns)
        cached = self._get_cached_chunk(shard.path, chunk_idx, columns_key)
        if cached is not None:
            return cached
        self._ensure_chunks(shard, [chunk_idx], columns)
        cached = self._get_cached_chunk(shard.path, chunk_idx, columns_key)
        if cached is None:
            raise RuntimeError(f"Failed to cache chunk {chunk_idx} for {shard.path}")
        return cached

    def _resolve_batch_rows(
        self,
        sample_indices: np.ndarray,
    ) -> Dict[int, Dict[int, List[Tuple[int, int]]]]:
        """Group requests by shard and chunk for single-image loading."""
        shard_groups: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for batch_pos, sample_idx in enumerate(sample_indices):
            shard_idx, local_row = self._locate(int(sample_idx))
            shard = self._shards[shard_idx]
            chunk_idx, row_in_chunk = self._locate_chunk(shard, local_row)
            shard_groups[shard_idx][chunk_idx].append((batch_pos, row_in_chunk))
        return shard_groups

    def _resolve_physical_batch_rows(
        self,
        sample_indices: np.ndarray,
    ) -> Dict[str, Dict[int, List[Tuple[int, int]]]]:
        """Group single-image requests by shard path and chunk from manifest coords."""
        shard_groups: Dict[str, Dict[int, List[Tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for batch_pos, manifest_idx in enumerate(sample_indices):
            manifest_idx = int(manifest_idx)
            shard_path = self._manifest_shard_paths[manifest_idx].as_py()
            chunk_idx = int(self._manifest_chunk_indices[manifest_idx])
            row_in_chunk = int(self._manifest_row_in_chunk[manifest_idx])
            shard_groups[shard_path][chunk_idx].append((batch_pos, row_in_chunk))
        return shard_groups

    def _resolve_multi_batch_rows(
        self,
        sample_indices: np.ndarray,
    ) -> Dict[int, Dict[int, Dict[int, List[Tuple[int, int]]]]]:
        """Group requests by shard, chunk, and row for multi-image loading."""
        shard_groups: Dict[int, Dict[int, Dict[int, List[Tuple[int, int]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for batch_pos, manifest_idx in enumerate(sample_indices):
            manifest_idx = int(manifest_idx)
            hf_row = int(self._sample_indices[manifest_idx])
            image_idx = int(self._image_indices[manifest_idx])
            shard_idx, local_row = self._locate(hf_row)
            shard = self._shards[shard_idx]
            chunk_idx, row_in_chunk = self._locate_chunk(shard, local_row)
            shard_groups[shard_idx][chunk_idx][row_in_chunk].append((batch_pos, image_idx))
        return shard_groups

    def _resolve_physical_multi_batch_rows(
        self,
        sample_indices: np.ndarray,
    ) -> Dict[str, Dict[int, Dict[int, List[Tuple[int, int]]]]]:
        """Group multi-image requests by shard path, chunk, and row from manifest coords."""
        shard_groups: Dict[str, Dict[int, Dict[int, List[Tuple[int, int]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for batch_pos, manifest_idx in enumerate(sample_indices):
            manifest_idx = int(manifest_idx)
            shard_path = self._manifest_shard_paths[manifest_idx].as_py()
            chunk_idx = int(self._manifest_chunk_indices[manifest_idx])
            row_in_chunk = int(self._manifest_row_in_chunk[manifest_idx])
            image_idx = int(self._image_indices[manifest_idx])
            shard_groups[shard_path][chunk_idx][row_in_chunk].append((batch_pos, image_idx))
        return shard_groups

    def load_batch(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Load a batch of images (and optionally text) by manifest index."""
        if self._is_multi:
            return self._load_batch_multi(sample_indices, group_slices)

        img_results: List[Optional[Image.Image]] = [None] * len(sample_indices)
        txt_results: List[Any] = [None] * len(sample_indices) if self.text_column else []

        columns = [self.image_column]
        if self.text_column:
            columns.append(self.text_column)

        if self._uses_physical_manifest:
            grouped_rows = self._resolve_physical_batch_rows(sample_indices).items()
        else:
            grouped_rows = (
                (self._shards[shard_idx].path, chunk_groups)
                for shard_idx, chunk_groups in self._resolve_batch_rows(sample_indices).items()
            )

        for shard_path, chunk_groups in grouped_rows:
            shard = self._get_shard_ref(shard_path)
            try:
                self._ensure_chunks(shard, list(chunk_groups.keys()), columns)
                for chunk_idx, positions in chunk_groups.items():
                    table = self._chunk_table(shard, chunk_idx, columns)
                    image_column = table.column(self.image_column)
                    text_column = table.column(self.text_column) if self.text_column else None
                    for batch_pos, row_in_chunk in positions:
                        try:
                            img_data = image_column[row_in_chunk].as_py()
                            img_results[batch_pos] = self._decode_image(img_data)
                            if text_column is not None:
                                txt_results[batch_pos] = text_column[row_in_chunk].as_py()
                        except Exception:
                            logger.warning(
                                f"Failed to read row {row_in_chunk} from chunk {chunk_idx} "
                                f"in {shard.path}",
                                exc_info=True,
                            )
            except Exception:
                logger.warning(f"Failed to read shard {shard.path}", exc_info=True)

        if self.text_column and group_slices is not None:
            txt_results = [txt_results[int(start)] for start, _end in group_slices]

        return img_results, txt_results if self.text_column else None

    def _load_batch_multi(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Load a multi-image batch using manifest-row indirection."""
        img_results: List[Optional[Image.Image]] = [None] * len(sample_indices)
        txt_results: List[Any] = [None] * len(sample_indices) if self.text_column else []

        columns = [self._image_list_column]
        if self.text_column:
            columns.append(self.text_column)

        if self._uses_physical_manifest:
            grouped_rows = self._resolve_physical_multi_batch_rows(sample_indices).items()
        else:
            grouped_rows = (
                (self._shards[shard_idx].path, chunk_groups)
                for shard_idx, chunk_groups in self._resolve_multi_batch_rows(sample_indices).items()
            )

        for shard_path, chunk_groups in grouped_rows:
            shard = self._get_shard_ref(shard_path)
            try:
                self._ensure_chunks(shard, list(chunk_groups.keys()), columns)
                for chunk_idx, row_map in chunk_groups.items():
                    table = self._chunk_table(shard, chunk_idx, columns)
                    image_column = table.column(self._image_list_column)
                    text_column = table.column(self.text_column) if self.text_column else None
                    for row_in_chunk, positions in row_map.items():
                        try:
                            img_list = image_column[row_in_chunk].as_py()
                            text_val = text_column[row_in_chunk].as_py() if text_column is not None else None
                            for batch_pos, img_pos in positions:
                                try:
                                    img_results[batch_pos] = self._decode_image(img_list[img_pos])
                                    if text_column is not None:
                                        txt_results[batch_pos] = text_val
                                except Exception:
                                    logger.warning(
                                        f"Failed to decode img_pos {img_pos} in row {row_in_chunk} "
                                        f"from chunk {chunk_idx} in {shard.path}",
                                        exc_info=True,
                                    )
                        except Exception:
                            logger.warning(
                                f"Failed to read row {row_in_chunk} from chunk {chunk_idx} "
                                f"in {shard.path}",
                                exc_info=True,
                            )
            except Exception:
                logger.warning(f"Failed to read shard {shard.path}", exc_info=True)

        if self.text_column and group_slices is not None:
            txt_results = [txt_results[int(start)] for start, _end in group_slices]

        return img_results, txt_results if self.text_column else None

    def close(self):
        with self._chunk_cache_lock:
            self._chunk_cache.clear()
        with self._parquet_cache_lock:
            self._parquet_file_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Image augmenter
# ---------------------------------------------------------------------------


class ImageAugmenter:
    """Optional CPU-only PIL transforms applied after load, before tokenization."""

    def __init__(
        self,
        horizontal_flip: float = 0.0,
        color_jitter: Optional[Dict[str, float]] = None,
    ):
        import torchvision.transforms as T

        transforms = []
        if horizontal_flip > 0:
            transforms.append(T.RandomHorizontalFlip(p=horizontal_flip))
        if color_jitter:
            transforms.append(T.ColorJitter(**color_jitter))

        self._transform = T.Compose(transforms) if transforms else None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._transform is None:
            return image
        return self._transform(image)

    def augment_batch(self, images: List[Optional[Image.Image]]) -> List[Optional[Image.Image]]:
        if self._transform is None:
            return images
        return [self(img) if img is not None else None for img in images]


def create_loader(cfg: Dict[str, Any]):
    """Factory to create the appropriate loader based on dataset_type."""
    dataset_type = cfg.get("dataset_type", "hf")
    text_column = cfg.get("text_column")

    if dataset_type == "wds":
        return WDSImageLoader(
            manifest_path=cfg["manifest_path"],
            text_field=text_column,
            max_open_files=cfg.get("max_open_files", 64),
        )
    if dataset_type == "hf":
        return HFImageLoader(
            input_pattern=cfg["input_pattern"],
            image_column=cfg.get("image_column", "image"),
            text_column=text_column,
            max_cached_chunks=cfg.get("max_cached_chunks", 32),
            manifest_path=cfg.get("manifest_path"),
            image_list_column=cfg.get("image_list_column"),
        )
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
