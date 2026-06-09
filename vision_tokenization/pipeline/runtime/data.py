"""Image loading (random-access) and optional augmentation for the distributed pipeline.

Two loader classes:
- ``WDSImageLoader``: Random-access via TarRandomAccessReader (byte offsets from manifest).
- ``HFImageLoader``: Reads from HF Arrow/Parquet shard files, preferring
  physical manifest coordinates when available.
- ``JSONLTarLoader``: Reads images from a JSONL+tar manifest, with document
  text loaded from JSONL by byte offsets.

Both support loading associated text for SFT / image-text-pair modes.

``ImageAugmenter``: Optional CPU-only PIL transforms applied after load, before tokenization.
"""

import logging
import os
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

import orjson

from vision_tokenization.indexing.manifest import (
    load_hf_manifest,
    load_interleave_manifest,
    load_wds_manifest,
)
from vision_tokenization.indexing.reader import TarRandomAccessReader
from vision_tokenization.indexing.scanners.hf import _discover_shards as _discover_hf_shards
from vision_tokenization.utils.interleave_documents import (
    extract_local_image_refs,
    parse_interleave_segments,
)
from vision_tokenization.utils.image_map_sft import (
    NORMALIZED_MESSAGES_KEY,
    image_map_lookup,
    normalize_messages_and_refs,
)
from vision_tokenization.utils.image_map_parquet import read_image_map_row_group_rows

logger = logging.getLogger(__name__)

DOC_NUM_IMAGES_KEY = "__doc_num_images__"


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
        document_format: Optional[str] = None,
        document_field: Optional[str] = None,
        local_image_prefixes: Optional[List[str]] = None,
        max_open_files: int = 64,
    ):
        self.manifest = load_wds_manifest(manifest_path)
        self.text_field = text_field
        self.document_format = document_format
        self.document_field = document_field
        self.local_image_prefixes = local_image_prefixes

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
        self._has_group_columns = all(
            name in self.manifest.column_names for name in ("group_id", "image_index")
        )
        if self._has_group_columns:
            self._group_ids = self.manifest.column("group_id").to_numpy()
            self._image_indices = self.manifest.column("image_index").to_numpy()
            unique_group_ids, group_counts = np.unique(
                self._group_ids.astype(np.int64, copy=False),
                return_counts=True,
            )
            self._group_sizes = {
                int(group_id): int(count)
                for group_id, count in zip(unique_group_ids.tolist(), group_counts.tolist())
            }
        else:
            self._group_ids = None
            self._image_indices = None
            self._group_sizes = None

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

    def _read_single_text_payload(
        self,
        manifest_idx: int,
        *,
        extract_text_field: bool = True,
    ) -> Optional[Any]:
        offset = int(self._text_offsets[manifest_idx])
        if offset < 0:
            return None

        tar_path = self._tar_paths[manifest_idx].as_py()
        size = int(self._text_sizes[manifest_idx])
        ext = self._text_exts[manifest_idx].as_py()

        try:
            raw = self._reader.read_bytes(tar_path, offset, size)
            if ext == "json":
                parsed = orjson.loads(raw)
                if extract_text_field and self.text_field and isinstance(parsed, dict):
                    return parsed.get(self.text_field)
                return parsed
            return raw.decode("utf-8")
        except Exception:
            logger.warning(
                f"Failed to read text sidecar at offset {offset} in {tar_path}",
                exc_info=True,
            )
            return None

    def _load_texts(self, sample_indices: np.ndarray) -> List[Optional[Any]]:
        texts: List[Optional[Any]] = []
        for i in sample_indices:
            i = int(i)
            texts.append(self._read_single_text_payload(i, extract_text_field=True))
        return texts

    def _load_single_text(self, manifest_idx: int) -> Optional[Any]:
        return self._read_single_text_payload(manifest_idx, extract_text_field=True)

    def _load_group_text(
        self,
        sample_indices: np.ndarray,
        start: int,
        end: int,
    ) -> Optional[Any]:
        manifest_idx = int(sample_indices[start])
        if self.document_format is None:
            return self._load_single_text(manifest_idx)

        raw_payload = self._read_single_text_payload(
            manifest_idx, extract_text_field=False
        )
        if raw_payload is None:
            return None

        if (
            isinstance(raw_payload, dict)
            and self.document_field is None
            and self.text_field
        ):
            raw_payload = raw_payload.get(self.text_field)

        try:
            segments = list(
                parse_interleave_segments(
                    raw_payload,
                    document_format=self.document_format,
                    document_field=self.document_field,
                    local_prefixes=self.local_image_prefixes,
                    num_images=max(0, int(end) - int(start)),
                )
            )
        except Exception:
            logger.warning(
                "Failed to parse grouped WDS interleave document at manifest row %d",
                manifest_idx,
                exc_info=True,
            )
            return None

        if self._has_group_columns:
            fragment_rows = [int(sample_indices[row_idx]) for row_idx in range(start, end)]
            fragment_group_ids = {int(self._group_ids[row_idx]) for row_idx in fragment_rows}
            if len(fragment_group_ids) != 1:
                logger.warning(
                    "Grouped WDS text load saw mixed group_ids at manifest rows %s",
                    fragment_rows,
                )
                return None
            group_id = next(iter(fragment_group_ids))
            expected_images = int(self._group_sizes[group_id])
            parsed_images = sum(1 for seg in segments if seg.get("type") == "image")
            if parsed_images != expected_images:
                logger.warning(
                    "WDS interleave document/image count mismatch for group %d at manifest row %d: "
                    "parsed_images=%d expected_images=%d",
                    group_id,
                    manifest_idx,
                    parsed_images,
                    expected_images,
                )
                return None
            fragment_image_indices = sorted(
                int(self._image_indices[row_idx]) for row_idx in fragment_rows
            )
            if (
                len(fragment_image_indices) != len(set(fragment_image_indices))
                or fragment_image_indices[0] < 0
                or fragment_image_indices[-1] >= expected_images
            ):
                logger.warning(
                    "Grouped WDS image_index mismatch at manifest rows %s: "
                    "manifest=%s expected_range=[0,%d)",
                    fragment_rows,
                    fragment_image_indices,
                    expected_images,
                )
                return None

        return segments

    def _load_texts_grouped(
        self,
        sample_indices: np.ndarray,
        group_slices: np.ndarray,
    ) -> List[Optional[Any]]:
        return [
            self._load_group_text(sample_indices, int(start), int(end))
            for start, end in group_slices
        ]

    def close(self):
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class JSONLTarLoader:
    """Load images and optional text from a JSONL+tar manifest."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        document_format: Optional[str] = None,
        document_field: Optional[str] = None,
        local_image_prefixes: Optional[List[str]] = None,
        mode: str = "interleave",
        text_column: Optional[str] = None,
        max_open_files: int = 64,
    ):
        self.manifest = load_interleave_manifest(manifest_path)
        self.mode = mode
        self.document_format = document_format
        self.document_field = document_field
        self.local_image_prefixes = local_image_prefixes
        self.text_column = text_column

        if self.mode == "interleave":
            if self.document_format is None:
                raise ValueError("JSONLTarLoader(mode='interleave') requires document_format")
        elif self.text_column is None:
            raise ValueError(
                "JSONLTarLoader requires text_column for non-interleave modes"
            )

        self._tar_paths = self.manifest.column("tar_path")
        self._offsets = self.manifest.column("offset_data").to_numpy()
        self._sizes = self.manifest.column("file_size").to_numpy()
        self._jsonl_paths = self.manifest.column("jsonl_path")
        self._line_starts = self.manifest.column("line_start").to_numpy()
        self._line_lengths = self.manifest.column("line_length").to_numpy()
        self._image_refs = self.manifest.column("image_ref")
        self._image_indices = self.manifest.column("image_index").to_numpy()
        self._has_segment_ranges = all(
            name in self.manifest.column_names
            for name in ("segment_start_index", "segment_end_index")
        )
        if self._has_segment_ranges:
            self._segment_starts = self.manifest.column("segment_start_index").to_numpy()
            self._segment_ends = self.manifest.column("segment_end_index").to_numpy()
        else:
            self._segment_starts = None
            self._segment_ends = None

        self._reader = TarRandomAccessReader(max_open_files=max_open_files)
        self._max_open_files = max(1, int(max_open_files))
        self._local = threading.local()
        self._all_fd_caches_lock = threading.Lock()
        self._all_fd_caches: list[OrderedDict[str, int]] = []

    def _get_thread_fd_cache(self) -> OrderedDict[str, int]:
        """Return the per-thread LRU fd cache, creating it on first access."""
        if not hasattr(self._local, "jsonl_fds"):
            self._local.jsonl_fds = OrderedDict()
            with self._all_fd_caches_lock:
                self._all_fd_caches.append(self._local.jsonl_fds)
        return self._local.jsonl_fds

    def _get_jsonl_fd(self, jsonl_path: str) -> int:
        fds = self._get_thread_fd_cache()
        cached = fds.get(jsonl_path)
        if cached is not None:
            fds.move_to_end(jsonl_path)
            return cached

        while len(fds) >= self._max_open_files:
            _old_path, old_fd = fds.popitem(last=False)
            os.close(old_fd)

        fd = os.open(jsonl_path, os.O_RDONLY)
        fds[jsonl_path] = fd
        return fd

    def _read_jsonl_line(self, jsonl_path: str, line_start: int, line_length: int) -> bytes:
        fd = self._get_jsonl_fd(jsonl_path)
        raw = os.pread(fd, line_length, line_start)
        if len(raw) != line_length:
            raise IOError(
                f"Short pread for {jsonl_path}: expected {line_length} bytes at offset {line_start}, "
                f"got {len(raw)}"
            )
        return raw

    def _load_text_value(self, manifest_idx: int) -> Optional[Any]:
        jsonl_path = self._jsonl_paths[manifest_idx].as_py()
        line_start = int(self._line_starts[manifest_idx])
        line_length = int(self._line_lengths[manifest_idx])

        try:
            raw = self._read_jsonl_line(jsonl_path, line_start, line_length)
            sample = orjson.loads(raw)
            if not isinstance(sample, dict):
                logger.warning(
                    "Expected JSON object row for %s at offset %d when loading text_column=%r",
                    jsonl_path,
                    line_start,
                    self.text_column,
                )
                return None
            return sample.get(self.text_column) if self.text_column is not None else None
        except Exception:
            logger.warning(
                "Failed to load JSONL row at offset %d in %s",
                line_start,
                jsonl_path,
                exc_info=True,
            )
            return None

    def _load_group_text(
        self,
        sample_indices: np.ndarray,
        start: int,
        end: int,
    ) -> Optional[Any]:
        if self.mode != "interleave":
            return self._load_text_value(int(sample_indices[start]))

        manifest_idx = int(sample_indices[start])
        jsonl_path = self._jsonl_paths[manifest_idx].as_py()
        line_start = int(self._line_starts[manifest_idx])
        line_length = int(self._line_lengths[manifest_idx])

        try:
            raw = self._read_jsonl_line(jsonl_path, line_start, line_length)
            sample = orjson.loads(raw)
            # Interleave rows are reparsed at load time so text and manifest stay consistent.
            segments = parse_interleave_segments(
                sample,
                document_format=self.document_format,
                document_field=self.document_field,
                local_prefixes=self.local_image_prefixes,
            )
            if self._has_segment_ranges:
                segment_start = int(self._segment_starts[manifest_idx])
                segment_end = int(self._segment_ends[manifest_idx])
                if segment_start < 0 or segment_end > len(segments) or segment_start >= segment_end:
                    logger.warning(
                        "Invalid interleave segment range for %s at offset %d: [%d, %d) with %d segments",
                        jsonl_path,
                        line_start,
                        segment_start,
                        segment_end,
                        len(segments),
                    )
                    return None
                segments = list(segments[segment_start:segment_end])
            else:
                segments = list(segments)

            refs = extract_local_image_refs(segments)
            fragment_rows = [int(sample_indices[row_idx]) for row_idx in range(start, end)]
            manifest_pairs = [
                (
                    int(self._image_indices[manifest_idx]),
                    self._image_refs[manifest_idx].as_py(),
                )
                for manifest_idx in fragment_rows
            ]
            mismatched = [
                (image_index, image_ref)
                for image_index, image_ref in manifest_pairs
                if image_index < 0
                or image_index >= len(refs)
                or refs[image_index] != image_ref
            ]
            if mismatched:
                fragment_refs = [image_ref for _image_index, image_ref in manifest_pairs]
                logger.warning(
                    "Interleave manifest/text mismatch for %s at offset %d: "
                    "parsed=%s fragment_manifest=%s mismatched=%s",
                    jsonl_path,
                    line_start,
                    refs,
                    fragment_refs,
                    mismatched,
                )
                return None
            return segments
        except Exception:
            logger.warning(
                "Failed to load interleave document at offset %d in %s",
                line_start,
                jsonl_path,
                exc_info=True,
            )
            return None

    def _load_texts_grouped(
        self,
        sample_indices: np.ndarray,
        group_slices: np.ndarray,
    ) -> List[Optional[Any]]:
        return [
            self._load_group_text(sample_indices, int(start), int(end))
            for start, end in group_slices
        ]

    def _load_texts_flat(self, sample_indices: np.ndarray) -> List[Optional[Any]]:
        return [self._load_text_value(int(manifest_idx)) for manifest_idx in sample_indices]

    def load_batch(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        refs = [
            (self._tar_paths[int(i)].as_py(), int(self._offsets[i]), int(self._sizes[i]))
            for i in sample_indices
        ]
        images = self._reader.read_batch(refs)

        texts = None
        if self.mode == "interleave":
            if group_slices is not None:
                texts = self._load_texts_grouped(sample_indices, group_slices)
        elif group_slices is None:
            texts = self._load_texts_flat(sample_indices)
        else:
            texts = self._load_texts_grouped(sample_indices, group_slices)

        return images, texts

    def close(self):
        self._reader.close()
        with self._all_fd_caches_lock:
            for fds in self._all_fd_caches:
                for fd in fds.values():
                    os.close(fd)
                fds.clear()
            self._all_fd_caches.clear()

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
        image_map_column: Optional[str] = None,
        message_column: Optional[str] = None,
        parser: Optional[str] = None,
        parser_columns: Optional[List[str]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        parser_kind: Optional[str] = None,
    ):
        self.input_pattern = str(input_pattern)
        self.image_column = image_column
        self.text_column = text_column
        self._image_list_column = image_list_column
        self._image_map_column = image_map_column
        self._message_column = message_column
        self._parser = parser
        self._parser_columns = parser_columns or []
        self._parser_args = dict(parser_args or {})
        self._parser_kind = parser_kind

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

        if self._image_list_column and self._image_map_column:
            raise ValueError("Set only one of image_list_column or image_map_column")
        if self._image_map_column and not self._message_column:
            raise ValueError("image_map_column requires message_column")

        if manifest_path is not None:
            manifest_schema = pq.read_schema(str(manifest_path))
            has_physical = all(name in manifest_schema.names for name in self._PHYSICAL_COLUMNS)
            if has_physical:
                self._load_physical_manifest(manifest_path, manifest_schema)
            else:
                self._build_shard_index()
                if "image_index" in manifest_schema.names:
                    if not (image_list_column or image_map_column):
                        raise ValueError(
                            "Multi-image manifest (has image_index column) requires "
                            "image_list_column or image_map_column to be set."
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
                        f"image_list_column={image_list_column!r}, "
                        f"image_map_column={image_map_column!r})"
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
            if not (self._image_list_column or self._image_map_column):
                raise ValueError(
                    "Multi-image manifest (has image_index column) requires "
                    "image_list_column or image_map_column to be set."
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
                f"image_list_column={self._image_list_column!r}, "
                f"image_map_column={self._image_map_column!r})"
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

    def _chunk_table_for_rows(
        self,
        shard: _HFShardInfo,
        chunk_idx: int,
        columns: List[str],
        row_indices,
    ) -> tuple[pa.Table, dict[int, int]]:
        if self._image_map_column and shard.suffix == ".parquet":
            parquet_file = self._get_parquet_file(shard.path)
            return read_image_map_row_group_rows(
                parquet_file,
                chunk_idx,
                columns,
                row_indices,
            )

        self._ensure_chunks(shard, [chunk_idx], columns)
        table = self._chunk_table(shard, chunk_idx, columns)
        return table, {int(row_idx): int(row_idx) for row_idx in row_indices}

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

    def _iter_single_grouped_rows(
        self,
        sample_indices: np.ndarray,
    ):
        if self._uses_physical_manifest:
            return self._resolve_physical_batch_rows(sample_indices).items()
        return (
            (self._shards[shard_idx].path, chunk_groups)
            for shard_idx, chunk_groups in self._resolve_batch_rows(sample_indices).items()
        )

    def _iter_multi_grouped_rows(
        self,
        sample_indices: np.ndarray,
    ):
        if self._uses_physical_manifest:
            return self._resolve_physical_multi_batch_rows(sample_indices).items()
        return (
            (self._shards[shard_idx].path, chunk_groups)
            for shard_idx, chunk_groups in self._resolve_multi_batch_rows(sample_indices).items()
        )

    def _load_text_values_for_rows(
        self,
        table: pa.Table,
        row_indices: List[int],
        *,
        shard_path: str,
        chunk_idx: int,
    ) -> List[Any]:
        """Load a chunk's text rows in bulk, falling back to per-row logging."""
        if not row_indices:
            return []

        try:
            selection = pa.array(row_indices, type=pa.int32())
            return table.take(selection).column(self.text_column).to_pylist()
        except Exception:
            logger.warning(
                "Falling back to per-row text reads for chunk %d in %s",
                chunk_idx,
                shard_path,
                exc_info=True,
            )

        text_column = table.column(self.text_column)
        values: List[Any] = []
        for row_in_chunk in row_indices:
            try:
                values.append(text_column[row_in_chunk].as_py())
            except Exception:
                logger.warning(
                    f"Failed to read text row {row_in_chunk} from chunk {chunk_idx} "
                    f"in {shard_path}",
                    exc_info=True,
                )
                values.append(None)
        return values

    def _parse_row_dicts(
        self,
        row_dict_by_batch_pos: Dict[int, dict[str, Any]],
        *,
        batch_size: int,
        group_slices: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Parse cached row dicts into structured text payloads.

        ``HFImageLoader`` supports two parser families:
        - ``interleave`` parsers return ordered ``text/image`` segments.
        - ``sft`` parsers return canonical conversation/message lists.

        SFT parsers receive ``num_images`` from the row's stored full image
        count so fragmented multi-image documents render the right placeholder
        count rather than the current fragment size.
        """
        if not self._parser:
            return []

        if self._parser_kind == "interleave":
            from vision_tokenization.parsers import (
                parse_segments as parse_fn,
                is_row_shaped_interleave_parser,
            )

            row_shaped = is_row_shaped_interleave_parser(self._parser)
            if not row_shaped:
                # Document parsers take one column's value as the positional payload.
                if len(self._parser_columns) != 1:
                    raise ValueError(
                        f"Interleave parser {self._parser!r} is document-shaped and "
                        f"requires exactly one parser_columns entry, got "
                        f"{self._parser_columns!r}"
                    )
                payload_column = self._parser_columns[0]

            def _parse(row_dict: dict[str, Any], num_images: int) -> Any:
                if row_shaped:
                    return parse_fn(
                        row_dict,
                        parser=self._parser,
                        num_images=num_images,
                        **self._parser_args,
                    )
                payload = row_dict.get(payload_column)
                return parse_fn(
                    payload,
                    parser=self._parser,
                    num_images=num_images,
                    **self._parser_args,
                )
        elif self._parser_kind == "sft":
            from vision_tokenization.parsers import parse_sft_messages as parse_fn

            def _parse(row_dict: dict[str, Any], num_images: int) -> Any:
                return parse_fn(
                    row_dict,
                    parser=self._parser,
                    num_images=num_images,
                    parser_args=self._parser_args,
                )
        else:
            raise ValueError(f"Unknown parser_kind: {self._parser_kind!r}")

        def _doc_num_images(row_dict: dict[str, Any], fragment_count: int) -> int:
            return int(row_dict.get(DOC_NUM_IMAGES_KEY, fragment_count))

        if group_slices is None:
            parsed_results: List[Any] = []
            # ``row_dict_by_batch_pos`` is sparse when earlier reads fail, so
            # iterate the requested batch width instead of the populated dict.
            for batch_pos in range(batch_size):
                row_dict = row_dict_by_batch_pos.get(batch_pos)
                if row_dict is None:
                    parsed_results.append(None)
                    continue
                try:
                    parsed_results.append(_parse(row_dict, _doc_num_images(row_dict, 1)))
                except Exception:
                    logger.warning("Parser failed for sample", exc_info=True)
                    parsed_results.append(None)
            return parsed_results

        parsed_results = []
        for start, end in group_slices:
            start, end = int(start), int(end)
            row_dict = row_dict_by_batch_pos.get(start)
            if row_dict is None:
                parsed_results.append(None)
                continue
            try:
                parsed_results.append(
                    _parse(row_dict, _doc_num_images(row_dict, end - start))
                )
            except Exception:
                logger.warning("Parser failed for group", exc_info=True)
                parsed_results.append(None)
        return parsed_results

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
        txt_results: List[Any] = [None] * len(sample_indices) if (self.text_column or self._parser) else []

        columns = [self.image_column]
        if self.text_column:
            columns.append(self.text_column)
        for col in self._parser_columns:
            if col not in columns:
                columns.append(col)

        parser_row_by_batch_pos: Dict[int, dict[str, Any]] = {}

        grouped_rows = self._iter_single_grouped_rows(sample_indices)

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
                            if self._parser:
                                row_dict: dict[str, Any] = {}
                                for col in self._parser_columns:
                                    try:
                                        row_dict[col] = table.column(col)[row_in_chunk].as_py()
                                    except Exception:
                                        row_dict[col] = None
                                parser_row_by_batch_pos[batch_pos] = row_dict
                        except Exception:
                            logger.warning(
                                f"Failed to read row {row_in_chunk} from chunk {chunk_idx} "
                                f"in {shard.path}",
                                exc_info=True,
                            )
            except Exception:
                logger.warning(f"Failed to read shard {shard.path}", exc_info=True)

        if self._parser:
            txt_results = self._parse_row_dicts(
                parser_row_by_batch_pos,
                batch_size=len(sample_indices),
                group_slices=group_slices,
            )
        elif self.text_column and group_slices is not None:
            txt_results = [txt_results[int(start)] for start, _end in group_slices]

        return img_results, txt_results if (self.text_column or self._parser) else None

    def _load_batch_multi(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Load a multi-image batch using manifest-row indirection."""
        img_results: List[Optional[Image.Image]] = [None] * len(sample_indices)
        txt_results: List[Any] = [None] * len(sample_indices) if (self.text_column or self._parser) else []

        if self._image_map_column:
            columns = [self._image_map_column, self._message_column]
        else:
            columns = [self._image_list_column]
        if self.text_column:
            columns.append(self.text_column)
        for col in self._parser_columns:
            if col not in columns:
                columns.append(col)

        parser_row_by_batch_pos: Dict[int, dict] = {}
        # Multiple batch positions can map to the same parquet row (multi-image
        # docs); cache parser-row dicts per chunk to avoid re-reading columns.
        _row_dict_cache: Dict[int, dict] = {}

        grouped_rows = self._iter_multi_grouped_rows(sample_indices)

        for shard_path, chunk_groups in grouped_rows:
            shard = self._get_shard_ref(shard_path)
            try:
                for chunk_idx, row_map in chunk_groups.items():
                    table, row_positions = self._chunk_table_for_rows(
                        shard,
                        chunk_idx,
                        columns,
                        row_map.keys(),
                    )
                    image_column = table.column(self._image_map_column or self._image_list_column)
                    message_column = (
                        table.column(self._message_column)
                        if self._image_map_column
                        else None
                    )
                    text_column = table.column(self.text_column) if self.text_column else None

                    if self._parser:
                        _row_dict_cache.clear()
                        skip_cols = (
                            {self._message_column} if self._image_map_column else set()
                        )
                        for row_in_chunk in row_map:
                            if row_in_chunk not in _row_dict_cache:
                                table_pos = row_positions.get(row_in_chunk)
                                if table_pos is None:
                                    continue
                                rd = {}
                                for col in self._parser_columns:
                                    if col in skip_cols:
                                        continue
                                    try:
                                        rd[col] = table.column(col)[table_pos].as_py()
                                    except Exception:
                                        rd[col] = None
                                _row_dict_cache[row_in_chunk] = rd

                    for row_in_chunk, positions in row_map.items():
                        try:
                            table_pos = row_positions.get(row_in_chunk)
                            if table_pos is None:
                                continue
                            if self._image_map_column:
                                raw_messages = message_column[table_pos].as_py()
                                normalized_messages, image_refs = normalize_messages_and_refs(
                                    raw_messages
                                )
                                images_by_ref = image_map_lookup(
                                    image_column[table_pos].as_py()
                                )
                                doc_num_images = len(image_refs)
                            else:
                                img_list = image_column[table_pos].as_py()
                                doc_num_images = len(img_list)
                                image_refs = []
                                images_by_ref = {}

                            text_val = text_column[table_pos].as_py() if text_column is not None else None
                            if self._parser:
                                _row_dict_cache[row_in_chunk][DOC_NUM_IMAGES_KEY] = doc_num_images
                                if self._image_map_column:
                                    _row_dict_cache[row_in_chunk][NORMALIZED_MESSAGES_KEY] = normalized_messages

                            for batch_pos, img_pos in positions:
                                try:
                                    if self._image_map_column:
                                        img_ref = image_refs[img_pos]
                                        img_results[batch_pos] = self._decode_image(
                                            images_by_ref.get(img_ref)
                                        )
                                    else:
                                        img_results[batch_pos] = self._decode_image(img_list[img_pos])
                                    if text_column is not None:
                                        txt_results[batch_pos] = text_val
                                    if self._parser:
                                        parser_row_by_batch_pos[batch_pos] = _row_dict_cache[row_in_chunk]
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

        if self._parser:
            txt_results = self._parse_row_dicts(
                parser_row_by_batch_pos,
                batch_size=len(sample_indices),
                group_slices=group_slices,
            )
        elif self.text_column and group_slices is not None:
            txt_results = [txt_results[int(start)] for start, _end in group_slices]

        return img_results, txt_results if (self.text_column or self._parser) else None

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

_LEGACY_INTERLEAVE_DATASET_TYPES = {
    "hf_interleave": ("hf", "interleave"),
    "jsonl_tar_interleave": ("jsonl_tar", "interleave"),
}


def _normalize_loader_storage_and_mode(cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Keep legacy alias types working while configs move to storage-only dataset_type."""
    dataset_type = cfg.get("dataset_type", "hf")
    mode = cfg.get("mode")

    alias_target = _LEGACY_INTERLEAVE_DATASET_TYPES.get(dataset_type)
    if alias_target is not None:
        normalized_type, alias_mode = alias_target
        if mode is not None and mode != alias_mode:
            raise ValueError(
                f"dataset_type={dataset_type!r} is incompatible with mode={mode!r}; "
                f"use dataset_type={normalized_type!r} mode={alias_mode!r} instead"
            )
        return normalized_type, alias_mode

    if mode is None:
        if dataset_type == "wds" and cfg.get("parser"):
            mode = "interleave"
        elif dataset_type == "jsonl_tar" and cfg.get("document_format"):
            mode = "interleave"

    return dataset_type, mode


def create_loader(cfg: Dict[str, Any]):
    """Factory to create the appropriate loader based on dataset_type."""
    dataset_type, mode = _normalize_loader_storage_and_mode(cfg)
    text_column = cfg.get("text_column")
    parser = cfg.get("parser")

    if dataset_type == "jsonl_tar" and parser and mode != "interleave":
        raise ValueError(
            "jsonl_tar datasets do not support parser-backed loading yet; use text_column directly"
        )

    parser_kind = None
    if dataset_type == "hf" and mode == "interleave":
        parser_kind = "interleave"
    elif dataset_type == "hf" and mode == "sft" and parser:
        parser_kind = "sft"

    if dataset_type == "hf" and mode == "interleave" and parser is None:
        raise ValueError("hf interleave datasets require parser to be set")
    if parser_kind and not cfg.get("parser_columns"):
        raise ValueError(
            f"{dataset_type} dataset with parser={parser!r} requires parser_columns to be set"
        )

    if dataset_type == "wds":
        document_format = cfg.get("document_format")
        if mode == "interleave":
            document_format = document_format or parser
        return WDSImageLoader(
            manifest_path=cfg["manifest_path"],
            text_field=text_column,
            document_format=document_format,
            document_field=cfg.get("document_field"),
            local_image_prefixes=cfg.get("local_image_prefixes"),
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
            image_map_column=cfg.get("image_map_column"),
            message_column=cfg.get("message_column"),
            parser=parser if parser_kind else None,
            parser_columns=cfg.get("parser_columns") if parser_kind else None,
            parser_args=cfg.get("parser_args") if parser_kind else None,
            parser_kind=parser_kind,
        )
    if dataset_type == "jsonl_tar":
        interleave_format = cfg.get("document_format")
        if mode == "interleave":
            interleave_format = interleave_format or parser
        return JSONLTarLoader(
            manifest_path=cfg["manifest_path"],
            document_format=interleave_format,
            document_field=cfg.get("document_field"),
            local_image_prefixes=cfg.get("local_image_prefixes"),
            mode=mode or "interleave",
            text_column=text_column,
            max_open_files=cfg.get("max_open_files", 64),
        )
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
