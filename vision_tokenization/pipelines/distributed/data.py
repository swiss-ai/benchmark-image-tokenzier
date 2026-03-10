"""Image loading (random-access) and optional augmentation for the distributed pipeline.

Two loader classes:
- ``WDSImageLoader``: Random-access via TarRandomAccessReader (byte offsets from manifest).
- ``HFImageLoader``: Reads from HF arrow files by sample_index.

Both support loading associated text for SFT / image-text-pair modes.

``ImageAugmenter``: Optional CPU-only PIL transforms applied after load, before tokenization.
"""

import json
import logging
from collections import OrderedDict, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from PIL import Image

from vision_tokenization.indexing.manifest import load_wds_manifest, load_hf_manifest
from vision_tokenization.indexing.reader import TarRandomAccessReader

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

        # Materialise columns we need for random access.
        # Numeric columns: zero-copy numpy views.
        # String columns: keep as Arrow ChunkedArray (avoids N Python objects);
        # individual lookups via [i].as_py() are O(1).
        self._tar_paths = self.manifest.column("tar_path")
        self._offsets = self.manifest.column("offset_data").to_numpy()
        self._sizes = self.manifest.column("file_size").to_numpy()

        # Text sidecar columns (only present in manifests created with text scanning)
        self._has_text_columns = "offset_text" in self.manifest.column_names
        if text_field is not None and not self._has_text_columns:
            raise ValueError(
                "WDS manifest does not contain text sidecar columns "
                "(offset_text, text_file_size, text_ext). "
                "Re-create the manifest with text_extensions enabled in scan_wds_dataset()."
            )
        if self._has_text_columns:
            # offset_text=-1 means no sidecar for that sample; use numpy for fast access
            self._text_offsets = self.manifest.column("offset_text").to_numpy()
            self._text_sizes = self.manifest.column("text_file_size").to_numpy()
            self._text_exts = self.manifest.column("text_ext")

        self._reader = TarRandomAccessReader(max_open_files=max_open_files)

    def load_batch(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Load a batch of images (and optionally text) by manifest index.

        Args:
            sample_indices: Flat array of manifest row indices (one per image).
            group_slices: Optional ``(num_groups, 2)`` array of ``[start, end)``
                slices into *sample_indices*.  When provided, texts are loaded
                once per group (from the first row's sidecar) instead of once
                per image.

        Returns:
            ``(images, texts)`` where texts is ``None`` if text_field is not set.
            When *group_slices* is provided, ``len(texts) == num_groups``.
            Individual images may be ``None`` on read failure.
        """
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
        """Read text sidecars from tar for each sample.

        Uses ``offset_text`` / ``text_file_size`` / ``text_ext`` columns from
        the manifest to seek into the tar and read raw bytes, then decodes:

        - ``.json`` → ``json.loads()``, then extracts ``self.text_field`` key
          if set, otherwise returns the full parsed object.
        - ``.txt`` (or other) → UTF-8 decoded string.

        Samples whose ``offset_text == -1`` (no sidecar found during scan)
        produce ``None``.
        """
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
        """Read one text sidecar by manifest row index."""
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
        """Load one text per group (from the first row's sidecar)."""
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


class HFImageLoader:
    """Load images (and optional text) from HuggingFace arrow/parquet files.

    Builds a shard-offset index at init: maps global_row → (shard_file, local_row).
    Supports both Arrow IPC (``.arrow``) and Parquet (``.parquet``) files.

    **Single-image mode** (default): ``sample_indices`` passed to
    :meth:`load_batch` are arrow row indices and images are read from
    ``image_column``.

    **Multi-image mode** (``manifest_path`` with ``image_index`` column):
    ``sample_indices`` are *manifest* row indices.  The manifest's
    ``sample_index`` / ``image_index`` columns are used to locate the
    correct arrow row and position within ``image_list_column``.
    """

    def __init__(
        self,
        arrow_dir: Union[str, Path],
        image_column: str = "image",
        text_column: Optional[str] = None,
        max_cached_shards: int = 4,
        manifest_path: Optional[Union[str, Path]] = None,
        image_list_column: Optional[str] = None,
    ):
        self.arrow_dir = Path(arrow_dir)
        self.image_column = image_column
        self.text_column = text_column

        # Build shard index: list of (shard_path, num_rows, cumulative_start)
        self._shards: List[Tuple[str, int, int]] = []
        self._total_rows = 0
        self._build_shard_index()

        # Multi-image support: when a manifest with image_index is provided,
        # sample_indices from BatchAssignment are manifest row indices, not
        # arrow row indices.  We need to map via sample_index / image_index.
        self._is_multi = False
        if manifest_path is not None:
            manifest = load_hf_manifest(manifest_path)
            if "image_index" in manifest.column_names:
                if not image_list_column:
                    raise ValueError(
                        "Multi-image manifest (has image_index column) requires "
                        "image_list_column to be set."
                    )
                self._is_multi = True
                self._sample_indices = manifest.column("sample_index").to_numpy()
                self._image_indices = manifest.column("image_index").to_numpy()
                self._image_list_column = image_list_column
                logger.info(
                    f"HFImageLoader: multi-image mode enabled "
                    f"({len(self._sample_indices):,} manifest rows, "
                    f"image_list_column={image_list_column!r})"
                )

        # LRU cache for shard tables (avoids re-reading the same shard)
        self._table_cache: OrderedDict[str, pa.Table] = OrderedDict()
        self._max_cached = max_cached_shards

    @staticmethod
    def _decode_image(img_data) -> Image.Image:
        """Decode a single image from an arrow cell value."""
        if isinstance(img_data, dict) and img_data.get("bytes") is not None:
            return Image.open(BytesIO(img_data["bytes"]))
        if isinstance(img_data, dict) and img_data.get("path") is not None:
            return Image.open(img_data["path"])
        if isinstance(img_data, bytes):
            return Image.open(BytesIO(img_data))
        return img_data  # Already PIL or similar

    def _count_rows(self, path: Path) -> int:
        """Count rows in an arrow or parquet file."""
        if path.suffix == ".arrow":
            with pa.memory_map(str(path), "r") as source:
                try:
                    reader = ipc.open_file(source)
                    return sum(
                        reader.get_batch(i).num_rows
                        for i in range(reader.num_record_batches)
                    )
                except pa.ArrowInvalid:
                    # HF save_to_disk() writes Arrow IPC stream format, not file format
                    source.seek(0)
                    reader = ipc.open_stream(source)
                    return sum(batch.num_rows for batch in reader)
        else:
            pf = pq.ParquetFile(str(path))
            return pf.metadata.num_rows

    def _build_shard_index(self):
        """Scan arrow/parquet files and build cumulative row offset index."""
        arrow_files = sorted(self.arrow_dir.glob("*.arrow"))
        parquet_files = sorted(self.arrow_dir.glob("*.parquet"))
        all_files = arrow_files + parquet_files
        if not all_files:
            raise FileNotFoundError(f"No arrow/parquet files found in {self.arrow_dir}")

        cum = 0
        for f in all_files:
            n = self._count_rows(f)
            self._shards.append((str(f), n, cum))
            cum += n
        self._total_rows = cum
        logger.info(
            f"HFImageLoader: {len(self._shards)} shards "
            f"({len(arrow_files)} arrow, {len(parquet_files)} parquet), "
            f"{self._total_rows:,} total rows"
        )

    def _locate(self, global_row: int) -> Tuple[str, int]:
        """Map global row index to (shard_file, local_row)."""
        for shard_path, num_rows, start in self._shards:
            if global_row < start + num_rows:
                return shard_path, global_row - start
        raise IndexError(f"global_row {global_row} out of range (total {self._total_rows})")

    def _read_shard(self, shard_path: str, columns: List[str]) -> pa.Table:
        """Read a shard table with LRU caching."""
        if shard_path in self._table_cache:
            self._table_cache.move_to_end(shard_path)
            return self._table_cache[shard_path]

        # Evict oldest if at capacity
        while len(self._table_cache) >= self._max_cached:
            self._table_cache.popitem(last=False)

        if shard_path.endswith(".arrow"):
            with pa.memory_map(shard_path, "r") as source:
                try:
                    reader = ipc.open_file(source)
                except pa.ArrowInvalid:
                    source.seek(0)
                    reader = ipc.open_stream(source)
                table = reader.read_all().select(columns)
        else:
            table = pq.read_table(shard_path, columns=columns)

        self._table_cache[shard_path] = table
        return table

    def load_batch(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Load a batch of images (and optionally text) by sample_index.

        Args:
            sample_indices: Flat array of manifest row indices.
            group_slices: Optional ``(num_groups, 2)`` array of ``[start, end)``
                slices into *sample_indices*.  When provided, texts are loaded
                once per group (from the first row) instead of once per image.

        Returns:
            ``(images, texts)`` where texts is ``None`` if text_column is
            not set.  When *group_slices* is provided,
            ``len(texts) == num_groups``.
        """
        if self._is_multi:
            return self._load_batch_multi(sample_indices, group_slices)

        # --- Single-image path (no manifest or single-image manifest) ---
        # Group indices by shard for efficient reads
        shard_groups: Dict[str, List[Tuple[int, int]]] = {}
        for batch_pos, idx in enumerate(sample_indices):
            idx = int(idx)
            shard_path, local_row = self._locate(idx)
            shard_groups.setdefault(shard_path, []).append((batch_pos, local_row))

        # Pre-allocate results
        img_results: List[Optional[Image.Image]] = [None] * len(sample_indices)
        txt_results: List[Any] = [None] * len(sample_indices) if self.text_column else []

        columns = [self.image_column]
        if self.text_column:
            columns.append(self.text_column)

        for shard_path, positions in shard_groups.items():
            try:
                table = self._read_shard(shard_path, columns)
                for batch_pos, local_row in positions:
                    try:
                        img_data = table.column(self.image_column)[local_row].as_py()
                        img_results[batch_pos] = self._decode_image(img_data)

                        if self.text_column:
                            txt_results[batch_pos] = table.column(self.text_column)[local_row].as_py()
                    except Exception:
                        logger.warning(f"Failed to read row {local_row} from {shard_path}", exc_info=True)
            except Exception:
                logger.warning(f"Failed to read shard {shard_path}", exc_info=True)

        # When group_slices is provided, collapse per-image texts to per-group
        if self.text_column and group_slices is not None:
            txt_results = [txt_results[int(start)] for start, _end in group_slices]

        return img_results, txt_results if self.text_column else None

    def _load_batch_multi(
        self,
        sample_indices: np.ndarray,
        group_slices: Optional[np.ndarray] = None,
    ) -> Tuple[List[Optional[Image.Image]], Optional[List[Any]]]:
        """Multi-image load: sample_indices are *manifest* row indices.

        Each manifest row maps to (hf_row, image_position_in_list) via
        ``_sample_indices`` and ``_image_indices``.

        Groups requests by (shard, local_row) so the image list is
        deserialized only once per HF sample, even when multiple manifest
        rows reference different images within the same sample.
        """
        # Group by (shard, local_row) so we deserialize each image list once.
        shard_row_groups: Dict[str, Dict[int, List[Tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for batch_pos, midx in enumerate(sample_indices):
            midx = int(midx)
            hf_row = int(self._sample_indices[midx])
            img_pos = int(self._image_indices[midx])
            shard_path, local_row = self._locate(hf_row)
            shard_row_groups[shard_path][local_row].append((batch_pos, img_pos))

        img_results: List[Optional[Image.Image]] = [None] * len(sample_indices)
        txt_results: List[Any] = [None] * len(sample_indices) if self.text_column else []

        columns = [self._image_list_column]
        if self.text_column:
            columns.append(self.text_column)

        for shard_path, row_map in shard_row_groups.items():
            try:
                table = self._read_shard(shard_path, columns)
                for local_row, positions in row_map.items():
                    try:
                        img_list = table.column(self._image_list_column)[local_row].as_py()
                        text_val = (
                            table.column(self.text_column)[local_row].as_py()
                            if self.text_column
                            else None
                        )
                        for batch_pos, img_pos in positions:
                            try:
                                img_results[batch_pos] = self._decode_image(img_list[img_pos])
                                if self.text_column:
                                    txt_results[batch_pos] = text_val
                            except Exception:
                                logger.warning(
                                    f"Failed to decode img_pos {img_pos} in row {local_row} "
                                    f"from {shard_path}",
                                    exc_info=True,
                                )
                    except Exception:
                        logger.warning(
                            f"Failed to read row {local_row} from {shard_path}",
                            exc_info=True,
                        )
            except Exception:
                logger.warning(f"Failed to read shard {shard_path}", exc_info=True)

        # Collapse per-image texts to per-group when group_slices is provided
        if self.text_column and group_slices is not None:
            txt_results = [txt_results[int(start)] for start, _end in group_slices]

        return img_results, txt_results if self.text_column else None

    def close(self):
        self._table_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Image augmenter
# ---------------------------------------------------------------------------


class ImageAugmenter:
    """Optional CPU-only PIL transforms applied after load, before tokenization.

    Args:
        horizontal_flip: Probability of random horizontal flip.
        color_jitter: Dict of ColorJitter kwargs, or None to disable.
    """

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
        """Apply augmentation to a batch, skipping None entries."""
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
    elif dataset_type == "hf":
        return HFImageLoader(
            arrow_dir=cfg["arrow_dir"],
            image_column=cfg.get("image_column", "image"),
            text_column=text_column,
            manifest_path=cfg.get("manifest_path"),
            image_list_column=cfg.get("image_list_column"),
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
