"""HF dataset scanner — iterates a HuggingFace dataset and writes an HF manifest."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import imagesize

from vision_tokenization.indexing.manifest import save_hf_manifest

logger = logging.getLogger(__name__)


def _get_image_dimensions(img_data) -> Tuple[int, int]:
    """Get dimensions from raw (undecoded) HF image cell via imagesize.

    Handles the dict formats produced by ``datasets.Image(decode=False)``
    (``{"bytes": ..., "path": ...}``), raw ``bytes``, and PIL fallback.
    """
    if isinstance(img_data, dict) and img_data.get("bytes") is not None:
        return imagesize.get(BytesIO(img_data["bytes"]))
    if isinstance(img_data, dict) and img_data.get("path") is not None:
        return imagesize.get(img_data["path"])
    if isinstance(img_data, bytes):
        return imagesize.get(BytesIO(img_data))
    if hasattr(img_data, "size"):  # PIL fallback
        return img_data.size
    return (-1, -1)


def scan_hf_dataset(
    dataset_name: str,
    dataset_split: str = "train",
    output_manifest: Union[str, Path] = "hf_manifest.parquet",
    image_column: str = "image",
    image_list_column: Optional[str] = None,
    config_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dataset_load_method: str = "default",
    data_files: Optional[str] = None,
    num_workers: int = 8,
) -> str:
    """Scan an HF dataset and write width/height to a Parquet manifest.

    Image dimensions are extracted from raw bytes via ``imagesize`` (header-
    only, no pixel decode).  The dataset image column is cast to
    ``Image(decode=False)`` so PIL is never instantiated.  Scanning is
    parallelised with ``Dataset.map(num_proc=num_workers)``.

    **Multi-image mode** (``image_list_column`` is set):
    The column contains ``List[Image]``.  Each list element becomes a
    separate manifest row sharing the same ``group_id = sample_index``,
    with ``image_index`` = position in the list.

    Args:
        dataset_name: HuggingFace dataset name or local path.
        dataset_split: Split to scan (e.g. ``"train"``).
        output_manifest: Destination Parquet path.
        image_column: Column name containing a single image.
        image_list_column: Column name containing ``List[Image]``
            for multi-image samples.  Mutually exclusive with
            ``image_column`` when set.
        config_name: Dataset config/subset name.
        cache_dir: HF cache directory.
        dataset_load_method: Loading method (``"default"``, ``"builder_load"``, ``"disk_load"``).
        data_files: Comma-separated data file paths.
        num_workers: Number of parallel workers for ``Dataset.map``.

    Returns:
        The output manifest path as a string.
    """
    from vision_tokenization.utils.dataset_loader import load_hf_dataset

    logger.info(f"Loading HF dataset: {dataset_name} split={dataset_split}")
    dataset = load_hf_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        split=dataset_split,
        cache_dir=cache_dir,
        method=dataset_load_method,
        streaming=False,
        data_files=data_files,
    )

    is_multi = image_list_column is not None
    col = image_list_column if is_multi else image_column

    # ------------------------------------------------------------------
    # Disable PIL decoding — gives raw {"bytes": ..., "path": ...} dicts
    # ------------------------------------------------------------------
    try:
        from datasets import Image as HFImage, Sequence

        if is_multi:
            dataset = dataset.cast_column(col, Sequence(HFImage(decode=False)))
        else:
            dataset = dataset.cast_column(col, HFImage(decode=False))
        logger.info("Cast image column to decode=False (raw bytes mode)")
    except Exception:
        logger.warning(
            "cast_column(decode=False) failed, falling back to PIL path",
            exc_info=True,
        )

    # ------------------------------------------------------------------
    # Extract dimensions via Dataset.map
    # ------------------------------------------------------------------
    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names,
        "num_proc": num_workers if num_workers > 1 else None,
        "desc": "Scanning image dimensions",
    }

    if is_multi:

        def _extract_dims_multi(batch, indices):
            sample_indices, widths, heights = [], [], []
            group_ids, image_indices = [], []
            for idx, img_list in zip(indices, batch[col]):
                for img_idx, img_data in enumerate(img_list):
                    w, h = _get_image_dimensions(img_data)
                    sample_indices.append(idx)
                    widths.append(w)
                    heights.append(h)
                    group_ids.append(idx)
                    image_indices.append(img_idx)
            return {
                "sample_index": sample_indices,
                "width": widths,
                "height": heights,
                "group_id": group_ids,
                "image_index": image_indices,
            }

        mapped = dataset.map(_extract_dims_multi, with_indices=True, **map_kwargs)
    else:

        def _extract_dims(batch, indices):
            widths, heights = [], []
            for img_data in batch[col]:
                w, h = _get_image_dimensions(img_data)
                widths.append(w)
                heights.append(h)
            return {
                "sample_index": list(indices),
                "width": widths,
                "height": heights,
            }

        mapped = dataset.map(_extract_dims, with_indices=True, **map_kwargs)

    # ------------------------------------------------------------------
    # Pass Arrow table directly to avoid materializing Python dicts
    # (85M+ rows would exhaust RAM as a list of dicts)
    # ------------------------------------------------------------------
    table = mapped.data.table
    logger.info(
        f"HF scan complete: {len(table):,} {'rows' if is_multi else 'samples'}"
    )

    if is_multi:
        from vision_tokenization.indexing.manifest import HF_SCHEMA_MULTI_IMAGE

        return save_hf_manifest(table, output_manifest, schema=HF_SCHEMA_MULTI_IMAGE)
    return save_hf_manifest(table, output_manifest)
