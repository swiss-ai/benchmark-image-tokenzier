"""Single-tar scan function — must be a top-level function for ProcessPoolExecutor."""

import os
import re
import tarfile
from io import BytesIO
from typing import Dict, FrozenSet, List, Optional, Tuple

import imagesize

# Default image extensions to look for in tar files
DEFAULT_IMAGE_EXTENSIONS: FrozenSet[str] = frozenset(
    {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"}
)

# Default text sidecar extensions (json for structured data, txt for plain text)
DEFAULT_TEXT_EXTENSIONS: FrozenSet[str] = frozenset({"json", "txt"})

_HEADER_BYTES = 4096  # bytes to read for imagesize header detection


def scan_single_tar(
    tar_path: str,
    image_extensions: FrozenSet[str] = DEFAULT_IMAGE_EXTENSIONS,
    text_extensions: Optional[FrozenSet[str]] = None,
    image_field_pattern: Optional[str] = None,
    multi_image: bool = False,
) -> List[Dict]:
    """Scan a single tar file and extract image metadata without full decode.

    For each image member, records the byte offset, file size, and image
    dimensions (via header-only ``imagesize`` read).  When *text_extensions*
    is provided, also records byte offset / size of the first matching text
    sidecar that shares the same sample key (WDS convention: same stem,
    different extension).

    ``image_field_pattern`` only controls how image member names are parsed.
    Set ``multi_image=True`` to emit grouped rows with ``group_id`` and
    ``image_index``. With ``multi_image=False``, the scanner still strips
    the ``.{pattern}{N}`` suffix from image stems when present so single-image
    datasets named like ``sample.img1.jpg`` can match ``sample.json`` sidecars.

    **Grouped mode** (``image_field_pattern`` is set and ``multi_image=True``):

    Naming convention: ``{sample_key}.{prefix}{N}.{ext}``
    e.g. ``000042.img0.jpg``, ``000042.img1.jpg``, ``000042.json``.

    Produces one manifest row per image with ``group_id`` (local, per
    unique sample_key) and ``image_index`` (order within group).

    Args:
        tar_path: Path to the tar file.
        image_extensions: Set of lowercase extensions (without dot) to treat
            as images.
        text_extensions: Optional set of lowercase extensions to treat as text
            sidecars.  If ``None``, text sidecars are ignored and the returned
            dicts contain only image columns.
        image_field_pattern: Prefix for multi-image field names (e.g.
            ``"img"``). When set, image members named like
            ``{sample_key}.{pattern}{N}.{ext}`` are normalized to the base
            ``sample_key`` even if ``multi_image=False``.
        multi_image: Whether to emit grouped manifest rows with ``group_id``
            and ``image_index``.

    Returns:
        List of dicts.  Always contains: ``sample_key``, ``tar_path``,
        ``offset_data``, ``file_size``, ``width``, ``height``, ``image_ext``.
        When *text_extensions* is not ``None``, additionally contains:
        ``offset_text`` (int or -1), ``text_file_size`` (int or 0),
        ``text_ext`` (str or ``""``).
        When *multi_image* is true, additionally contains:
        ``group_id`` (int, local) and ``image_index`` (int).
    """
    if multi_image and image_field_pattern is None:
        raise ValueError(
            "WDS grouped scanning requires image_field_pattern to parse image members."
        )

    if multi_image:
        return _scan_single_tar_multi_image(
            tar_path, image_extensions, text_extensions, image_field_pattern
        )

    # --- Single-image mode ---
    field_re = _compile_image_field_re(image_field_pattern)
    image_entries: List[Dict] = []
    text_entries: Dict[str, tuple] = {}  # sample_key -> (offset_data, size, ext)

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue

            basename = os.path.basename(member.name)
            if "." not in basename:
                continue
            stem, ext = basename.rsplit(".", 1)
            ext = ext.lower()

            if ext in image_extensions:
                sample_key, _image_index = _parse_sample_key_and_index(stem, field_re)
                # Read header bytes for dimension detection
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                header = fobj.read(_HEADER_BYTES)

                width, height = imagesize.get(BytesIO(header))
                if width < 0 or height < 0:
                    continue

                image_entries.append({
                    "sample_key": sample_key,
                    "tar_path": tar_path,
                    "offset_data": member.offset_data,
                    "file_size": member.size,
                    "width": width,
                    "height": height,
                    "image_ext": ext,
                })

            elif text_extensions is not None and ext in text_extensions:
                sample_key, _image_index = _parse_sample_key_and_index(stem, field_re)
                # Keep first text sidecar per sample_key (prefer earlier in tar)
                if sample_key not in text_entries:
                    text_entries[sample_key] = (member.offset_data, member.size, ext)

    # Build results — one per image, optionally joined with text sidecar
    results: List[Dict] = []
    for img_info in image_entries:
        sample_key = img_info["sample_key"]
        if text_extensions is not None:
            text = text_entries.get(sample_key)
            if text is not None:
                img_info["offset_text"] = text[0]
                img_info["text_file_size"] = text[1]
                img_info["text_ext"] = text[2]
            else:
                img_info["offset_text"] = -1
                img_info["text_file_size"] = 0
                img_info["text_ext"] = ""
        results.append(img_info)

    return results


def _compile_image_field_re(image_field_pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if image_field_pattern is None:
        return None
    return re.compile(rf"^{re.escape(image_field_pattern)}(\d+)$")


def _parse_sample_key_and_index(
    stem: str,
    field_re: Optional[re.Pattern[str]],
) -> Tuple[str, Optional[int]]:
    """Parse ``sample_key`` and optional image index from an image stem."""
    if field_re is None or "." not in stem:
        return stem, None

    sample_key, field = stem.rsplit(".", 1)
    match = field_re.match(field)
    if match is None:
        return stem, None
    return sample_key, int(match.group(1))


def _scan_single_tar_multi_image(
    tar_path: str,
    image_extensions: FrozenSet[str],
    text_extensions: Optional[FrozenSet[str]],
    image_field_pattern: str,
) -> List[Dict]:
    """Multi-image scanning: one row per image, grouped by sample_key.

    Naming: ``{sample_key}.{pattern}{N}.{ext}`` for images,
    ``{sample_key}.{text_ext}`` for text sidecars.
    """
    field_re = _compile_image_field_re(image_field_pattern)
    assert field_re is not None

    # sample_key -> {field_name: (offset, size, w, h, ext)}
    groups: Dict[str, Dict[str, tuple]] = {}
    text_entries: Dict[str, tuple] = {}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue

            basename = os.path.basename(member.name)
            if "." not in basename:
                continue

            # Split from the right: we need at least stem.ext
            # For multi-image: stem = sample_key.field
            parts = basename.rsplit(".", 1)
            ext = parts[1].lower()

            if ext in image_extensions:
                stem = parts[0]
                sample_key, image_index = _parse_sample_key_and_index(stem, field_re)
                if image_index is None:
                    continue

                # Multi-image: e.g. 000042.img0
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                header = fobj.read(_HEADER_BYTES)
                width, height = imagesize.get(BytesIO(header))
                if width < 0 or height < 0:
                    continue

                groups.setdefault(sample_key, {})[image_index] = (
                    member.offset_data, member.size, width, height, ext,
                )

            elif text_extensions is not None and ext in text_extensions:
                stem = parts[0]
                sample_key, _image_index = _parse_sample_key_and_index(stem, field_re)
                if sample_key not in text_entries:
                    text_entries[sample_key] = (member.offset_data, member.size, ext)

    # Build results: one row per image, sorted by sample_key then field name
    results: List[Dict] = []
    # Assign local group_id per unique sample_key (sorted for determinism)
    for local_gid, sample_key in enumerate(sorted(groups.keys())):
        fields = groups[sample_key]
        for img_idx, image_index in enumerate(sorted(fields.keys())):
            offset, size, w, h, ext = fields[image_index]
            rec: Dict = {
                "sample_key": sample_key,
                "tar_path": tar_path,
                "offset_data": offset,
                "file_size": size,
                "width": w,
                "height": h,
                "image_ext": ext,
                "group_id": local_gid,
                "image_index": img_idx,
            }
            if text_extensions is not None:
                text = text_entries.get(sample_key)
                if text is not None:
                    rec["offset_text"] = text[0]
                    rec["text_file_size"] = text[1]
                    rec["text_ext"] = text[2]
                else:
                    rec["offset_text"] = -1
                    rec["text_file_size"] = 0
                    rec["text_ext"] = ""
            results.append(rec)

    return results
