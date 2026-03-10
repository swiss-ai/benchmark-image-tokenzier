"""Single-tar scan function — must be a top-level function for ProcessPoolExecutor."""

import os
import tarfile
from io import BytesIO
from typing import Dict, FrozenSet, List, Optional

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
) -> List[Dict]:
    """Scan a single tar file and extract image metadata without full decode.

    For each image member, records the byte offset, file size, and image
    dimensions (via header-only ``imagesize`` read).  When *text_extensions*
    is provided, also records byte offset / size of the first matching text
    sidecar that shares the same sample key (WDS convention: same stem,
    different extension).

    **Multi-image mode** (``image_field_pattern`` is set):

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
            ``"img"``).  When ``None``, single-image mode (backward compat).

    Returns:
        List of dicts.  Always contains: ``sample_key``, ``tar_path``,
        ``offset_data``, ``file_size``, ``width``, ``height``, ``image_ext``.
        When *text_extensions* is not ``None``, additionally contains:
        ``offset_text`` (int or -1), ``text_file_size`` (int or 0),
        ``text_ext`` (str or ``""``).
        When *image_field_pattern* is set, additionally contains:
        ``group_id`` (int, local) and ``image_index`` (int).
    """
    if image_field_pattern is not None:
        return _scan_single_tar_multi_image(
            tar_path, image_extensions, text_extensions, image_field_pattern
        )

    # --- Single-image mode (original) ---
    image_entries: Dict[str, Dict] = {}  # sample_key -> image record
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
                # Read header bytes for dimension detection
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                header = fobj.read(_HEADER_BYTES)

                width, height = imagesize.get(BytesIO(header))
                if width < 0 or height < 0:
                    continue

                image_entries[stem] = {
                    "sample_key": stem,
                    "tar_path": tar_path,
                    "offset_data": member.offset_data,
                    "file_size": member.size,
                    "width": width,
                    "height": height,
                    "image_ext": ext,
                }

            elif text_extensions is not None and ext in text_extensions:
                # Keep first text sidecar per sample_key (prefer earlier in tar)
                if stem not in text_entries:
                    text_entries[stem] = (member.offset_data, member.size, ext)

    # Build results — one per image, optionally joined with text sidecar
    results: List[Dict] = []
    for sample_key, img_info in image_entries.items():
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
    import re

    # Regex for field name: pattern followed by digits
    field_re = re.compile(rf"^{re.escape(image_field_pattern)}(\d+)$")

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
                # Try to split stem into sample_key.field
                if "." in stem:
                    sample_key, field = stem.rsplit(".", 1)
                    m = field_re.match(field)
                    if m:
                        # Multi-image: e.g. 000042.img0
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        header = fobj.read(_HEADER_BYTES)
                        width, height = imagesize.get(BytesIO(header))
                        if width < 0 or height < 0:
                            continue

                        groups.setdefault(sample_key, {})[field] = (
                            member.offset_data, member.size, width, height, ext,
                        )

            elif text_extensions is not None and ext in text_extensions:
                stem = parts[0]
                # Text sidecar: sample_key.json (no field part)
                if stem not in text_entries:
                    text_entries[stem] = (member.offset_data, member.size, ext)

    # Build results: one row per image, sorted by sample_key then field name
    results: List[Dict] = []
    # Assign local group_id per unique sample_key (sorted for determinism)
    for local_gid, sample_key in enumerate(sorted(groups.keys())):
        fields = groups[sample_key]
        for img_idx, field_name in enumerate(
            sorted(fields.keys(), key=lambda f: int(field_re.match(f).group(1)))
        ):
            offset, size, w, h, ext = fields[field_name]
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
