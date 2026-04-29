"""Shared tar-member indexing helpers for JSONL+tar scanners."""

from __future__ import annotations

import logging
import os
import tarfile
from typing import Dict, Sequence

from vision_tokenization.indexing.scanners._workers.wds import (
    DEFAULT_IMAGE_EXTENSIONS,
    _get_image_dims,
)

logger = logging.getLogger(__name__)


def build_tar_index(
    tar_paths: Sequence[str],
    image_extensions: Sequence[str] = tuple(DEFAULT_IMAGE_EXTENSIONS),
) -> Dict[str, dict]:
    """Build ``member_name -> tar metadata`` for random-access image loading."""
    index: Dict[str, dict] = {}
    image_exts = frozenset(ext.lower().lstrip(".") for ext in image_extensions)

    for tar_path in tar_paths:
        try:
            tf = tarfile.open(tar_path, "r")
        except Exception:
            logger.warning("Skipping unreadable tar: %s", tar_path, exc_info=True)
            continue
        try:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name
                basename = os.path.basename(name)
                if "." not in basename:
                    continue
                _stem, ext = basename.rsplit(".", 1)
                ext = ext.lower()
                if ext not in image_exts:
                    continue
                if name in index:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                width, height = _get_image_dims(fobj, ext)
                if width < 0 or height < 0:
                    continue
                index[name] = {
                    "tar_path": tar_path,
                    "offset_data": int(member.offset_data),
                    "file_size": int(member.size),
                    "width": int(width),
                    "height": int(height),
                }
        except Exception:
            logger.warning("Error reading tar (truncated?): %s", tar_path, exc_info=True)
        finally:
            tf.close()
    return index
