"""Shared tar-member indexing helpers for JSONL+tar scanners."""

from __future__ import annotations

import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, FrozenSet, Sequence

from vision_tokenization.indexing.scanners._workers.wds import (
    DEFAULT_IMAGE_EXTENSIONS,
    _get_image_dims,
)

logger = logging.getLogger(__name__)


def _index_one_tar(
    tar_path: str,
    image_exts: FrozenSet[str],
) -> tuple[str, Dict[str, dict]]:
    """Walk one tar archive, returning ``(tar_path, member_name -> metadata)``.

    Per-member work: extract first ~4 KB of payload to read image header for
    width/height. Sequential within a single tar (tarfile is a stream).
    """
    index: Dict[str, dict] = {}
    try:
        tf = tarfile.open(tar_path, "r")
    except Exception:
        logger.warning("Skipping unreadable tar: %s", tar_path, exc_info=True)
        return tar_path, index
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
    return tar_path, index


def build_tar_index(
    tar_paths: Sequence[str],
    image_extensions: Sequence[str] = tuple(DEFAULT_IMAGE_EXTENSIONS),
    workers: int = 64,
) -> Dict[str, dict]:
    """Build ``member_name -> tar metadata`` for random-access image loading.

    Parallelizes across tars (one process per tar, up to ``workers`` concurrent).
    Within each tar the walk is sequential because tarfile is a stream.

    When member names collide across tars, last-merged wins (per-tar order
    follows completion order, not input order — so for collision-prone trees
    callers should ensure globally-unique member names upstream).
    """
    image_exts = frozenset(ext.lower().lstrip(".") for ext in image_extensions)
    paths = list(tar_paths)
    if not paths:
        return {}

    # Cap workers at min(requested, len(paths)) — no benefit beyond #tars.
    n = min(workers, len(paths))
    logger.info("Indexing %d tar archives with %d parallel workers", len(paths), n)

    unified: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(_index_one_tar, p, image_exts): p for p in paths}
        for f in as_completed(futs):
            try:
                tar_path, idx = f.result()
            except Exception:
                logger.exception("worker crashed for %s", futs[f])
                continue
            unified.update(idx)
            logger.info("  %s — %d members", tar_path, len(idx))
    return unified
