"""Scan metadata: record timing, worker count, and stats alongside manifests."""

from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict

from vision_tokenization.utils.json import json_dump


def write_scan_metadata(
    manifest_path: str | Path,
    *,
    num_workers: int,
    elapsed_seconds: float,
    total_rows: int,
    dataset_type: str,
    extra: Dict[str, Any] | None = None,
) -> str:
    """Write a JSON sidecar with scan metadata next to the manifest."""
    manifest_path = Path(manifest_path)
    meta_path = manifest_path.with_name(manifest_path.stem + "_meta.json")

    payload = {
        "manifest_path": str(manifest_path),
        "dataset_type": dataset_type,
        "num_workers": num_workers,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "total_rows": total_rows,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
    }
    if extra:
        payload.update(extra)

    json_dump(payload, meta_path)
    return str(meta_path)


class ScanTimer:
    """Context manager that records elapsed time."""

    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
