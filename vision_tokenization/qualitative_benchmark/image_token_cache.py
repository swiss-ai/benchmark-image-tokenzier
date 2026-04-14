"""On-disk cache for image tokenization results used by qualitative benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


CACHE_VERSION = 1


def _normalize_path(value: Any) -> Any:
    """Resolve filesystem paths while leaving other values untouched."""
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _normalize_tokenizer_kwargs(tokenizer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only cache-relevant tokenizer arguments in a stable format."""
    normalized: Dict[str, Any] = {}
    for key in ("min_pixels", "max_pixels", "model_path", "tokenizer_path"):
        value = tokenizer_kwargs.get(key)
        if value is None:
            continue
        normalized[key] = _normalize_path(value) if key.endswith("_path") else value
    return normalized


class ImageTokenCache:
    """Store encoded image tokens on disk and invalidate them on source/config changes."""

    def __init__(self, cache_dir: str | Path, tokenizer_type: str, tokenizer_kwargs: Dict[str, Any]):
        self.cache_dir = Path(cache_dir)
        self.tokenizer_type = tokenizer_type.lower()
        self.tokenizer_kwargs = _normalize_tokenizer_kwargs(tokenizer_kwargs)
        signature_payload = {
            "cache_version": CACHE_VERSION,
            "tokenizer_type": self.tokenizer_type,
            "tokenizer_kwargs": self.tokenizer_kwargs,
        }
        signature_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
        self.tokenizer_signature = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()[:16]

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        return Path(image_path).expanduser().resolve()

    def _source_signature(self, image_path: str | Path) -> Dict[str, Any]:
        resolved_path = self._resolve_image_path(image_path)
        stat = resolved_path.stat()
        return {
            "path": str(resolved_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _cache_path(self, image_path: str | Path) -> Path:
        resolved_path = self._resolve_image_path(image_path)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_path.stem)[:48] or "image"
        image_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / self.tokenizer_signature / f"{safe_stem}-{image_hash}.pt"

    def inspect_entry(self, image_path: str | Path) -> Dict[str, Any]:
        """Return cache status for a single image without encoding it."""
        try:
            source_signature = self._source_signature(image_path)
        except FileNotFoundError:
            return {"hit": False, "reason": "missing_source", "path": str(image_path)}

        cache_path = self._cache_path(image_path)
        if not cache_path.exists():
            return {"hit": False, "reason": "missing_cache", "path": str(image_path), "cache_path": str(cache_path)}

        try:
            record = torch.load(cache_path, map_location="cpu")
        except Exception:
            return {"hit": False, "reason": "unreadable_cache", "path": str(image_path), "cache_path": str(cache_path)}

        if record.get("cache_version") != CACHE_VERSION:
            return {"hit": False, "reason": "cache_version_mismatch", "path": str(image_path), "cache_path": str(cache_path)}
        if record.get("tokenizer_signature") != self.tokenizer_signature:
            return {
                "hit": False,
                "reason": "tokenizer_signature_mismatch",
                "path": str(image_path),
                "cache_path": str(cache_path),
            }
        if record.get("source") != source_signature:
            return {"hit": False, "reason": "stale_source", "path": str(image_path), "cache_path": str(cache_path)}
        if "indices" not in record or "metadata" not in record:
            return {"hit": False, "reason": "invalid_record", "path": str(image_path), "cache_path": str(cache_path)}

        return {
            "hit": True,
            "path": str(image_path),
            "cache_path": str(cache_path),
            "record": record,
        }

    def inspect_many(self, image_paths: Iterable[str | Path]) -> Dict[str, Any]:
        """Summarize cache coverage for a list of images."""
        summary = {
            "hits": 0,
            "misses": 0,
            "reasons": {},
        }
        for image_path in image_paths:
            result = self.inspect_entry(image_path)
            if result["hit"]:
                summary["hits"] += 1
                continue
            summary["misses"] += 1
            reason = result["reason"]
            summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1
        return summary

    def load(self, image_path: str | Path) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Return cached indices and metadata if a valid entry exists."""
        result = self.inspect_entry(image_path)
        if not result["hit"]:
            return None
        record = result["record"]
        return record["indices"], record["metadata"]

    def save(
        self,
        image_path: str | Path,
        tokenizer_name: str,
        indices: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> Path:
        """Persist encoded image tokens for later reuse."""
        cache_path = self._cache_path(image_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        record = {
            "cache_version": CACHE_VERSION,
            "tokenizer_signature": self.tokenizer_signature,
            "tokenizer_name": tokenizer_name,
            "source": self._source_signature(image_path),
            "metadata": metadata,
            "indices": indices.detach().cpu(),
        }
        torch.save(record, temp_path)
        temp_path.replace(cache_path)
        return cache_path
