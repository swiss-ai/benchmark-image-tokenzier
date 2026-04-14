#!/usr/bin/env python3
"""Convert Argimi-Ardian-Finance-10k to JSONL+tar interleave format.

Reads per-document tar.gz files, builds column-aware structured markdown
from segments.json, and writes WDS-compatible output:
  - content_image/ tars with page images
  - JSONL with structured markdown + <img> references

Usage::

    python -m vision_tokenization.preprocess.convert_argimi_finance \
        --input-dir /path/to/hf___artefactory___Argimi-Ardian-Finance-10k-text-image/data \
        --output-dir /path/to/output \
        --num-workers 288
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-aware segment sorting
# ---------------------------------------------------------------------------

def _segment_top(segment: dict) -> float:
    return float(segment["bbox"][1])


def _is_full_width_segment(segment: dict) -> bool:
    x0, _y0, x1, _y1 = segment["bbox"]
    width = x1 - x0
    return width > 0.5 or (x0 < 0.15 and x1 > 0.6)


def _column_aware_sort(segments: List[dict], col_threshold: float = 0.4) -> List[dict]:
    """Sort segments respecting multi-column layout.

    Pages are read in column order (left then right), while full-width blocks
    are injected at their vertical position instead of being hoisted above the
    earlier column text they follow.
    """
    full_width: List[dict] = []
    left_column: List[dict] = []
    right_column: List[dict] = []

    for segment in segments:
        if _is_full_width_segment(segment):
            full_width.append(segment)
            continue
        x0, _y0, x1, _y1 = segment["bbox"]
        x_mid = (x0 + x1) / 2
        if x_mid < col_threshold:
            left_column.append(segment)
        else:
            right_column.append(segment)

    full_width.sort(key=_segment_top)
    left_column.sort(key=_segment_top)
    right_column.sort(key=_segment_top)

    ordered: List[dict] = []
    left_idx = 0
    right_idx = 0
    for segment in full_width:
        full_y = _segment_top(segment)
        while left_idx < len(left_column) and _segment_top(left_column[left_idx]) < full_y:
            ordered.append(left_column[left_idx])
            left_idx += 1
        while right_idx < len(right_column) and _segment_top(right_column[right_idx]) < full_y:
            ordered.append(right_column[right_idx])
            right_idx += 1
        ordered.append(segment)

    ordered.extend(left_column[left_idx:])
    ordered.extend(right_column[right_idx:])
    return ordered


# ---------------------------------------------------------------------------
# Build structured markdown from segments
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "section_header": "## ",
    "text": "",
    "caption": "*",
    "footnote": "> ",
    "table": "",
    "picture": "",
}


def _build_page_markdown(segments: List[dict]) -> str:
    """Build structured markdown from a page's segments."""
    sorted_segs = _column_aware_sort(segments)
    parts = []
    for s in sorted_segs:
        label = s.get("label", "text")
        text = s.get("text", "").strip()
        if not text:
            continue
        prefix = _LABEL_MAP.get(label, "")
        if label == "caption":
            parts.append(f"*{text}*")
        else:
            parts.append(f"{prefix}{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Process one document tar.gz
# ---------------------------------------------------------------------------

def _process_document(
    tar_path: str,
    doc_id: str,
) -> Optional[Tuple[str, List[Tuple[str, bytes, int, int]], str]]:
    """Process one document tar.gz.

    Returns (doc_id, [(page_image_name, image_bytes, width, height), ...], markdown_text)
    or None on failure.
    """
    try:
        tf = tarfile.open(tar_path, "r:gz")
        members = tf.getmembers()
    except Exception:
        return None

    # Organize members by page number
    pages: Dict[int, Dict[str, tarfile.TarInfo]] = {}
    for m in members:
        name = m.name
        parts = name.rsplit("-", 1)
        if len(parts) != 2:
            continue
        page_part = parts[1]
        try:
            page_num = int(page_part.split(".")[0])
        except ValueError:
            continue
        ext = page_part.split(".", 1)[1]
        pages.setdefault(page_num, {})[ext] = m

    if not pages:
        tf.close()
        return None

    image_entries: List[Tuple[str, bytes]] = []
    md_parts = []

    for page_num in sorted(pages.keys()):
        page = pages[page_num]
        img_name = f"content_image/{doc_id}-page{page_num}.png"
        has_png = False

        # Extract page image + dimensions
        if "png" in page:
            f = tf.extractfile(page["png"])
            if f:
                has_png = True
                img_bytes = f.read()
                # Get dimensions from PNG header (first 24 bytes)
                w, h = 0, 0
                if len(img_bytes) >= 24 and img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                    import struct
                    w = struct.unpack('>I', img_bytes[16:20])[0]
                    h = struct.unpack('>I', img_bytes[20:24])[0]
                image_entries.append((img_name, img_bytes, w, h))

        # Build page markdown from segments
        page_text = ""
        if "segments.json" in page:
            f = tf.extractfile(page["segments.json"])
            if f:
                try:
                    segments = json.loads(f.read())
                    page_text = _build_page_markdown(segments)
                except Exception:
                    pass

        # Fall back to raw .txt if no segments
        if not page_text and "txt" in page:
            f = tf.extractfile(page["txt"])
            if f:
                page_text = f.read().decode("utf-8", errors="replace").strip()

        if not has_png:
            continue

        # Append page as image + text
        md_parts.append(f"<img src='{img_name}'>\n\n{page_text}")

    tf.close()

    if not image_entries:
        return None

    markdown = "\n\n".join(md_parts)
    return (doc_id, image_entries, markdown)


# ---------------------------------------------------------------------------
# Write output shards
# ---------------------------------------------------------------------------

def _write_shard(
    shard_id: int,
    results: List[Tuple[str, List[Tuple[str, bytes, int, int]], str]],
    output_dir: Path,
    group_id_start: int,
) -> Tuple[int, int, int, list]:
    """Write one output shard (tar + jsonl). Returns (docs, pages, chars, manifest_rows)."""
    tar_path = output_dir / f"part-{shard_id:06d}.tar"
    jsonl_path = output_dir / f"part-{shard_id:06d}.jsonl"

    total_pages = 0
    total_chars = 0
    manifest_rows = []
    group_id = group_id_start

    with tarfile.open(str(tar_path), "w") as tf, open(str(jsonl_path), "wb") as jf:
        for doc_id, image_entries, markdown in results:
            # Record JSONL line offset before writing
            jsonl_offset = jf.tell()
            doc_rows = []

            # Write images to tar, recording offsets
            for img_idx, (img_name, img_bytes, width, height) in enumerate(image_entries):
                info = tarfile.TarInfo(name=img_name)
                info.size = len(img_bytes)
                tar_offset = tf.offset
                offset_data = _tar_data_offset(tf, info, tar_offset)
                tf.addfile(info, io.BytesIO(img_bytes))

                row = {
                    "tar_path": str(tar_path),
                    "offset_data": offset_data,
                    "file_size": len(img_bytes),
                    "width": width,
                    "height": height,
                    "group_id": group_id,
                    "image_index": img_idx,
                    "jsonl_path": str(jsonl_path),
                    "line_start": jsonl_offset,
                    "image_ref": img_name,
                }
                doc_rows.append(row)
                manifest_rows.append(row)
                total_pages += 1

            # Write JSONL line
            record = {"id": doc_id, "md": markdown}
            line = json.dumps(record, ensure_ascii=False).encode("utf-8")
            line_length = len(line)
            jf.write(line + b"\n")
            total_chars += len(markdown)

            for row in doc_rows:
                row["line_length"] = line_length

            group_id += 1

    return len(results), total_pages, total_chars, manifest_rows


def _tar_data_offset(tf: tarfile.TarFile, info: tarfile.TarInfo, start_offset: int) -> int:
    """Return the payload offset for a tar member, including any extra header blocks."""
    header_size = len(info.tobuf(tf.format, tf.encoding, tf.errors))
    return start_offset + header_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(
    input_dir: str,
    output_dir: str,
    num_workers: int = 128,
    docs_per_shard: int = 100,
) -> dict:
    """Convert all documents to JSONL+tar format.

    Processes in shard-sized batches to bound memory: at most
    ``docs_per_shard`` document results live in memory at once.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all document tar.gz paths
    doc_paths = []
    for hash_dir in sorted(input_dir.iterdir()):
        if not hash_dir.is_dir():
            continue
        for f in sorted(hash_dir.iterdir()):
            if f.name.endswith(".tar.gz"):
                doc_id = f.stem.replace(".tar", "").replace("document-", "")
                doc_paths.append((str(f), doc_id))

    logger.info(f"Found {len(doc_paths):,} documents")

    total_docs = 0
    total_pages = 0
    total_chars = 0
    failed = 0
    shard_id = 0
    group_id_counter = 0
    all_manifest_rows = []

    # Single pool for the entire run — avoid per-batch pool creation overhead.
    # Bounded submission: process docs_per_shard at a time, write shard, free memory.
    logger.info(f"Processing with {num_workers} workers, {docs_per_shard} docs/shard")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for batch_start in range(0, len(doc_paths), docs_per_shard):
            batch_paths = doc_paths[batch_start : batch_start + docs_per_shard]

            futures = [
                pool.submit(_process_document, path, doc_id)
                for path, doc_id in batch_paths
            ]
            shard_results = []
            for future in futures:
                result = future.result()
                if result is not None:
                    shard_results.append(result)
                else:
                    failed += 1

            if shard_results:
                docs, pages, chars, manifest_rows = _write_shard(
                    shard_id, shard_results, output_dir, group_id_counter,
                )
                total_docs += docs
                total_pages += pages
                total_chars += chars
                all_manifest_rows.extend(manifest_rows)
                group_id_counter += docs
                shard_id += 1

            del shard_results, futures

            if shard_id % 50 == 0:
                logger.info(
                    f"  Shard {shard_id}: {total_docs:,} docs, "
                    f"{total_pages:,} pages, {failed:,} failed"
                )

    # Write manifest parquet
    if all_manifest_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq

        manifest_path = output_dir / "manifest.parquet"
        table = pa.table({
            "tar_path": pa.array([r["tar_path"] for r in all_manifest_rows], type=pa.dictionary(pa.int32(), pa.string())),
            "offset_data": pa.array([r["offset_data"] for r in all_manifest_rows], type=pa.int64()),
            "file_size": pa.array([r["file_size"] for r in all_manifest_rows], type=pa.int64()),
            "width": pa.array([r["width"] for r in all_manifest_rows], type=pa.int32()),
            "height": pa.array([r["height"] for r in all_manifest_rows], type=pa.int32()),
            "group_id": pa.array([r["group_id"] for r in all_manifest_rows], type=pa.int64()),
            "image_index": pa.array([r["image_index"] for r in all_manifest_rows], type=pa.int16()),
            "jsonl_path": pa.array([r["jsonl_path"] for r in all_manifest_rows], type=pa.dictionary(pa.int32(), pa.string())),
            "line_start": pa.array([r["line_start"] for r in all_manifest_rows], type=pa.int64()),
            "line_length": pa.array([r["line_length"] for r in all_manifest_rows], type=pa.int64()),
            "image_ref": pa.array([r["image_ref"] for r in all_manifest_rows], type=pa.dictionary(pa.int32(), pa.string())),
        })
        pq.write_table(table, str(manifest_path))
        logger.info(f"Manifest: {len(all_manifest_rows):,} rows -> {manifest_path}")
        del all_manifest_rows

    stats = {
        "total_documents": total_docs,
        "total_pages": total_pages,
        "total_chars": total_chars,
        "failed": failed,
        "shards": shard_id,
        "output_dir": str(output_dir),
    }

    logger.info(
        f"Done: {total_docs:,} documents, {total_pages:,} pages, "
        f"{total_chars:,} chars, {shard_id} shards"
    )

    with open(output_dir / "convert_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___artefactory___Argimi-Ardian-Finance-10k-text-image/data",
    )
    parser.add_argument(
        "--output-dir",
        default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/argimi_finance_interleave",
    )
    parser.add_argument("--num-workers", type=int, default=288)
    parser.add_argument("--docs-per-shard", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        docs_per_shard=args.docs_per_shard,
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
