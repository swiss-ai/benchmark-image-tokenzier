"""Parquet media scanner with raw example artifacts.

The scanner emits media candidates plus row-to-media references. Task-specific
text/view conversion belongs to a later view builder.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vision_tokenization.indexing.scanners._parallel import run_ordered_pool
from vision_tokenization.indexing.scanners._workers.arrow_media import (
    ImageColumnView,
    image_bytes,
    image_dimensions,
    image_media_id,
    image_path,
    image_raw_length,
)

MEDIA_OCC_SCHEMA = pa.schema([
    pa.field("media_id", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("raw_length_bytes", pa.int64()),
    pa.field("raw_ext", pa.string()),
    pa.field("source", pa.string()),
    pa.field("source_path", pa.string()),
    pa.field("row_group", pa.int32()),
    pa.field("row_index", pa.int64()),
    pa.field("image_index", pa.int32()),
])

MEDIA_SCAN_SCHEMA = pa.schema([
    pa.field("media_id", pa.string()),
    pa.field("width", pa.int32()),
    pa.field("height", pa.int32()),
    pa.field("raw_length_bytes", pa.int64()),
    pa.field("source", pa.string()),
])

ROW_MEDIA_REFS_SCHEMA = pa.schema([
    pa.field("source_path", pa.string()),
    pa.field("row_group", pa.int32()),
    pa.field("row_index", pa.int64()),
    pa.field("source", pa.string()),
    pa.field("media_refs", pa.list_(pa.string())),
])


@dataclass
class ParquetMediaScanResult:
    n_source_rows: int = 0
    n_row_refs: int = 0
    n_media_candidates: int = 0


@dataclass
class MediaDedupResult:
    n_unique_media: int = 0
    n_invalid_media: int = 0
    n_valid_rows: int = 0
    n_filtered_rows: int = 0
    media_unique_path: str = ""
    scan_path: str = ""
    row_refs_path: str = ""


@dataclass
class LazyRawMedia:
    media_id: str
    raw_length_bytes: int
    raw_ext: str
    width: int
    height: int
    source: str
    source_path: str
    row_group: int
    row_index: int
    image_index: int

    @property
    def raw(self) -> bytes:
        return _read_source_image_bytes(
            self.source_path,
            self.row_group,
            self.row_index,
            self.image_index,
        )


_SOURCE_IMAGE_CACHE: OrderedDict[tuple[str, int], ImageColumnView] = OrderedDict()
_SOURCE_IMAGE_CACHE_MAX_ROW_GROUPS = 4


def _source_image_view(source_path: str, row_group: int) -> ImageColumnView:
    key = (source_path, int(row_group))
    cached = _SOURCE_IMAGE_CACHE.get(key)
    if cached is not None:
        _SOURCE_IMAGE_CACHE.move_to_end(key)
        return cached

    parquet_file = pq.ParquetFile(source_path)
    batch = next(parquet_file.iter_batches(
        row_groups=[int(row_group)],
        columns=["image"],
        batch_size=max(1, parquet_file.metadata.row_group(int(row_group)).num_rows),
    ))
    view = ImageColumnView(batch.column(0))
    _SOURCE_IMAGE_CACHE[key] = view
    if len(_SOURCE_IMAGE_CACHE) > _SOURCE_IMAGE_CACHE_MAX_ROW_GROUPS:
        _SOURCE_IMAGE_CACHE.popitem(last=False)
    return view


def _read_source_image_bytes(
    source_path: str,
    row_group: int,
    row_index: int,
    image_index: int,
) -> bytes:
    images = _source_image_view(source_path, row_group).row(int(row_index))
    if isinstance(images, list):
        img = images[int(image_index)]
    else:
        img = images
    return image_bytes(img)


class _PylistPartWriter:
    def __init__(self, root: Path, schema: pa.Schema | None = None, rows_per_file: int = 4096):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.rows_per_file = rows_per_file
        self._buf: list[dict] = []
        self._seq = 0

    def add(self, row: dict) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.rows_per_file:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        path = self.root / f"part-{self._seq:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(self._buf, schema=self.schema), path)
        self._seq += 1
        self._buf = []


class _OccurrenceWriter(_PylistPartWriter):
    def __init__(self, root: Path, *, worker_id: int, rows_per_file: int = 4096):
        super().__init__(
            Path(root) / f"worker-{worker_id:05d}",
            schema=MEDIA_OCC_SCHEMA,
            rows_per_file=rows_per_file,
        )


def _partition_row_groups(n_row_groups: int, workers: int) -> list[list[int]]:
    workers = max(1, min(workers, n_row_groups))
    out = [[] for _ in range(workers)]
    for row_group in range(n_row_groups):
        out[row_group % workers].append(row_group)
    return out


def _iter_rows(path: Path, row_groups: list[int], columns: list[str], batch_size: int):
    parquet_file = pq.ParquetFile(path)
    for row_group in row_groups:
        row_index = 0
        for batch in parquet_file.iter_batches(
            row_groups=[row_group],
            columns=columns,
            batch_size=batch_size,
        ):
            col_by_name = {
                name: batch.column(i)
                for i, name in enumerate(batch.schema.names)
            }
            images = ImageColumnView(col_by_name["image"])
            payload_cols = {
                name: col
                for name, col in col_by_name.items()
                if name != "image"
            }
            for local_idx in range(batch.num_rows):
                row = {
                    name: col[local_idx].as_py()
                    for name, col in payload_cols.items()
                }
                row["image"] = images.row(local_idx)
                yield row_group, row_index, row
                row_index += 1


def _images_of(row: dict):
    img = row["image"]
    return img if isinstance(img, list) else [img]


def _worker_scan(args: tuple) -> dict:
    (
        path_str,
        build_dir_str,
        worker_id,
        row_groups,
        columns,
        batch_size,
    ) = args
    path = Path(path_str)
    build_dir = Path(build_dir_str)
    row_refs = _PylistPartWriter(
        build_dir / "row_media_refs" / f"worker-{worker_id:05d}",
        schema=ROW_MEDIA_REFS_SCHEMA,
    )
    occ = _OccurrenceWriter(build_dir / "media_occ", worker_id=worker_id)
    emitted: set[str] = set()
    n_source_rows = 0
    n_row_refs = 0
    n_media_candidates = 0

    for row_group, row_index, row in _iter_rows(path, row_groups, columns, batch_size):
        n_source_rows += 1
        refs = []
        pending: dict[str, tuple[int, int, int, str, int]] = {}
        for image_index, img in enumerate(_images_of(row)):
            media_id = image_media_id(img)
            refs.append(media_id)
            if media_id in emitted or media_id in pending:
                continue
            raw_length = image_raw_length(img)
            width, height = image_dimensions(img)
            raw_ext = (image_path(img) or "bin").rsplit(".", 1)[-1]
            pending[media_id] = (raw_length, width, height, raw_ext, image_index)

        row_refs.add({
            "source_path": str(path),
            "row_group": row_group,
            "row_index": row_index,
            "source": str(row.get("source-id", "")),
            "media_refs": refs,
        })
        n_row_refs += 1

        for media_id, (raw_length, width, height, raw_ext, image_index) in pending.items():
            occ.add({
                "media_id": media_id,
                "width": width,
                "height": height,
                "raw_length_bytes": raw_length,
                "raw_ext": raw_ext,
                "source": str(row.get("source-id", "")),
                "source_path": str(path),
                "row_group": row_group,
                "row_index": row_index,
                "image_index": image_index,
            })
            emitted.add(media_id)
            n_media_candidates += 1

    row_refs.flush()
    occ.flush()
    return {
        "n_source_rows": n_source_rows,
        "n_row_refs": n_row_refs,
        "n_media_candidates": n_media_candidates,
    }


def scan_parquet_media_refs_many(
    paths: list[Path],
    build_dir: Path,
    *,
    workers: int,
    batch_size: int = 1024,
    source_column: str = "source-id",
) -> ParquetMediaScanResult:
    """Row-group-parallel Parquet scanner for media facts and row refs.

    Occurrences are written in one flat layout: ``media_occ/worker-NNNNN/``.
    ``dedup_media_scan`` consumes exactly that layout.
    """
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("scan_parquet_media_refs_many requires at least one parquet path")
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        columns = ["image"]
        if source_column in parquet_file.schema_arrow.names:
            columns.append(source_column)
        row_group_parts = _partition_row_groups(parquet_file.num_row_groups, workers)
        for row_groups in row_group_parts:
            if not row_groups:
                continue
            jobs.append((
                str(path),
                str(build_dir),
                len(jobs),
                row_groups,
                columns,
                batch_size,
            ))

    results: list[dict | None] = [None] * len(jobs)
    pool_workers = max(1, min(int(workers), len(jobs)))
    if len(jobs) == 1:
        results[0] = _worker_scan(jobs[0])
    else:
        def submit(pool, idx):
            return pool.submit(_worker_scan, jobs[idx])

        def emit(idx, result):
            results[idx] = result

        run_ordered_pool(len(jobs), submit, emit, pool_workers)

    out = ParquetMediaScanResult()
    for result in results:
        if result is None:
            continue
        out.n_source_rows += result["n_source_rows"]
        out.n_row_refs += result["n_row_refs"]
        out.n_media_candidates += result["n_media_candidates"]
    return out


def scan_parquet_media_refs(
    path: Path,
    build_dir: Path,
    *,
    workers: int,
    batch_size: int = 1024,
    source_column: str = "source-id",
) -> ParquetMediaScanResult:
    return scan_parquet_media_refs_many(
        [Path(path)],
        build_dir,
        workers=workers,
        batch_size=batch_size,
        source_column=source_column,
    )


def _dedup_occurrences_in_memory(files: list[Path]) -> pa.Table:
    if not files:
        return pa.Table.from_pylist([], schema=MEDIA_OCC_SCHEMA)

    table = pa.concat_tables([pq.read_table(path) for path in files])
    if len(table) == 0:
        return table
    table = table.sort_by([
        ("media_id", "ascending"),
        ("source_path", "ascending"),
        ("row_group", "ascending"),
        ("row_index", "ascending"),
        ("image_index", "ascending"),
    ])
    media_ids = table.column("media_id").to_pylist()
    keep = []
    last = None
    for media_id in media_ids:
        is_first = media_id != last
        keep.append(is_first)
        last = media_id
    return table.filter(pa.array(keep))


def _write_row_refs(build_dir: Path, publish_dir: Path, valid_media: set[str]) -> tuple[int, int, str]:
    rows = []
    filtered = 0
    for path in sorted((build_dir / "row_media_refs").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            refs = row["media_refs"]
            if any(ref not in valid_media for ref in refs):
                filtered += 1
                continue
            rows.append(row)
    rows.sort(key=lambda row: (row["source_path"], row["row_group"], row["row_index"]))
    out = publish_dir / "row_media_refs.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=ROW_MEDIA_REFS_SCHEMA), out)
    return len(rows), filtered, str(out)


def dedup_media_scan(
    build_dir: Path,
    publish_dir: Path,
    *,
    min_side: int = 16,
    max_in_memory_bytes: int = 4 << 30,
) -> MediaDedupResult:
    """Exact global dedup and filtered row-ref materialization.

    Dedup is in-memory over the flat ``media_occ/worker-*/`` layout the scanner
    writes. ``max_in_memory_bytes`` is a hard ceiling, not a path selector: an
    occurrence set larger than it fails loud rather than silently degrading.
    Streaming out-of-core dedup is intentionally not implemented until a real
    dataset reaches that scale (occurrences are small metadata rows — 4 GiB is
    ~20M images).
    """
    build_dir = Path(build_dir)
    publish_dir = Path(publish_dir)
    publish_dir.mkdir(parents=True, exist_ok=True)

    occ_files = sorted((build_dir / "media_occ").rglob("*.parquet"))
    occ_bytes = sum(path.stat().st_size for path in occ_files)
    if occ_bytes > int(max_in_memory_bytes):
        raise RuntimeError(
            f"media occurrence set is {occ_bytes / (1 << 30):.2f} GiB across "
            f"{len(occ_files)} files, over the {max_in_memory_bytes / (1 << 30):.2f} GiB "
            f"in-memory dedup ceiling; streaming out-of-core dedup is not "
            f"implemented (no current dataset reaches this scale)"
        )
    media_table = _dedup_occurrences_in_memory(occ_files)

    n_unique_before_filter = len(media_table)
    valid_mask = pc.and_(
        pc.greater_equal(media_table.column("width"), min_side),
        pc.greater_equal(media_table.column("height"), min_side),
    )
    media_table = media_table.filter(valid_mask)

    media_unique = publish_dir / "media_unique.parquet"
    pq.write_table(media_table, media_unique)
    scan_path = publish_dir / "scan.parquet"
    scan_table = media_table.select([
        "media_id",
        "width",
        "height",
        "raw_length_bytes",
        "source",
    ]).cast(MEDIA_SCAN_SCHEMA)
    pq.write_table(scan_table, scan_path)

    valid_media = set(media_table.column("media_id").to_pylist())
    n_valid_rows, n_filtered, row_refs_path = _write_row_refs(build_dir, publish_dir, valid_media)
    return MediaDedupResult(
        n_unique_media=len(media_table),
        n_invalid_media=n_unique_before_filter - len(media_table),
        n_valid_rows=n_valid_rows,
        n_filtered_rows=n_filtered,
        media_unique_path=str(media_unique),
        scan_path=str(scan_path),
        row_refs_path=row_refs_path,
    )


def load_media_inventory(media_unique_path: Path) -> list[LazyRawMedia]:
    rows = pq.read_table(media_unique_path).to_pylist()
    return [
        LazyRawMedia(
            media_id=row["media_id"],
            raw_length_bytes=row["raw_length_bytes"],
            raw_ext=row["raw_ext"],
            width=row["width"],
            height=row["height"],
            source=row["source"],
            source_path=row["source_path"],
            row_group=row["row_group"],
            row_index=row["row_index"],
            image_index=row["image_index"],
        )
        for row in rows
    ]
