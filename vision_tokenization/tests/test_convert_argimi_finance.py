from __future__ import annotations

import io
import tarfile

from PIL import Image

from vision_tokenization.indexing.reader import TarRandomAccessReader
from vision_tokenization.preprocess.convert_argimi_finance import (
    _build_page_markdown,
    _process_document,
    _write_shard,
)


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _write_tar_gz(tar_path, members: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_build_page_markdown_keeps_full_width_footer_after_column_text():
    segments = [
        {"bbox": [0.08, 0.10, 0.34, 0.18], "label": "text", "text": "Left column text"},
        {"bbox": [0.62, 0.14, 0.84, 0.22], "label": "text", "text": "Right column text"},
        {"bbox": [0.05, 0.92, 0.95, 0.98], "label": "footnote", "text": "Footer note"},
    ]

    markdown = _build_page_markdown(segments)

    assert markdown == "Left column text\n\nRight column text\n\n> Footer note"


def test_process_document_skips_pages_missing_png(tmp_path):
    tar_path = tmp_path / "document-doc.tar.gz"
    _write_tar_gz(
        tar_path,
        {
            "document-doc-0.png": _png_bytes(16, 16),
            "document-doc-0.txt": b"First page text",
            "document-doc-1.txt": b"Missing image page",
        },
    )

    result = _process_document(str(tar_path), "doc")

    assert result is not None
    doc_id, image_entries, markdown = result
    assert doc_id == "doc"
    assert len(image_entries) == 1
    assert image_entries[0][0] == "content_image/doc-page0.png"
    assert "<img src='content_image/doc-page0.png'>" in markdown
    assert "First page text" in markdown
    assert "doc-page1" not in markdown
    assert "Missing image page" not in markdown


def test_write_shard_uses_real_tar_payload_offset_for_long_names(tmp_path):
    img_name = "content_image/" + ("a" * 220) + ".png"
    img_bytes = _png_bytes(8, 8)

    docs, pages, _chars, manifest_rows = _write_shard(
        shard_id=0,
        results=[("doc", [(img_name, img_bytes, 8, 8)], "<img src='x'>")],
        output_dir=tmp_path,
        group_id_start=0,
    )

    assert docs == 1
    assert pages == 1
    assert len(manifest_rows) == 1

    row = manifest_rows[0]
    reader = TarRandomAccessReader(max_open_files=1)
    try:
        payload = reader.read_bytes(row["tar_path"], row["offset_data"], row["file_size"])
    finally:
        reader.close()

    assert payload == img_bytes
