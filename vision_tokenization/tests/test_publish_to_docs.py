import json
from pathlib import Path

from vision_tokenization.qualitative_benchmark.publish_to_docs import publish_result_to_docs


def test_publish_result_to_docs_copies_assets_and_updates_manifest(tmp_path):
    benchmark_dir = tmp_path / "qualitative_benchmark"
    docs_dir = tmp_path / "docs"
    results_dir = benchmark_dir / "results"
    assets_dir = benchmark_dir / "assets"
    results_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    (assets_dir / "sample.png").write_bytes(b"fake-png")
    result_payload = {
        "mode": "captioning",
        "runs": [
            {
                "image": {"path": "assets/sample.png", "tags": ["general"]},
                "caption": "test caption",
            }
        ],
    }
    result_path = results_dir / "captioning_test.json"
    result_path.write_text(json.dumps(result_payload))

    summary = publish_result_to_docs(
        result_path=result_path,
        docs_dir=docs_dir,
        benchmark_dir=benchmark_dir,
    )

    assert (docs_dir / "results" / "captioning_test.json").exists()
    assert (docs_dir / "assets" / "sample.png").read_bytes() == b"fake-png"
    manifest = json.loads((docs_dir / "results" / "manifest.json").read_text())
    assert manifest["experiments"] == [{"file": "captioning_test.json", "name": "Captioning Test"}]
    assert summary["missing_assets"] == []
