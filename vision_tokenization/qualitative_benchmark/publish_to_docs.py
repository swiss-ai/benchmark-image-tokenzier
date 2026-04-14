"""Publish qualitative benchmark results into the repo's GitHub Pages docs/ tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Set


def _iter_asset_paths(value: Any) -> Iterable[str]:
    """Yield all relative asset paths referenced inside a result payload."""
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path.startswith("assets/"):
            yield path
        for nested in value.values():
            yield from _iter_asset_paths(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_asset_paths(item)


def _manifest_name(result_filename: str) -> str:
    return Path(result_filename).stem.replace("_", " ").replace("qa ", "QA - ").title()


def update_docs_manifest(docs_dir: str | Path) -> Path:
    """Regenerate docs/results/manifest.json from published result files."""
    docs_dir = Path(docs_dir)
    results_dir = docs_dir / "results"
    experiments = []
    for result_file in sorted(results_dir.glob("*.json")):
        if result_file.name == "manifest.json":
            continue
        experiments.append({"file": result_file.name, "name": _manifest_name(result_file.name)})

    manifest = {"experiments": experiments}
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def publish_result_to_docs(
    result_path: str | Path,
    docs_dir: str | Path,
    benchmark_dir: str | Path,
) -> dict:
    """Copy one result JSON plus its referenced assets into docs/ and refresh the manifest."""
    result_path = Path(result_path).resolve()
    docs_dir = Path(docs_dir).resolve()
    benchmark_dir = Path(benchmark_dir).resolve()

    result_payload = json.loads(result_path.read_text())
    docs_results_dir = docs_dir / "results"
    docs_assets_dir = docs_dir / "assets"
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    docs_assets_dir.mkdir(parents=True, exist_ok=True)

    published_result_path = docs_results_dir / result_path.name
    shutil.copy2(result_path, published_result_path)

    copied_assets: Set[str] = set()
    missing_assets = []

    for asset_rel_path in sorted(set(_iter_asset_paths(result_payload))):
        source_asset = benchmark_dir / asset_rel_path
        if not source_asset.exists():
            missing_assets.append(asset_rel_path)
            continue
        dest_asset = docs_dir / asset_rel_path
        dest_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, dest_asset)
        copied_assets.add(asset_rel_path)

    manifest_path = update_docs_manifest(docs_dir)
    return {
        "result": str(published_result_path),
        "assets_copied": sorted(copied_assets),
        "missing_assets": missing_assets,
        "manifest": str(manifest_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Publish a qualitative benchmark result into docs/ for GitHub Pages")
    parser.add_argument("--result", required=True, help="Path to the benchmark result JSON to publish")
    parser.add_argument("--docs-dir", required=True, help="Path to the repo docs/ directory")
    parser.add_argument(
        "--benchmark-dir",
        default=str(Path(__file__).resolve().parent),
        help="Path to the qualitative_benchmark directory that owns assets/",
    )
    args = parser.parse_args()

    summary = publish_result_to_docs(
        result_path=args.result,
        docs_dir=args.docs_dir,
        benchmark_dir=args.benchmark_dir,
    )
    print(f"Published result: {summary['result']}")
    print(f"Updated manifest: {summary['manifest']}")
    print(f"Copied {len(summary['assets_copied'])} asset(s)")
    for asset in summary["assets_copied"]:
        print(f"  - {asset}")
    if summary["missing_assets"]:
        print("Missing assets:")
        for asset in summary["missing_assets"]:
            print(f"  - {asset}")


if __name__ == "__main__":
    main()
