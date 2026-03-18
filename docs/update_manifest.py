#!/usr/bin/env python3
"""Generate manifest.json from all JSON files in results/."""
import json
from pathlib import Path

results_dir = Path(__file__).parent / "results"
experiments = []

for f in sorted(results_dir.glob("*.json")):
    if f.name == "manifest.json":
        continue
    # Create a readable name from filename
    name = f.stem.replace("_", " ").replace("qa ", "QA - ").title()
    experiments.append({"file": f.name, "name": name})

manifest = {"experiments": experiments}
manifest_path = results_dir / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Written {len(experiments)} experiments to {manifest_path}")
for exp in experiments:
    print(f"  - {exp['name']} ({exp['file']})")
