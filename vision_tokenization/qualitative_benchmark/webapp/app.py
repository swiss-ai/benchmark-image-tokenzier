"""
Flask webapp for exploring VLM benchmark results.

This webapp provides an interface to:
1. Browse multiple experiments
2. View results with images, prompts, and model outputs
3. Filter results by tags
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# Configuration - these will be set from command-line arguments
RESULTS_BASE_DIR = None
ASSETS_BASE_DIR = None


def get_experiments():
    """Get list of all experiments (JSON files in results folder)."""
    if not RESULTS_BASE_DIR.exists():
        return []

    experiments = []
    for item in RESULTS_BASE_DIR.iterdir():
        # Look for JSON files directly in the results folder
        if item.is_file() and item.suffix == ".json":
            # Extract experiment name from filename (without .json extension)
            experiment_name = item.stem
            experiments.append({"name": experiment_name, "path": str(item), "file": item.name})

    return experiments


def load_experiment_results(experiment_name):
    """Load results for a specific experiment."""
    # Look for the JSON file named <experiment_name>.json in the results folder
    result_file = RESULTS_BASE_DIR / f"{experiment_name}.json"

    if not result_file.exists():
        return None

    with open(result_file, "r") as f:
        return json.load(f)


def get_all_tags(results):
    """Extract all unique tags from results."""
    image_tags = set()
    prompt_tags = set()

    for run in results.get("runs", []):
        image_tags.update(run["image"].get("tags", []))
        # For image completion mode, there are no prompts
        if "prompt" in run:
            prompt_tags.update(run["prompt"].get("tags", []))

    return {"image_tags": sorted(list(image_tags)), "prompt_tags": sorted(list(prompt_tags))}


@app.route("/")
def index():
    """Home page showing list of experiments."""
    experiments = get_experiments()
    return render_template("index.html", experiments=experiments)


@app.route("/experiment/<experiment_name>")
def view_experiment(experiment_name):
    """View results for a specific experiment."""
    results = load_experiment_results(experiment_name)

    if results is None:
        return "Experiment not found", 404

    # Detect if this is an image completion benchmark
    is_completion = results.get("mode") == "image_completion"

    # Get filter parameters
    image_tag_filter = request.args.get("image_tag", None)
    prompt_tag_filter = request.args.get("prompt_tag", None) if not is_completion else None
    percentage_filter = request.args.get("percentage", None) if is_completion else None

    # Filter results if requested
    filtered_runs = results.get("runs", [])
    if image_tag_filter:
        filtered_runs = [r for r in filtered_runs if image_tag_filter in r["image"].get("tags", [])]
    if prompt_tag_filter:
        filtered_runs = [r for r in filtered_runs if prompt_tag_filter in r.get("prompt", {}).get("tags", [])]
    if percentage_filter:
        try:
            percentage = int(percentage_filter)
            filtered_runs = [r for r in filtered_runs if r.get("completion_percentage") == percentage]
        except ValueError:
            pass

    # Get all available tags and percentages for filtering UI
    all_tags = get_all_tags(results)
    all_percentages = sorted(list(set(results.get("completion_percentages", [])))) if is_completion else []

    return render_template(
        "experiment.html",
        experiment_name=experiment_name,
        results=results,
        filtered_runs=filtered_runs,
        all_tags=all_tags,
        all_percentages=all_percentages,
        current_image_tag=image_tag_filter,
        current_prompt_tag=prompt_tag_filter,
        current_percentage=percentage_filter,
        is_completion=is_completion,
    )


@app.route("/compare")
def compare_experiments():
    """Compare multiple experiments side-by-side."""
    all_experiments = get_experiments()

    # Get selected experiments from query parameters
    selected_experiments = request.args.getlist("experiments")

    if not selected_experiments or len(selected_experiments) < 2:
        # Show experiment selection page
        return render_template(
            "compare.html", experiments=all_experiments, selected_experiments=selected_experiments, comparison_data=None
        )

    # Load results for all selected experiments
    experiment_results = {}
    for exp_name in selected_experiments:
        results = load_experiment_results(exp_name)
        if results:
            experiment_results[exp_name] = results

    if len(experiment_results) < 2:
        return render_template(
            "compare.html",
            experiments=all_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
            error="Could not load results for selected experiments",
        )

    # Find common image-prompt combinations
    comparison_data = find_common_runs(experiment_results)

    return render_template(
        "compare.html",
        experiments=all_experiments,
        selected_experiments=selected_experiments,
        experiment_results=experiment_results,
        comparison_data=comparison_data,
    )


def find_common_runs(experiment_results):
    """
    Find image-prompt combinations that exist in all experiments.

    Returns a list of dicts with structure:
    {
        'image_path': 'assets/image.jpg',
        'prompt_text': 'Describe this image',
        'outputs': {
            'experiment1': 'output text 1',
            'experiment2': 'output text 2',
            ...
        }
    }
    """
    if not experiment_results:
        return []

    # Get all experiment names
    exp_names = list(experiment_results.keys())

    # Build a mapping of (image_path, prompt_text) -> experiment outputs
    combinations = {}

    for exp_name, results in experiment_results.items():
        for run in results.get("runs", []):
            key = (run["image"]["path"], run["prompt"]["text"])

            if key not in combinations:
                combinations[key] = {"image": run["image"], "prompt": run["prompt"], "outputs": {}}

            combinations[key]["outputs"][exp_name] = run["output"]

    # Filter to only include combinations present in ALL experiments
    common_runs = []
    for key, data in combinations.items():
        if len(data["outputs"]) == len(exp_names):
            common_runs.append(
                {
                    "image_path": data["image"]["path"],
                    "image_tags": data["image"].get("tags", []),
                    "prompt_text": data["prompt"]["text"],
                    "prompt_tags": data["prompt"].get("tags", []),
                    "outputs": data["outputs"],
                }
            )

    return common_runs


@app.route("/api/experiments")
def api_experiments():
    """API endpoint to get list of experiments."""
    return jsonify(get_experiments())


@app.route("/api/experiment/<experiment_name>")
def api_experiment_results(experiment_name):
    """API endpoint to get results for a specific experiment."""
    results = load_experiment_results(experiment_name)
    if results is None:
        return jsonify({"error": "Experiment not found"}), 404
    return jsonify(results)


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    """Serve image assets."""
    return send_from_directory(ASSETS_BASE_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    """Serve result images (e.g., completion images from experiments)."""
    return send_from_directory(RESULTS_BASE_DIR, filename)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Flask webapp for exploring VLM benchmark results")
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results/",
        help="Path to results folder containing experiment subdirectories (default: results/)",
    )
    parser.add_argument(
        "--assets_folder",
        type=str,
        default=None,
        help="Path to assets folder containing images (default: auto-detected relative to results folder)",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run the webapp on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the webapp on (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set global configuration from arguments
    # Convert relative paths to absolute paths based on the qualitative_benchmark directory
    benchmark_dir = Path(__file__).parent.parent

    if Path(args.results_folder).is_absolute():
        RESULTS_BASE_DIR = Path(args.results_folder)
    else:
        RESULTS_BASE_DIR = benchmark_dir / args.results_folder

    # Auto-detect assets folder if not provided
    if args.assets_folder is None:
        ASSETS_BASE_DIR = benchmark_dir / "assets"
    elif Path(args.assets_folder).is_absolute():
        ASSETS_BASE_DIR = Path(args.assets_folder)
    else:
        ASSETS_BASE_DIR = benchmark_dir / args.assets_folder

    print("=" * 60)
    print("VLM Benchmark Results Viewer")
    print("=" * 60)
    print(f"Results directory: {RESULTS_BASE_DIR.absolute()}")
    print(f"Assets directory:  {ASSETS_BASE_DIR.absolute()}")
    print(f"Server URL:        http://{args.host}:{args.port}")
    print("=" * 60)

    # Create results directory if it doesn't exist
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    experiments = get_experiments()
    print(f"Found {len(experiments)} experiments")

    if experiments:
        print("\nAvailable experiments:")
        for exp in experiments:
            print(f"  - {exp['name']} ({exp['file']})")
    else:
        print("\nNo experiments found. Run vlm_benchmark.py to generate results.")

    print("\n" + "=" * 60)
    print("Starting server...")
    print("=" * 60 + "\n")

    app.run(debug=args.debug, host=args.host, port=args.port)
