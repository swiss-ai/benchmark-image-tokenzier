"""
Flask webapp for exploring VLM benchmark results.

This webapp provides an interface to:
1. Browse multiple experiments
2. View results with images, prompts, and model outputs
3. Filter results by tags
4. Separate views for VLM, Captioning, and Image Completion benchmarks
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

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


def get_experiments_by_type():
    """Separate experiments into VLM, captioning, and completion types."""
    all_experiments = get_experiments()
    vlm_experiments = []
    captioning_experiments = []
    completion_experiments = []

    for exp in all_experiments:
        results = load_experiment_results(exp["name"])
        if results:
            mode = results.get("mode")
            if mode == "image_completion":
                completion_experiments.append(exp)
            elif mode == "captioning":
                captioning_experiments.append(exp)
            else:
                vlm_experiments.append(exp)

    return vlm_experiments, captioning_experiments, completion_experiments


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


def _get_run_metrics(run):
    """
    Get metrics from a run, handling both old (perceptual_metrics) and new (metrics) format.

    Args:
        run: Run dictionary from results

    Returns:
        Dictionary of metrics, or None if no metrics present
    """
    # Try new format first, then fall back to old format
    metrics = run.get("metrics")
    if metrics:
        return metrics
    return run.get("perceptual_metrics")


def aggregate_metrics(runs):
    """
    Aggregate metrics across runs with min/max/avg/count statistics.

    Args:
        runs: List of run dictionaries

    Returns:
        Dictionary with structure:
        {
            "total_samples": int,
            "metrics": {
                "metric_name": {
                    "min": float,
                    "max": float,
                    "avg": float,
                    "count": int  # samples with this metric
                },
                ...
            }
        }
    """
    total_samples = len(runs)

    # Collect all numeric metric values
    metric_values = {}
    for run in runs:
        m = _get_run_metrics(run)
        if m:
            for key, value in m.items():
                if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)

    # Compute statistics for each metric
    metrics_summary = {}
    for key, values in metric_values.items():
        if values:
            metrics_summary[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values),
            }

    return {
        "total_samples": total_samples,
        "metrics": metrics_summary,
    }


def calculate_captioning_summary(results):
    """Calculate summary stats for captioning benchmark including aggregated metrics."""
    runs = results.get("runs", [])
    total_runs = len(runs)

    # Aggregate metrics across all runs
    metrics_agg = aggregate_metrics(runs)

    return {
        "total_runs": total_runs,
        "aggregated_metrics": metrics_agg["metrics"],
    }


def calculate_completion_summary(results):
    """Calculate summary stats for completion benchmark: total/valid/invalid runs, success rate by percentage."""
    runs = results.get("runs", [])
    total_runs = len(runs)
    valid_runs = sum(1 for r in runs if r.get("is_valid", False))
    invalid_runs = total_runs - valid_runs

    # Calculate success rate by percentage
    percentages = results.get("completion_percentages", [])
    stats_by_percentage = {}
    for pct in percentages:
        pct_runs = [r for r in runs if r.get("completion_percentage") == pct]
        pct_valid = sum(1 for r in pct_runs if r.get("is_valid", False))
        pct_total = len(pct_runs)
        stats_by_percentage[pct] = {
            "total": pct_total,
            "valid": pct_valid,
            "invalid": pct_total - pct_valid,
            "success_rate": (pct_valid / pct_total * 100) if pct_total > 0 else 0,
        }

    # Calculate row count difference metrics
    row_diffs = []
    for r in runs:
        generated = r.get("generated_rows", 0)
        expected = r.get("expected_rows", 0)
        row_diffs.append(abs(generated - expected))

    # Average including correct (diff=0)
    avg_row_diff_all = sum(row_diffs) / len(row_diffs) if row_diffs else 0

    # Average excluding correct (diff=0)
    non_zero_diffs = [d for d in row_diffs if d > 0]
    avg_row_diff_excluding_correct = sum(non_zero_diffs) / len(non_zero_diffs) if non_zero_diffs else 0
    correct_row_count = len(row_diffs) - len(non_zero_diffs)

    # Aggregate metrics using the generic function (includes min/max/avg/count)
    metrics_agg = aggregate_metrics(runs)
    aggregated_metrics = metrics_agg["metrics"]

    # Per-percentage metrics with full stats
    metrics_by_percentage = {}
    for pct in percentages:
        pct_runs = [r for r in runs if r.get("completion_percentage") == pct]
        pct_agg = aggregate_metrics(pct_runs)
        metrics_by_percentage[pct] = pct_agg["metrics"]

    return {
        "total_runs": total_runs,
        "valid_runs": valid_runs,
        "invalid_runs": invalid_runs,
        "success_rate": (valid_runs / total_runs * 100) if total_runs > 0 else 0,
        "stats_by_percentage": stats_by_percentage,
        "avg_row_diff_all": avg_row_diff_all,
        "avg_row_diff_excluding_correct": avg_row_diff_excluding_correct,
        "correct_row_count": correct_row_count,
        "incorrect_row_count": len(non_zero_diffs),
        "aggregated_metrics": aggregated_metrics,
        "metrics_by_percentage": metrics_by_percentage,
    }


def find_common_runs(experiment_results):
    """
    Find image-prompt combinations that exist in all experiments (for VLM benchmarks).

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


def find_common_completion_runs(experiment_results):
    """
    Find common (image_path, completion_percentage) across experiments for completion comparison.

    Returns a list of dicts with structure:
    {
        'image_path': 'assets/image.jpg',
        'image_tags': ['tag1', 'tag2'],
        'completion_percentage': 30,
        'runs': {
            'experiment1': {run_data},
            'experiment2': {run_data},
            ...
        }
    }
    """
    if not experiment_results:
        return []

    exp_names = list(experiment_results.keys())

    # Build mapping of (image_path, percentage) -> experiment runs
    combinations = {}

    for exp_name, results in experiment_results.items():
        for run in results.get("runs", []):
            key = (run["image"]["path"], run.get("completion_percentage", 0))

            if key not in combinations:
                combinations[key] = {
                    "image": run["image"],
                    "completion_percentage": run.get("completion_percentage", 0),
                    "runs": {},
                }

            combinations[key]["runs"][exp_name] = run

    # Filter to only include combinations present in ALL experiments
    common_runs = []
    for key, data in combinations.items():
        if len(data["runs"]) == len(exp_names):
            common_runs.append(
                {
                    "image_path": data["image"]["path"],
                    "image_tags": data["image"].get("tags", []),
                    "completion_percentage": data["completion_percentage"],
                    "runs": data["runs"],
                }
            )

    # Sort by image path then by percentage
    common_runs.sort(key=lambda x: (x["image_path"], x["completion_percentage"]))

    return common_runs


def find_common_captioning_runs(experiment_results):
    """
    Find common images across captioning experiments.

    Returns a list of dicts with structure:
    {
        'image_path': 'assets/image.jpg',
        'image_tags': ['tag1', 'tag2'],
        'init_phrase': 'This',
        'captions': {
            'experiment1': 'caption text',
            'experiment2': 'caption text',
            ...
        }
    }
    """
    if not experiment_results:
        return []

    exp_names = list(experiment_results.keys())

    # Build mapping of image_path -> experiment captions
    combinations = {}

    for exp_name, results in experiment_results.items():
        for run in results.get("runs", []):
            key = run["image"]["path"]

            if key not in combinations:
                combinations[key] = {
                    "image": run["image"],
                    "init_phrase": run.get("init_phrase", ""),
                    "captions": {},
                }

            combinations[key]["captions"][exp_name] = run.get("caption", "")

    # Filter to only include images present in ALL experiments
    common_runs = []
    for key, data in combinations.items():
        if len(data["captions"]) == len(exp_names):
            common_runs.append(
                {
                    "image_path": data["image"]["path"],
                    "image_tags": data["image"].get("tags", []),
                    "init_phrase": data["init_phrase"],
                    "captions": data["captions"],
                }
            )

    # Sort by image path
    common_runs.sort(key=lambda x: x["image_path"])

    return common_runs


# =============================================================================
# Landing Page
# =============================================================================


@app.route("/")
def index():
    """Landing page with links to VLM, Captioning, and Completion benchmarks."""
    vlm_experiments, captioning_experiments, completion_experiments = get_experiments_by_type()
    return render_template(
        "index.html",
        vlm_experiments=vlm_experiments,
        captioning_experiments=captioning_experiments,
        completion_experiments=completion_experiments,
    )


# =============================================================================
# VLM Benchmark Routes
# =============================================================================


@app.route("/vlm")
def vlm_index():
    """VLM experiments list page."""
    vlm_experiments, _, _ = get_experiments_by_type()
    return render_template("vlm/index.html", experiments=vlm_experiments)


@app.route("/vlm/<experiment_name>")
def vlm_experiment(experiment_name):
    """View results for a specific VLM experiment."""
    results = load_experiment_results(experiment_name)

    if results is None:
        return "Experiment not found", 404

    # Verify this is a VLM experiment (mode is None or not set)
    mode = results.get("mode")
    if mode == "image_completion":
        return redirect(url_for("completion_experiment", experiment_name=experiment_name))
    if mode == "captioning":
        return redirect(url_for("captioning_experiment", experiment_name=experiment_name))

    # Get filter parameters
    image_tag_filter = request.args.get("image_tag", None)
    prompt_tag_filter = request.args.get("prompt_tag", None)

    # Filter results if requested
    filtered_runs = results.get("runs", [])
    if image_tag_filter:
        filtered_runs = [r for r in filtered_runs if image_tag_filter in r["image"].get("tags", [])]
    if prompt_tag_filter:
        filtered_runs = [r for r in filtered_runs if prompt_tag_filter in r.get("prompt", {}).get("tags", [])]

    # Get all available tags for filtering UI
    all_tags = get_all_tags(results)

    return render_template(
        "vlm/experiment.html",
        experiment_name=experiment_name,
        results=results,
        filtered_runs=filtered_runs,
        all_tags=all_tags,
        current_image_tag=image_tag_filter,
        current_prompt_tag=prompt_tag_filter,
    )


@app.route("/vlm/compare")
def vlm_compare():
    """Compare multiple VLM experiments side-by-side."""
    vlm_experiments, _, _ = get_experiments_by_type()

    # Get selected experiments from query parameters
    selected_experiments = request.args.getlist("experiments")

    if not selected_experiments or len(selected_experiments) < 2:
        # Show experiment selection page
        return render_template(
            "vlm/compare.html",
            experiments=vlm_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
        )

    # Load results for all selected experiments
    experiment_results = {}
    for exp_name in selected_experiments:
        results = load_experiment_results(exp_name)
        if results and results.get("mode") is None:
            experiment_results[exp_name] = results

    if len(experiment_results) < 2:
        return render_template(
            "vlm/compare.html",
            experiments=vlm_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
            error="Could not load results for selected experiments",
        )

    # Find common image-prompt combinations
    comparison_data = find_common_runs(experiment_results)

    return render_template(
        "vlm/compare.html",
        experiments=vlm_experiments,
        selected_experiments=selected_experiments,
        experiment_results=experiment_results,
        comparison_data=comparison_data,
    )


# =============================================================================
# Captioning Benchmark Routes
# =============================================================================


@app.route("/captioning")
def captioning_index():
    """Captioning experiments list page."""
    _, captioning_experiments, _ = get_experiments_by_type()

    # Load summary stats for each experiment
    experiments_with_stats = []
    for exp in captioning_experiments:
        results = load_experiment_results(exp["name"])
        if results:
            summary = calculate_captioning_summary(results)
            experiments_with_stats.append({
                **exp,
                "summary": summary,
            })

    return render_template("captioning/index.html", experiments=experiments_with_stats)


@app.route("/captioning/<experiment_name>")
def captioning_experiment(experiment_name):
    """View results for a specific captioning experiment."""
    results = load_experiment_results(experiment_name)

    if results is None:
        return "Experiment not found", 404

    # Verify this is a captioning experiment
    if results.get("mode") != "captioning":
        mode = results.get("mode")
        if mode == "image_completion":
            return redirect(url_for("completion_experiment", experiment_name=experiment_name))
        return redirect(url_for("vlm_experiment", experiment_name=experiment_name))

    # Get filter parameters
    image_tag_filter = request.args.get("image_tag", None)

    # Filter results if requested
    filtered_runs = results.get("runs", [])
    if image_tag_filter:
        filtered_runs = [r for r in filtered_runs if image_tag_filter in r["image"].get("tags", [])]

    # Get all available tags for filtering UI
    all_tags = get_all_tags(results)

    # Calculate summary statistics including aggregated metrics
    summary = calculate_captioning_summary(results)

    return render_template(
        "captioning/experiment.html",
        experiment_name=experiment_name,
        results=results,
        filtered_runs=filtered_runs,
        all_tags=all_tags,
        current_image_tag=image_tag_filter,
        summary=summary,
    )


@app.route("/captioning/compare")
def captioning_compare():
    """Compare multiple captioning experiments side-by-side."""
    _, captioning_experiments, _ = get_experiments_by_type()

    # Get selected experiments from query parameters
    selected_experiments = request.args.getlist("experiments")

    if not selected_experiments or len(selected_experiments) < 2:
        # Show experiment selection page
        return render_template(
            "captioning/compare.html",
            experiments=captioning_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
        )

    # Load results for all selected experiments
    experiment_results = {}
    for exp_name in selected_experiments:
        results = load_experiment_results(exp_name)
        if results and results.get("mode") == "captioning":
            experiment_results[exp_name] = results

    if len(experiment_results) < 2:
        return render_template(
            "captioning/compare.html",
            experiments=captioning_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
            error="Could not load results for selected experiments",
        )

    # Find common images across experiments
    comparison_data = find_common_captioning_runs(experiment_results)

    return render_template(
        "captioning/compare.html",
        experiments=captioning_experiments,
        selected_experiments=selected_experiments,
        experiment_results=experiment_results,
        comparison_data=comparison_data,
    )


# =============================================================================
# Image Completion Benchmark Routes
# =============================================================================


@app.route("/completion")
def completion_index():
    """Completion experiments list page."""
    _, _, completion_experiments = get_experiments_by_type()

    # Load summary stats for each experiment
    experiments_with_stats = []
    for exp in completion_experiments:
        results = load_experiment_results(exp["name"])
        if results:
            summary = calculate_completion_summary(results)
            experiments_with_stats.append(
                {
                    **exp,
                    "summary": summary,
                    "completion_percentages": results.get("completion_percentages", []),
                }
            )

    return render_template("completion/index.html", experiments=experiments_with_stats)


@app.route("/completion/<experiment_name>")
def completion_experiment(experiment_name):
    """View results for a specific completion experiment."""
    results = load_experiment_results(experiment_name)

    if results is None:
        return "Experiment not found", 404

    # Verify this is a completion experiment
    if results.get("mode") != "image_completion":
        mode = results.get("mode")
        if mode == "captioning":
            return redirect(url_for("captioning_experiment", experiment_name=experiment_name))
        return redirect(url_for("vlm_experiment", experiment_name=experiment_name))

    # Get filter parameters
    image_tag_filter = request.args.get("image_tag", None)
    percentage_filter = request.args.get("percentage", None)
    validity_filter = request.args.get("validity", None)

    # Filter results if requested
    filtered_runs = results.get("runs", [])
    if image_tag_filter:
        filtered_runs = [r for r in filtered_runs if image_tag_filter in r["image"].get("tags", [])]
    if percentage_filter:
        try:
            percentage = int(percentage_filter)
            filtered_runs = [r for r in filtered_runs if r.get("completion_percentage") == percentage]
        except ValueError:
            pass
    if validity_filter:
        if validity_filter == "valid":
            filtered_runs = [r for r in filtered_runs if r.get("is_valid", False)]
        elif validity_filter == "invalid":
            filtered_runs = [r for r in filtered_runs if not r.get("is_valid", False)]

    # Get all available tags and percentages for filtering UI
    all_tags = get_all_tags(results)
    all_percentages = sorted(list(set(results.get("completion_percentages", []))))

    # Calculate summary statistics
    summary = calculate_completion_summary(results)

    return render_template(
        "completion/experiment.html",
        experiment_name=experiment_name,
        results=results,
        filtered_runs=filtered_runs,
        all_tags=all_tags,
        all_percentages=all_percentages,
        current_image_tag=image_tag_filter,
        current_percentage=percentage_filter,
        current_validity=validity_filter,
        summary=summary,
    )


@app.route("/completion/compare")
def completion_compare():
    """Compare multiple completion experiments side-by-side."""
    _, _, completion_experiments = get_experiments_by_type()

    # Get selected experiments from query parameters
    selected_experiments = request.args.getlist("experiments")

    if not selected_experiments or len(selected_experiments) < 2:
        # Show experiment selection page
        return render_template(
            "completion/compare.html",
            experiments=completion_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
        )

    # Load results for all selected experiments
    experiment_results = {}
    for exp_name in selected_experiments:
        results = load_experiment_results(exp_name)
        if results and results.get("mode") == "image_completion":
            experiment_results[exp_name] = results

    if len(experiment_results) < 2:
        return render_template(
            "completion/compare.html",
            experiments=completion_experiments,
            selected_experiments=selected_experiments,
            comparison_data=None,
            error="Could not load results for selected experiments",
        )

    # Find common (image, percentage) combinations
    comparison_data = find_common_completion_runs(experiment_results)

    # Calculate summaries for each experiment
    experiment_summaries = {}
    for exp_name, results in experiment_results.items():
        experiment_summaries[exp_name] = calculate_completion_summary(results)

    return render_template(
        "completion/compare.html",
        experiments=completion_experiments,
        selected_experiments=selected_experiments,
        experiment_results=experiment_results,
        experiment_summaries=experiment_summaries,
        comparison_data=comparison_data,
    )


# =============================================================================
# Legacy Routes (redirect based on type)
# =============================================================================


@app.route("/experiment/<experiment_name>")
def view_experiment(experiment_name):
    """Legacy route: redirect to appropriate section based on experiment type."""
    results = load_experiment_results(experiment_name)

    if results is None:
        return "Experiment not found", 404

    mode = results.get("mode")
    if mode == "image_completion":
        return redirect(url_for("completion_experiment", experiment_name=experiment_name))
    elif mode == "captioning":
        return redirect(url_for("captioning_experiment", experiment_name=experiment_name))
    else:
        return redirect(url_for("vlm_experiment", experiment_name=experiment_name))


@app.route("/compare")
def compare_experiments():
    """Legacy route: redirect to VLM compare page."""
    # Pass through query parameters
    return redirect(url_for("vlm_compare", **request.args))


# =============================================================================
# API Routes
# =============================================================================


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


# =============================================================================
# Static File Routes
# =============================================================================


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    """Serve image assets."""
    return send_from_directory(ASSETS_BASE_DIR, filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    """Serve result images (e.g., completion images from experiments)."""
    return send_from_directory(RESULTS_BASE_DIR, filename)


# =============================================================================
# Main
# =============================================================================


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
