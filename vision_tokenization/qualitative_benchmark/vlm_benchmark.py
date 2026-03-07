"""
Script to generate qualitative results for VLM inference based on single-turn image prompts.
Prompts and images are loaded from json files. Results are stored in an experiment folder together with prompts and images.

For now only supports EMU3 inferencer.

This module serves as the CLI entry point for running benchmarks.
Benchmark implementations are in the benchmarks/ module.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from vision_tokenization.qualitative_benchmark.inferencers import create_inferencer
from vision_tokenization.qualitative_benchmark.utils.prompt_formatter import CHAT_TRANFORMS, PROMPT_BUILDERS
from vision_tokenization.qualitative_benchmark.vlm import VLM, InferenceArgs

base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "Tokenizer"))

# Set environment variables to prevent OOM and process issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use only GPU 0 for the LLM

# Import benchmark registry
from vision_tokenization.qualitative_benchmark.benchmarks import get_benchmark

# Legacy class definitions kept for backward compatibility
# New code should use the registry-based benchmarks from benchmarks/ module


class VLMBenchmark:
    """Scaffolding for running qualitative VLM benchmarks with images and prompts."""

    def __init__(
        self,
        images_config_path: str,
        prompts_config_path: str,
        vlm: VLM,
        results_dir: str = "results",
        debug: bool = False,
    ):
        """
        Initialize the benchmark system.

        Args:
            images_config_path: Path to JSON file containing image configurations
            prompts_config_path: Path to JSON file containing prompt configurations
            results_dir: Directory where results will be stored
            debug: Enable debug mode for detailed token output
        """
        self.images = self._load_json(images_config_path)
        self.prompts = self._load_json(prompts_config_path)
        self.results_dir = Path(results_dir)
        self.vlm = vlm
        self.debug = debug
        print(f"Loaded {len(self.images)} images and {len(self.prompts)} prompts.")

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON configuration file."""
        with open(path, "r") as f:
            return json.load(f)

    def _tags_match(self, image_tags: List[str], prompt_tags: List[str], require_all: bool = False) -> bool:
        """
        Check if image and prompt tags match.

        Args:
            image_tags: Tags from the image
            prompt_tags: Tags from the prompt
            require_all: If True, all prompt tags must be in image tags.
                        If False, at least one overlap is sufficient.

        Returns:
            True if tags match according to the requirement
        """
        if require_all:
            return set(prompt_tags).issubset(set(image_tags))
        else:
            return bool(set(image_tags) & set(prompt_tags))

    def run_benchmark(self, output_filename: str = None) -> Dict[str, Any]:
        """
        Run the benchmark by matching images with compatible prompts.

        Args:
            output_filename: Optional custom filename for results

        Returns:
            Dictionary containing all benchmark results
        """
        # Serialize inference args to dict
        inference_args_dict = {
            "apply_chat_template": self.vlm.inf_args.apply_chat_template,
            "prompt_builder": self.vlm.inf_args.prompt_builder,
            "temperature": self.vlm.inf_args.temperature,
            "top_p": self.vlm.inf_args.top_p,
            "max_new_tokens": self.vlm.inf_args.max_new_tokens,
            "max_emu_aspect_ratio": self.vlm.inf_args.max_emu_aspect_ratio,
            "min_emu_aspect_ratio": self.vlm.inf_args.min_emu_aspect_ratio,
            "stop_token_ids": self.vlm.inf_args.stop_token_ids,
        }

        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.vlm.model_path,
            "tokenizer_path": self.vlm.tokenizer_path,
            "vision_tokenizer": self.vlm.vision_tokenizer.name,
            "inferencer": type(self.vlm.inferencer).__name__,
            "inference_args": inference_args_dict,
            "total_runs": 0,
            "runs": [],
        }

        # Track sample number for debug mode (only first 3 samples)
        sample_num = 0

        # Match images with prompts based on tags
        for image in tqdm(self.images, desc="Processing images"):
            for prompt in self.prompts:
                require_all = prompt.get("require_all_tags", False)
                if self._tags_match(image.get("tags", []), prompt.get("tags", []), require_all):
                    sample_num += 1

                    # Enable debug for first 3 samples
                    debug_this_sample = self.debug and sample_num <= 3

                    if debug_this_sample:
                        print(f"\n{'='*60}")
                        print(f"[DEBUG] Sample {sample_num}: {Path(image['path']).name}")
                        print(f"[DEBUG] Prompt: {prompt['text'][:80]}...")
                        print(f"{'='*60}")

                    # Run VLM generation
                    final_prompt = self.vlm.preprocess(image["path"], prompt["text"])
                    output = self.vlm.generate(final_prompt, debug=debug_this_sample)

                    # Store result
                    result_entry = {
                        "image": {"path": image["path"], "tags": image.get("tags", [])},
                        "prompt": {"text": prompt["text"], "tags": prompt.get("tags", [])},
                        "output": output,
                    }

                    results["runs"].append(result_entry)
                    results["total_runs"] += 1

        # Save results
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_results_{timestamp}.json"

        output_path = self.results_dir / output_filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark complete! {results['total_runs']} runs saved to {output_path}")
        return results


class ImageCompletionBenchmark:
    """Scaffolding for running image completion benchmarks."""

    def __init__(self, images_config_path: str, vlm: VLM, results_dir: str = "results", debug: bool = False):
        """
        Initialize the image completion benchmark system.

        Args:
            images_config_path: Path to JSON file containing image configurations
            vlm: VLM instance for running completion
            results_dir: Directory where results will be stored
            debug: Enable debug mode for detailed output
        """
        self.images = self._load_json(images_config_path)
        self.results_dir = Path(results_dir)
        self.vlm = vlm
        self.debug = debug
        print(f"Loaded {len(self.images)} images for completion benchmark.")

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON configuration file."""
        with open(path, "r") as f:
            return json.load(f)

    def run_completion_benchmark(
        self, percentages: List[int], output_filename: str, strict_row_count: bool = False
    ) -> Dict[str, Any]:
        """
        Run image completion benchmark for all images and percentages.

        Args:
            percentages: List of completion percentages (e.g., [20, 40, 60, 80])
            output_filename: Filename for results JSON
            strict_row_count: If True, require exact row count match for validity

        Returns:
            Dictionary containing all benchmark results
        """
        # Serialize inference args
        inference_args_dict = {
            "apply_chat_template": self.vlm.inf_args.apply_chat_template,
            "prompt_builder": self.vlm.inf_args.prompt_builder,
            "temperature": self.vlm.inf_args.temperature,
            "top_p": self.vlm.inf_args.top_p,
            "max_new_tokens": self.vlm.inf_args.max_new_tokens,
            "max_emu_aspect_ratio": self.vlm.inf_args.max_emu_aspect_ratio,
            "min_emu_aspect_ratio": self.vlm.inf_args.min_emu_aspect_ratio,
            "stop_token_ids": self.vlm.inf_args.stop_token_ids,
        }

        results = {
            "timestamp": datetime.now().isoformat(),
            "mode": "image_completion",
            "model_path": self.vlm.model_path,
            "tokenizer_path": self.vlm.tokenizer_path,
            "vision_tokenizer": self.vlm.vision_tokenizer.name,
            "completion_percentages": percentages,
            "strict_row_count": strict_row_count,
            "inference_args": inference_args_dict,
            "total_runs": 0,
            "runs": [],
        }

        # Create completion_images subdirectory
        experiment_name = Path(output_filename).stem
        images_dir = self.results_dir / experiment_name / "completion_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Track sample number for debug mode (only first 3 samples)
        sample_num = 0

        # Run completion for each image and percentage
        for image_config in tqdm(self.images, desc="Processing images"):
            image_path = image_config["path"]
            image_tags = image_config.get("tags", [])

            for percentage in percentages:
                sample_num += 1
                print(f"\n{'='*60}")
                print(f"Sample {sample_num}: Image: {Path(image_path).name}, Completion: {percentage}%")
                print(f"{'='*60}")

                # Enable debug for first 3 samples
                debug_this_sample = self.debug and sample_num <= 3

                try:
                    # Generate completion
                    completion_result = self.vlm.generate_image_completion(
                        image_path, percentage, strict_row_count, debug=debug_this_sample
                    )

                    # Prepare result entry
                    result_entry = {
                        "image": {
                            "path": image_path,
                            "tags": image_tags,
                            "token_dimensions": {
                                "height": completion_result["original_image_rows"],
                                "width": completion_result["original_image_width"],
                            },
                        },
                        "completion_percentage": percentage,
                        "original_image_rows": completion_result["original_image_rows"],
                        "given_rows": completion_result["given_rows"],
                        "generated_rows": completion_result["generated_rows"],
                        "expected_rows": completion_result["expected_rows"],
                        "is_valid": completion_result["is_valid"],
                        "validation": completion_result["validation"],
                        "statistics": completion_result["statistics"],
                    }

                    # If valid, reconstruct and save result image
                    if completion_result["is_valid"]:
                        result_image_path, metrics_dict = self._save_result_image(
                            completion_result,
                            image_path,
                            percentage,
                            images_dir,
                        )
                        result_entry["result_image_path"] = str(
                            Path(experiment_name) / "completion_images" / Path(result_image_path).name
                        )
                        if metrics_dict:
                            result_entry["perceptual_metrics"] = metrics_dict
                        print(f"   ✓ Valid completion - image saved")
                    else:
                        result_entry["result_image_path"] = None
                        failed_checks = [k for k, v in completion_result["validation"].items() if not v]
                        print(f"   ✗ Invalid completion - failed checks: {', '.join(failed_checks)}")

                except Exception as e:
                    print(f"   ✗ Error during completion: {e}")
                    import traceback

                    traceback.print_exc()

                    # Store error result
                    result_entry = {
                        "image": {"path": image_path, "tags": image_tags},
                        "completion_percentage": percentage,
                        "is_valid": False,
                        "error": str(e),
                    }

                results["runs"].append(result_entry)
                results["total_runs"] += 1

        # Save results
        output_path = self.results_dir / output_filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Benchmark complete! {results['total_runs']} runs saved to {output_path}")
        valid_count = sum(1 for r in results["runs"] if r.get("is_valid", False))
        print(f"Valid completions: {valid_count}/{results['total_runs']}")
        print(f"{'='*60}")

        return results

    def _save_result_image(
        self, completion_result: Dict[str, Any], original_image_path: str, percentage: int, images_dir: Path
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Reconstruct image, compute perceptual metrics, and save with red boundary line.
        Expects a valid formatted image (equally sized full rows).

        Args:
            completion_result: Result from generate_image_completion()
            original_image_path: Path to original image
            percentage: Completion percentage
            images_dir: Directory to save result images

        Returns:
            Tuple of (path to saved result image, perceptual metrics dict)
        """
        from PIL import ImageDraw

        # Combine given and generated indices
        given_indices = completion_result["given_indices"]
        generated_indices = completion_result["generated_indices"]
        width = completion_result["original_image_width"]  # width expected to be the same to be valid!
        given_rows = completion_result["given_rows"]
        original_height = completion_result["original_image_rows"]
        expected_rows = completion_result["expected_rows"]

        # We reconstruct with all shapes that are valid (method assumes valid shapes)
        # Specifically we don't pad with 0s or truncate extra rows
        all_indices = given_indices + generated_indices

        # Calculate actual height from combined data
        actual_height = len(all_indices) // width
        assert len(all_indices) == actual_height * width

        # Reconstruct completed image using vision tokenizer
        tokenizer = self.vlm.vision_tokenizer.tokenizer
        indices_tensor = torch.tensor(all_indices, dtype=torch.long, device=tokenizer.device)
        indices_tensor = indices_tensor.reshape(1, actual_height, width)

        with torch.no_grad():
            reconstructed_tensor = tokenizer.decode(indices_tensor)

        reconstructed_pil = tokenizer.postprocess(reconstructed_tensor)

        # Reconstruct full reference image from original indices (isolates completion quality from tokenizer noise)
        original_all_indices = completion_result["original_all_indices"]
        ref_tensor = torch.tensor(original_all_indices, dtype=torch.long, device=tokenizer.device)
        ref_tensor = ref_tensor.reshape(1, original_height, width)

        with torch.no_grad():
            ref_decoded = tokenizer.decode(ref_tensor)

        reference_pil = tokenizer.postprocess(ref_decoded)

        # Calculate boundary and pixel dimensions
        pixels_per_row = reconstructed_pil.height / actual_height
        boundary_y = int(given_rows * pixels_per_row)
        expected_generated_pixel_height = int(expected_rows * pixels_per_row)
        actual_generated_pixel_height = reconstructed_pil.height - boundary_y

        # Compute perceptual metrics before drawing the red line
        metrics_dict = {}
        try:
            from vision_tokenization.qualitative_benchmark.utils.metrics import compute_completion_metrics

            metrics_device = str(tokenizer.device) if hasattr(tokenizer, "device") else "cpu"
            metrics_dict = compute_completion_metrics(
                completed_image=reconstructed_pil,
                reference_image=reference_pil,
                boundary_y=boundary_y,
                expected_generated_height=expected_generated_pixel_height,
                actual_generated_height=actual_generated_pixel_height,
                device=metrics_device,
            )
            print(
                f"   Metrics: PSNR_full={metrics_dict['psnr_full']:.2f}, "
                f"SSIM_full={metrics_dict['ssim_full']:.4f}, "
                f"LPIPS_full={metrics_dict['lpips_full']:.4f}"
            )
            if metrics_dict.get("psnr_generated") is not None:
                print(
                    f"   Metrics (gen): PSNR={metrics_dict['psnr_generated']:.2f}, "
                    f"SSIM={metrics_dict['ssim_generated']:.4f}, "
                    f"LPIPS={metrics_dict['lpips_generated']:.4f}"
                )
        except Exception as e:
            print(f"   Warning: Failed to compute perceptual metrics: {e}")

        # Draw red line at boundary
        draw = ImageDraw.Draw(reconstructed_pil)
        img_width = reconstructed_pil.width
        draw.line([(0, boundary_y), (img_width, boundary_y)], fill="red", width=3)

        # Save annotated image
        image_stem = Path(original_image_path).stem
        result_filename = f"{image_stem}_{percentage}pct.png"
        result_path = images_dir / result_filename
        reconstructed_pil.save(result_path)

        return result_path, metrics_dict


class CaptioningBenchmark:
    """Scaffolding for running image captioning benchmarks."""

    def __init__(
        self,
        images_config_path: str,
        vlm: VLM,
        results_dir: str = "results",
        init_phrase: str = None,
        debug: bool = False,
    ):
        """
        Initialize the captioning benchmark system.

        Args:
            images_config_path: Path to JSON file containing image configurations
            vlm: VLM instance for running captioning
            results_dir: Directory where results will be stored
            init_phrase: Optional initialization phrase for caption generation
            debug: Enable debug mode for detailed token output
        """
        self.images = self._load_json(images_config_path)
        self.results_dir = Path(results_dir)
        self.vlm = vlm
        self.init_phrase = init_phrase or ""
        self.debug = debug
        print(f"Loaded {len(self.images)} images for captioning benchmark.")
        if self.init_phrase:
            print(f"Using init phrase: '{self.init_phrase}'")

    def _load_json(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON configuration file."""
        with open(path, "r") as f:
            return json.load(f)

    def run_captioning_benchmark(self, output_filename: str) -> Dict[str, Any]:
        """
        Run captioning benchmark for all images.

        Args:
            output_filename: Filename for results JSON

        Returns:
            Dictionary containing all benchmark results
        """
        # Serialize inference args
        inference_args_dict = {
            "apply_chat_template": self.vlm.inf_args.apply_chat_template,
            "prompt_builder": self.vlm.inf_args.prompt_builder,
            "temperature": self.vlm.inf_args.temperature,
            "top_p": self.vlm.inf_args.top_p,
            "max_new_tokens": self.vlm.inf_args.max_new_tokens,
            "max_emu_aspect_ratio": self.vlm.inf_args.max_emu_aspect_ratio,
            "min_emu_aspect_ratio": self.vlm.inf_args.min_emu_aspect_ratio,
            "stop_token_ids": self.vlm.inf_args.stop_token_ids,
        }

        results = {
            "timestamp": datetime.now().isoformat(),
            "mode": "captioning",
            "model_path": self.vlm.model_path,
            "tokenizer_path": self.vlm.tokenizer_path,
            "vision_tokenizer": self.vlm.vision_tokenizer.name,
            "inferencer": type(self.vlm.inferencer).__name__,
            "caption_init_phrase": self.init_phrase if self.init_phrase else None,
            "inference_args": inference_args_dict,
            "total_runs": 0,
            "runs": [],
        }

        # Track sample number for debug mode (only first 3 samples)
        sample_num = 0

        for image in tqdm(self.images, desc="Generating captions"):
            sample_num += 1

            # Enable debug for first 3 samples
            debug_this_sample = self.debug and sample_num <= 3

            if debug_this_sample:
                print(f"\n{'='*60}")
                print(f"[DEBUG] Sample {sample_num}: {Path(image['path']).name}")
                print(f"{'='*60}")

            final_prompt = self.vlm.preprocess(image["path"], self.init_phrase)
            caption = self.vlm.generate(final_prompt, debug=debug_this_sample)

            result_entry = {
                "image": {"path": image["path"], "tags": image.get("tags", [])},
                "init_phrase": self.init_phrase,
                "caption": caption,
            }

            results["runs"].append(result_entry)
            results["total_runs"] += 1

        output_path = self.results_dir / output_filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Captioning benchmark complete! {results['total_runs']} captions saved to {output_path}")
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM benchmark with configurable vision v_tokenizers and utils")

    # Core arguments
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to text tokenizer supporting SFT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to VLM model")
    parser.add_argument(
        "--chat-format",
        type=str,
        choices=list(CHAT_TRANFORMS.keys()),
        help="Chat format to use. Method to transform input into format hat works with a models chat template",
    )
    parser.add_argument(
        "--prompt-builder",
        type=str,
        choices=list(PROMPT_BUILDERS.keys()),
        default=None,
        help="Custom prompt builder that bypasses apply_chat_template entirely (e.g., emu3 for BAAI/Emu3-Chat)",
    )
    parser.add_argument("--results_folder", type=str, default="results/", help="Path to save results")
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Optional subfolder within results_folder to organize results (e.g., 'project_a' saves to results/project_a/)",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of experiment, used for saving results"
    )
    parser.add_argument("--image_list", type=str, default="images.json", help="Path to image list JSON")
    parser.add_argument("--prompt_list", type=str, default="prompts.json", help="Path to prompt list JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results file if it exists")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (prints prompt token IDs and decoded text for first 3 samples in all benchmark modes)",
    )

    # Image completion benchmark arguments
    parser.add_argument(
        "--image-completion", action="store_true", help="Run image completion benchmark instead of VLM Q&A"
    )
    parser.add_argument(
        "--completion-percentages",
        type=str,
        default=None,
        help="Comma-separated percentages of image rows to provide as context (required for --image-completion). "
        "Model will complete the remaining rows. E.g., '20,40,60,80' means give 20%%, 40%%, 60%%, 80%% of rows "
        "and generate the rest.",
    )
    parser.add_argument(
        "--strict-row-count",
        action="store_true",
        help="Require exact row count match for completion to be valid (default: False)",
    )

    # Captioning benchmark arguments
    parser.add_argument(
        "--captioning",
        action="store_true",
        help="Run captioning benchmark instead of VLM Q&A",
    )
    parser.add_argument(
        "--caption-init-phrase",
        type=str,
        default=None,
        help="Optional initialization phrase for caption generation (e.g., 'The image shows')",
    )

    # Vision tokenizer arguments
    tokenizer_group = parser.add_argument_group("vision tokenizer arguments", "Configuration for vision tokenizer")
    tokenizer_group.add_argument(
        "--vision-tokenizer-type",
        type=str,
        default="emu3",
        choices=["emu3", "emu3.5", "emu3.5-ibq"],
        help="Type of vision tokenizer to use (default: emu3)",
    )
    tokenizer_group.add_argument(
        "--vision-tokenizer-path", type=str, default=None, help="Path to vision tokenizer model (required for emu3.5)"
    )
    tokenizer_group.add_argument(
        "--min-tokenizer-pixels",
        type=int,
        default=256 * 256,
        help="Minimum pixel count for vision tokenizer (default: 65536)",
    )
    tokenizer_group.add_argument(
        "--max-tokenizer-pixels",
        type=int,
        default=1024 * 1024,
        help="Maximum pixel count for vision tokenizer (default: 262144)",
    )

    # Inference generation arguments
    inference_group = parser.add_argument_group("inference arguments", "Arguments controlling model inference behavior")
    inference_group.add_argument(
        "--no_chat_template", action="store_true", help="Do not apply chat template to prompts"
    )
    inference_group.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    inference_group.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter (default: 0.9)")
    inference_group.add_argument(
        "--max_new_tokens", type=int, default=300, help="Maximum number of tokens to generate (default: 300)"
    )
    inference_group.add_argument(
        "--inferencer-type",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
        help="Inference backend: 'vllm' (faster) or 'hf' (HuggingFace, more compatible) (default: vllm)",
    )
    inference_group.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model (default: 8192)",
    )
    inference_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (vLLM only, ignored by HF backend) (default: 1)",
    )
    inference_group.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (sets temperature=0, top_p=1.0)",
    )
    inference_group.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV caching during generation (HF backend only). "
        "Slower but fixes compatibility issues with some models (e.g., Emu3)",
    )
    return parser.parse_args()


def validate_args(args):
    """
    Validate argument combinations for benchmark modes.

    Ensures only one mode is selected, required args are provided, and warns about unused args.
    """
    # Check that only one mode is selected
    modes_selected = []
    if args.image_completion:
        modes_selected.append("--image-completion")
    if args.captioning:
        modes_selected.append("--captioning")

    if len(modes_selected) > 1:
        print(f"ERROR: Multiple benchmark modes selected: {', '.join(modes_selected)}")
        print("Please select only one mode: --image-completion, --captioning, or neither for VLM Q&A")
        sys.exit(1)

    # Determine current mode
    if args.image_completion:
        current_mode = "image-completion"
    elif args.captioning:
        current_mode = "captioning"
    else:
        current_mode = "vlm-qa"

    # Warn if --prompt-builder is used together with --chat-format
    if args.prompt_builder is not None and args.chat_format is not None:
        print("WARNING: --prompt-builder overrides --chat-format; --chat-format will be ignored")

    # Check required arguments for each mode
    if current_mode == "image-completion":
        if args.completion_percentages is None:
            print("ERROR: --completion-percentages is required when using --image-completion")
            print("Example: --completion-percentages 20,40,60,80")
            sys.exit(1)

    # Check for unused mode-specific arguments and warn
    # Image completion specific args
    if current_mode != "image-completion":
        if args.completion_percentages is not None:
            print(f"WARNING: --completion-percentages ignored in {current_mode} mode")
        if args.strict_row_count:
            print(f"WARNING: --strict-row-count ignored in {current_mode} mode")

    # Captioning specific args
    if current_mode != "captioning":
        if args.caption_init_phrase is not None:
            print(f"WARNING: --caption-init-phrase ignored in {current_mode} mode")

    # VLM Q&A specific args
    if current_mode != "vlm-qa":
        if args.prompt_list != "prompts.json":
            print(f"WARNING: --prompt_list ignored in {current_mode} mode")

    return current_mode


def setup_vlm_inferencer(args):
    """
    Setup VLM with pluggable vision tokenizer and inferencer components.

    Handles backward compatibility with deprecated arguments.
    """
    from vision_tokenization.qualitative_benchmark.v_tokenizers import create_vision_tokenizer

    # Handle backward compatibility for pixel arguments
    min_pixels = args.min_tokenizer_pixels
    max_pixels = args.max_tokenizer_pixels

    if hasattr(args, "min_emu_aspect_ratio") and args.min_emu_aspect_ratio is not None:
        print("WARNING: --min_emu_aspect_ratio is deprecated, use --min-tokenizer-pixels")
        min_pixels = args.min_emu_aspect_ratio
    if hasattr(args, "max_emu_aspect_ratio") and args.max_emu_aspect_ratio is not None:
        print("WARNING: --max_emu_aspect_ratio is deprecated, use --max-tokenizer-pixels")
        max_pixels = args.max_emu_aspect_ratio

    # Create vision tokenizer
    print(f"\nCreating vision tokenizer: {args.vision_tokenizer_type}")

    vision_tokenizer_kwargs = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "tokenizer_path": args.tokenizer_path,  # Pass text tokenizer path for vision mapping
    }

    # Add model_path for v_tokenizers that need it
    if args.vision_tokenizer_type in ["emu3.5", "emu3.5-ibq"]:
        if args.vision_tokenizer_path:
            vision_tokenizer_kwargs["model_path"] = args.vision_tokenizer_path
        else:
            # Will use default path from EMU35IBQVisionTokenizer.DEFAULT_MODEL_PATH
            print("WARNING: vision tokenizer path not given, using default path...")
    elif args.vision_tokenizer_path:
        # Optional for emu3
        vision_tokenizer_kwargs["model_path"] = args.vision_tokenizer_path

    vision_tokenizer = create_vision_tokenizer(args.vision_tokenizer_type, **vision_tokenizer_kwargs)

    # Create inferencer based on selected backend
    inferencer_type = getattr(args, "inferencer_type", "vllm")
    print(f"Creating inferencer: {inferencer_type}")

    inferencer_kwargs = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "max_seq_len": getattr(args, "max_seq_len", 8192),
    }
    if inferencer_type == "vllm":
        inferencer_kwargs["tp_size"] = getattr(args, "tp_size", 1)
    elif inferencer_type == "hf":
        inferencer_kwargs["device"] = "cuda:0"
        inferencer_kwargs["use_cache"] = not getattr(args, "no_kv_cache", False)

    inferencer = create_inferencer(inferencer_type, **inferencer_kwargs)

    # Create inference args
    # For captioning mode, disable chat template by default (unless explicitly enabled)
    apply_chat = not args.no_chat_template
    if args.captioning and not args.no_chat_template and args.prompt_builder is None:
        # Captioning mode defaults to no chat template (unless prompt_builder handles it)
        apply_chat = False
        print("Note: Chat template disabled for captioning mode (use explicit prompts if needed)")

    # For image completion mode, set max_new_tokens to max_pixels by default
    # (image completion can generate many tokens - up to the full image token count)
    max_new_tokens = args.max_new_tokens
    if args.image_completion:
        max_new_tokens = max_pixels
        print(f"Note: max_new_tokens set to {max_new_tokens} for image completion mode")

    inference_args = InferenceArgs(
        apply_chat_template=apply_chat,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=[],  # Will be set by VLM __init__
        max_new_tokens=max_new_tokens,
        max_emu_aspect_ratio=max_pixels,
        min_emu_aspect_ratio=min_pixels,
        chat_transform=args.chat_format or None,
        prompt_builder=args.prompt_builder,
    )

    # Create VLM with pluggable components
    vlm = VLM(
        vision_tokenizer=vision_tokenizer,
        inferencer=inferencer,
        inf_args=inference_args,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
    )

    return vlm


if __name__ == "__main__":
    """
    Example usage:
    python vlm_benchmark.py --tokenizer_path /path/to/tokenizer --model_path /path/to/model --experiment_name my_experiment
    """
    args = parse_args()

    # Apply greedy decoding settings if requested
    if args.greedy:
        args.temperature = 0.0
        args.top_p = 1.0
        print("Greedy decoding enabled: temperature=0, top_p=1.0")

    # Validate argument combinations
    validate_args(args)

    # Setup results directory and output file path
    results_dir = Path(args.results_folder)
    if args.subfolder:
        results_dir = results_dir / args.subfolder
    results_dir.mkdir(exist_ok=True, parents=True)
    output_file = results_dir / f"{args.experiment_name}.json"

    # Check if results file already exists
    if output_file.exists() and not args.overwrite:
        print(f"ERROR: Results file already exists: {output_file}")
        print(f"Use --overwrite to overwrite existing results, or choose a different experiment name.")
        sys.exit(1)

    if output_file.exists() and args.overwrite:
        print(f"WARNING: Overwriting existing results file: {output_file}")

    vlm = setup_vlm_inferencer(args)

    # Check which benchmark mode to run using registry
    if args.image_completion:
        # Parse percentages
        percentages = [int(p.strip()) for p in args.completion_percentages.split(",")]
        print(f"Running image completion benchmark with percentages: {percentages}")
        print(f"Strict row count: {args.strict_row_count}")
        if args.debug:
            print("Debug mode enabled (will print tokens for first 3 samples)")

        # Create and run image completion benchmark using registry
        benchmark = get_benchmark(
            "image_completion",
            images_config_path=args.image_list,
            vlm=vlm,
            results_dir=str(results_dir),
            debug=args.debug,
        )
        results = benchmark.run(
            percentages=percentages,
            output_filename=f"{args.experiment_name}.json",
            strict_row_count=args.strict_row_count,
        )
    elif args.captioning:
        print("Running captioning benchmark")
        if args.caption_init_phrase:
            print(f"Init phrase: '{args.caption_init_phrase}'")
        if args.debug:
            print("Debug mode enabled (will print tokens for first 3 samples)")

        # Create and run captioning benchmark using registry
        benchmark = get_benchmark(
            "captioning",
            images_config_path=args.image_list,
            vlm=vlm,
            results_dir=str(results_dir),
            init_phrase=args.caption_init_phrase,
            debug=args.debug,
        )
        results = benchmark.run(output_filename=f"{args.experiment_name}.json")
    else:
        # Run standard VLM Q&A benchmark using registry
        if args.debug:
            print("Debug mode enabled (will print tokens for first 3 samples)")

        benchmark = get_benchmark(
            "vlm",
            images_config_path=args.image_list,
            prompts_config_path=args.prompt_list,
            vlm=vlm,
            results_dir=str(results_dir),
            debug=args.debug,
        )
        results = benchmark.run(output_filename=f"{args.experiment_name}.json")

    print("Done!")
