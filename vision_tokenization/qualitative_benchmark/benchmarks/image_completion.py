"""
Image completion benchmark for evaluating partial image generation.

Tests VLM ability to complete images given a partial context (top portion).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch
from PIL import ImageDraw
from tqdm import tqdm

from vision_tokenization.qualitative_benchmark.benchmarks import register_benchmark
from vision_tokenization.qualitative_benchmark.benchmarks.base import BaseBenchmark


@register_benchmark("image_completion")
class ImageCompletionBenchmark(BaseBenchmark):
    """
    Benchmark for image completion tasks.

    Tests VLM ability to complete images given a percentage of the top rows.
    Computes perceptual quality metrics (PSNR, SSIM, LPIPS) on valid completions.

    Args:
        images_config_path: Path to JSON file containing image configurations
        vlm: VLM instance for running completion
        results_dir: Directory where results will be stored
        debug: Enable debug mode for detailed output
    """

    name: ClassVar[str] = "image_completion"
    METRICS: ClassVar[List[str]] = ["completion_quality"]

    def __init__(
        self,
        images_config_path: str,
        vlm,
        results_dir: str = "results",
        debug: bool = False,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize the image completion benchmark.

        Args:
            images_config_path: Path to JSON file containing image configurations
            vlm: VLM instance for running completion
            results_dir: Directory where results will be stored
            debug: Enable debug mode for detailed output
            metrics: Optional list of metric names to use
        """
        super().__init__(vlm=vlm, results_dir=results_dir, debug=debug, metrics=metrics)

        self.images = self._load_json(images_config_path)
        print(f"Loaded {len(self.images)} images for completion benchmark.")

    def run(
        self,
        percentages: List[int],
        output_filename: str,
        strict_row_count: bool = False,
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
        results = self._create_base_results()
        results["mode"] = "image_completion"
        results["completion_percentages"] = percentages
        results["strict_row_count"] = strict_row_count

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
                            result_entry["metrics"] = metrics_dict
                        print(f"   Valid completion - image saved")
                    else:
                        result_entry["result_image_path"] = None
                        failed_checks = [k for k, v in completion_result["validation"].items() if not v]
                        print(f"   Invalid completion - failed checks: {', '.join(failed_checks)}")

                except Exception as e:
                    print(f"   Error during completion: {e}")
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
        output_path = self._save_results(results, output_filename)

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

        Args:
            completion_result: Result from generate_image_completion()
            original_image_path: Path to original image
            percentage: Completion percentage
            images_dir: Directory to save result images

        Returns:
            Tuple of (path to saved result image, metrics dict)
        """
        # Combine given and generated indices
        given_indices = completion_result["given_indices"]
        generated_indices = completion_result["generated_indices"]
        width = completion_result["original_image_width"]
        given_rows = completion_result["given_rows"]
        original_height = completion_result["original_image_rows"]
        expected_rows = completion_result["expected_rows"]

        # Reconstruct with all valid shapes
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

        # Reconstruct full reference image from original indices
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

        # Compute metrics using the registry-based system
        metrics_dict = {}
        if self.metrics:
            try:
                metrics_data = {
                    "completed_image": reconstructed_pil,
                    "reference_image": reference_pil,
                    "boundary_y": boundary_y,
                    "expected_generated_height": expected_generated_pixel_height,
                    "actual_generated_height": actual_generated_pixel_height,
                }
                metrics_dict = self._compute_metrics(metrics_data)

                if metrics_dict.get("psnr_full") is not None:
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
