"""
Image captioning benchmark for evaluating caption generation quality.

Tests VLM ability to generate descriptive captions for images.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from vision_tokenization.qualitative_benchmark.benchmarks import register_benchmark
from vision_tokenization.qualitative_benchmark.benchmarks.base import BaseBenchmark


@register_benchmark("captioning")
class CaptioningBenchmark(BaseBenchmark):
    """
    Benchmark for image captioning tasks.

    Tests VLM ability to generate descriptive captions for images.
    Optionally computes CLIP score to measure image-caption alignment.

    Args:
        images_config_path: Path to JSON file containing image configurations
        vlm: VLM instance for running captioning
        results_dir: Directory where results will be stored
        init_phrase: Optional initialization phrase for caption generation
        debug: Enable debug mode for detailed token output
    """

    name: ClassVar[str] = "captioning"
    METRICS: ClassVar[List[str]] = ["clip_score", "polos_score"]

    def __init__(
        self,
        images_config_path: str,
        vlm,
        results_dir: str = "results",
        init_phrase: Optional[str] = None,
        debug: bool = False,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize the captioning benchmark.

        Args:
            images_config_path: Path to JSON file containing image configurations
            vlm: VLM instance for running captioning
            results_dir: Directory where results will be stored
            init_phrase: Optional initialization phrase for caption generation
            debug: Enable debug mode for detailed token output
            metrics: Optional list of metric names to use
        """
        super().__init__(vlm=vlm, results_dir=results_dir, debug=debug, metrics=metrics)

        self.images = self._load_json(images_config_path)
        self.init_phrase = init_phrase or ""
        print(f"Loaded {len(self.images)} images for captioning benchmark.")
        if self.init_phrase:
            print(f"Using init phrase: '{self.init_phrase}'")

    def run(self, output_filename: str) -> Dict[str, Any]:
        """
        Run captioning benchmark for all images.

        Args:
            output_filename: Filename for results JSON

        Returns:
            Dictionary containing all benchmark results
        """
        results = self._create_base_results()
        results["mode"] = "captioning"
        results["caption_init_phrase"] = self.init_phrase if self.init_phrase else None

        # Track sample number for debug mode (only first 3 samples)
        sample_num = 0

        for image_config in tqdm(self.images, desc="Generating captions"):
            sample_num += 1

            # Enable debug for first 3 samples
            debug_this_sample = self.debug and sample_num <= 3

            if debug_this_sample:
                print(f"\n{'='*60}")
                print(f"[DEBUG] Sample {sample_num}: {Path(image_config['path']).name}")
                print(f"{'='*60}")

            image_path = image_config["path"]

            # Generate caption
            final_prompt = self.vlm.preprocess(image_path, self.init_phrase)
            caption = self.vlm.generate(final_prompt, debug=debug_this_sample)

            # Prepare result entry
            result_entry = {
                "image": {"path": image_path, "tags": image_config.get("tags", [])},
                "init_phrase": self.init_phrase,
                "caption": caption,
            }

            # Compute CLIP score if metrics are enabled
            if self.metrics:
                try:
                    # Load image for CLIP scoring
                    pil_image = Image.open(image_path).convert("RGB")

                    metrics_data = {
                        "image": pil_image,
                        "caption": caption,
                    }
                    metrics_dict = self._compute_metrics(metrics_data)

                    if metrics_dict:
                        result_entry["metrics"] = metrics_dict

                        if metrics_dict.get("clip_score") is not None:
                            print(f"   CLIP score: {metrics_dict['clip_score']:.4f}")
                        if metrics_dict.get("polos_score") is not None:
                            print(f"   POLOS score: {metrics_dict['polos_score']:.4f}")

                except Exception as e:
                    print(f"   Warning: Failed to compute metrics: {e}")

            results["runs"].append(result_entry)
            results["total_runs"] += 1

        # Save results
        output_path = self._save_results(results, output_filename)

        print(f"Captioning benchmark complete! {results['total_runs']} captions saved to {output_path}")
        return results
