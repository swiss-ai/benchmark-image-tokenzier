"""
VLM (Vision-Language Model) benchmark for image-based question answering.

Tests VLM performance on image-prompt pairs with tag-based matching.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from tqdm import tqdm

from vision_tokenization.qualitative_benchmark.benchmarks import register_benchmark
from vision_tokenization.qualitative_benchmark.benchmarks.base import BaseBenchmark


@register_benchmark("vlm")
class VLMBenchmark(BaseBenchmark):
    """
    Benchmark for VLM question-answering tasks.

    Tests VLM on image-prompt pairs where prompts are matched to images
    based on tag compatibility. Results include the model's generated responses.

    Args:
        images_config_path: Path to JSON file containing image configurations
        prompts_config_path: Path to JSON file containing prompt configurations
        vlm: VLM instance for running inference
        results_dir: Directory where results will be stored
        debug: Enable debug mode for detailed token output
    """

    name: ClassVar[str] = "vlm"
    METRICS: ClassVar[List[str]] = []  # VLM benchmark has no automatic metrics

    def __init__(
        self,
        images_config_path: str,
        prompts_config_path: str,
        vlm,
        results_dir: str = "results",
        debug: bool = False,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize the VLM benchmark.

        Args:
            images_config_path: Path to JSON file containing image configurations
            prompts_config_path: Path to JSON file containing prompt configurations
            vlm: VLM instance for running inference
            results_dir: Directory where results will be stored
            debug: Enable debug mode for detailed token output
            metrics: Optional list of metric names (not typically used for VLM)
        """
        super().__init__(vlm=vlm, results_dir=results_dir, debug=debug, metrics=metrics)

        self.images = self._load_json(images_config_path)
        self.prompts = self._load_json(prompts_config_path)
        print(f"Loaded {len(self.images)} images and {len(self.prompts)} prompts.")

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

    def run(self, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the VLM benchmark by matching images with compatible prompts.

        Args:
            output_filename: Optional custom filename for results

        Returns:
            Dictionary containing all benchmark results
        """
        results = self._create_base_results()

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

        output_path = self._save_results(results, output_filename)

        print(f"Benchmark complete! {results['total_runs']} runs saved to {output_path}")
        return results
