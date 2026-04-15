"""
Image captioning benchmark for evaluating caption generation quality.

Tests VLM ability to generate descriptive captions for images.
"""

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
    CLIP_METRIC: ClassVar[str] = "clip_score"
    METRICS: ClassVar[List[str]] = [CLIP_METRIC]
    FALLBACK_INIT_PHRASE: ClassVar[str] = "The image shows "

    def __init__(
        self,
        images_config_path: str,
        vlm,
        results_dir: str = "results",
        init_phrase: Optional[str] = None,
        debug: bool = False,
        metrics: Optional[List[str]] = None,
        retry_clip_threshold: Optional[float] = None,
        retry_max_attempts: int = 1,
        retry_base_seed: Optional[int] = None,
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
            retry_clip_threshold: If set, regenerate captions whose CLIP score is
                below this threshold (re-rolls junk outputs). ``None`` disables
                retries.
            retry_max_attempts: Maximum total attempts per image (default 1 = no
                retry). Only the *last* attempt is kept — the one that either
                cleared the threshold or exhausted the budget.
            retry_base_seed: Base seed for retry attempts; attempt ``k`` uses
                ``base_seed + k`` so re-rolls are deterministic. ``None`` =
                backend default randomness.
        """
        super().__init__(vlm=vlm, results_dir=results_dir, debug=debug, metrics=metrics)

        self.images = self._load_json(images_config_path)
        self.init_phrase = init_phrase or ""
        if retry_max_attempts < 1:
            raise ValueError(
                f"retry_max_attempts must be >= 1, got {retry_max_attempts}"
            )
        self.retry_clip_threshold = retry_clip_threshold
        self.retry_max_attempts = int(retry_max_attempts)
        self.retry_base_seed = retry_base_seed
        self._retry_enabled = (
            retry_clip_threshold is not None
            and self.retry_max_attempts > 1
            and self.CLIP_METRIC in self.metrics
        )

        print(f"Loaded {len(self.images)} images for captioning benchmark.")
        if self.init_phrase:
            print(f"Using init phrase: '{self.init_phrase}'")
        if self._retry_enabled:
            print(
                f"Retry-on-low-CLIP enabled: threshold={retry_clip_threshold}, "
                f"max_attempts={self.retry_max_attempts}, base_seed={retry_base_seed}"
            )

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
        if self._retry_enabled:
            results["retry_config"] = {
                "clip_threshold": self.retry_clip_threshold,
                "max_attempts": self.retry_max_attempts,
                "base_seed": self.retry_base_seed,
            }

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
            pil_image = Image.open(image_path).convert("RGB") if self.metrics else None

            gen = self._generate_with_retries(image_path, pil_image, debug_this_sample)

            result_entry = {
                "image": {"path": image_path, "tags": image_config.get("tags", [])},
                "init_phrase": gen["init_phrase"],
                "caption": gen["caption"],
            }
            if gen["metrics"]:
                result_entry["metrics"] = gen["metrics"]
                if gen["metrics"].get(self.CLIP_METRIC) is not None:
                    print(f"   CLIP score: {gen['metrics'][self.CLIP_METRIC]:.4f}")
            if gen["retry_stats"] is not None:
                result_entry["retry_stats"] = gen["retry_stats"]

            results["runs"].append(result_entry)
            results["total_runs"] += 1

        # Save results
        output_path = self._save_results(results, output_filename)

        print(f"Captioning benchmark complete! {results['total_runs']} captions saved to {output_path}")
        return results

    def _generate_with_retries(
        self, image_path: str, pil_image: Optional[Image.Image], debug_this_sample: bool,
    ) -> Dict[str, Any]:
        """Generate a caption, optionally re-rolling junk outputs.

        Re-rolls happen when ``clip_score < retry_clip_threshold`` and there is
        retry budget left. The accepted caption is whichever attempt cleared
        the threshold first; if no attempt does, the last attempt is kept (we
        ran out of budget — give up).

        Returns ``{"caption", "init_phrase", "metrics", "retry_stats"}`` where
        ``retry_stats`` is ``None`` when retries are disabled (keeps default
        JSON unchanged).
        """
        final_prompt = self.vlm.preprocess(image_path, self.init_phrase)
        fallback_prompt: Optional[str] = None
        fallback_init_phrase = (
            self.FALLBACK_INIT_PHRASE if self._retry_enabled and not self.init_phrase else None
        )

        attempts_log: List[Dict[str, Any]] = []
        caption: Optional[str] = None
        metrics: Dict[str, Any] = {}
        used_init_phrase = self.init_phrase

        for k in range(self.retry_max_attempts):
            seed = None if self.retry_base_seed is None else self.retry_base_seed + k
            use_fallback_prompt = (
                fallback_init_phrase is not None
                and k == self.retry_max_attempts - 1
                and k > 0
            )
            if use_fallback_prompt:
                if fallback_prompt is None:
                    print(
                        f"   Final retry {k}: switching init phrase to "
                        f"{fallback_init_phrase!r} to anchor text generation."
                    )
                    fallback_prompt = self.vlm.preprocess(image_path, fallback_init_phrase)
                attempt_prompt = fallback_prompt
                attempt_init_phrase = fallback_init_phrase
            else:
                attempt_prompt = final_prompt
                attempt_init_phrase = self.init_phrase

            caption = self.vlm.generate(
                attempt_prompt, debug=(debug_this_sample and k == 0), seed=seed
            )
            used_init_phrase = attempt_init_phrase

            metrics = {}
            if self.metrics and pil_image is not None:
                try:
                    metrics = self._compute_metrics({"image": pil_image, "caption": caption})
                except Exception as e:
                    print(f"   Warning: metrics failed on attempt {k}: {e}")

            attempts_log.append(
                {
                    "attempt": k,
                    "seed": seed,
                    self.CLIP_METRIC: metrics.get(self.CLIP_METRIC),
                    "init_phrase": attempt_init_phrase,
                }
            )

            if not self._retry_enabled:
                break
            clip = metrics.get(self.CLIP_METRIC)
            if clip is None:
                print(f"   CLIP unavailable on attempt {k}; stopping retries.")
                break
            if clip >= self.retry_clip_threshold:
                if k > 0:
                    print(f"   Accepted on retry {k} (clip={clip:.4f} >= {self.retry_clip_threshold}).")
                break
            if k < self.retry_max_attempts - 1:
                print(
                    f"   Retry {k + 1}/{self.retry_max_attempts - 1}: "
                    f"clip={clip:.4f} < {self.retry_clip_threshold}, re-rolling."
                )
            else:
                print(
                    f"   Retry budget exhausted ({self.retry_max_attempts} attempts); "
                    f"keeping last caption with clip={clip:.4f}."
                )

        retry_stats: Optional[Dict[str, Any]] = None
        if self._retry_enabled:
            retry_stats = {
                "threshold": self.retry_clip_threshold,
                "max_attempts": self.retry_max_attempts,
                "base_seed": self.retry_base_seed,
                "num_attempts": len(attempts_log),
                "attempts": attempts_log,
            }

        return {
            "caption": caption,
            "init_phrase": used_init_phrase,
            "metrics": metrics,
            "retry_stats": retry_stats,
        }
