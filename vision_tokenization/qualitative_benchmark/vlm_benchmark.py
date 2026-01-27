"""
Script to generate qualitative results for VLM inference based on single-turn image prompts.
Prompts and images are loaded from json files. Results are stored in an experiment folder together with prompts and images.

For now only supports EMU3 inferencer.
"""

import argparse
import json
import os
import re

# Add paths for imports
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import PIL
import torch
from PIL import Image

from vision_tokenization.qualitative_benchmark.utils.prompt_formatter import CHAT_TRANFORMS, PromptFormatter
from vision_tokenization.qualitative_benchmark.utils.vllm_inferencer import VLLMInferencer
from vision_tokenization.qualitative_benchmark.v_tokenizers import VLMVisionTokenizer

base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "Tokenizer"))

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Set environment variables to prevent OOM and process issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use only GPU 0 for the LLM


class InferenceArgs:
    """Class to store inference arguments."""

    def __init__(
        self,
        apply_chat_template: bool,
        temperature: float,
        top_p: float,
        stop_token_ids: List[int],
        max_new_tokens: int,
        max_emu_aspect_ratio,
        min_emu_aspect_ratio,
        chat_transform: str = None,  # method out of registered ones in prompt_formatter to prepare input to chat template
        img_right: bool = False,  # Will image be after the prompt or before
    ):
        self.apply_chat_template = apply_chat_template
        self.img_right = img_right
        self.temperature = temperature
        self.top_p = top_p
        self.stop_token_ids = stop_token_ids
        self.max_new_tokens = max_new_tokens
        self.chat_transform = chat_transform
        self.max_emu_aspect_ratio = max_emu_aspect_ratio
        self.min_emu_aspect_ratio = min_emu_aspect_ratio


class VLM(object):
    """Class to initialize and run VLM inference with pluggable vision v_tokenizers and utils."""

    def __init__(
        self,
        vision_tokenizer: VLMVisionTokenizer,
        inferencer,
        inf_args: InferenceArgs,
        tokenizer_path: str,
        model_path: str,
    ):
        """
        Initialize VLM with vision tokenizer and inferencer.

        Args:
            vision_tokenizer: VLMVisionTokenizer instance for encoding images
            inferencer: VLMInferencer instance for running generation
            inf_args: InferenceArgs with generation parameters
            tokenizer_path: Path to text tokenizer
            model_path: Path to VLM model
        """
        self.vision_tokenizer = vision_tokenizer
        self.prompt_formatter = PromptFormatter(tokenizer_path)
        self.inferencer = inferencer
        self.inf_args = inf_args
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # Extract stop tokens from tokenizer
        stop_tokens = []
        if hasattr(self.inferencer.txt_tokenizer, "eos_token_id") and self.inferencer.txt_tokenizer.eos_token_id:
            stop_tokens.append(self.inferencer.txt_tokenizer.eos_token_id)
        self.inf_args.stop_token_ids = stop_tokens

        # Determine inferencer name for logging
        inferencer_name = type(self.inferencer).__name__
        print(f"VLM initialized with {self.vision_tokenizer.name} tokenizer and {inferencer_name}")

    def _load_image(self, image_path: str, resize: Tuple[int, int] = None):
        """Load image from path."""
        img = Image.open(image_path).convert("RGB")
        if resize is not None:
            img = img.resize(resize, Image.LANCZOS)
        return img

    def _prepare_image_text_tokens(self, img: Image.Image) -> str:
        """
        Encode image to vision tokens and format for VLM.

        Uses the vision tokenizer to encode the image and format it as a string
        suitable for insertion into the chat template.
        """
        # Encode image using vision tokenizer
        indices, metadata = self.vision_tokenizer.encode_for_vlm(img)

        # Log token dimensions if available
        if "height" in metadata and "width" in metadata:
            h, w = metadata["height"], metadata["width"]
            print(f"   Token dimensions: {h}×{w} = {metadata.get('num_tokens', h*w)} tokens")

        # Format tokens for chat using tokenizer-specific logic
        # EMU3 uses string tokens, not IDs, so we pass an empty dict
        img_tokens_str = self.vision_tokenizer.format_tokens_for_chat(indices, metadata, {})

        return img_tokens_str

    def _prepare_final_prompt(self, image_token_string, prompt):
        """
        Prepare final prompt for VLM inference.

        Uses PromptFormatter to create the chat input string.
        """
        if self.inf_args.apply_chat_template:
            # Apply chat template and replace <|image|> with actual image tokens
            formatted_prompt = self.prompt_formatter.prepare_chat_prompt(
                prompt,
                image_token_string,
                chat_transform=self.inf_args.chat_transform,
                add_generation_prompt=True,
                img_right=self.inf_args.img_right,
            )
        else:
            # Simple concatenation without chat template
            formatted_prompt = self.prompt_formatter.prepare_non_chat_prompt(
                prompt, image_token_string, img_right=self.inf_args.img_right
            )

        return formatted_prompt

    def preprocess(self, img_path, prompt):
        """Prepare image and prompt for VLM inference. Returns formatted string."""
        img = self._load_image(img_path)
        img_tokens_str = self._prepare_image_text_tokens(img)
        formatted_prompt_str = self._prepare_final_prompt(img_tokens_str, prompt)
        return formatted_prompt_str  # Return string, not token IDs

    def generate(self, prompt_string: str, debug: bool = False):
        """Run VLM inference on formatted prompt string."""
        result = self.inferencer.run_inference(
            prompt_string,
            return_text=True,
            sampling_tmp=self.inf_args.temperature,
            sampling_topp=self.inf_args.top_p,
            sampling_max_tok=self.inf_args.max_new_tokens,
            sampling_min_tok=1,
            sampling_stop_token_ids=self.inf_args.stop_token_ids,
            debug=debug,
        )
        return result["generated_text"]

    def generate_image_completion(
        self, image_path: str, given_percentage: int, strict_row_count: bool = False, debug: bool = False
    ) -> Dict[str, Any]:
        """
        Generate image completion from partial image tokens.

        Args:
            image_path: Path to the image file
            given_percentage: Percentage of image rows to provide as context (e.g., 20, 40, 60, 80)
            strict_row_count: If True, require exact row count match for validity
            debug: If True, print detailed token information

        Returns:
            Dictionary with:
                - given_indices: List of token indices provided as context
                - generated_indices: List of generated token indices
                - original_image_rows: Total rows in original image
                - given_rows: Number of rows given as context
                - generated_rows: Number of rows generated
                - expected_rows: Number of rows expected
                - is_valid: Whether generation is valid
                - validation: Detailed validation breakdown
                - statistics: Token counts and special token counts
                - metadata: Image token dimensions
        """
        img = self._load_image(image_path)
        indices, metadata = self.vision_tokenizer.encode_for_vlm(img)

        height = metadata["height"]
        width = metadata["width"]
        visual_indices = indices[0].flatten().tolist() if hasattr(indices[0], "flatten") else list(indices[0])

        print(f"   Original image tokenized: {height}×{width} = {len(visual_indices)} tokens")

        given_rows = max(1, int(height * given_percentage / 100))
        expected_rows = height - given_rows

        print(f"   Using {given_percentage}%: {given_rows} given rows, {expected_rows} expected rows")

        # Get given indices for debug
        given_indices = visual_indices[: given_rows * width]

        # Debug: Print last few tokens of given rows
        if debug:
            print(f"\n   [DEBUG] Given rows: {given_rows}")
            print(f"   [DEBUG] Last 10 tokens of given rows: {given_indices[-10:]}")

        prompt = self._create_partial_image_prompt(visual_indices, height, width, given_rows)

        # 4. Generate completion
        max_tokens = expected_rows * width + 200  # Extra tokens for structure tokens
        result = self.inferencer.run_inference(
            prompt,
            return_text=False,  # We need token IDs, not text
            sampling_tmp=self.inf_args.temperature,
            sampling_topp=self.inf_args.top_p,
            sampling_max_tok=max_tokens,
            sampling_min_tok=1,
            sampling_stop_token_ids=self.inf_args.stop_token_ids,
            debug=debug,
        )

        generated_token_ids = result["generated_ids"]
        print(f"   Generated {len(generated_token_ids)} tokens")

        # Debug: Print generated token IDs and text
        if debug:
            # Decode tokens to text
            decoded_text = self.inferencer.txt_tokenizer.decode(
                generated_token_ids, skip_special_tokens=False
            )
            print(f"\n   [DEBUG] First 10 generated token IDs: {generated_token_ids[:10]}")
            print(f"   [DEBUG] First 100 text characters: {decoded_text[:100]}")
            print(f"   [DEBUG] Last 10 generated token IDs: {generated_token_ids[-10:]}")
            print(f"   [DEBUG] Last 100 generated textcharacters: {decoded_text[-100:]}")

        # 5. Extract and validate generated tokens
        from emu3_reconstruct_helper import extract_visual_tokens_by_row

        # Get special token IDs from inferencer
        special_token_ids = self._get_special_token_ids()

        # Extract tokens by row
        rows_generated, stats = extract_visual_tokens_by_row(
            generated_token_ids, self.vision_tokenizer.vision_mapping, special_token_ids
        )

        # Flatten generated indices
        generated_indices = [idx for row in rows_generated for idx in row]

        # Debug: Print extracted visual indices
        if debug:
            print(f"\n   [DEBUG] Number of rows extracted: {len(rows_generated)}")
            print(f"   [DEBUG] First 20 extracted visual indices: {generated_indices[:20]}")
            print(f"   [DEBUG] Last 20 extracted visual indices: {generated_indices[-20:]}")

        # 6. Validate the completion
        validation_result = self._validate_completion(
            generated_token_ids, special_token_ids, expected_rows, len(rows_generated), strict_row_count
        )

        return {
            "given_indices": given_indices,
            "generated_indices": generated_indices,
            "original_image_rows": height,
            "original_image_width": width,
            "given_rows": given_rows,
            "generated_rows": len(rows_generated),
            "expected_rows": expected_rows,
            "is_valid": validation_result["is_valid"],
            "validation": validation_result["validation"],
            "statistics": {
                "given_tokens": len(given_indices),
                "generated_tokens": len(generated_indices),
                "expected_tokens": expected_rows * width,
                "total_tokens": len(given_indices) + len(generated_indices),
                "unique_generated_tokens": len(set(generated_indices)),
                **validation_result["statistics"],
            },
            "metadata": metadata,
        }

    def _create_partial_image_prompt(self, visual_indices: List[int], height: int, width: int, given_rows: int) -> str:
        """
        Create a prompt with partial image tokens.

        Args:
            visual_indices: Full list of visual token indices
            height: Height in tokens
            width: Width in tokens
            given_rows: Number of rows to include in prompt

        Returns:
            Formatted prompt string
        """
        return self.vision_tokenizer.create_partial_prompt(visual_indices, height, width, given_rows)

    def _get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs from the inferencer."""
        # This will vary by tokenizer, but for EMU3:
        tokenizer = self.inferencer.txt_tokenizer
        special_tokens = {}

        # Try to get EMU3-specific tokens
        token_names = ["img_start", "img_end", "img_token_start", "img_end_of_row", "img_end_of_frame"]
        for token_name in token_names:
            token_str = f"<|{token_name}|>"
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
                    special_tokens[token_name] = token_id

        return special_tokens

    def _validate_completion(
        self,
        generated_token_ids: List[int],
        special_token_ids: Dict[str, int],
        expected_rows: int,
        generated_rows: int,
        strict_row_count: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate image completion generation.

        Args:
            generated_token_ids: Generated token IDs
            special_token_ids: Dictionary of special token IDs
            expected_rows: Expected number of rows
            generated_rows: Actually generated number of rows
            strict_row_count: If True, require exact row count match

        Returns:
            Dictionary with validation results
        """
        from emu3_reconstruct_helper import extract_visual_tokens_by_row

        # Extract tokens by row to check consistency
        rows, stats = extract_visual_tokens_by_row(
            generated_token_ids, self.vision_tokenizer.vision_mapping, special_token_ids
        )

        # Count special tokens
        eol_count = generated_token_ids.count(special_token_ids.get("img_end_of_row", -1))
        eof_count = generated_token_ids.count(special_token_ids.get("img_end_of_frame", -1))
        eoi_count = generated_token_ids.count(special_token_ids.get("img_end", -1))

        # Check all criteria
        structure_consistent = stats["consistent"]  # All rows same width
        has_proper_eol = eol_count == generated_rows
        has_eof = eof_count >= 1
        has_eoi = eoi_count >= 1
        row_count_match = generated_rows == expected_rows

        # Check token order (EOF before EOI)
        eof_idx = -1
        eoi_idx = -1
        try:
            if eof_count > 0:
                eof_idx = generated_token_ids.index(special_token_ids["img_end_of_frame"])
            if eoi_count > 0:
                eoi_idx = generated_token_ids.index(special_token_ids["img_end"])
        except (ValueError, KeyError):
            pass

        proper_order = (eof_idx < eoi_idx) if (eof_idx >= 0 and eoi_idx >= 0) else False

        # Build list of required checks
        required_checks = [structure_consistent, has_proper_eol, has_eof, has_eoi, proper_order]

        # Only require row count match if strict_row_count flag is set
        if strict_row_count:
            required_checks.append(row_count_match)

        is_valid = all(required_checks)

        return {
            "is_valid": is_valid,
            "validation": {
                "structure_consistent": structure_consistent,
                "has_proper_eol_tokens": has_proper_eol,
                "has_eof_token": has_eof,
                "has_eoi_token": has_eoi,
                "proper_token_order": proper_order,
                "row_count_match": row_count_match,  # Always recorded
            },
            "statistics": {"eol_count": eol_count, "eof_count": eof_count, "eoi_count": eoi_count},
        }


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
                        result_image_path = self._save_result_image(
                            completion_result,
                            image_path,
                            percentage,
                            images_dir,
                        )
                        result_entry["result_image_path"] = str(
                            Path(experiment_name) / "completion_images" / Path(result_image_path).name
                        )
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
    ) -> Path:
        """
        Reconstruct image and save with red boundary line.
        Expects a valid formatted image (equally sized full rows).

        Args:
            completion_result: Result from generate_image_completion()
            original_image_path: Path to original image
            percentage: Completion percentage
            images_dir: Directory to save result images

        Returns:
            Path to saved result image
        """
        from PIL import ImageDraw

        # Combine given and generated indices
        given_indices = completion_result["given_indices"]
        generated_indices = completion_result["generated_indices"]
        width = completion_result["original_image_width"]  # width expected to be the same to be valid!
        given_rows = completion_result["given_rows"]

        # We reconstruct with all shapes that are valid (method assumes valid shapes)
        # Specifically we don't pad with 0s or truncate extra rows
        all_indices = given_indices + generated_indices

        # Calculate actual height from combined data
        actual_height = len(all_indices) // width
        assert len(all_indices) == actual_height * width

        # Reconstruct image using vision tokenizer
        tokenizer = self.vlm.vision_tokenizer.tokenizer
        indices_tensor = torch.tensor(all_indices, dtype=torch.long, device=tokenizer.device)
        indices_tensor = indices_tensor.reshape(1, actual_height, width)

        with torch.no_grad():
            reconstructed_tensor = tokenizer.decode(indices_tensor)

        reconstructed_pil = tokenizer.postprocess(reconstructed_tensor)

        # Draw red line at boundary
        # Calculate pixels per token row dynamically (varies by tokenizer, e.g., 8 for emu3, 16 for emu3.5)
        pixels_per_row = reconstructed_pil.height / actual_height
        boundary_y = int(given_rows * pixels_per_row)
        draw = ImageDraw.Draw(reconstructed_pil)
        img_width = reconstructed_pil.width
        draw.line([(0, boundary_y), (img_width, boundary_y)], fill="red", width=3)

        # Save image
        image_stem = Path(original_image_path).stem
        result_filename = f"{image_stem}_{percentage}pct.png"
        result_path = images_dir / result_filename
        reconstructed_pil.save(result_path)

        return result_path


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
    parser.add_argument("--results_folder", type=str, default="results/", help="Path to save results")
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
        default=512 * 512,
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

    if inferencer_type == "vllm":
        inferencer = VLLMInferencer(
            model_path=args.model_path, tokenizer_path=args.tokenizer_path, tp_size=1, max_seq_len=8192
        )
    elif inferencer_type == "hf":
        from vision_tokenization.qualitative_benchmark.utils.hf_inferencer import HFInferencer

        inferencer = HFInferencer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=8192,
            device="cuda:0",
        )
    else:
        raise ValueError(f"Unknown inferencer type: {inferencer_type}")

    # Create inference args
    # For captioning mode, disable chat template by default (unless explicitly enabled)
    apply_chat = not args.no_chat_template
    if args.captioning and not args.no_chat_template:
        # Captioning mode defaults to no chat template
        apply_chat = False
        print("Note: Chat template disabled for captioning mode (use explicit prompts if needed)")

    inference_args = InferenceArgs(
        apply_chat_template=apply_chat,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=[],  # Will be set by VLM __init__
        max_new_tokens=args.max_new_tokens,
        max_emu_aspect_ratio=max_pixels,
        min_emu_aspect_ratio=min_pixels,
        chat_transform=args.chat_format or None,
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

    # Validate argument combinations
    validate_args(args)

    # Setup results directory and output file path
    results_dir = Path(args.results_folder)
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

    # Check which benchmark mode to run
    if args.image_completion:
        # Parse percentages
        percentages = [int(p.strip()) for p in args.completion_percentages.split(",")]
        print(f"Running image completion benchmark with percentages: {percentages}")
        print(f"Strict row count: {args.strict_row_count}")
        if args.debug:
            print("Debug mode enabled (will print tokens for first 3 samples)")

        # Create and run image completion benchmark
        benchmark = ImageCompletionBenchmark(
            images_config_path=args.image_list, vlm=vlm, results_dir=str(results_dir), debug=args.debug
        )
        results = benchmark.run_completion_benchmark(
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

        # Create and run captioning benchmark
        benchmark = CaptioningBenchmark(
            images_config_path=args.image_list,
            vlm=vlm,
            results_dir=str(results_dir),
            init_phrase=args.caption_init_phrase,
            debug=args.debug,
        )
        results = benchmark.run_captioning_benchmark(output_filename=f"{args.experiment_name}.json")
    else:
        # Run standard VLM Q&A benchmark
        if args.debug:
            print("Debug mode enabled (will print tokens for first 3 samples)")

        benchmark = VLMBenchmark(
            images_config_path=args.image_list,
            prompts_config_path=args.prompt_list,
            vlm=vlm,
            results_dir=str(results_dir),
            debug=args.debug,
        )
        results = benchmark.run_benchmark(output_filename=f"{args.experiment_name}.json")

    print("Done!")
