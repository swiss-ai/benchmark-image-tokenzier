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

import PIL
import numpy as np
import torch
from PIL import Image

from vision_tokenization.qualitative_benchmark.utils.prompt_formatter import PromptFormatter, CHAT_TRANFORMS
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
        chat_transform: str = None, # method out of registered ones in prompt_formatter to prepare input to chat template
        img_right: bool = False # Will image be after the prompt or before
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

    def __init__(self, vision_tokenizer: VLMVisionTokenizer, inferencer, inf_args: InferenceArgs,
                 tokenizer_path: str, model_path: str):
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
        if hasattr(self.inferencer.txt_tokenizer, 'eos_token_id') and self.inferencer.txt_tokenizer.eos_token_id:
            stop_tokens.append(self.inferencer.txt_tokenizer.eos_token_id)
        self.inf_args.stop_token_ids = stop_tokens

        print(f"VLM initialized with {self.vision_tokenizer.name} tokenizer and vLLM inferencer")

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
        if 'height' in metadata and 'width' in metadata:
            h, w = metadata['height'], metadata['width']
            print(f"   Token dimensions: {h}×{w} = {metadata.get('num_tokens', h*w)} tokens")

        # Format tokens for chat using tokenizer-specific logic
        # EMU3 uses string tokens, not IDs, so we pass an empty dict
        img_tokens_str = self.vision_tokenizer.format_tokens_for_chat(
            indices, metadata, {}
        )

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
                img_right=self.inf_args.img_right
            )
        else:
            # Simple concatenation without chat template
            formatted_prompt = self.prompt_formatter.prepare_non_chat_prompt(
                prompt,
                image_token_string,
                img_right=self.inf_args.img_right
            )

        return formatted_prompt

    def preprocess(self, img_path, prompt):
        """Prepare image and prompt for VLM inference. Returns formatted string."""
        img = self._load_image(img_path)
        img_tokens_str = self._prepare_image_text_tokens(img)
        formatted_prompt_str = self._prepare_final_prompt(img_tokens_str, prompt)
        return formatted_prompt_str  # Return string, not token IDs

    def generate(self, prompt_string: str):
        """Run VLM inference on formatted prompt string."""
        result = self.inferencer.run_inference(
            prompt_string,
            return_text=True,
            sampling_tmp=self.inf_args.temperature,
            sampling_topp=self.inf_args.top_p,
            sampling_max_tok=self.inf_args.max_new_tokens,
            sampling_min_tok=1,
            sampling_stop_token_ids=self.inf_args.stop_token_ids
        )
        return result["generated_text"]


class VLMBenchmark:
    """Scaffolding for running qualitative VLM benchmarks with images and prompts."""

    def __init__(self, images_config_path: str, prompts_config_path: str, vlm: VLM, results_dir: str = "results"):
        """
        Initialize the benchmark system.

        Args:
            images_config_path: Path to JSON file containing image configurations
            prompts_config_path: Path to JSON file containing prompt configurations
            results_dir: Directory where results will be stored
        """
        self.images = self._load_json(images_config_path)
        self.prompts = self._load_json(prompts_config_path)
        self.results_dir = Path(results_dir)
        self.vlm = vlm
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
            "inferencer": "vllm",
            "inference_args": inference_args_dict,
            "total_runs": 0,
            "runs": [],
        }

        # Match images with prompts based on tags
        for image in tqdm(self.images, desc="Processing images"):
            for prompt in self.prompts:
                require_all = prompt.get("require_all_tags", False)
                if self._tags_match(image.get("tags", []), prompt.get("tags", []), require_all):
                    # Run VLM generation
                    final_prompt = self.vlm.preprocess(image["path"], prompt["text"])
                    output = self.vlm.generate(final_prompt)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VLM benchmark with configurable vision v_tokenizers and utils"
    )

    # Core arguments
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to text tokenizer supporting SFT")
    parser.add_argument("--model_path", type=str, required=True, help="Path to VLM model")
    parser.add_argument("--chat-format", type=str, choices=list(CHAT_TRANFORMS.keys()), help="Chat format to use. Method to transform input into format hat works with a models chat template")
    parser.add_argument("--results_folder", type=str, default="results/", help="Path to save results")
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of experiment, used for saving results"
    )
    parser.add_argument("--image_list", type=str, default="images.json", help="Path to image list JSON")
    parser.add_argument("--prompt_list", type=str, default="prompts.json", help="Path to prompt list JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results file if it exists")

    # Vision tokenizer arguments
    tokenizer_group = parser.add_argument_group("vision tokenizer arguments", "Configuration for vision tokenizer")
    tokenizer_group.add_argument(
        "--vision-tokenizer-type",
        type=str,
        default="emu3",
        choices=["emu3", "emu3.5", "emu3.5-ibq"],
        help="Type of vision tokenizer to use (default: emu3)"
    )
    tokenizer_group.add_argument(
        "--vision-tokenizer-path",
        type=str,
        default=None,
        help="Path to vision tokenizer model (required for emu3.5)"
    )
    tokenizer_group.add_argument(
        "--min-tokenizer-pixels",
        type=int,
        default=256 * 256,
        help="Minimum pixel count for vision tokenizer (default: 65536)"
    )
    tokenizer_group.add_argument(
        "--max-tokenizer-pixels",
        type=int,
        default=512 * 512,
        help="Maximum pixel count for vision tokenizer (default: 262144)"
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

    return parser.parse_args()


def setup_vlm_inferencer(args):
    """
    Setup VLM with pluggable vision tokenizer and inferencer components.

    Handles backward compatibility with deprecated arguments.
    """
    from vision_tokenization.qualitative_benchmark.v_tokenizers import create_vision_tokenizer

    # Handle backward compatibility for pixel arguments
    min_pixels = args.min_tokenizer_pixels
    max_pixels = args.max_tokenizer_pixels

    if hasattr(args, 'min_emu_aspect_ratio') and args.min_emu_aspect_ratio is not None:
        print("WARNING: --min_emu_aspect_ratio is deprecated, use --min-tokenizer-pixels")
        min_pixels = args.min_emu_aspect_ratio
    if hasattr(args, 'max_emu_aspect_ratio') and args.max_emu_aspect_ratio is not None:
        print("WARNING: --max_emu_aspect_ratio is deprecated, use --max-tokenizer-pixels")
        max_pixels = args.max_emu_aspect_ratio

    # Create vision tokenizer
    print(f"\nCreating vision tokenizer: {args.vision_tokenizer_type}")

    vision_tokenizer_kwargs = {
        'min_pixels': min_pixels,
        'max_pixels': max_pixels,
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu'
    }

    # Add model_path for v_tokenizers that need it
    if args.vision_tokenizer_type in ['emu3.5', 'emu3.5-ibq']:
        if not args.vision_tokenizer_path:
            raise ValueError(
                f"--vision-tokenizer-path is required for {args.vision_tokenizer_type}"
            )
        vision_tokenizer_kwargs['model_path'] = args.vision_tokenizer_path
    elif args.vision_tokenizer_path:
        # Optional for emu3
        vision_tokenizer_kwargs['model_path'] = args.vision_tokenizer_path

    vision_tokenizer = create_vision_tokenizer(
        args.vision_tokenizer_type,
        **vision_tokenizer_kwargs
    )

    # Create inferencer
    print(f"Creating inferencer: vllm")
    inferencer = VLLMInferencer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        tp_size=1,
        max_seq_len=8192
    )

    # Create inference args
    inference_args = InferenceArgs(
        apply_chat_template=not args.no_chat_template,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=[],  # Will be set by VLM __init__
        max_new_tokens=args.max_new_tokens,
        max_emu_aspect_ratio=max_pixels,
        min_emu_aspect_ratio=min_pixels,
        chat_transform=CHAT_TRANFORMS[args.chat_format] if args.chat_format else None
    )

    # Create VLM with pluggable components
    vlm = VLM(
        vision_tokenizer=vision_tokenizer,
        inferencer=inferencer,
        inf_args=inference_args,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path
    )

    return vlm


if __name__ == "__main__":
    """
    Example usage:
    python vlm_benchmark.py --tokenizer_path /path/to/tokenizer --model_path /path/to/model --experiment_name my_experiment
    """
    args = parse_args()

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

    benchmark = VLMBenchmark(
        images_config_path=args.image_list, prompts_config_path=args.prompt_list, vlm=vlm, results_dir=str(results_dir)
    )
    results = benchmark.run_benchmark(output_filename=f"{args.experiment_name}.json")
    print("Done!")
