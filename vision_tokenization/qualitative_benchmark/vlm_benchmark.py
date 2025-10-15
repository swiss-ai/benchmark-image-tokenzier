"""
Script to generate qualitative results for VLM inference based on single-turn image prompts.
Prompts and images are loaded from json files. Results are stored in an experiment folder together with prompts and images.

For now only supports EMU3 inferencer.
"""
import argparse
import json
import re
import time
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

# Add paths for imports
import sys
base_dir = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / 'Tokenizer'))

import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm


# Set environment variables to prevent OOM and process issues
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use only GPU 0 for the LLM


class InferenceArgs:
    """Class to store inference arguments."""
    def __init__(self, apply_chat_template: bool, temperature: float, top_p: float, stop_token_ids: List[int], max_new_tokens: int, max_emu_aspect_ratio, min_emu_aspect_ratio):
        self.apply_chat_template = apply_chat_template
        self.temperature = temperature
        self.top_p = top_p
        self.stop_token_ids = stop_token_ids
        self.max_new_tokens = max_new_tokens
        self.max_emu_aspect_ratio = max_emu_aspect_ratio
        self.min_emu_aspect_ratio = min_emu_aspect_ratio


class VLM(object):
    """Class to initialize and run VLM inference. TODO: Add abstract class"""
    def __init__(self, model_path: str, tokenizer_path: str, inf_args: InferenceArgs):
        """Initialize VLM with model and tokenizer paths."""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.inf_args = inf_args

        print("Initializing EMU3 Vision Tokenizer...")
        from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer
        self.emu3_tokenizer = Emu3VisionTokenizer(
            min_pixels=self.inf_args.min_emu_aspect_ratio,
            max_pixels=self.inf_args.max_emu_aspect_ratio
        )
        if torch.cuda.is_available():
            self.emu3_tokenizer.model = self.emu3_tokenizer.model.to("cuda:1")
            self.emu3_tokenizer.device = "cuda:1"

        print("Initializing EMU3 Inferencer...")
        from emu3_vllm_inferencer import EMU3Inferencer
        self.inferencer = EMU3Inferencer(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            tensor_parallel_size=1,  # 3B model fits on 1 GPU
            max_model_len=8192
        )

        self.inf_args.stop_token_ids = [self.inferencer.special_token_ids["end_of_turn"]]

    def _load_image(self, image_path: str, resize: Tuple[int, int] = None):
        """Load image from path."""
        img = Image.open(image_path).convert('RGB')
        if resize is not None:
            img = img.resize(resize, Image.LANCZOS)
        return img

    def _prepare_image_text_tokens(self, img):
        """Prepare image tokens as text to insert into chat for VLM inference."""
        img_tensor = self.emu3_tokenizer.preprocess(img)
        img_tensor = img_tensor.to(self.emu3_tokenizer.device)

        with torch.no_grad():
            indices, _ = self.emu3_tokenizer.encode(img_tensor)

        h, w = indices.shape[1], indices.shape[2]
        visual_indices = indices[0].flatten().cpu().tolist()
        print(f"   Token dimensions: {h}×{w} = {len(visual_indices)} tokens")

        img_tokens_str = f"<|img_start|>{h}*{w}<|img_token_start|>"

        # Add all image tokens
        for row in range(h):
            row_start = row * w
            row_end = row_start + w
            row_tokens = visual_indices[row_start:row_end]

            for token_idx in row_tokens:
                img_tokens_str += f"<|visual token {token_idx:06d}|>"
            img_tokens_str += "<|img_end_of_row|>"

        # End image and add text prompt
        img_tokens_str += "<|img_end_of_frame|><|img_end|>"
        return img_tokens_str

    def _prepare_final_prompt(self, image_token_string, prompt):
        """
        Prepare final prompt for VLM inference.
        Loads Image and text prompt. Puts Everything into chat template or vanilla format
        and encodes it into the list of tokens needed for inference.
        """
        if self.inf_args.apply_chat_template:
            my_prompt = [{
                "role": "user",
                "content": [
                    {"type": "image"},  # This becomes <|image|> token (ID 128263)
                    # {"type": "text", "text": "What do you see in this image?"}
                    {"type": "text", "text": prompt}
                ]
            }]
            msg_template = self.inferencer.tokenizer.apply_chat_template(
                my_prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Not applying the chat template, just add image token and text prompt one after another
            msg_template = f"<|image|>{prompt}"


        chat_text = re.sub(r"<\|image\|>", image_token_string, msg_template)
        return self.inferencer.tokenizer.encode(chat_text, add_special_tokens=not self.inf_args.apply_chat_template)

    def preprocess(self, img_path, prompt):
        """Prepare image and prompt for VLM inference."""
        img = self._load_image(img_path)
        img_tokens_str = self._prepare_image_text_tokens(img)
        tokens = self._prepare_final_prompt(img_tokens_str, prompt)
        return tokens

    def generate(self, tokens: List[int]):
        """Run VLM inference on given tokens."""
        result = self.inferencer.generate(
            tokens,
            max_tokens=self.inf_args.max_new_tokens,
            min_tokens=1,
            temperature=self.inf_args.temperature,  # Higher temperature for more creative text
            top_p=self.inf_args.top_p,
            stop_token_ids=self.inf_args.stop_token_ids,
        )
        generated_ids = result['generated_token_ids']
        decoded = self.inferencer.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return decoded


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
        with open(path, 'r') as f:
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
            "stop_token_ids": self.vlm.inf_args.stop_token_ids
        }

        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.vlm.model_path,
            "tokenizer_path": self.vlm.tokenizer_path,
            "inference_args": inference_args_dict,
            "total_runs": 0,
            "runs": []
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
                        "image": {
                            "path": image["path"],
                            "tags": image.get("tags", [])
                        },
                        "prompt": {
                            "text": prompt["text"],
                            "tags": prompt.get("tags", [])
                        },
                        "output": output
                    }

                    results["runs"].append(result_entry)
                    results["total_runs"] += 1

        # Save results
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_results_{timestamp}.json"

        output_path = self.results_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark complete! {results['total_runs']} runs saved to {output_path}")
        return results



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer supporting SFT and Images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--results_folder", type=str, default="results/", help="Path to save results")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of experiment, used for saving results")
    parser.add_argument("--image_list", type=str, default="images.json", help="Path to image list")
    parser.add_argument("--prompt_list", type=str, default="prompts.json", help="Path to prompt list")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results file if it exists")
    return parser.parse_args()


def setup_vlm_inferencer(args):
    inference_args = InferenceArgs(
        apply_chat_template=True,
        temperature=0.3,
        top_p=0.9,
        stop_token_ids=[],
        max_new_tokens=300,
        max_emu_aspect_ratio=512*512,
        min_emu_aspect_ratio=256*256
    )
    vlm = VLM(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        inf_args=inference_args
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
        images_config_path=args.image_list,
        prompts_config_path=args.prompt_list,
        vlm=vlm,
        results_dir=str(results_dir)
    )
    results = benchmark.run_benchmark(output_filename=f"{args.experiment_name}.json")
    print("Done!")