"""
HuggingFace inference backend for VLM benchmarks.
Provides the same interface as VLLMInferencer but uses HuggingFace transformers directly.
This is slower but more compatible with models not yet supported by vLLM.
"""

import logging
from typing import Dict, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import BaseInferencer

logger = logging.getLogger(__name__)


class HFInferencer(BaseInferencer):
    """
    HuggingFace inference class with the same interface as VLLMInferencer.
    Provides compatibility with models not yet supported by vLLM (e.g., Apertus on older vLLM versions).
    """

    REQUIRED_ARGS = ["model_path"]
    OPTIONAL_ARGS = {
        "tokenizer_path": None,
        "max_seq_len": 8192,
        "model_dtype": "auto",
        "device": "cuda",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        # Map dtype string to torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.model_dtype, "auto")

        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        self._txt_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)

        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self.model.eval()

        logger.info(f"HFInferencer initialized on {self.device}")

    @property
    def txt_tokenizer(self):
        return self._txt_tokenizer

    def run_inference(
        self,
        prompt: Union[str, List[int]],
        return_text: bool = False,
        sampling_tmp: float = 0.3,
        sampling_topp: float = 0.95,
        sampling_max_tok: int = 500,
        sampling_min_tok: int = 3,
        sampling_stop_token_ids: List[int] = None,
        debug: bool = False,
    ) -> Dict:
        """
        Run inference on the given prompt. Same signature as VLLMInferencer.run_inference.

        Args:
            prompt: Input prompt as string or list of token IDs
            return_text: Whether to decode output tokens to text
            sampling_tmp: Sampling temperature
            sampling_topp: Top-p sampling parameter
            sampling_max_tok: Maximum tokens to generate
            sampling_min_tok: Minimum tokens to generate
            sampling_stop_token_ids: Token IDs that stop generation
            debug: If True, print prompt token IDs and decoded text

        Returns:
            Dictionary with 'generated_ids' and optionally 'generated_text'
        """
        # Convert string prompt to token IDs if needed
        if isinstance(prompt, str):
            # Check if prompt already has special tokens (from chat template)
            has_special_tokens = self._txt_tokenizer.bos_token is not None and prompt.startswith(
                self._txt_tokenizer.bos_token
            )

            if has_special_tokens:
                # Chat template already added special tokens, don't add again
                input_ids = self._txt_tokenizer.encode(prompt, add_special_tokens=False)
            else:
                # No chat template used, add BOS
                input_ids = self._txt_tokenizer.encode(prompt, add_special_tokens=True)

            # Remove EOS token if present at the end (we want to generate, not stop)
            if input_ids and input_ids[-1] == self._txt_tokenizer.eos_token_id:
                input_ids = input_ids[:-1]
        else:
            input_ids = prompt

        # Debug: Print prompt token IDs and text
        if debug:
            print("\n" + "=" * 80)
            print("[HF INFERENCER DEBUG] Prompt Token IDs:")
            print(f"  Total prompt length: {len(input_ids)} tokens")
            print(f"  First 10 token IDs: {input_ids[:10]}")
            print(f"  Last 10 token IDs: {input_ids[-10:]}")

            # Decode first and last 10 tokens
            first_10_text = self._txt_tokenizer.decode(input_ids[:10], skip_special_tokens=False)
            last_10_text = self._txt_tokenizer.decode(input_ids[-10:], skip_special_tokens=False)
            print(f"\n  First 10 tokens as text:\n    {repr(first_10_text)}")
            print(f"\n  Last 10 tokens as text:\n    {repr(last_10_text)}")
            print("=" * 80 + "\n")

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        prompt_len = input_tensor.shape[1]

        # Build generation config
        gen_config = GenerationConfig(
            max_new_tokens=sampling_max_tok,
            min_new_tokens=sampling_min_tok,
            temperature=sampling_tmp if sampling_tmp > 0 else None,
            top_p=sampling_topp if sampling_tmp > 0 else None,
            do_sample=sampling_tmp > 0,
            eos_token_id=sampling_stop_token_ids if sampling_stop_token_ids else self._txt_tokenizer.eos_token_id,
            pad_token_id=self._txt_tokenizer.pad_token_id or self._txt_tokenizer.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                generation_config=gen_config,
            )

        # Extract only generated tokens (exclude prompt)
        gen_ids = output[0, prompt_len:].tolist()

        return_dict = {
            "generated_ids": gen_ids,
        }

        if return_text:
            return_dict["generated_text"] = self._txt_tokenizer.decode(gen_ids, skip_special_tokens=False)

        return return_dict
