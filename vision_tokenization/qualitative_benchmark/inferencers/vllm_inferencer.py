"""
Simple wrapper for VLLM hosted LLM. Takes model path, load the model into VLLM and offers interfaces for inference.
"""

import logging
from typing import List, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logger = logging.getLogger(__name__)


class VLLMInferencer:
    """
    Simple VLLM inference class. Can be initialized with or without tokenizer.
    If tokenizer not given, only inference from token ids is allowed.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        tp_size: int = 1,
        max_seq_len: int = 8192,
        model_dtype: str = "auto",
        model_quantization: str = None,
        seed: int = None,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        if tokenizer_path is not None:
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.txt_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.warning(
                "No tokenizer given, vllm will initialize from model tokenizer which might be unintended or fail!"
            )
            self.txt_tokenizer = None

        logger.info(f"Loading model from {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_seq_len,
            skip_tokenizer_init=self.txt_tokenizer is not None,
            dtype=model_dtype,
            quantization=model_quantization,
            seed=seed,
        )

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
    ) -> dict:
        """
        Takes a string/token ids as input and runs inference on it. No chat template is applied.
        If return_text is True, tries to decode the output tokens to text before return.
        If no txt tokenizer given always returns only generated ids.
        Can only work with one sample currently!

        Args:
            prompt: Input prompt as string or list of token IDs
            return_text: Whether to decode output tokens to text
            sampling_tmp: Sampling temperature
            sampling_topp: Top-p sampling parameter
            sampling_max_tok: Maximum tokens to generate
            sampling_min_tok: Minimum tokens to generate
            sampling_stop_token_ids: Token IDs that stop generation
            debug: If True, print prompt token IDs and decoded text
        """
        sampling_params = SamplingParams(
            temperature=sampling_tmp,
            top_p=sampling_topp,
            max_tokens=sampling_max_tok,
            min_tokens=sampling_min_tok,
            stop_token_ids=sampling_stop_token_ids,
            skip_special_tokens=False,
        )

        if isinstance(prompt, str):
            if self.txt_tokenizer is not None:
                # Check if prompt already has special tokens (from chat template)
                # Chat templates include BOS token as text like "<|begin_of_text|>"
                has_special_tokens = self.txt_tokenizer.bos_token is not None and prompt.startswith(
                    self.txt_tokenizer.bos_token
                )

                if has_special_tokens:
                    # Chat template already added special tokens, don't add again
                    prompt = self.txt_tokenizer.encode(prompt, add_special_tokens=False)
                else:
                    # No chat template used, add BOS but strip EOS since we're generating
                    prompt = self.txt_tokenizer.encode(prompt, add_special_tokens=True)
                    # Remove EOS token if present at the end
                # sanity check: remove eod token if exists as we want to generate
                if prompt and prompt[-1] == self.txt_tokenizer.eos_token_id:
                    prompt = prompt[:-1]
            else:
                logger.warning(
                    "Specific Tokenizer(VLLMInferencer.txt_tokenizer) not given, vllm will try to use default model tokenizer which might be unintended or fail!"
                )

        # Debug: Print prompt token IDs and text
        if debug and isinstance(prompt, list) and self.txt_tokenizer is not None:
            print("\n" + "=" * 80)
            print("[VLLM INFERENCER DEBUG] Prompt Token IDs:")
            print(f"  Total prompt length: {len(prompt)} tokens")
            print(f"  First 10 token IDs: {prompt[:10]}")
            print(f"  Last 10 token IDs: {prompt[-10:]}")

            # Decode first and last 10 tokens
            first_10_text = self.txt_tokenizer.decode(prompt[:10], skip_special_tokens=False)
            last_10_text = self.txt_tokenizer.decode(prompt[-10:], skip_special_tokens=False)
            print(f"\n  First 10 tokens as text:\n    {repr(first_10_text)}")
            print(f"\n  Last 10 tokens as text:\n    {repr(last_10_text)}")
            print("=" * 80 + "\n")

        # Wrap token IDs in TokensPrompt for vLLM
        if isinstance(prompt, list):
            prompt = TokensPrompt(prompt_token_ids=prompt)

        # Single sample output
        output = self.llm.generate([prompt], sampling_params=sampling_params)[0]

        gen_ids = output.outputs[0].token_ids

        return_dict = {
            "generated_ids": gen_ids,
        }

        if return_text:
            if self.txt_tokenizer is not None:
                return_dict["generated_text"] = self.txt_tokenizer.decode(gen_ids, skip_special_tokens=False)
            else:
                return_dict["generated_text"] = output.outputs[0].text

        return return_dict
