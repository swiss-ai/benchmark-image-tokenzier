from typing import List, Tuple, Dict, Any

from PIL import Image

from vision_tokenization.qualitative_benchmark.utils.prompt_formatter import PromptFormatter
from vision_tokenization.qualitative_benchmark.v_tokenizers import VLMVisionTokenizer


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
        prompt_builder: str = None,  # registered prompt builder name, bypasses apply_chat_template entirely
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
        self.prompt_builder = prompt_builder

    def __repr__(self) -> str:
        """Return a string representation of the inference arguments."""
        return (
            f"InferenceArgs("
            f"apply_chat_template={self.apply_chat_template}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"max_emu_aspect_ratio={self.max_emu_aspect_ratio}, "
            f"min_emu_aspect_ratio={self.min_emu_aspect_ratio}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"chat_transform={self.chat_transform!r}, "
            f"img_right={self.img_right}, "
            f"prompt_builder={self.prompt_builder!r}"
            f")"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation of the inference arguments."""
        lines = [
            "Inference Configuration:",
            f"  Apply chat template: {self.apply_chat_template}",
            f"  Temperature: {self.temperature}",
            f"  Top-p: {self.top_p}",
            f"  Max new tokens: {self.max_new_tokens}",
            f"  Max EMU aspect ratio: {self.max_emu_aspect_ratio}",
            f"  Min EMU aspect ratio: {self.min_emu_aspect_ratio}",
            f"  Stop token IDs: {self.stop_token_ids}",
            f"  Chat transform: {self.chat_transform if self.chat_transform else 'None'}",
            f"  Image right: {self.img_right}",
            f"  Prompt builder: {self.prompt_builder if self.prompt_builder else 'None'}",
        ]
        return "\n".join(lines)


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

        # Extract stop tokens from tokenizer TODO: hardcoded for now, may later want to try to load from GenerationConfig
        stop_tokens = []
        if hasattr(self.inferencer.txt_tokenizer, "eos_token_id") and self.inferencer.txt_tokenizer.eos_token_id:
            stop_tokens.append(self.inferencer.txt_tokenizer.eos_token_id)
        if hasattr(self.inferencer.txt_tokenizer.init_kwargs, "sft_eot_token") and self.inferencer.txt_tokenizer.init_kwargs["sft_eot_token"]:
            stop_tokens.append(self.inferencer.txt_tokenizer.init_kwargs["sft_eot_token"])
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

    def _prepare_final_prompt(self, image_token_string, prompt) -> str:
        """
        Prepare final prompt for VLM inference.

        Uses PromptFormatter to create the chat input string.
        Priority: prompt_builder > apply_chat_template > simple concatenation.
        """
        if self.inf_args.prompt_builder is not None:
            # Custom prompt builder — bypasses apply_chat_template entirely
            formatted_prompt = self.prompt_formatter.prepare_custom_prompt(
                prompt,
                image_token_string,
                prompt_builder_name=self.inf_args.prompt_builder,
                img_right=self.inf_args.img_right,
            )
        elif self.inf_args.apply_chat_template:
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
            decoded_text = self.inferencer.txt_tokenizer.decode(generated_token_ids, skip_special_tokens=False)
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
            "original_all_indices": visual_indices,
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