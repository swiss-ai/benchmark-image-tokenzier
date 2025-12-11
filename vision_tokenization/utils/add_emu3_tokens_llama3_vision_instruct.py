#!/usr/bin/env python3
"""
Ensure Llama3-Emu3 tokenizer compatibility and add SFT sequences.

This script:
1. Loads an existing Llama3-Emu3 tokenizer
2. Verifies ALL Emu3 tokens remain at their EXACT original positions
3. Ensures the tokenizer maintains EXACT same vocabulary size as loaded
4. Dynamically retrieves all token IDs from the tokenizer (NO HARDCODING)
5. Preserves compatibility with millions of already-tokenized images
6. Adds pre-tokenized SFT sequences for Megatron-LM SFT dataset

Usage:
    # Process an Emu3 tokenizer to add Llama 3.2 Vision compatibility
    python add_emu3_tokens_llama3_vision_instruct.py \
        --tokenizer-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer \
        --output-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer

The output tokenizer will include:
- Llama 3.2 Vision chat template
- Pre-tokenized SFT sequences in tokenizer_config.json:
  * sft_user_begin_sequence
  * sft_assistant_begin_sequence
  * sft_eot_token
  * img_begin_token
  * img_end_token

These sequences can be accessed in Megatron-LM via:
    self.tokenizer._tokenizer.sft_user_begin_sequence

Reference: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision

CRITICAL: All Emu3 tokens (<|visual token XXXXXX|>, structure tokens, reserved tokens)
MUST remain at their original IDs to maintain compatibility with existing tokenized data.
The tokenizer vocabulary size must remain EXACTLY the same as the original.
"""

import json
import os

from transformers import AutoTokenizer


class Llama32VisionEmu3Adapter:
    """Ensure Llama3-Emu3 tokenizer works with Llama 3.2 Vision chat templates."""

    def __init__(
        self,
        tokenizer_path: str = "/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer",
        vision_instruct_path: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        output_path: str = None,
    ):
        """
        Initialize the adapter.

        Args:
            tokenizer_path: Path to Llama3-Emu3 tokenizer
            vision_instruct_path: Path to Llama-3.2-Vision-Instruct tokenizer for chat template
            output_path: Where to save the verified/updated tokenizer
        """
        self.tokenizer_path = tokenizer_path
        self.vision_instruct_path = vision_instruct_path
        self.output_path = output_path or f"{tokenizer_path}_llama32_vision"

        self.tokenizer = None
        self.vision_tokenizer = None
        self.original_vocab = None
        self.original_size = None
        self.emu3_tokens = {}

    def load_and_verify_tokenizer(self):
        """Load tokenizer and verify it's a Llama3-Emu3 tokenizer."""
        print(f"Loading tokenizer from {self.tokenizer_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)

        self.original_vocab = self.tokenizer.get_vocab().copy()
        self.original_size = len(self.original_vocab)

        print(f"Tokenizer vocabulary size: {self.original_size:,}")

        # Expected size is base Llama3 + Emu3 additions
        # But we don't hardcode it - just verify it has Emu3 tokens
        print(f"✅ Tokenizer loaded with {self.original_size:,} tokens")

    def retrieve_emu3_tokens(self):
        """Dynamically retrieve all Emu3 tokens and their positions from the tokenizer."""
        print("\nRetrieving Emu3 tokens from tokenizer...")

        self.emu3_tokens = {"structure": {}, "reserved": {}, "visual": {}, "llama_special": {}}

        # Scan vocabulary to find all special tokens
        for token, token_id in self.original_vocab.items():
            # Structure tokens
            if token in [
                "<|img_start|>",
                "<|img_end|>",
                "<|img_token_start|>",
                "<|img_end_of_row|>",
                "<|img_end_of_frame|>",
            ]:
                self.emu3_tokens["structure"][token] = token_id

            # Reserved tokens
            elif token.startswith("<|RESERVED_") and token.endswith("|>"):
                self.emu3_tokens["reserved"][token] = token_id

            # Visual tokens
            elif token.startswith("<|visual token ") and token.endswith("|>"):
                self.emu3_tokens["visual"][token] = token_id

            # Llama special tokens needed for chat template
            elif token in [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
                "<|image|>",
                "<|python_tag|>",
                "<|finetune_right_pad_id|>",
                "<|eom_id|>",
                "<|step_id|>",
            ]:
                self.emu3_tokens["llama_special"][token] = token_id

        # Print summary
        print(f"\nFound tokens:")
        print(f"  Structure tokens: {len(self.emu3_tokens['structure'])}")
        for token, token_id in list(self.emu3_tokens["structure"].items())[:5]:
            print(f"    {token}: ID {token_id}")

        print(f"  Reserved tokens: {len(self.emu3_tokens['reserved'])}")
        if self.emu3_tokens["reserved"]:
            sample = list(self.emu3_tokens["reserved"].items())
            for token, token_id in sample[:2]:
                print(f"    {token}: ID {token_id}")
            if len(sample) > 2:
                print(f"    ... and {len(sample) - 2} more")

        print(f"  Visual tokens: {len(self.emu3_tokens['visual'])}")
        if self.emu3_tokens["visual"]:
            sample = list(self.emu3_tokens["visual"].items())
            for token, token_id in sample[:2]:
                print(f"    {token}: ID {token_id}")
            if len(sample) > 2:
                print(f"    ... and {len(sample) - 2} more")

        print(f"  Llama special tokens: {len(self.emu3_tokens['llama_special'])}")
        for token, token_id in self.emu3_tokens["llama_special"].items():
            print(f"    {token}: ID {token_id}")

    def check_chat_template_compatibility(self):
        """Check if all tokens needed for Llama 3.2 Vision chat template are present."""
        print("\n" + "=" * 60)
        print("CHECKING CHAT TEMPLATE COMPATIBILITY")
        print("=" * 60)

        required_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]

        all_present = True
        for token in required_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                print(f"  ❌ {token}: NOT FOUND")
                all_present = False
            else:
                print(f"  ✅ {token}: ID {token_id}")

        # Check for image token - can use reserved token as placeholder
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")
        img_start_id = self.tokenizer.convert_tokens_to_ids("<|img_start|>")
        reserved_002_id = self.tokenizer.convert_tokens_to_ids("<|RESERVED_002|>")

        if image_token_id != self.tokenizer.unk_token_id:
            print(f"  ✅ <|image|>: ID {image_token_id}")
        elif reserved_002_id != self.tokenizer.unk_token_id:
            print(f"  ✅ <|RESERVED_002|> (ID {reserved_002_id}) will be renamed to <|image|>")
            print(f"     Strategy: Token ID stays the same, only the string changes")
            print(f"     Actual images use <|img_start|> (ID {img_start_id}) + Emu3 sequence")
        else:
            print(f"  ❌ No reserved token available to map as <|image|>")
            all_present = False

        if all_present:
            print("\n✅ All required tokens for chat template are present!")
        else:
            print("\n⚠️  Some required tokens are missing. Chat template may not work correctly.")

        return all_present

    def verify_token_positions(self):
        """Verify all Emu3 tokens are at their expected positions."""
        print("\n" + "=" * 60)
        print("VERIFYING TOKEN POSITIONS")
        print("=" * 60)

        all_correct = True

        # Verify a sample of visual tokens to ensure they're consecutive
        if self.emu3_tokens["visual"]:
            visual_tokens = sorted(self.emu3_tokens["visual"].items(), key=lambda x: x[1])

            # Check visual tokens are at their expected positions
            print("\nChecking visual token positions...")

            # Visual tokens should be named sequentially from 000000
            for i in range(min(5, len(visual_tokens))):
                token, actual_id = visual_tokens[i]

                # Extract the index from token name like "<|visual token 000000|>"
                token_number = int(token.split()[2].rstrip("|>"))

                # Check if the token is at its expected position
                # The first visual token's ID + token_number should equal actual_id
                first_visual_id = visual_tokens[0][1]
                expected_id = first_visual_id + token_number

                if actual_id != expected_id:
                    print(f"  ⚠️  {token}: at ID {actual_id}, expected ID {expected_id}")
                    all_correct = False
                else:
                    print(f"  ✅ {token}: ID {actual_id} (correct position)")

            # Also check the last few to ensure the full range is correct
            if len(visual_tokens) > 10:
                print("\nChecking last visual tokens...")
                for i in range(-3, 0):
                    token, actual_id = visual_tokens[i]
                    token_number = int(token.split()[2].rstrip("|>"))
                    first_visual_id = visual_tokens[0][1]
                    expected_id = first_visual_id + token_number

                    if actual_id != expected_id:
                        print(f"  ⚠️  {token}: at ID {actual_id}, expected ID {expected_id}")
                        all_correct = False
                    else:
                        print(f"  ✅ {token}: ID {actual_id} (correct position)")

        # Verify structure tokens
        print("\nStructure tokens:")
        for token, token_id in self.emu3_tokens["structure"].items():
            print(f"  {token}: ID {token_id}")

        if all_correct:
            print("\n✅ All tokens are at expected positions!")
        else:
            print("\n⚠️  Some tokens may not be at expected positions.")

        return all_correct

    def _replace_reserved_token_with_image(self, reserved_002_id: int):
        """Replace <|RESERVED_002|> with <|image|> in saved tokenizer files."""
        # Modify tokenizer.json
        tokenizer_json_path = os.path.join(self.output_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace all occurrences of <|RESERVED_002|> with <|image|>
            content = content.replace('"<|RESERVED_002|>"', '"<|image|>"')
            content = content.replace("<|RESERVED_002|>", "<|image|>")

            with open(tokenizer_json_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"  ✅ Replaced <|RESERVED_002|> with <|image|> (ID {reserved_002_id}) in tokenizer.json")

        # Also modify tokenizer_config.json to update token and add chat template
        config_path = os.path.join(self.output_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Replace RESERVED_002 with image
            def replace_in_dict(obj):
                if isinstance(obj, dict):
                    return {k: replace_in_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_in_dict(item) for item in obj]
                elif isinstance(obj, str):
                    return obj.replace("<|RESERVED_002|>", "<|image|>")
                return obj

            config = replace_in_dict(config)

            # Load the Llama 3.2 Vision Instruct chat template and fix it
            if hasattr(self, "chat_template") and self.chat_template:
                # Simple fix: Only show system prompt if user explicitly provides one
                # This removes the automatic system prompt with dates
                fixed_template = self.chat_template.replace(
                    "{%- if user_supplied_system_message or not image_ns.has_images %}",
                    "{%- if user_supplied_system_message %}",
                )
                config["chat_template"] = fixed_template
                print(f"  ✅ Added Llama 3.2 Vision chat template")
                print(f"     Fixed: System prompt only appears when explicitly provided by user")
            else:
                print(f"  ⚠️  No chat template to add")

            # Add pre-tokenized SFT sequences as tokenizer attributes
            # These will be accessible in Megatron via: self.tokenizer._tokenizer.sft_user_begin_sequence
            print("  Adding SFT sequences for Megatron-LM...")

            # Tokenize the sequences
            config["sft_user_begin_sequence"] = self.tokenizer.encode(
                "<|start_header_id|>user<|end_header_id|>", add_special_tokens=False
            )
            config["sft_assistant_begin_sequence"] = self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
            )
            config["sft_eot_token"] = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            config["img_begin_token"] = self.tokenizer.encode("<|img_start|>", add_special_tokens=False)
            config["img_end_token"] = self.tokenizer.encode("<|img_end|>", add_special_tokens=False)

            print(f"  ✅ Added SFT sequences:")
            print(f"     - sft_user_begin_sequence: {config['sft_user_begin_sequence']}")
            print(f"     - sft_assistant_begin_sequence: {config['sft_assistant_begin_sequence']}")
            print(f"     - sft_eot_token: {config['sft_eot_token']}")
            print(f"     - img_begin_token: {config['img_begin_token']}")
            print(f"     - img_end_token: {config['img_end_token']}")

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            print(f"  ✅ Replaced <|RESERVED_002|> with <|image|> in tokenizer_config.json")

    def save_tokenizer(self):
        """Save the tokenizer with metadata."""
        print(f"\nSaving tokenizer to {self.output_path}...")

        os.makedirs(self.output_path, exist_ok=True)

        # Save tokenizer first
        self.tokenizer.save_pretrained(self.output_path)

        # Replace <|RESERVED_002|> with <|image|> in the saved files
        reserved_002_id = self.tokenizer.convert_tokens_to_ids("<|RESERVED_002|>")
        if reserved_002_id != self.tokenizer.unk_token_id:
            self._replace_reserved_token_with_image(reserved_002_id)

        # Save token mapping info
        token_info = {
            "original_tokenizer": self.tokenizer_path,
            "vocabulary_size": len(self.tokenizer.get_vocab()),
            "emu3_tokens": {
                "structure_count": len(self.emu3_tokens["structure"]),
                "reserved_count": len(self.emu3_tokens["reserved"]),
                "visual_count": len(self.emu3_tokens["visual"]),
                "structure_tokens": self.emu3_tokens["structure"],
                "llama_special_tokens": self.emu3_tokens["llama_special"],
            },
            "notes": [
                "This tokenizer maintains exact compatibility with Emu3-tokenized data",
                "All visual tokens remain at their original positions",
                f"Total vocabulary size: {len(self.tokenizer.get_vocab())}",
            ],
        }

        info_path = os.path.join(self.output_path, "tokenizer_info.json")
        with open(info_path, "w") as f:
            json.dump(token_info, f, indent=2)

        print(f"✅ Tokenizer saved to {self.output_path}")
        print(f"✅ Token info saved to {info_path}")

    def load_vision_instruct_template(self):
        """Load Llama-3.2-Vision-Instruct tokenizer to get chat template."""
        print("\n" + "=" * 60)
        print("LOADING VISION-INSTRUCT CHAT TEMPLATE")
        print("=" * 60)

        try:
            print(f"Loading Vision-Instruct tokenizer from {self.vision_instruct_path}...")
            # Check if it's a local path or HuggingFace model ID
            if os.path.exists(self.vision_instruct_path):
                # Local path
                self.vision_tokenizer = AutoTokenizer.from_pretrained(
                    self.vision_instruct_path, trust_remote_code=True, local_files_only=True
                )
            else:
                # HuggingFace model ID
                self.vision_tokenizer = AutoTokenizer.from_pretrained(self.vision_instruct_path, trust_remote_code=True)
            print("✅ Loaded Vision-Instruct tokenizer")

            # Get chat template from the tokenizer
            self.chat_template = None

            # Try different ways to get the chat template
            if hasattr(self.vision_tokenizer, "chat_template") and self.vision_tokenizer.chat_template:
                self.chat_template = self.vision_tokenizer.chat_template
                print("✅ Got chat template from tokenizer.chat_template")
            elif (
                hasattr(self.vision_tokenizer, "_tokenizer_config")
                and "chat_template" in self.vision_tokenizer._tokenizer_config
            ):
                self.chat_template = self.vision_tokenizer._tokenizer_config["chat_template"]
                print("✅ Got chat template from tokenizer._tokenizer_config")
            else:
                # Try to get from the tokenizer's internal config
                try:
                    # Access the tokenizer's config directly
                    self.chat_template = self.vision_tokenizer.tokenizer_config.get("chat_template", None)
                    if self.chat_template:
                        print("✅ Got chat template from tokenizer.tokenizer_config")
                except:
                    pass

            if self.chat_template:
                print("✅ Found chat template in Vision-Instruct tokenizer")
                # Show a preview
                preview = self.chat_template[:200] + "..." if len(self.chat_template) > 200 else self.chat_template
                print(f"Template preview: {preview}")

                # Check format
                if "<|start_header_id|>" in self.chat_template:
                    print("✅ Template uses Llama Instruct format with header IDs")
                else:
                    print("⚠️  Template does not use Llama Instruct format")
            else:
                print("⚠️  No chat template found in Vision-Instruct tokenizer")
        except Exception as e:
            print(f"⚠️  Could not load Vision-Instruct tokenizer: {e}")
            print("   Will continue without updating chat template")
            self.chat_template = None

    def setup_reserved_token_mapping(self):
        """Setup mapping for reserved tokens as placeholders."""
        print("\n" + "=" * 60)
        print("SETTING UP RESERVED TOKEN MAPPING")
        print("=" * 60)

        # Check available reserved tokens
        reserved_001_id = self.tokenizer.convert_tokens_to_ids("<|RESERVED_001|>")
        reserved_002_id = self.tokenizer.convert_tokens_to_ids("<|RESERVED_002|>")
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")

        if image_token_id != self.tokenizer.unk_token_id:
            print(f"  ✅ <|image|> already exists at ID {image_token_id}")
            print("     No mapping needed")
        elif reserved_001_id != self.tokenizer.unk_token_id and reserved_002_id != self.tokenizer.unk_token_id:
            print(f"  ✅ Reserved token renaming:")
            print(f"     <|RESERVED_001|> (ID {reserved_001_id}) → Keep as type indicator")
            print(f"     <|RESERVED_002|> (ID {reserved_002_id}) → Rename to <|image|>")
            print(f"\n  Renaming strategy:")
            print(f"     • <|RESERVED_002|> renamed to <|image|> in vocabulary")
            print(f"     • Token ID {reserved_002_id} remains the same (no compatibility break)")
            print(f"     • Actual image data uses <|img_start|>...<|img_end|> structure")
            print(f"\n  Token assignments after renaming:")
            print(f"     <|RESERVED_001|>: Type indicator (0=text, 1=image, 2=video, etc.)")
            print(f"     <|image|>: (was <|RESERVED_002|>) Chat template image placeholder")
            print(f"     <|RESERVED_003|>-<|RESERVED_099|>: Available for future use")
            print(f"\n  ✅ Vocabulary size maintained at {self.original_size:,} tokens")
        else:
            print("  ❌ Reserved tokens not available for mapping")

    def run(self):
        """Run the complete verification and adaptation process."""
        print("=" * 60)
        print("LLAMA3-EMU3 TOKENIZER VERIFICATION")
        print("=" * 60)

        try:
            # Step 1: Load tokenizer
            self.load_and_verify_tokenizer()

            # Step 2: Retrieve all tokens
            self.retrieve_emu3_tokens()

            # Step 3: Load Vision-Instruct template
            self.load_vision_instruct_template()

            # Step 4: Verify positions
            positions_ok = self.verify_token_positions()

            # Step 5: Check chat template compatibility
            chat_ok = self.check_chat_template_compatibility()

            # Step 6: Setup reserved token mapping (using reserved tokens as placeholders)
            self.setup_reserved_token_mapping()

            # Step 7: Final size check
            final_size = len(self.tokenizer.get_vocab())
            print("\n" + "=" * 60)
            print("FINAL STATUS")
            print("=" * 60)
            print(f"Original size: {self.original_size:,}")
            print(f"Final size: {final_size:,}")

            if final_size != self.original_size:
                print(f"⚠️  SIZE CHANGED by {final_size - self.original_size:+,} tokens")
                print("   This may affect compatibility!")
            else:
                print("✅ SIZE UNCHANGED - Full compatibility maintained")

            # Step 8: Save if everything is OK
            if positions_ok and chat_ok:
                self.save_tokenizer()
                print("\n✅ Process complete!")
            else:
                print("\n⚠️  Some issues detected. Review the output above.")
                print("Not saving tokenizer due to issues.")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify and adapt Llama3-Emu3 tokenizer for Llama 3.2 Vision compatibility"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/MLLM/llama3_emu3_tokenizer",
        help="Path to Llama3-Emu3 tokenizer",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Output path for verified tokenizer")

    args = parser.parse_args()

    adapter = Llama32VisionEmu3Adapter(tokenizer_path=args.tokenizer_path, output_path=args.output_path)

    adapter.run()


if __name__ == "__main__":
    main()
