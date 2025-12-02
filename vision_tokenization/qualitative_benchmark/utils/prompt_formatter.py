"""
Class that enables formatting prompts, bring together image and text, allows to apply chat template or not and
outputs final string or token id list
"""

import logging
from transformers import AutoTokenizer
import re

logger = logging.getLogger(__name__)

########################################################################
# Simple transforms for simple single turn prompt to model chat format #
########################################################################
def to_apertus_format(text: str, img_right: bool = False) -> dict:
    conv_dict = {
                "role": "user",
                "content": {"parts": []},
            }
    if img_right:
        conv_dict["content"]["parts"].append({"type": "image"})
        conv_dict["content"]["parts"].append({"type": "text", "text": text})
    else:
        conv_dict["content"]["parts"].append({"type": "text", "text": text})
        conv_dict["content"]["parts"].append({"type": "image"})
    return conv_dict


# Mapper methods to convert standard prompts to format needed by chat template of models tokenizer
CHAT_TRANFORMS = {"to_apertus": to_apertus_format}


class PromptFormatter:
    def __init__(self, tokenizer_path: str = None):
        if tokenizer_path is not None:
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def prepare_chat_prompt(self, conversation_text: str, img_token_string: str, chat_transform: str = None, add_generation_prompt: bool = True, img_right: bool = False) -> str:
        """
        Takes conversation dict and list of img tokens and returns formatted prompt.
        Conversation dict will be transformed through chat_transform indicated by string arg "chat_transform".
        img_right determines whether image comes before or after the prompt.
        TODO: For now only one img supported!
        """
        assert self.tokenizer is not None, "Tokenizer not loaded, please provide path to tokenizer"

        if chat_transform is not None:
            conversation_text = CHAT_TRANFORMS[chat_transform](conversation_text, img_right=img_right)

        chat_txt = self.tokenizer.apply_chat_template(conversation_text, tokenize=False, add_generation_prompt=add_generation_prompt)

        if img_token_string is not None:
            # TODO: img placeholder might change for different tokenizers/chat templates
            chat_txt = re.sub(r"<\|image\|>", img_token_string, chat_txt)

        return chat_txt

    def prepare_non_chat_prompt(self, txt_string: str, img_token_string: str = None, img_right: bool = False) -> str:
        """
        Prepare prompt with one prompt text and optionally an image token string.
        Can decode whether img can be on right or left side of prompt.
        """
        prompt = txt_string
        if img_token_string is not None:
            if img_right:
                prompt = prompt + img_token_string
            else:
                prompt = img_token_string + prompt
        return prompt
