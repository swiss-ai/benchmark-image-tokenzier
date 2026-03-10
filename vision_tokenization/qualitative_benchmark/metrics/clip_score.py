"""
CLIP score metric for measuring image-text similarity.

Uses CLIP model to compute cosine similarity between image and text embeddings,
useful for evaluating caption quality in captioning benchmarks.
"""

from typing import ClassVar, Dict, Set

import torch
from PIL import Image

from vision_tokenization.qualitative_benchmark.metrics import register_metric
from vision_tokenization.qualitative_benchmark.metrics.base import BaseMetric

# Lazy-loaded global CLIP model (shared across instances)
_clip_model = None
_clip_processor = None
_clip_device = None


def _get_clip_model(device):
    """Lazy-load and cache the CLIP model and processor."""
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None or _clip_device != device:
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_name).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model.eval()
        _clip_device = device
    return _clip_model, _clip_processor


@register_metric("clip_score")
class CLIPScoreMetric(BaseMetric):
    """
    CLIP score metric for image-caption similarity.

    Computes cosine similarity between CLIP embeddings of the image and caption.
    Score ranges from -1 to 1, with higher values indicating better alignment.

    Required input keys:
        image: PIL Image to evaluate
        caption: Text caption to compare against

    Output keys:
        clip_score: Cosine similarity between image and text embeddings (float)
    """

    name: ClassVar[str] = "clip_score"
    REQUIRED_KEYS: ClassVar[Set[str]] = {"image", "caption"}

    def _setup(self, **kwargs):
        """Pre-load the CLIP model for efficiency."""
        _get_clip_model(self.device)

    def __call__(self, data: dict) -> dict:
        """
        Compute CLIP score between image and caption.

        Args:
            data: Dictionary with "image" (PIL Image) and "caption" (str)

        Returns:
            Dictionary with "clip_score" key
        """
        data = self.validate_input(data)

        image = data["image"]
        caption = data["caption"]

        # Handle empty caption
        if not caption or not caption.strip():
            return {"clip_score": None}

        model, processor = _get_clip_model(self.device)

        # Process inputs
        inputs = processor(
            text=[caption],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            clip_score = float(torch.sum(image_embeds * text_embeds).item())

        return {"clip_score": clip_score}
