"""Inference backends for VLM benchmarking."""

from .base import BaseInferencer


def create_inferencer(inferencer_type: str, **kwargs) -> BaseInferencer:
    """Factory function to create an inferencer by type.

    Args:
        inferencer_type: ``"vllm"`` or ``"hf"``.
        **kwargs: Forwarded to the inferencer constructor.
            See each class's ``REQUIRED_ARGS`` and ``OPTIONAL_ARGS``
            for supported parameters.

    Returns:
        A :class:`BaseInferencer` subclass instance.

    Raises:
        ValueError: If *inferencer_type* is not recognised.
    """
    if inferencer_type == "vllm":
        from .vllm_inferencer import VLLMInferencer

        return VLLMInferencer(**kwargs)
    elif inferencer_type == "hf":
        from .hf_inferencer import HFInferencer

        return HFInferencer(**kwargs)
    else:
        raise ValueError(f"Unknown inferencer type: {inferencer_type!r}. Choose 'vllm' or 'hf'.")


__all__ = ["BaseInferencer", "create_inferencer"]
