from omegaconf import OmegaConf

from vision_tokenization.tokenize import _build_tokenizer_kwargs


def test_build_tokenizer_kwargs_prefers_tokenizer_overrides():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "torch_compile": False,
                "torch_compile_mode": "max-autotune",
                "max_sequence_tokens": 1234,
            },
            "tokenizer": {
                "torch_compile": True,
                "torch_compile_mode": "max-autotune-no-cudagraphs",
            },
        }
    )

    kwargs = _build_tokenizer_kwargs(cfg)

    assert kwargs["torch_compile"] is True
    assert kwargs["torch_compile_mode"] == "max-autotune-no-cudagraphs"
    assert kwargs["max_sequence_tokens"] == 1234


def test_build_tokenizer_kwargs_falls_back_to_dataset_defaults():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "torch_compile": True,
                "torch_compile_mode": "reduce-overhead",
            },
            "tokenizer": {},
        }
    )

    kwargs = _build_tokenizer_kwargs(cfg)

    assert kwargs["torch_compile"] is True
    assert kwargs["torch_compile_mode"] == "reduce-overhead"
