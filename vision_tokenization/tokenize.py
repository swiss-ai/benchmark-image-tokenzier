#!/usr/bin/env python3
"""
Main entry point for vision tokenization.
Unified interface for different tokenization pipelines.

Args and Config precedences:
    - defaults < json config file < CLI args

Usage:
    # Using configuration file (recommended)
    python -m vision_tokenization.tokenize hf --config /iopsstor/scratch/cscs/xyixuan/benchmark-image-tokenzier/vision_tokenization/configs/example_hf_image_only.json

    # Resume from previous checkpoint (skips completed shards)
    python -m vision_tokenization.tokenize hf --config path/to/config.json --resume

    # View all available options
    python -m vision_tokenization.tokenize hf --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vision_tokenization.pipelines import HFDatasetPipeline, WDSPipeline
from vision_tokenization.utils.parse_utils import parse_resolution


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_hf_parser(subparsers):
    """Create parser for HuggingFace datasets."""
    parser = subparsers.add_parser("hf", help="Tokenize HuggingFace datasets")

    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--dataset-name", type=str, required=False, help="HuggingFace dataset name")
    parser.add_argument("--dataset-split", type=str, required=False, help="Dataset split to process")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image_only", "image2text", "text2image", "sft"],
        required=False,
        help="Tokenization mode",
    )
    parser.add_argument("--config-name", type=str, help="Dataset configuration/subset name (e.g., for FineVision)")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for downloaded datasets")
    parser.add_argument("--num-proc", type=int, help="Number of processes for dataset loading")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument(
        "--num-shards", type=int, help="Number of shards for distributed processing and checkpointing (required)"
    )
    parser.add_argument("--image-field", type=str, default="images", help="Name of image field in dataset")
    parser.add_argument(
        "--text-field",
        type=str,
        default="texts",
        help="Name of text field in dataset (for image_text_pair and SFT modes)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing checkpoint by skipping completed shards"
    )
    parser.add_argument(
        "--image-transforms",
        type=str,
        help='Comma-separated list of image transforms to apply (e.g., "convert_rgb,resize_max")',
    )
    parser.add_argument(
        "--text-transforms",
        type=str,
        help='Comma-separated list of text transforms to apply (e.g., "strip_whitespace")',
    )
    parser.add_argument(
        "--dataset-load-method",
        type=str,
        choices=["default", "builder_load", "disk_load"],
        default="default",
        help=(
            "Method for loading HuggingFace datasets. "
            '"default": use load_dataset() (requires HF hub cache). '
            '"builder_load": use load_dataset_builder().as_dataset() '
            "(requires pre-prepared dataset, no hub cache needed). "
            '"disk_load": use load_from_disk() for datasets saved with '
            "dataset.save_to_disk() (dataset_name is a local path)"
        ),
    )
    parser.add_argument(
        "--conversation-transform",
        type=str,
        help=(
            "(Optional!) Conversation transform to apply in SFT tokenizer worker. Conversation transforms are applied "
            "to conversation structure of a dataset, to convert to a format that is compatible with tokenizer chat template."
        ),
    )
    parser.add_argument(
        "--image-field-pattern",
        type=str,
        default=None,
        help=(
            "Pattern prefix to auto-discover multiple image fields per sample. "
            'For HF datasets, matches column names (e.g., "img" matches img1, img2, img3). '
            'For WDS datasets, matches decoded sample keys (e.g., "img" matches img1.png, img2.png). '
            "Keys are sorted alphabetically for consistent ordering. "
            "Incompatible with batching (batch_mode) and image_only mode."
        ),
    )
    parser.add_argument(
        "--dataset-streamed", action="store_true", help="If set, and a HF dataset is loaded with mode 'load_dataset'"
    )
    parser.add_argument(
        "--data-files",
        type=str,
        help=(
            "Explicit data file path(s) or pattern(s) to pass to load_dataset(). "
            'Accepts a single path/glob (e.g., "*.tar") or a comma-separated list '
            '(e.g., "shard_001.parquet,shard_002.parquet"). '
            "Useful for loading webdatasets or custom file collections through the HF datasets API."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help=(
            "Local directory containing the raw dataset files. "
            "Passed as data_dir to load_dataset()/load_dataset_builder() so the builder "
            "script uses local files instead of downloading from the Hub."
        ),
    )
    return parser


def create_wds_parser(subparsers):
    """Create parser for WebDataset."""
    parser = subparsers.add_parser("wds", help="Tokenize WebDataset tar files")

    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument(
        "--input-pattern",
        type=str,
        required=False,
        help='Pattern for input tar files (e.g., "/path/to/*.tar" or "data_{000..100}.tar")',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image_only", "image2text", "text2image", "sft"],
        required=False,
        help="Tokenization mode",
    )
    parser.add_argument(
        "--image-field",
        type=str,
        default="jpg;png;jpeg;webp",
        help="Semicolon-separated image extension keys to try in fallback order",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="txt",
        help="Semicolon-separated text extension keys to try in fallback order",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing checkpoint by skipping completed shards"
    )
    parser.add_argument(
        "--skip-sample-count",
        action="store_true",
        help="Skip counting samples in tar files before processing. "
        "Useful for large numbers of tars where the header scan is slow. "
        "Progress will show absolute counts without percentages or ETA.",
    )
    parser.add_argument(
        "--image-transforms",
        type=str,
        help='Comma-separated list of image transforms to apply (e.g., "convert_rgb,resize_max")',
    )
    parser.add_argument(
        "--text-transforms",
        type=str,
        help='Comma-separated list of text transforms to apply (e.g., "strip_whitespace")',
    )
    parser.add_argument(
        "--conversation-transform",
        type=str,
        help="Conversation transform to apply in SFT tokenizer worker",
    )
    parser.add_argument(
        "--image-field-pattern",
        type=str,
        default=None,
        help=(
            "Pattern prefix to auto-discover multiple image fields per sample. "
            'For WDS, matches decoded sample keys (e.g., "img" matches img1.png, img2.png). '
            "Incompatible with batching (batch_mode) and image_only mode."
        ),
    )

    return parser


def parse_args():
    """Main entry point for parsing CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Tokenization Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Common arguments
    parser.add_argument("--tokenizer-path", type=str, required=False, help="Path to the tokenizer")
    parser.add_argument("--output-dir", type=str, required=False, help="Output directory for tokenized data")
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        help="Number of GPUs for tokenization. In a multinode setup this should be the number of GPUs per node.",
    )
    parser.add_argument("--device", type=str, required=False, choices=["cuda", "cpu"], help="Device for tokenization")
    parser.add_argument(
        "--min-tokenizer-pixels",
        type=parse_resolution,
        help='Minimum pixels for tokenizer preprocessing (e.g., "384*384" or "147456")',
    )
    parser.add_argument(
        "--max-tokenizer-pixels",
        type=parse_resolution,
        help='Maximum pixels for tokenizer preprocessing (e.g., "1024*1024" or "1048576")',
    )
    parser.add_argument(
        "--min-image-pixels", type=parse_resolution, help='Minimum pixels to filter images (e.g., "256*256")'
    )
    parser.add_argument(
        "--max-image-pixels", type=parse_resolution, help='Maximum pixels to filter images (e.g., "2048*2048")'
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Log progress every N samples when output is piped to file (default: 1000, 0 = log every batch)",
    )
    parser.add_argument(
        "--slurm-time-limit",
        type=str,
        help='Override SLURM time limit detection (e.g., "12:00:00" or "43200" for 12 hours)',
    )

    # Subparsers for different data formats
    subparsers = parser.add_subparsers(dest="data_format", help="Data format to process", required=True)

    # Add format-specific parsers
    create_hf_parser(subparsers)
    create_wds_parser(subparsers)

    return parser.parse_args(), parser


def get_defaults(parser, data_format):
    """Get defaults for a subparser by parsing minimal required args."""
    defaults, _ = parser.parse_known_args([data_format])
    return defaults


def main():
    """Main entry point."""
    args, parser = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    defaults = vars(get_defaults(parser, args.data_format))

    # Build config container from args
    config = vars(args).copy()

    # Load config file if exists.
    if args.config:
        logger.info(f"Loading config from {args.config}")
        with open(args.config) as f:
            json_config = json.load(f)
        logger.info(f"Config loaded: {list(json_config.keys())}")
        config.update(json_config)

        # CLI args override config file (only where non default values)
        cli_overrides = {k: v for k, v in vars(args).items() if v is not None and v != defaults.get(k, None)}
        logger.info(f"CLI overrides: {list(cli_overrides.keys())}")
        config.update(cli_overrides)

    # Parse resolution strings like "384*384" (from config file or CLI)
    res_keys = ["min_tokenizer_pixels", "max_tokenizer_pixels", "min_image_pixels", "max_image_pixels"]
    parsed_resolutions = {}
    original_resolutions = {}
    for key in res_keys:
        value = config.get(key)
        if value is not None:
            original_resolutions[key] = str(value)
        if isinstance(value, str):
            value = parse_resolution(value)
        if isinstance(value, dict):
            config[key] = value["pixels"]
            parsed_resolutions[key] = value["dims"]

    # Parse SLURM time limit if provided
    slurm_time_limit = None
    if config.get("slurm_time_limit"):
        from vision_tokenization.utils.time_utils import parse_slurm_time_limit

        value = config["slurm_time_limit"]
        # Accept both raw seconds (int) and time string
        if isinstance(value, int):
            slurm_time_limit = value
        else:
            slurm_time_limit = parse_slurm_time_limit(value)
            if slurm_time_limit is None:
                logger.warning(f"Invalid SLURM time limit format: {value}")

        if slurm_time_limit is not None:
            config["slurm_time_limit"] = slurm_time_limit
        else:
            config.pop("slurm_time_limit", None)

    # Pretty print final config
    logger.info("=" * 80)
    logger.info("Final Configuration (after CLI overrides and parsing):")
    logger.info("=" * 80)
    for key, value in sorted(config.items()):
        if key in res_keys and key in original_resolutions:
            logger.info(f"  {key}: {value:,} pixels (from: {original_resolutions[key]})")
        elif value is not None:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    # Validate multi-image configuration
    image_field_pattern = config.get("image_field_pattern")
    if image_field_pattern:
        batch_mode = config.get("batch_mode")
        if batch_mode is not None:
            raise ValueError(
                "Multi-image mode (image_field_pattern) is incompatible with batching (batch_mode). "
                "Set batch_mode to null/None or remove it from the config."
            )
        mode = config.get("mode")
        if mode == "image_only":
            raise ValueError(
                "Multi-image mode (image_field_pattern) is incompatible with image_only mode. "
                "Use image2text, text2image, or sft mode instead."
            )

    # Extract data format and remove non-pipeline keys
    data_format = args.data_format
    for key in ["config", "verbose", "data_format"]:
        config.pop(key, None)

    # Validate required parameters
    required = ["tokenizer_path", "output_dir", "num_gpus", "device"]
    if data_format == "hf":
        required += ["dataset_name", "dataset_split", "mode", "num_shards"]
    elif data_format == "wds":
        required += ["input_pattern", "mode"]

    missing = [k for k in required if not config.get(k)]
    if missing:
        logger.error(f"Missing required parameters: {', '.join(missing)}")
        logger.error("Provide them via config file or command line arguments")
        sys.exit(1)

    try:
        if data_format == "hf":
            logger.info("Running HuggingFace dataset pipeline")
            # Remove None values and pass to pipeline
            pipeline_config = {k: v for k, v in config.items() if v is not None}
            # Multi-image requires no batching — override pipeline default batch_mode="sorted"
            if config.get("image_field_pattern"):
                pipeline_config["batch_mode"] = None
            # Add resolution dims for directory naming
            if parsed_resolutions.get("min_image_pixels"):
                pipeline_config["min_image_dims"] = parsed_resolutions["min_image_pixels"]
            if parsed_resolutions.get("max_image_pixels"):
                pipeline_config["max_image_dims"] = parsed_resolutions["max_image_pixels"]
            pipeline = HFDatasetPipeline(**pipeline_config)
            result = pipeline.run()

        elif data_format == "wds":
            logger.info("Running WebDataset pipeline")
            pipeline_config = {k: v for k, v in config.items() if v is not None}
            # Multi-image requires no batching — override pipeline default batch_mode="sorted"
            if config.get("image_field_pattern"):
                pipeline_config["batch_mode"] = None
            # Remove HF-specific keys that may leak from shared args
            for hf_key in [
                "dataset_name",
                "dataset_split",
                "config_name",
                "cache_dir",
                "num_proc",
                "num_shards",
                "max_samples",
                "dataset_load_method",
                "dataset_streamed",
            ]:
                pipeline_config.pop(hf_key, None)
            pipeline = WDSPipeline(**pipeline_config)
            result = pipeline.run()

        else:
            raise ValueError(f"Unknown data format: {data_format}")

        # Report results
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
