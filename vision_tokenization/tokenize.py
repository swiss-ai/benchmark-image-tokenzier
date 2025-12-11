#!/usr/bin/env python3
"""
Main entry point for vision tokenization.

This script provides a unified interface for different tokenization pipelines.

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

from vision_tokenization.pipelines import HFDatasetPipeline, WebDatasetPipeline
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

    return parser


def create_wds_parser(subparsers):
    """Create parser for WebDataset."""
    parser = subparsers.add_parser("wds", help="Tokenize WebDataset format")

    parser.add_argument(
        "--input-pattern",
        type=str,
        required=False,
        help='Pattern for input webdataset files (e.g., "data_{000..100}.tar")',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image_only", "image_text_pair", "sft"],
        default="image_text_pair",
        help="Tokenization mode",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")

    return parser


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vision Tokenization Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Common arguments
    parser.add_argument("--tokenizer-path", type=str, required=False, help="Path to the tokenizer")
    parser.add_argument("--output-dir", type=str, required=False, help="Output directory for tokenized data")
    parser.add_argument("--num-gpus", type=int, required=False, help="Number of GPUs for tokenization")
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

    # Subparsers for different data formats
    subparsers = parser.add_subparsers(dest="data_format", help="Data format to process", required=True)

    # Add format-specific parsers
    create_hf_parser(subparsers)
    # TODO: Add WebDataset parser when implemented
    # create_wds_parser(subparsers)

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config file or use CLI args
    if args.config:
        logger.info(f"Loading config from {args.config}")
        with open(args.config) as f:
            config = json.load(f)

        logger.info(f"Config loaded: {list(config.keys())}")

        # Parse resolution strings like "384*384"
        res_keys = ["min_tokenizer_pixels", "max_tokenizer_pixels", "min_image_pixels", "max_image_pixels"]
        parsed_resolutions = {}
        for key in res_keys:
            if isinstance(config.get(key), str):
                parsed = parse_resolution(config[key])
                config[key] = parsed["pixels"]
                parsed_resolutions[key] = parsed["dims"]

        # CLI args override config file
        cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
        logger.info(f"CLI overrides: {list(cli_overrides.keys())}")
        config.update(cli_overrides)
        logger.info(f"Final config keys: {list(config.keys())}")
    else:
        config = vars(args)

    # Extract data format and remove non-pipeline keys
    data_format = args.data_format
    for key in ["config", "verbose", "data_format"]:
        config.pop(key, None)

    # Validate required parameters
    required = ["tokenizer_path", "output_dir", "num_gpus", "device"]
    if data_format == "hf":
        required += ["dataset_name", "dataset_split", "mode", "num_shards"]

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
            # Add resolution dims for directory naming
            if parsed_resolutions.get("min_image_pixels"):
                pipeline_config["min_image_dims"] = parsed_resolutions["min_image_pixels"]
            if parsed_resolutions.get("max_image_pixels"):
                pipeline_config["max_image_dims"] = parsed_resolutions["max_image_pixels"]
            pipeline = HFDatasetPipeline(**pipeline_config)
            result = pipeline.run()

        elif data_format == "wds":
            logger.error("WebDataset pipeline not yet implemented")
            sys.exit(1)
            # TODO: Implement WebDataset pipeline
            # logger.info("Running WebDataset pipeline")
            # pipeline_config.update({...})
            # pipeline = WebDatasetPipeline(**pipeline_config)
            # result = pipeline.run()

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
