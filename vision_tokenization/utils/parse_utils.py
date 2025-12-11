#!/usr/bin/env python3
"""
Parsing utilities for command line arguments.
"""

import argparse
import re


def parse_resolution(value):
    """
    Parse resolution like '256*256' or '65536'.
    Returns dict with 'pixels' (int) and 'dims' (tuple or None).
    """
    if "*" in value:
        parts = value.split("*")
        w, h = int(parts[0]), int(parts[1])
        return {"pixels": w * h, "dims": (w, h)}
    else:
        return {"pixels": int(value), "dims": None}


def add_emu3_tokenization_args(parser=None, description="EMU3 tokenization"):
    """Add standard EMU3 tokenization arguments to parser."""
    if parser is None:
        parser = argparse.ArgumentParser(description=description)

    # Basic arguments
    parser.add_argument("--input-pattern", required=True, help="Input shard pattern")
    parser.add_argument("--output-dir", required=True, help="Output directory for tokenized data")
    parser.add_argument("--tokenizer-path", required=True, help="EMU3 tokenizer path")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs")

    # Image filtering arguments
    parser.add_argument(
        "--min-resolution",
        type=parse_resolution,
        default=None,
        help="Minimum image resolution (e.g., '256*256' or '65536')",
    )
    parser.add_argument(
        "--max-resolution",
        type=parse_resolution,
        default=None,
        help="Maximum image resolution (e.g., '1024*1024' or '1048576')",
    )

    # Range argument for processing subset of shards
    parser.add_argument(
        "--range", type=str, default=None, help="Process specific range of shards (e.g., '0:100', '100:200')"
    )

    return parser
