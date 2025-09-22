#!/usr/bin/env python3
"""
Parsing utilities for command line arguments.
"""

import re
import argparse


def parse_resolution(value):
    """Parse resolution like '256*256' or '65536'."""
    # Match pattern: number*number or just number
    match = re.match(r'^(\d+)(?:\*(\d+))?$', value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid resolution: {value}")

    a = int(match.group(1))
    b = int(match.group(2)) if match.group(2) else 1
    return a * b


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
    parser.add_argument("--min-resolution", type=parse_resolution, default=None,
                        help="Minimum image resolution (e.g., '256*256' or '65536')")
    parser.add_argument("--max-resolution", type=parse_resolution, default=None,
                        help="Maximum image resolution (e.g., '1024*1024' or '1048576')")

    # Range argument for processing subset of shards
    parser.add_argument("--range", type=str, default=None,
                        help="Process specific range of shards (e.g., '0:100', '100:200')")

    return parser