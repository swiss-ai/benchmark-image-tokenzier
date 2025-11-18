#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader

Downloads HuggingFace datasets to determined datasets folder.
Leverages huggingface datasets library with additional logic to handle http rate limit and fails.

Example Usage:
    # Download IBM Duroc dataset to local test cache
    python download_hf_dataset.py \
        --dataset-name "ibm-research/duorc" \
        --subset-name "ParaphraseRC" \
        --split "train" \
        --cache-dir "./test_cache" \
        --max-retries 5 \
        --backoff-factor 1.0
"""

import sys
import os
import argparse
import traceback
from datasets import load_dataset_builder, DownloadMode, VerificationMode
from huggingface_hub import configure_http_backend
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets to cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset repository name (e.g., HuggingFaceM4/FineVision)",
    )

    parser.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Dataset subset/config name (optional)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Cache directory path for storing downloaded data",
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for downloading (default: auto-detect)",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download even if cached",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts for failed downloads (default: 5)",
    )

    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=1.0,
        help="Exponential backoff multiplier in seconds (default: 1.0)",
    )

    return parser.parse_args()


def check_hf_authentication():
    """Check if HuggingFace token is set and display authentication status."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_token:
        print("✓ HuggingFace authentication: Token detected")
        print("  Rate limit: ~1,000-2,500 requests per 5 minutes")
        return True
    else:
        print("⚠ HuggingFace authentication: No token detected")
        print("  Rate limit: ~500 requests per 5 minutes (reduced)")
        print("  Tip: Set HF_TOKEN environment variable to increase rate limit")
        return False


def setup_http_retry_backend(
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    timeout: int = 900,
):
    """
    Configure HTTP backend with retry logic for HuggingFace libraries.

    This configures the underlying HTTP session used by huggingface_hub and datasets
    libraries. All subsequent HTTP requests will automatically retry on failures with
    exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        backoff_factor: Exponential backoff multiplier in seconds (default: 1.0)
            Sleep time = backoff_factor * (2 ** (retry_number - 1))
        timeout: Request timeout in seconds (default: 900 = 15 minutes)

    Retries on:
        - Connection errors (network failures)
        - Read/timeout errors
        - HTTP 500, 502, 503, 504 (server errors)
        - HTTP 429 (rate limiting) - respects Retry-After header
    """
    # Configure retry with urllib3.Retry
    retries = Retry(
        total=max_retries,              # Total number of retries
        connect=max_retries,             # Connection errors
        read=max_retries,                # Read errors
        status=max_retries,              # Status code errors
        backoff_factor=backoff_factor,   # Exponential backoff
        status_forcelist=(500, 502, 503, 504, 429),  # HTTP codes to retry
        raise_on_status=False,           # Don't raise exception, return response
        respect_retry_after_header=True  # Respect server's Retry-After header
    )

    class TimeoutHTTPAdapter(HTTPAdapter):
        """HTTPAdapter with default timeout for all requests."""

        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.pop("timeout", timeout)
            super().__init__(*args, **kwargs)

        def send(self, request, **kwargs):
            """Override send to apply default timeout."""
            kwargs["timeout"] = kwargs.get("timeout", self.timeout)
            return super().send(request, **kwargs)

    def backend_factory() -> Session:
        """Factory function to create HTTP session with retry logic."""
        session = Session()

        # Create adapter with retry configuration and timeout
        adapter = TimeoutHTTPAdapter(max_retries=retries, timeout=timeout)

        # Mount adapter for both HTTP and HTTPS
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    # Configure huggingface_hub to use this backend
    configure_http_backend(backend_factory=backend_factory)

    # Calculate example backoff times
    backoff_times = [backoff_factor * (2 ** i) for i in range(min(max_retries, 5))]
    backoff_str = ", ".join([f"{t:.1f}s" for t in backoff_times])

    print(f"✓ HTTP retry configured: {max_retries} retries, {backoff_factor}s backoff")
    print(f"  Retry delays: {backoff_str}")
    print(f"  Retries on: connection errors, timeouts, HTTP 429/500/502/503/504")


def main():
    args = parse_args()

    print("=" * 80)
    print("Downloading dataset via HuggingFace DatasetBuilder")
    print("=" * 80)

    # Check authentication status
    check_hf_authentication()
    print()

    # Configure HTTP backend with retry logic BEFORE any downloads
    setup_http_retry_backend(
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        timeout=900,
    )
    print()

    download_mode = (
        DownloadMode.FORCE_REDOWNLOAD
        if args.force_redownload
        else DownloadMode.REUSE_CACHE_IF_EXISTS
    )

    # All checks only useful to test deterioration over time
    # (hf hub files have hash as filename so could run check directly on it manually)
    verification_mode = VerificationMode.BASIC_CHECKS

    print(f"Dataset:       {args.dataset_name}")
    print(f"Subset:        {args.subset_name or '<none>'}")
    print(f"Cache dir:     {args.cache_dir}")
    print(f"Num processes: {args.num_proc or 'auto'}")
    print(f"Download mode: {download_mode}")
    print(f"Max retries:   {args.max_retries}")
    print(f"Backoff factor: {args.backoff_factor}")
    print()

    try:
        # Initialize dataset builder
        builder = load_dataset_builder(
            args.dataset_name,
            name=args.subset_name,
            cache_dir=args.cache_dir,
        )
        print(f"Builder loaded: {builder.info.builder_name}")

        # Download and prepare dataset (resumable and cached)
        builder.download_and_prepare(
            download_mode=download_mode,
            num_proc=args.num_proc,
            verification_mode=verification_mode,
        )

        print("")
        print("=" * 80)
        print("✅ DOWNLOAD SUCCESSFUL!")
        print("=" * 80)
        print(f"Dataset: {args.dataset_name}")
        if args.subset_name:
            print(f"Subset: {args.subset_name}")
        print(f"Cached in: {args.cache_dir}")
        print(f"Splits: {builder.info.splits.keys()}")
        print(f"Features: {builder.info.features}")
        print("=" * 80)

        return 0

    except Exception as e:
        print("")
        print("=" * 80)
        print("❌ DOWNLOAD FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())