"""Time formatting and SLURM detection utilities for progress tracking."""

import logging
import os
import re
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """
    Format seconds into human-readable duration string.

    Args:
        seconds: Duration in seconds (can be float)

    Returns:
        Formatted string like "1h 23m 45s", "45m 12s", "23s", or "0s"
    """
    # Handle edge cases
    if seconds <= 0:
        return "0s"

    # Round to nearest second
    total_seconds = int(round(seconds))

    # Calculate time components
    days = total_seconds // 86400
    remaining = total_seconds % 86400
    hours = remaining // 3600
    remaining = remaining % 3600
    minutes = remaining // 60
    secs = remaining % 60

    # Build time string with non-zero components
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if nothing else, or if non-zero
        parts.append(f"{secs}s")

    return " ".join(parts)


def parse_slurm_time_limit(time_string: str) -> Optional[int]:
    """
    Parse SLURM TimeLimit string to seconds.

    Args:
        time_string: SLURM time format string

    Returns:
        Total seconds, or None if invalid format

    Formats supported:
        - "HH:MM:SS" -> 3600*HH + 60*MM + SS
        - "MM:SS" -> 60*MM + SS
        - "DD-HH:MM:SS" -> 86400*DD + 3600*HH + 60*MM + SS
        - "DD-HH:MM" -> 86400*DD + 3600*HH + 60*MM
        - "UNLIMITED" -> None (no time limit)
    """
    if not time_string:
        return None

    # Handle special cases
    if time_string.upper() in ("UNLIMITED", "INFINITE"):
        return None

    try:
        # Pattern: DD-HH:MM:SS or DD-HH:MM
        if "-" in time_string:
            match = re.match(r'^(\d+)-(\d+):(\d+)(?::(\d+))?$', time_string)
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))
                minutes = int(match.group(3))
                seconds = int(match.group(4)) if match.group(4) else 0
                return days * 86400 + hours * 3600 + minutes * 60 + seconds

        # Pattern: HH:MM:SS or MM:SS
        parts = time_string.split(":")
        if len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds

        # Try parsing as raw seconds
        return int(time_string)
    except (ValueError, AttributeError):
        return None


def get_slurm_time_limit() -> Optional[int]:
    """
    Auto-detect SLURM time limit from environment.

    Returns:
        Time limit in seconds, or None if:
        - Not running in SLURM environment
        - scontrol command fails
        - TimeLimit is UNLIMITED
        - Cannot parse time format

    Process:
        1. Check for SLURM_JOB_ID environment variable
        2. Run: scontrol show job $SLURM_JOB_ID
        3. Parse output for "TimeLimit=" line
        4. Extract and parse time limit string
    """
    # Check if running in SLURM environment
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        logger.debug("SLURM_JOB_ID not found in environment - not in SLURM job")
        return None

    try:
        # Run scontrol with timeout
        result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.debug(f"scontrol command failed with return code {result.returncode}")
            return None

        # Parse output for TimeLimit
        output = result.stdout
        match = re.search(r'TimeLimit=([^\s]+)', output)
        if not match:
            logger.debug("Could not find TimeLimit in scontrol output")
            return None

        time_limit_str = match.group(1)
        time_limit_seconds = parse_slurm_time_limit(time_limit_str)

        if time_limit_seconds is None:
            logger.debug(f"SLURM time limit is UNLIMITED or invalid: {time_limit_str}")
        else:
            logger.debug(f"Detected SLURM time limit: {time_limit_str} ({time_limit_seconds} seconds)")

        return time_limit_seconds

    except subprocess.TimeoutExpired:
        logger.debug("scontrol command timed out after 5 seconds")
        return None
    except FileNotFoundError:
        logger.debug("scontrol command not found - not in SLURM environment")
        return None
    except Exception as e:
        logger.debug(f"Error detecting SLURM time limit: {e}")
        return None
