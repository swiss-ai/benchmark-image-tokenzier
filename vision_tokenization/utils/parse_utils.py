#!/usr/bin/env python3
"""Parsing utilities for configuration values."""


def parse_resolution(value):
    """Parse resolution like '256*256' or '65536'.

    Returns dict with 'pixels' (int) and 'dims' (tuple or None).
    """
    if "*" in value:
        parts = value.split("*")
        w, h = int(parts[0]), int(parts[1])
        return {"pixels": w * h, "dims": (w, h)}
    else:
        return {"pixels": int(value), "dims": None}
