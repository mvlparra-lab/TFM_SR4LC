#!/usr/bin/env python3

"""
Author: Victoria León
Project: SR4LC

Description:
Utility functions for standardized output naming
and timestamp generation.
"""

from datetime import datetime


# -----------------------
# 1. Generate timestamp
# -----------------------
def generate_timestamp():
    """
    Generate current timestamp for output naming.
    """

    return datetime.now().strftime("%Y%m%d_%H%M")


# -----------------------
# 2. Build run name
# -----------------------
def build_run_name(prefix, location_code=None, polygon_name=None):
    """
    Build standardized run name.

    If a polygon name is provided, it is used.
    Otherwise, the coordinate based location code is used.
    """

    timestamp = generate_timestamp()

    if polygon_name:
        location = polygon_name
    elif location_code:
        location = location_code
    else:
        location = "UnknownArea"

    return f"{prefix}_{location}_{timestamp}"