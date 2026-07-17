#!/usr/bin/env python3

"""
Author: Victoria León
Project: SR4LC

Description:
Common helper functions used across the SR4LC workflow.
Includes utilities for size estimation and general formatting.
"""

# -----------------------
# 1. Format byte size
# -----------------------
def format_bytes(n: int) -> str:
    """
    Convert bytes into a human-readable format (KB, MB, GB, etc.).
    """

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)

    for u in units:
        if f < 1024 or u == units[-1]:

            if u == "B":
                return f"{int(f)} B"

            return f"{f:.2f} {u}"

    return f"{n} B"


# -----------------------
# 2. Estimate lazy cube size
# -----------------------
def lazy_data_size(da):
    """
    Estimate the size of a lazy xarray object.
    """

    return format_bytes(da.nbytes)