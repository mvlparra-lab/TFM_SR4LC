#!/usr/bin/env python3

"""
Author: Victoria León
Project: SR4LC

Description:
Utility functions for standardized output
directory creation and workflow path management.
"""

from pathlib import Path


# -----------------------
# 1. Create output directory
# -----------------------
def create_output_dir(base_dir, workflow_step, run_name):
    """
    Create standardized output directory.
    """

    output_dir = Path(base_dir) / "outputs" / workflow_step / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir