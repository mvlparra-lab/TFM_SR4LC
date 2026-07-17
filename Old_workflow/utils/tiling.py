#!/usr/bin/env python3

"""
Author: Victoria León
Project: SR4LC

Description:
Utility functions for tiled image processing.
Includes tools to create patch grids, extract patches,
and support patch-based super-resolution workflows.
"""

# -----------------------
# 1. Create patch grid
# -----------------------
def create_patch_grid(width, height, patch_size, overlap):

    step = patch_size - overlap
    windows = []

    for row in range(0, height, step):
        for col in range(0, width, step):

            win_width = min(patch_size, width - col)
            win_height = min(patch_size, height - row)

            windows.append((col, row, win_width, win_height))

    print("Total patches:", len(windows))

    return windows


# -----------------------
# 2. Extract patch
# -----------------------
def extract_patch(image, window):
    """
    Extract a patch from a raster array using a window definition.
    """

    col, row, win_width, win_height = window

    return image[
        :,
        row:row + win_height,
        col:col + win_width
    ]


# -----------------------
# 3. Pad patch to size
# -----------------------
def pad_patch_to_size(patch, patch_size):
    import numpy as np

    bands, height, width = patch.shape

    padded_patch = np.zeros((bands, patch_size, patch_size), dtype=patch.dtype)

    padded_patch[:, :height, :width] = patch

    return padded_patch, height, width


# -----------------------
# 4. Crop SR patch
# -----------------------
def crop_sr_patch(super_patch, original_height, original_width, scale_factor=4):

    sr_height = original_height * scale_factor
    sr_width = original_width * scale_factor

    return super_patch[:, :sr_height, :sr_width]