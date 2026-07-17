#!/usr/bin/env python3

"""
Compute dataset normalization statistics.

This script computes the per-band mean and standard deviation of the
super-resolved Sentinel-2 RGBN dataset. The statistics are calculated
incrementally over all valid pixels and are intended for image
normalization during model training and inference.

Author: Victoria León
Project: SR4LC
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import rasterio

# =============================================================================
# 2. INPUT DATA
# =============================================================================

INPUT_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
    "outputs/S2_SEN2SR/RGBN_COG"
)

# =============================================================================
# 3. INITIALIZATION
# =============================================================================

image_paths = sorted(INPUT_DIR.glob("*.tif"))

if not image_paths:
    raise FileNotFoundError(
        f"No .tif files found in {INPUT_DIR}"
    )

band_sum = np.zeros(4, dtype=np.float64)
band_sq_sum = np.zeros(4, dtype=np.float64)
pixel_count = np.zeros(4, dtype=np.int64)

print(f"Found {len(image_paths)} images", flush=True)

# =============================================================================
# 4. COMPUTE DATASET STATISTICS
# =============================================================================

for image_index, image_path in enumerate(image_paths, start=1):

    print(
        f"[{image_index}/{len(image_paths)}] {image_path.name}",
        flush=True,
    )

    with rasterio.open(image_path) as src:

        if src.count != 4:
            raise ValueError(
                f"{image_path.name} has {src.count} bands instead of 4"
            )

        for _, window in src.block_windows(1):

            data = src.read(window=window).astype(np.float64)

            for band_index in range(4):

                values = data[band_index]
                valid = np.isfinite(values)

                if src.nodata is not None:
                    valid &= values != src.nodata

                values = values[valid]

                band_sum[band_index] += values.sum()
                band_sq_sum[band_index] += np.square(values).sum()
                pixel_count[band_index] += values.size

    print("  finished", flush=True)

# =============================================================================
# 5. COMPUTE MEAN AND STANDARD DEVIATION
# =============================================================================

means = band_sum / pixel_count
variances = band_sq_sum / pixel_count - np.square(means)

# Avoid tiny negative values caused by floating-point precision.
variances = np.maximum(variances, 0)

stds = np.sqrt(variances)

# =============================================================================
# 6. PRINT RESULTS
# =============================================================================

print("\nRESULTS")
print("MEANS  =", means.tolist())
print("STDS   =", stds.tolist())
print("PIXELS =", pixel_count.tolist())