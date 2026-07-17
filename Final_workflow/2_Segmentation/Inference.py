#!/usr/bin/env python3

"""
Based on the memory-efficient inference workflow

Original author: Planetek
Adapted by: Victoria León
Project: SR4LC

Description
-----------
Run semantic segmentation inference on super-resolved Sentinel-2 RGBN imagery
using a trained UNet model and the LargeImagePredictor class.

Workflow
--------
1. Load the trained UNet checkpoint.
2. Apply memory-efficient tiled inference on large GeoTIFF images.
3. Preserve the original georeferencing.
4. Export the predicted land-cover masks as GeoTIFF files.
"""

# =============================================================================
# 0. PYTHON PATH SETUP
# =============================================================================
#
# Add the project directory to the Python path so local packages can be imported.
#

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/teamspace/studios/this_studio")


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pk_seg.inference.large_image_predictor import LargeImagePredictor
from pK_seg.models.lightning_wrapper.lightning_model import LightningSegModel
from pK_seg.training_datasets.stats_archive import ItalyS2SRTile1024


# =============================================================================
# 2. INPUT / OUTPUT PATHS
# =============================================================================
#
# Define the input Sentinel-2 imagery, the trained model checkpoint,
# and the output directory where the predicted masks will be stored.
#

INPUT_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/S2_SEN2SR/RGBN_COG"
)

OUTPUT_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/Segmentation/inference"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/Segmentation/training/"
    "unet_lr0.0001_schedcosine_lossce_gsd2.5_from_scratch/"
    "unet_lr0.0001_schedcosine_lossce_gsd2.5_from_scratch/"
    "2026-07-15T14-57-58.898+00-00/checkpoints/"
    "best-epoch=14-val_miou=0.2672.ckpt"
)


# =============================================================================
# 3. DATASET NORMALIZATION
# =============================================================================
#
# Mean and standard deviation computed from the Italian SR4LC dataset.
# These values must match those used during training.
#

TRAIN_MEANS = ItalyS2SRTile1024.means
TRAIN_STDS = ItalyS2SRTile1024.stds


# =============================================================================
# 4. INFERENCE CONFIGURATION
# =============================================================================
#
# Parameters used by the tiled predictor.
#

SOURCE_PIXEL_RANGE = (0, 1)
TARGET_PIXEL_RANGE = (0, 1)

# Sentinel-2 RGBN band order:
# B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
BAND_INDICES = [0, 1, 2, 3]

NUM_CLASSES = 8
NODATA_VALUE = 0

WINDOW_SIZE = 512
WINDOW_BUFFER = 64

BATCH_SIZE = 16
MAX_RAM_GB = 4.0


# =============================================================================
# 5. LOAD TRAINED MODEL
# =============================================================================
#
# Restore the trained UNet checkpoint and move the model to the
# available device (GPU if available, otherwise CPU).
#

def load_model(ckpt_path: Path):

    model = LightningSegModel.load_from_checkpoint(
        ckpt_path,
        weights_only=False
    )

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return model.to(device)


# =============================================================================
# 6. RUN INFERENCE
# =============================================================================
#
# Perform memory-efficient tiled inference on a single GeoTIFF image.
# The resulting prediction is saved as a GeoTIFF while preserving
# the original spatial reference.
#

def run_inference(
    model,
    tiff_path: Path,
    out_path: Path,
) -> None:

    # Skip files that have already been processed.
    if out_path.exists():
        print(f"Skipping existing prediction: {out_path.name}")
        return

    predictor = LargeImagePredictor(
        model=model,
        means=TRAIN_MEANS,
        stds=TRAIN_STDS,
        num_classes=NUM_CLASSES,
        source_pixel_range=SOURCE_PIXEL_RANGE,
        target_pixel_range=TARGET_PIXEL_RANGE,
        window_size=WINDOW_SIZE,
        window_buffer=WINDOW_BUFFER,
        batch_size=BATCH_SIZE,
        target_band_mapping=BAND_INDICES,
        max_ram_gb=MAX_RAM_GB,
        nodata_value=NODATA_VALUE,
    )

    predictor.execute(
        tiff_path=tiff_path,
        out_path=out_path,
        band_indices=BAND_INDICES,
    )

    print(f"Saved: {out_path.name}")


# =============================================================================
# 7. MAIN WORKFLOW
# =============================================================================
#
# Iterate through all GeoTIFF images in the input directory,
# run inference and save the predicted masks.
#

def main() -> None:

    print("=" * 60)
    print("SR4LC - UNET INFERENCE")
    print("=" * 60)
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Input dir  : {INPUT_DIR}")
    print(f"Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)

    tiff_paths = sorted(INPUT_DIR.glob("*.tif")) + sorted(INPUT_DIR.glob("*.tiff"))

    if not tiff_paths:
        print(f"No TIFF files found in {INPUT_DIR}")
        return

    for tiff_path in tiff_paths:

        print(f"\nProcessing: {tiff_path.name}")

        out_path = OUTPUT_DIR / f"{tiff_path.stem}_pred.tif"

        run_inference(
            model=model,
            tiff_path=tiff_path,
            out_path=out_path,
        )

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()