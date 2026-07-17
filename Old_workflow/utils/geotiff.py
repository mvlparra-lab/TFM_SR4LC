#!/usr/bin/env python3

"""
Author: Victoria León
Project: SR4LC

Description:
Utility functions for GeoTIFF handling.
Includes tools to load, save, export and update
GeoTIFF files for QGIS validation and SR outputs.
"""

# -----------------------
# 1. Imports
# -----------------------
import rasterio


# -----------------------
# 2. Load GeoTIFF
# -----------------------
def load_geotiff(input_path):
    """
    Load a GeoTIFF as a NumPy array and return its raster profile.
    """

    with rasterio.open(input_path) as src:
        image = src.read()
        profile = src.profile

    print("Input shape:", image.shape)

    return image, profile


# -----------------------
# 3. Export single cube date as GeoTIFF
# -----------------------
def export_single_date_geotiff(stac, output_path, time_index=0):
    """
    Export one time step from an xarray STAC cube as a GeoTIFF.

    This is mainly used for quick visual validation in QGIS.
    """

    stac_single = stac.isel(time=time_index)

    stac_single.rio.to_raster(str(output_path))

    print(f"GeoTIFF exported to: {output_path}")


# -----------------------
# 4. Prepare tensor
# -----------------------
def prepare_tensor(patch):
    """
    Normalize a raster patch and convert it to a PyTorch tensor.
    """

    import numpy as np
    import torch

    patch_np = (patch / 10000).astype("float32")

    patch_np = np.nan_to_num(
        patch_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X = torch.from_numpy(patch_np).float()

    return X


# -----------------------
# 5. Save GeoTIFF
# -----------------------
def save_geotiff(output_path, image, profile):
    """
    Save a NumPy array as a GeoTIFF using an existing raster profile.
    """

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image)

    print("Saved:", output_path)


# -----------------------
# 6. Update SR profile
# -----------------------
def update_sr_profile(profile, super_patch, window, scale_factor=4):
    """
    Update raster profile after super-resolution.

    The transform is adjusted to match the higher spatial resolution.
    """

    from rasterio.transform import Affine

    col, row, win_width, win_height = window

    sr_profile = profile.copy()

    original_transform = profile["transform"]

    patch_transform = original_transform * Affine.translation(col, row)
    sr_transform = patch_transform * Affine.scale(1 / scale_factor, 1 / scale_factor)

    sr_profile.update({
        "height": super_patch.shape[1],
        "width": super_patch.shape[2],
        "transform": sr_transform,
        "dtype": "float32"
    })

    return sr_profile