#!/usr/bin/env python3

"""
Based on: Arisoy, B., Betz, F., Stauch, G., Klein, D., Dech, S., & Ullmann, T. (2025). stac2cube (Version 1.3.0). 
          Zenodo. https://doi.org/10.5281/zenodo.18459201
Author: Mohajane Meriame
Adapted by: Victoria León
Project: SR4LC

Description:
Generate Sentinel-2 cloud mask layers from an existing LR data cube
and apply the selected cloud mask to create a cloud-masked LR cube.
"""


# =============================================================================
# 0. PYTHON PATH SETUP
# =============================================================================

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# =============================================================================
# 1. IMPORTS
# =============================================================================

from dataclasses import dataclass

import xarray as xr
from utils.cloud_masking import get_cloud_layers, mask_stac_clouds
from utils.common import lazy_data_size
from utils.naming import build_run_name
from utils.paths import create_output_dir
from utils.export_cfg import export_to_cogs

# =============================================================================
# 2. CONFIGURATION CLASS
# =============================================================================

@dataclass
class CloudMaskConfig:

    # Existing LR cube
    input_cube_path: str

    # AOI used for cloud layer generation
    polygon: str | list[float]

    # Acquisition date range
    daterange: tuple[str, str]

    # Selected cloud mask threshold
    threshold: int = 70

    # Selected mask layer
    mask_layer: str = "cloud_mask_70"

    # Raster clipping
    clip_raster: bool = False

    # Export paths
    output_clouds: str | None = None
    output_masked: str | None = None


# =============================================================================
# 3. LOAD DATAARRAY
# =============================================================================

def load_dataarray(
    netcdf_path: str,
    variable_name: str
) -> xr.DataArray:
    """
    Load a DataArray from a NetCDF file.
    """

    with xr.open_dataset(netcdf_path) as ds:
        return ds[variable_name].load()


# =============================================================================
# 4. BUILD CLOUD MASK CUBE
# =============================================================================

def build_cloud_cube(config: CloudMaskConfig) -> xr.DataArray:
    """
    Generate cloud probability and binary cloud mask layers.
    """

    cloud_stac = get_cloud_layers(
        polygon=config.polygon,
        daterange=list(config.daterange),
        output=config.output_clouds,
        clip_raster=config.clip_raster,
        threshold=config.threshold,
        masking=None,
    )

    if isinstance(cloud_stac, str):
        cloud_stac = load_dataarray(
            cloud_stac,
            "Cloud_Stack"
        )

    return cloud_stac


# =============================================================================
# 5. APPLY CLOUD MASK
# =============================================================================

def apply_cloud_mask(
    config: CloudMaskConfig,
    cloud_stac: xr.DataArray
) -> xr.DataArray:
    """
    Apply selected cloud mask layer to the LR cube.
    """

    masked_stac = mask_stac_clouds(
        stac=config.input_cube_path,
        cloud=cloud_stac,
        mask_layer=config.mask_layer,
        output=config.output_masked,
    )

    if isinstance(masked_stac, str):
        masked_stac = load_dataarray(
            masked_stac,
            "Spectral_Temporal_Stack"
        )

    return masked_stac


# =============================================================================
# 6. PRINT SUMMARY
# =============================================================================

def print_summary(
    cloud_stac: xr.DataArray,
    masked_stac: xr.DataArray
) -> None:
    """
    Print basic information about cloud and masked cubes.
    """

    print("\nCloud cube created successfully.")
    print(f"\nCloud cube dimensions: {dict(cloud_stac.sizes)}")
    print(f"\nCloud cube size: {lazy_data_size(cloud_stac)}")
    print(f"\nAvailable cloud bands: {list(cloud_stac.band.values)}")

    print("\nCloud mask applied successfully.")
    print(f"\nMasked cube dimensions: {dict(masked_stac.sizes)}")
    print(f"\nMasked cube size: {lazy_data_size(masked_stac)}")


# =============================================================================
# 7. MAIN WORKFLOW
# =============================================================================

def main() -> None:
    """
    Run cloud masking workflow.
    """

    base_dir = (
        "/teamspace/lightning_storage/"
        "pKq003_SR4LC_Data"
    )

    # Existing LR cube generated in the previous step
    input_cube_path = (
        "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/comparison_tests/Cube/LR_RGBN_Area2_20260521_1214/LR_RGBN_Area2_20260521_1214.nc"
    )

    # Same AOI used for LR cube generation
    polygon = (
        "/teamspace/lightning_storage/"
        "pKq003_SR4LC_Data/inputs/tests/Area2/Area2.shp"
    )

        # Automatic naming
    if isinstance(polygon, str):
        polygon_name = Path(polygon).stem
        location_code = None
    else:
        polygon_name = None
        location_code = "41N16E"

    # Detect spectral configuration from input cube name
    input_name = Path(input_cube_path).stem

    if "RGBN" in input_name:
        prefix = "CloudMask_RGBN"
    elif "FS" in input_name:
        prefix = "CloudMask_FS"

    run_name = build_run_name(
        prefix=prefix,
        location_code=location_code,
        polygon_name=polygon_name,
    )

    output_dir = create_output_dir(
        base_dir=base_dir,
        workflow_step="comparison_tests/CloudMasking",
        run_name=run_name,
    )

    config = CloudMaskConfig(

      # Existing LR cube
      input_cube_path=input_cube_path,

      # AOI used for cloud generation
      polygon=polygon,

      # Acquisition dates
      daterange=("2024-01-01", "2024-12-31"),

      # Cloud mask threshold
      threshold=70,

     # Selected cloud mask layer
      mask_layer="cloud_mask_70",

      # Raster clipping
      clip_raster=False,

      # Export paths
      output_clouds=str(
          output_dir / f"{run_name}_clouds.nc"
      ),

      output_masked=str(
          output_dir / f"{run_name}_masked.nc"
      ),
    )

    cloud_stac = build_cloud_cube(config)
    masked_stac = apply_cloud_mask(config, cloud_stac)

    print_summary(cloud_stac, masked_stac)
    
    # =========================================================
    # Export GeoTIFF/COGs for QGIS validation
    # =========================================================

    export_to_cogs(
        cloud_stac
        .sel(band=["cloud_prob"])
        .mean(dim="time", skipna=True),
        str(output_dir / "cogs_clouds")
    )

    export_to_cogs(
        masked_stac
        .mean(dim="time", skipna=True),
        str(output_dir / "cogs_masked")
    )

    print("\nGeoTIFF/COG exports completed.")


# =============================================================================
# 8. SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()