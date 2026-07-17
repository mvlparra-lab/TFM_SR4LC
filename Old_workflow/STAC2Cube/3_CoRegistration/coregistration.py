#!/usr/bin/env python3

"""
Based on: Arisoy, B., Betz, F., Stauch, G., Klein, D., Dech, S., & Ullmann, T. (2025). stac2cube (Version 1.3.0). 
          Zenodo. https://doi.org/10.5281/zenodo.18459201
Author: Mohajane Meriame
Adapted by: Victoria León
Project: SR4LC

Description:
Apply automatic co-registration to an existing Sentinel-2 data cube.

The workflow supports both:
- Original LR cubes
- Cloud-masked LR cubes

The co-registered cube is exported as NetCDF and
GeoTIFF/COGs for QGIS validation.
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

from utils.co_registration import coregister_cube
from utils.export_cfg import export_to_cogs
from utils.common import lazy_data_size
from utils.naming import build_run_name
from utils.paths import create_output_dir


# =============================================================================
# 2. CONFIGURATION CLASS
# =============================================================================

@dataclass
class CoregistrationConfig:

    # Input cube to be co-registered
    input_cube_path: str

    # AOI definition
    polygon: str | list[float]

    # Time period for co-registration
    time_period: tuple[str, str] | None

    # Coregistration parameters
    grid_size: int = 7
    max_cc: float = 5

    min_reliability_keep: float = 10.0
    min_reliability_update_ref: float = 70.0
    max_cloud_update_ref: float = 20.0

    first_scene_mode: str = "first"

    composite_window_days: int = 30
    iteration: int = 5

    # Export path
    output_path: str | None = None


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
# 4. RUN AUTOMATIC CO-REGISTRATION
# =============================================================================

def run_coregistration(
    config: CoregistrationConfig
) -> xr.DataArray:
    """
    Apply automatic co-registration.
    """

    input_stac = load_dataarray(
        config.input_cube_path,
        "Spectral_Temporal_Stack"
    )

    # Remove problematic transform metadata
    if "transform" in input_stac.attrs:
        del input_stac.attrs["transform"]

    coreg_stac, saved_path = coregister_cube(

        input_path=input_stac,

        grid_size=config.grid_size,

        max_cc=config.max_cc,

        time_period=list(config.time_period)
        if config.time_period
        else None,

        min_reliability_keep=config.min_reliability_keep,

        min_reliability_update_ref=config.min_reliability_update_ref,

        max_cloud_update_ref=config.max_cloud_update_ref,

        first_scene_mode=config.first_scene_mode,

        composite_window_days=config.composite_window_days,

        iteration=config.iteration,

        output_path=config.output_path,
    )

    if isinstance(coreg_stac, str):

        coreg_stac = load_dataarray(
            coreg_stac,
            "Spectral_Temporal_Stack"
        )

    if coreg_stac is None:
        raise RuntimeError(
            "Co-registration did not return a valid DataArray."
        )

    return coreg_stac


# =============================================================================
# 5. PRINT SUMMARY
# =============================================================================

def print_summary(
    coreg_stac: xr.DataArray
) -> None:
    """
    Print summary of co-registered cube.
    """

    print("\nCo-registration completed successfully.")

    print(
        f"\nCube dimensions: "
        f"{dict(coreg_stac.sizes)}"
    )

    print(
        f"\nCube size: "
        f"{lazy_data_size(coreg_stac)}"
    )


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main() -> None:
    """
    Run co-registration workflow.
    """

    base_dir = (
        "/teamspace/lightning_storage/"
    "pKq003_SR4LC_Data"
    )

    # =========================================================
    # Input cube
    # =========================================================

    input_cube_path = (
        "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/comparison_tests/CloudMasking/CloudMask_RGBN_Area2_20260521_1402/CloudMask_RGBN_Area2_20260521_1402_masked.nc"
    )

    # =========================================================
    # AOI definition
    # =========================================================

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
        prefix = "Coreg_RGBN"

    elif "FS" in input_name:
        prefix = "Coreg_FS"

    run_name = build_run_name(
        prefix=prefix,
        location_code=location_code,
        polygon_name=polygon_name,
    )
   
    output_dir = create_output_dir(
        base_dir=base_dir,
        workflow_step="comparison_tests/Coregistration",
        run_name=run_name,
    )

    # =========================================================
    # Workflow configuration
    # =========================================================

    config = CoregistrationConfig(

        # Input cube
        input_cube_path=input_cube_path,

        # AOI
        polygon=polygon,

        # Time period
        time_period=(
            "2024-01-01",
            "2024-12-31"
        ),

        # Coregistration parameters
        grid_size=7,

        max_cc=5,

        min_reliability_keep=10.0,

        min_reliability_update_ref=70.0,

        max_cloud_update_ref=20.0,

        first_scene_mode="first",

        composite_window_days=30,

        iteration=5,

        # Export path
        output_path=str(
            output_dir / f"{run_name}.nc"
        ),
    )

    # Run co-registration
    coreg_stac = run_coregistration(config)

    # Print summary
    print_summary(coreg_stac)

    # =========================================================
    # Export GeoTIFF/COGs for QGIS validation
    # =========================================================

    export_to_cogs(
        coreg_stac.mean(dim="time", skipna=True),
        str(output_dir / "cogs_coreg")
    )

    print("\nGeoTIFF/COG exports completed.")


# =============================================================================
# 7. SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()