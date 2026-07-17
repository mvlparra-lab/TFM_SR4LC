#!/usr/bin/env python3

"""
Based on: Arisoy, B., Betz, F., Stauch, G., Klein, D., Dech, S., & Ullmann, T. (2025). stac2cube (Version 1.3.0). 
          Zenodo. https://doi.org/10.5281/zenodo.18459201
Author: Mohajane Meriame
Adapted by: Victoria León
Project: SR4LC

Description:
Generate the initial Sentinel-2 Low Resolution (LR) data cube
using STAC2Cube.

The workflow exports the generated cube as NetCDF and
GeoTIFF/COGs for QGIS validation and further
super-resolution processing.
"""


# =============================================================================
# 0. PYTHON PATH SETUP
# =============================================================================

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]

sys.path.append(str(project_root))


# =============================================================================
# 1. IMPORTS
# =============================================================================

from dataclasses import dataclass
from typing import Sequence

import dask
import xarray as xr
from dask.diagnostics.progress import ProgressBar

# STAC2Cube utilities
from utils.main import get_stac_layers
from utils.export_cfg import export_stac, export_to_cogs

# Helper functions
from utils.common import lazy_data_size
from utils.naming import build_run_name
from utils.paths import create_output_dir


# =============================================================================
# 2. CONFIGURATION CLASS
# =============================================================================

@dataclass
class CubeConfig:

    mission: str = "s2"
    polygon: str | list[float] = "./data/polygons/test.gpkg"
    resolution: int = 10

    daterange: tuple[str, str] = (
        "2024-01-01",
        "2024-12-31"
    )

    bands: Sequence[str] = (
        "green",
        "red",
        "nir",
        "nir08",
        "rededge1",
        "rededge2",
        "rededge3",
        "swir16",
        "swir22", 
    )

    # Keep empty for base LR cube generation
    indices: Sequence[str] = ()

    clip_raster: bool = False
    max_cc: int = 30
    cloud_masking: bool = False

    stats: Sequence[str] | None = None
    aggregator: str | None = None

    q: bool | None = None
    compute: bool = False

    export_path: str | None = None
    export_cogs_dir: str | None = None


# =============================================================================
# 3. BUILD DATA CUBE
# =============================================================================

def build_data_cube(config: CubeConfig) -> xr.DataArray:
    """
    Build the initial Sentinel-2 LR cube from STAC.
    """

    stac = get_stac_layers(
        mission=config.mission,
        polygon=config.polygon,
        resolution=config.resolution,
        daterange=list(config.daterange),
        bands=list(config.bands),
        max_cc=config.max_cc,
        clip_raster=config.clip_raster,
        cloud_masking=config.cloud_masking,
        indices=list(config.indices),
        output=None,
        aggregator=config.aggregator,
        stats=config.stats,
        q=config.q,
    )

    # Optional computation of lazy cube
    if config.compute:
        with ProgressBar():
            stac = stac.compute()

    return stac


# =============================================================================
# 4. PRINT DATA CUBE SUMMARY
# =============================================================================

def print_summary(stac: xr.DataArray) -> None:
    """
    Print basic information about the generated cube.
    """

    print("\nData cube created successfully.")

    print(f"\nName: {stac.name}")

    print(f"\nDimensions: {dict(stac.sizes)}")

    print(f"\nEstimated size: {lazy_data_size(stac)}")

    # Print acquisition dates
    if "time" in stac.coords:

        print("\nAvailable dates:")

        for value in stac.time.values:
            print(f"  {value}")


# =============================================================================
# 5. EXPORT OUTPUTS
# =============================================================================

def export_outputs(
    stac: xr.DataArray,
    config: CubeConfig,
    run_name: str
) -> None:
    """
    Export the cube as NetCDF and GeoTIFF/COGs.
    """

    # Export NetCDF cube
    if config.export_path:

       # Create export-safe copy of the cube
       stac_to_export = stac.copy()

       # Convert metadata attributes to strings
       # to avoid NetCDF encoding issues
       stac_to_export.attrs = {
           key: str(value)
           for key, value in stac_to_export.attrs.items()
       }

       export_stac(
           stac_to_export,
           config.export_path
       )

    print(f"\nExported NetCDF: {config.export_path}")

    # Export GeoTIFF/COGs for QGIS validation
    if config.export_cogs_dir:

        Path(config.export_cogs_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        # Create temporal composite from all dates
        # to reduce empty/no-data areas
        stac_composite = stac.mean(
            dim="time",
            skipna=True
        )
    
        stac_composite.name = run_name

        export_to_cogs(
            stac_composite,
            config.export_cogs_dir
        )

        print(
            f"\nExported GeoTIFF/COGs to: "
            f"{config.export_cogs_dir}"
        )


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main() -> None:
    """
    Run initial LR cube generation workflow.
    """

    # =========================================================
    # Base Lightning storage directory
    # =========================================================

    base_dir = (
        "/teamspace/lightning_storage/"
        "pKq003_SR4LC_Data"
    )

    # =========================================================
    # Input area definition
    # =========================================================

    # ---------------------------------------------------------
    # Option 1: Bounding box
    # ---------------------------------------------------------

    #polygon = [
        #15.35,  # min lon
        #40.50,  # min lat
        #16.65,  # max lon
        #41.50   # max lat
    #]

    # ---------------------------------------------------------
    # Option 2: Polygon file
    # ---------------------------------------------------------

    polygon = (
        "/teamspace/lightning_storage/"
        "pKq003_SR4LC_Data/inputs/tests/Area2/Area2.shp"
    )

    # =========================================================
    # Automatic naming logic
    # =========================================================

    # If polygon is a file:
    # use polygon filename for outputs
    if isinstance(polygon, str):

        polygon_name = Path(polygon).stem
        location_code = None

    # If polygon is a bounding box:
    # use coordinate-based naming
    else:

        polygon_name = None
        location_code = "41N16E"

    # =========================================================
    # Build standardized run name
    # =========================================================

    run_name = build_run_name(
        prefix="LR_FS",
        location_code=location_code,
        polygon_name=polygon_name,
    )

    # =========================================================
    # Create standardized output directory
    # =========================================================

    output_dir = create_output_dir(
        base_dir=base_dir,
        workflow_step="comparison_tests/Cube",
        run_name=run_name,
    )

    # =========================================================
    # Cube configuration
    # =========================================================

    config = CubeConfig(

        # Satellite mission
        mission="s2",

        # Input geometry
        polygon=polygon,

        # Spatial resolution in meters
        resolution=10,

        # Acquisition date range
        daterange=(
            "2024-03-01",
            "2024-05-07"
        ),

        # Spectral bands
        bands=(
            "blue",
            "green",
            "red",
            "nir",
            "nir08",
            "rededge1",
            "rededge2",
            "rededge3",
            "swir16",
            "swir22",
        ),

        # Keep empty for base LR cube
        indices=(),

        # Raster clipping
        clip_raster=False,

        # Maximum cloud coverage
        max_cc=20,

        # SCL cloud masking
        cloud_masking=False,

        # Optional statistics
        stats=None,

        # Optional temporal aggregation
        aggregator=None,

        # Quiet mode
        q=None,

        # Compute lazy cube
        compute=False,

        # NetCDF export path
        export_path=str(
            output_dir / f"{run_name}.nc"
        ),

        # GeoTIFF/COG export directory
        export_cogs_dir=str(
            output_dir / "cogs"
        ),
    )

    # =========================================================
    # Build STAC cube
    # =========================================================

    stac = build_data_cube(config)

    # =========================================================
    # Print cube summary
    # =========================================================

    print_summary(stac)

    # =========================================================
    # Export outputs
    # =========================================================

    export_outputs(
        stac,
        config,
        run_name
    )


# =============================================================================
# 7. SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()