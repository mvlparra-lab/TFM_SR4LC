#!/usr/bin/env python3

""""
Based on: Aybar, C., Contreras, J., Donike, S., Portalés-Julià, E., Mateo-García, G., & Gómez-Chova, L. (2025).
          A Radiometrically and Spatially Consistent Super-Resolution Framework for Sentinel-2.
          SSRN. https://doi.org/10.2139/ssrn.5247739
Author: Victoria León
Project: SR4LC

SEN2SR Full Spectral
Workflow:
10 m TIFF + 20 m TIFF
-> initial 20 m alignment to 10 m grid
-> Reference_RSWIR_x2
-> SEN2SRLite Full Spectral
-> GeoTIFF
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path

import mlstac
import sen2sr
import torch
import numpy as np
import rioxarray as rxr
import rasterio
from rasterio.transform import from_bounds


# =============================================================================
# 2. DEFAULT PATHS
# =============================================================================

DEFAULT_OUTPUT_DIR = (
    "/teamspace/lightning_storage/"
    "pKq003_SR4LC_Data/outputs/S2_SEN2SR/FS"
)

FULL_MODEL_PATH = (
    "/teamspace/studios/this_studio/"
    "SR4LC/model/SEN2SR_Full"
)


# =============================================================================
# 3. OUTPUT NAMING
# =============================================================================

def build_output_path(input_10m_tif, output_dir=DEFAULT_OUTPUT_DIR):

    input_path = Path(input_10m_tif)
    output_dir = Path(output_dir)

    base_name = input_path.stem

    if base_name.endswith("_10m"):
        base_name = base_name[:-4]

    if base_name.endswith("_composite"):
        base_name = base_name[:-10]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir / f"SR_FS_{base_name}.tif"


# =============================================================================
# 4. LOAD GEOTIFF
# =============================================================================

def load_geotiff(input_tif):

    da_raw = rxr.open_rasterio(
        input_tif,
        masked=False,
    )

    if isinstance(da_raw, list):
        da = da_raw[0]
    else:
        da = da_raw

    return da.astype("float32")


# =============================================================================
# 5. SAVE GEOTIFF
# =============================================================================

def save_tensor_as_geotiff(
    tensor,
    reference_da,
    output_path,
):

    data = tensor.detach().cpu().numpy().astype("float32")

    bands, height, width = data.shape

    x = reference_da.x.values
    y = reference_da.y.values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    res_x = abs(x[1] - x[0])
    res_y = abs(y[1] - y[0])

    left = xmin - res_x / 2
    right = xmax + res_x / 2
    bottom = ymin - res_y / 2
    top = ymax + res_y / 2

    transform = from_bounds(
        left,
        bottom,
        right,
        top,
        width,
        height,
    )

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs=reference_da.rio.crs,
        transform=transform,
    ) as dst:
        dst.write(data)


# =============================================================================
# 6. BUILD INITIAL 10-BAND STACK
# =============================================================================

def build_initial_full_stack(
    input_10m_tif,
    input_20m_tif,
):
    """
    Expected input_10m_tif:
        1 -> B02
        2 -> B03
        3 -> B04
        4 -> B08

    Expected input_20m_tif:
        1 -> B05
        2 -> B06
        3 -> B07
        4 -> B8A
        5 -> B11
        6 -> B12

    Output order:
        B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    """

    print("Loading 10 m image...")
    da_10m = load_geotiff(input_10m_tif)

    print("Loading 20 m image...")
    da_20m = load_geotiff(input_20m_tif)

    print("Aligning 20 m bands to 10 m grid...")

    # Initial alignment required so the model receives one common 10 m tensor.
    da_20m_to_10m = da_20m.rio.reproject_match(
        da_10m,
    )

    b02 = da_10m.sel(band=1).load()
    b03 = da_10m.sel(band=2).load()
    b04 = da_10m.sel(band=3).load()
    b08 = da_10m.sel(band=4).load()

    b05 = da_20m_to_10m.sel(band=1).load()
    b06 = da_20m_to_10m.sel(band=2).load()
    b07 = da_20m_to_10m.sel(band=3).load()
    b8a = da_20m_to_10m.sel(band=4).load()
    b11 = da_20m_to_10m.sel(band=5).load()
    b12 = da_20m_to_10m.sel(band=6).load()

    stack = np.stack(
        [
            np.asarray(b02.to_numpy()),
            np.asarray(b03.to_numpy()),
            np.asarray(b04.to_numpy()),
            np.asarray(b05.to_numpy()),
            np.asarray(b06.to_numpy()),
            np.asarray(b07.to_numpy()),
            np.asarray(b08.to_numpy()),
            np.asarray(b8a.to_numpy()),
            np.asarray(b11.to_numpy()),
            np.asarray(b12.to_numpy()),
        ],
        axis=0,
    )

    stack = (stack / 10000.0).astype("float32")

    return stack, da_10m.sel(band=1)


# =============================================================================
# 7. FULL SPECTRAL WORKFLOW
# =============================================================================

def run_sen2sr_fs(
    input_10m_tif,
    input_20m_tif,
    output_dir=DEFAULT_OUTPUT_DIR,
    full_model_path=FULL_MODEL_PATH,
):

    output_tif = build_output_path(
        input_10m_tif=input_10m_tif,
        output_dir=output_dir,
    )

    stack, reference_da = build_initial_full_stack(
        input_10m_tif=input_10m_tif,
        input_20m_tif=input_20m_tif,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X = torch.from_numpy(stack).float().to(device)

    X = torch.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print("Loading SEN2SR Full Spectral model...")

    full_model = mlstac.load(
        full_model_path
    ).compiled_model(
        device=device
    )

    print("Running SEN2SR Full Spectral...")

    superX = sen2sr.predict_large(
        model=full_model,
        X=X,
        overlap=32,
    )

    print("Saving GeoTIFF...")

    save_tensor_as_geotiff(
        tensor=superX,
        reference_da=reference_da,
        output_path=output_tif,
    )

    print(f"Done: {output_tif}")

    return output_tif


# =============================================================================
# 8. TEST ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_sen2sr_fs(
        input_10m_tif="/path/to/S2_10m.tif",
        input_20m_tif="/path/to/S2_20m.tif",
    )