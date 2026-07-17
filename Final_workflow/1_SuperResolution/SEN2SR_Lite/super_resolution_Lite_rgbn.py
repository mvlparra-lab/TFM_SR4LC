#!/usr/bin/env python3

"""
Based on: Aybar, C., Contreras, J., Donike, S., Portalés-Julià, E., Mateo-García, G., & Gómez-Chova, L. (2025).
          A Radiometrically and Spatially Consistent Super-Resolution Framework for Sentinel-2.
          SSRN. https://doi.org/10.2139/ssrn.5247739
Author: Victoria León
Project: SR4LC


SEN2SRLite RGBN
GeoTIFF -> Super Resolution -> GeoTIFF
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
    "pKq003_SR4LC_Data/outputs/S2_SRLite/RGBN"
)

DEFAULT_MODEL_PATH = (
    "/teamspace/studios/this_studio/"
    "SR4LC/model/SEN2SRLite_RGBN"
)


# =============================================================================
# 3. OUTPUT NAMING
# =============================================================================

def build_output_path(
    input_tif,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Build output path.

    Example:
    input:
        S2A_MSIL2A_..._composite.tif

    output:
        SRL_S2A_MSIL2A_....tif
    """

    input_path = Path(input_tif)
    output_dir = Path(output_dir)

    base_name = input_path.stem

    if base_name.endswith("_composite"):
        base_name = base_name[:-10]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_tif = output_dir / f"SRL_RGB_{base_name}.tif"

    return output_tif


# =============================================================================
# 4. SAVE OUTPUT GEOTIFF
# =============================================================================

def save_tensor_as_geotiff(
    tensor,
    reference_da,
    output_path,
):
    """
    Save SR tensor as GeoTIFF using the original image georeferencing.
    """

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
# 5. INPUT VALIDATION
# =============================================================================

def validate_tif_readability(
    input_tif,
    step=512,
):
    """
    Validate that the TIFF can be read across the full image.

    If reading fails, raises an error with approximate column and row.
    """

    with rasterio.open(input_tif) as src:

        for row in range(0, src.height, step):
            for col in range(0, src.width, step):

                width = min(step, src.width - col)
                height = min(step, src.height - row)

                try:
                    src.read(
                        window=((row, row + height),
                                (col, col + width))
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Read failure near "
                        f"column {col}, row {row}. "
                        f"Original error: {e}"
                    )

# =============================================================================
# 6. SEN2SRLITE RGBN WORKFLOW
# =============================================================================

def run_sen2srlite_rgbn(
    input_tif,
    output_dir=DEFAULT_OUTPUT_DIR,
    model_path=DEFAULT_MODEL_PATH,
):
    """
    Run SEN2SRLite RGBN on a Sentinel-2 composite.

    Expected TIFF bands:
        1 -> B02
        2 -> B03
        3 -> B04
        4 -> B08
        5 -> B11
        6 -> B12
    """

    output_tif = build_output_path(
        input_tif=input_tif,
        output_dir=output_dir,
    )
    
    validate_tif_readability(input_tif)

    print("Loading image...")

    da_raw = rxr.open_rasterio(
        input_tif,
        masked=False,
    )

    if isinstance(da_raw, list):
        da = da_raw[0]
    else:
        da = da_raw

    da = da.astype("float32")

    # Model order:
    # B04, B03, B02, B08
    
    red = da.sel(band=3).load()
    green = da.sel(band=2).load()
    blue = da.sel(band=1).load()
    nir = da.sel(band=4).load()

    rgbn = np.stack(
        [
            np.asarray(red),
            np.asarray(green),
            np.asarray(blue),
            np.asarray(nir),
        ],
        axis=0,
    )

    rgbn = (rgbn / 10000.0).astype("float32")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X = torch.from_numpy(rgbn).float().to(device)

    X = torch.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print("Loading model...")

    model = mlstac.load(
        model_path
    ).compiled_model(
        device=device
    )

    print("Running Super-Resolution...")

    superX = sen2sr.predict_large(
        model=model,
        X=X,
        overlap=32,
    )

    print("Saving GeoTIFF...")

    save_tensor_as_geotiff(
        tensor=superX,
        reference_da=da.sel(band=1),
        output_path=output_tif,
    )

    print(f"Done: {output_tif}")

    return output_tif


# =============================================================================
# 7. BATCH PROCESSING
# =============================================================================

def run_sen2srlite_rgbn_folder(
    input_dir,
    output_dir=DEFAULT_OUTPUT_DIR,
    model_path=DEFAULT_MODEL_PATH,
):
    """
    Run SEN2SRLite RGBN for all TIFF files in a folder.

    If one image fails, continue with the next one.
    """

    input_dir = Path(input_dir)

    tif_files = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff"))
    )

    converted = []
    failed = []

    for tif_path in tif_files:

        output_tif = build_output_path(
            input_tif=tif_path,
            output_dir=output_dir,
        )
        
        if output_tif.exists():
            print("\n========================================")
            print(f"Already exists, skipping: {output_tif.name}")
            print("========================================")
            continue
        
        print("\n========================================")
        print(f"Processing image: {tif_path.name}")
        print("========================================")

        try:
            output_tif = run_sen2srlite_rgbn(
                input_tif=tif_path,
                output_dir=output_dir,
                model_path=model_path,
            )

            converted.append(output_tif)

        except Exception as e:
            failed.append(
                {
                    "image": tif_path.name,
                    "error": str(e),
                }
            )

            print(f"ERROR: could not process {tif_path.name}")
            print(f"Reason: {e}")
            print("Skipping to next image...")

    print("\n========================================")
    print("FINAL SUMMARY")
    print("========================================")
    print(f"Total converted images: {len(converted)}")
    print(f"Total failed images: {len(failed)}")

    for i, item in enumerate(failed, start=1):
        print(
            f"Failed image nº {i}: "
            f"{item['image']} - {item['error']}"
        )

    return converted, failed


# =============================================================================
# 8. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_sen2srlite_rgbn_folder(
        input_dir=(
            "/teamspace/lightning_storage/"
            "pKq003_SR4LC_Data/inputs/Sentinel2"
        )
    )