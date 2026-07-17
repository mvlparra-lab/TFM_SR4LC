#!/usr/bin/env python3

"""
SEN2SR super-resolution utilities for SR4LC.

Based on:
- stac2cube super-resolution workflow
- tacofoundation/SEN2SR model structure

Adapted by: Victoria León
Project: SR4LC
"""

import os
import io
import sys
import math
import ast
import contextlib

import mlstac
import torch
import torch.nn.functional as F
import xarray as xr
import numpy as np
import sen2sr
import rioxarray

from affine import Affine
from tqdm.auto import tqdm

from .get_spectral_indices import calculate_spectral_index


@contextlib.contextmanager
def suppress_tqdm():
    """
    Suppress tqdm stderr output.
    """

    with contextlib.redirect_stderr(io.StringIO()):
        yield


# =============================================================================
# SEN2SR MODEL CONFIGURATION
# =============================================================================

RGBN_REQUIRED = {
    "blue",
    "green",
    "red",
    "nir",
}

FULL_REQUIRED = {
    "blue",
    "green",
    "red",
    "nir",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "swir16",
    "swir22",
}

# IMPORTANT:
# SEN2SR expects this exact model input order

RGBN_MODEL_ORDER = [
    "red",
    "green",
    "blue",
    "nir",
]

FULL_MODEL_ORDER = [
    "red",
    "green",
    "blue",
    "nir",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "swir16",
    "swir22",
]


# =============================================================================
# SEN2SR MODEL PATHS
# =============================================================================

SEN2SR_MODEL_ROOT = (
    "/teamspace/studios/this_studio/"
    "STAC2Cube/4_SuperResolution/"
    "SEN2SR/model"
)

SEN2SR_RGBN_MODEL_PATH = (
    SEN2SR_MODEL_ROOT
    + "/NonReference_RGBN_x4"
)

SEN2SR_FULL_MODEL_PATH = (
    SEN2SR_MODEL_ROOT
    + "/SEN2SR_Full"
)


# =============================================================================
# HELPERS
# =============================================================================

def _bands_in_cube_order(
    da: xr.DataArray,
    band_set: set[str],
) -> list[str]:
    """
    Return bands present in cube preserving cube order.
    """

    cube_order = [
        str(band)
        for band in da.coords["band"].values
    ]

    return [
        band
        for band in cube_order
        if band in band_set
    ]


def _validate_required_bands(
    available: set[str],
    required: set[str],
    model_type: str,
) -> None:
    """
    Validate required bands.
    """

    missing = sorted(required - available)

    if missing:

        raise ValueError(
            f"model_type='{model_type}' requires:\n"
            f"required: {', '.join(sorted(required))}\n"
            f"missing: {', '.join(missing)}\n"
            f"present: {', '.join(sorted(available))}"
        )


def _parse_list_attr(value):
    """
    Parse list-like NetCDF attributes.
    """

    if value is None:
        return None

    if isinstance(value, str):

        try:
            return ast.literal_eval(value)

        except Exception:

            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

    if isinstance(
        value,
        (list, tuple, np.ndarray)
    ):
        return list(value)

    return None

def affine_from_xy_centers(
    x: np.ndarray,
    y: np.ndarray,
) -> Affine:
    """
    Build affine transform from x/y center coordinates.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))

    c = float(x[0] - dx * 0.5)
    f = float(y[0] - dy * 0.5)

    return Affine(
        dx,
        0.0,
        c,
        0.0,
        dy,
        f,
    )


def _copy_time_coords(
    src: xr.DataArray,
    dst: xr.DataArray,
) -> xr.DataArray:
    """
    Copy all 1D time-dependent coordinates.
    """

    if "time" not in src.dims or "time" not in dst.dims:
        return dst

    for name, coord in src.coords.items():

        if name == "time":
            continue

        if coord.dims == ("time",):

            dst = dst.assign_coords(
                {
                    name: (
                        "time",
                        src[name].values,
                    )
                }
            )

    return dst


def _extract_cf_crs_and_geotransform(
    ds: xr.Dataset,
    data_var: str,
):
    """
    Extract CRS and transform from CF NetCDF metadata.
    """

    da = ds[data_var]

    gm_name = da.attrs.get(
        "grid_mapping",
        None
    )

    if (
        gm_name is None
        and "spatial_ref" in ds.variables
    ):
        gm_name = "spatial_ref"

    crs_wkt = None
    geotransform = None

    # =========================================================
    # Read CRS info from spatial_ref
    # =========================================================

    if (
        gm_name is not None
        and gm_name in ds.variables
    ):

        gm = ds[gm_name]

        for key in (
            "spatial_ref",
            "crs_wkt",
            "WKT",
            "proj_wkt",
            "esri_pe_string",
        ):

            value = gm.attrs.get(
                key,
                None
            )

            if (
                isinstance(value, str)
                and value.strip()
            ):
                crs_wkt = value
                break

        gt = (
            gm.attrs.get("GeoTransform", None)
            or gm.attrs.get("geotransform", None)
        )

        if isinstance(gt, str):

            parts = [
                part
                for part in gt.replace(",", " ").split()
                if part
            ]

            if len(parts) == 6:

                geotransform = [
                    float(part)
                    for part in parts
                ]

        elif (
            isinstance(gt, (list, tuple))
            and len(gt) == 6
        ):

            geotransform = [
                float(part)
                for part in gt
            ]

    # =========================================================
    # Fallback CRS
    # =========================================================

    if crs_wkt is None:

        crs_wkt = (
            da.attrs.get("crs_wkt", None)
            or da.attrs.get("spatial_ref", None)
        )

    # =========================================================
    # Build affine transform
    # =========================================================

    transform = None

    if geotransform is not None:

        c, a, b, f, d, e = geotransform

        transform = Affine(
            a,
            b,
            c,
            d,
            e,
            f,
        )

    elif (
        "x" in da.coords
        and "y" in da.coords
    ):

        transform = affine_from_xy_centers(
            da["x"].values,
            da["y"].values,
        )

    return crs_wkt, transform


def dilate_mask_2d(
    mask: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """
    Dilate a 2D mask using torch max pooling.
    """

    if radius_px <= 0:
        return mask

    mask_tensor = torch.from_numpy(
        mask.astype(np.float32)
    )[None, None, :, :]

    kernel_size = 2 * radius_px + 1

    dilated = F.max_pool2d(
        mask_tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=radius_px,
    )

    return (
        dilated[0, 0] > 0
    ).cpu().numpy()

# =============================================================================
# SINGLE TIME STEP SUPER-RESOLUTION
# =============================================================================

def superresolve_single_time(
    da,
    crs_wkt,
    transform,
    model,
    device,
    bands_to_use,
    model_band_order,
    old_res=10.0,
    new_res=2.5,
    nan_pixel_buffer=8,
    edge_crop_px=8,
):
    """
    Super-resolve a single time slice.
    """

    da = da.sel(
        band=bands_to_use
    )

    da.rio.set_spatial_dims(
        "x",
        "y",
        inplace=True
    )

    # =========================================================
    # Preserve original cube band order
    # =========================================================

    orig_band_order = da.band.values

    orig_attrs = dict(da.attrs)

    orig_attrs.pop(
        "transform",
        None
    )

    orig_attrs.pop(
        "grid_mapping",
        None
    )

    time_coord = da.coords.get(
        "time",
        None
    )

    # =========================================================
    # Reorder into SEN2SR model order
    # =========================================================

    new_order = list(
        model_band_order
    )

    da_reordered = da.sel(
        band=new_order
    )

    # =========================================================
    # Pad image to square
    # =========================================================

    ny = da_reordered.sizes["y"]
    nx = da_reordered.sizes["x"]

    new_side = math.ceil(
        max(ny, nx) / 128
    ) * 128

    pad_y = new_side - ny
    pad_x = new_side - nx

    pad_dict = {
        "y": (
            pad_y // 2,
            pad_y - pad_y // 2,
        ),
        "x": (
            pad_x // 2,
            pad_x - pad_x // 2,
        ),
    }

    da_square = da_reordered.pad(
        pad_dict,
        constant_values=0,
    )

    # =========================================================
    # Build NaN mask
    # =========================================================

    mask_lr = (
        da_square
        .isnull()
        .any(dim="band")
        .compute()
        .to_numpy()
    )

    # =========================================================
    # Tensor conversion
    # =========================================================

    X = torch.from_numpy(
        da_square
        .compute()
        .to_numpy()
        .astype("float32")
    ).to(device)

    X = torch.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # =========================================================
    # SEN2SR inference
    # =========================================================

    with suppress_tqdm():

        superX = sen2sr.predict_large(
            model=model,
            X=X,
            overlap=32,
        )

    # =========================================================
    # Upsample NaN mask
    # =========================================================

    scale = int(
        round(old_res / new_res)
    )

    mask_lr_t = torch.from_numpy(
        mask_lr.astype(np.float32)
    )[None, None, :, :].to(device)

    mask_hr_t = F.interpolate(
        mask_lr_t,
        scale_factor=scale,
        mode="nearest",
    )

    mask_hr = mask_hr_t[
        0,
        0
    ].bool()

    # =========================================================
    # Build HR transform
    # =========================================================

    pad_y_top_hr = (
        pad_dict["y"][0]
        * scale
    )

    pad_x_left_hr = (
        pad_dict["x"][0]
        * scale
    )

    scaled_tf = (
        transform
        * Affine.scale(
            1 / scale,
            1 / scale,
        )
    )

    full_tf = (
        scaled_tf
        * Affine.translation(
            -pad_x_left_hr,
            -pad_y_top_hr,
        )
    )

    # =========================================================
    # Crop back to original extent
    # =========================================================

    orig_h_lr = da_reordered.sizes["y"]
    orig_w_lr = da_reordered.sizes["x"]

    y0 = pad_y_top_hr
    y1 = pad_y_top_hr + orig_h_lr * scale

    x0 = pad_x_left_hr
    x1 = pad_x_left_hr + orig_w_lr * scale

    superX = superX[
        :,
        y0:y1,
        x0:x1,
    ]

    mask_hr = mask_hr[
        y0:y1,
        x0:x1,
    ]

    cropped_tf = (
        full_tf
        * Affine.translation(
            x0,
            y0,
        )
    )

    # =========================================================
    # Edge crop
    # =========================================================

    if edge_crop_px > 0:

        H = superX.shape[1]
        W = superX.shape[2]

        if (
            2 * edge_crop_px < H
            and 2 * edge_crop_px < W
        ):

            superX = superX[
                :,
                edge_crop_px:-edge_crop_px,
                edge_crop_px:-edge_crop_px,
            ]

            mask_hr = mask_hr[
                edge_crop_px:-edge_crop_px,
                edge_crop_px:-edge_crop_px,
            ]

            cropped_tf = (
                cropped_tf
                * Affine.translation(
                    edge_crop_px,
                    edge_crop_px,
                )
            )

    # =========================================================
    # NaN buffer
    # =========================================================

    if nan_pixel_buffer > 0:

        mask_hr_np = (
            mask_hr
            .detach()
            .cpu()
            .numpy()
        )

        mask_hr_np = dilate_mask_2d(
            mask_hr_np,
            radius_px=int(
                nan_pixel_buffer
            ),
        )

        mask_hr = torch.from_numpy(
            mask_hr_np
        ).to(device)

    superX[
        :,
        mask_hr.bool()
    ] = float("nan")

    # =========================================================
    # Build xarray output
    # =========================================================

    arr = (
        superX
        .detach()
        .cpu()
        .numpy()
    )

    var_name = (
        da.name
        or "Spectral_Temporal_Stack"
    )

    da_hr = xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={
            "band": new_order,
            "y": np.arange(arr.shape[1]),
            "x": np.arange(arr.shape[2]),
        },
        name=var_name,
    )

    da_hr.attrs = orig_attrs

    if time_coord is not None:

        da_hr = da_hr.assign_coords(
            time=time_coord
        )

    ds_tmp = da_hr.to_dataset(
        name=var_name
    )

    ds_tmp.rio.set_spatial_dims(
        "x",
        "y",
        inplace=True
    )

    ds_tmp.rio.write_crs(
        crs_wkt,
        inplace=True
    )

    ds_tmp.rio.write_transform(
        cropped_tf,
        inplace=True
    )

    # =========================================================
    # Restore cube band order
    # =========================================================

    ds_tmp = ds_tmp.sel(
        band=orig_band_order
    )

    # =========================================================
    # World coordinates
    # =========================================================

    W = ds_tmp.sizes["x"]
    H = ds_tmp.sizes["y"]

    xs = (
        cropped_tf.c
        + cropped_tf.a
        * (np.arange(W) + 0.5)
    )

    ys = (
        cropped_tf.f
        + cropped_tf.e
        * (np.arange(H) + 0.5)
    )

    ds_tmp = ds_tmp.assign_coords(
        x=("x", xs.astype(np.float64)),
        y=("y", ys.astype(np.float64)),
    )

    da_super = ds_tmp[var_name]

    da_super.rio.set_spatial_dims(
        "x",
        "y",
        inplace=True
    )

    da_super.rio.write_crs(
        crs_wkt,
        inplace=True
    )

    da_super.rio.write_transform(
        cropped_tf,
        inplace=True
    )

    da_super.attrs.update(
        orig_attrs
    )

    return da_super

# =============================================================================
# FULL CUBE SUPER-RESOLUTION
# =============================================================================

def super_resolve_cube(
    input_path,
    output_path: str | None = None,
    var_name: str = "Spectral_Temporal_Stack",
    nan_pixel_buffer: int = 8,
    model_type: str | None = None,
):
    """
    Super-resolve a Sentinel-2 cube using SEN2SR.

    model_type:
    - None: auto-detect
    - "rgbn": uses NonReference_RGBN_x4
    - "full_spectral": uses SEN2SR_Full
    """

    # =========================================================
    # Output path
    # =========================================================

    if output_path is None:

        if isinstance(input_path, xr.DataArray):
            raise ValueError(
                "Provide output_path when input_path is a DataArray."
            )

        base, ext = os.path.splitext(input_path)

        if ext == "":
            ext = ".nc"

        output_path = f"{base}_sr{ext}"

    # =========================================================
    # Resolution settings
    # =========================================================

    old_res = 10.0
    new_res = 2.5
    edge_crop_px = 8

    if isinstance(input_path, xr.DataArray):
        raise ValueError(
            "This version expects a NetCDF file path."
        )

    # =========================================================
    # Load input cube
    # =========================================================

    ds_in = xr.open_dataset(input_path)

    dataarray = ds_in[var_name]

    crs_wkt, transform = _extract_cf_crs_and_geotransform(
        ds_in,
        var_name,
    )

    if crs_wkt is None:
        raise ValueError(
            "Could not extract CRS WKT from spatial_ref."
        )

    if transform is None:
        raise ValueError(
            "Could not extract/build transform."
        )

    # =========================================================
    # Read metadata
    # =========================================================

    indices = _parse_list_attr(
        dataarray.attrs.get("indices", None)
    )

    spectral_bands_attr = _parse_list_attr(
        dataarray.attrs.get("spectral_bands", None)
    )

    band_coord_set = {
        str(band)
        for band in dataarray.coords["band"].values
    }

    if spectral_bands_attr and len(spectral_bands_attr) > 0:

        spectral_set = {
            str(band)
            for band in spectral_bands_attr
        }

    else:

        idx_set = set(indices) if indices else set()
        spectral_set = band_coord_set - idx_set

    missing_from_coord = sorted(
        spectral_set - band_coord_set
    )

    if missing_from_coord:

        raise ValueError(
            "attrs['spectral_bands'] contains bands not present "
            "in the band coordinate:\n"
            f"{', '.join(missing_from_coord)}"
        )

    # =========================================================
    # Decide model type
    # =========================================================

    if model_type is None:

        if spectral_set.issubset(RGBN_REQUIRED):
            model_type_used = "rgbn"

        else:
            model_type_used = "full_spectral"

    else:

        model_type_used = str(
            model_type
        ).strip().lower()

        if model_type_used not in (
            "rgbn",
            "full_spectral",
        ):
            raise ValueError(
                "model_type must be None, 'rgbn', or 'full_spectral'."
            )

    # =========================================================
    # Select bands and model
    # =========================================================

    if model_type_used == "rgbn":

        _validate_required_bands(
            spectral_set,
            RGBN_REQUIRED,
            model_type_used,
        )

        bands_to_use = _bands_in_cube_order(
            dataarray,
            RGBN_REQUIRED,
        )

        model_band_order = RGBN_MODEL_ORDER
        model_path = SEN2SR_RGBN_MODEL_PATH

    else:

        _validate_required_bands(
            spectral_set,
            FULL_REQUIRED,
            model_type_used,
        )

        bands_to_use = _bands_in_cube_order(
            dataarray,
            FULL_REQUIRED,
        )

        model_band_order = FULL_MODEL_ORDER
        model_path = SEN2SR_FULL_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SEN2SR model path not found: {model_path}"
        )

    print(f"\nUsing SEN2SR model: {model_path}")

    dataarray_sub = dataarray.sel(
        band=bands_to_use
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = mlstac.load(
        model_path
    ).compiled_model(
        device=device
    )

    # =========================================================
    # Super-resolve time series or single image
    # =========================================================

    if "time" in dataarray_sub.dims:

        super_list = []

        for time_value in tqdm(
            dataarray_sub.time.values,
            desc=f"Super-resolving time steps ({model_type_used})",
            unit="date",
            file=sys.stdout,
            dynamic_ncols=False,
        ):

            da_t = dataarray_sub.sel(
                time=time_value
            )

            da_sr_t = superresolve_single_time(
                da=da_t,
                crs_wkt=crs_wkt,
                transform=transform,
                model=model,
                device=device,
                bands_to_use=bands_to_use,
                model_band_order=model_band_order,
                old_res=old_res,
                new_res=new_res,
                nan_pixel_buffer=nan_pixel_buffer,
                edge_crop_px=edge_crop_px,
            )

            super_list.append(da_sr_t)

        da_super_all = xr.concat(
            super_list,
            dim="time",
            coords="minimal",
            compat="override",
        )

        da_super_all = _copy_time_coords(
            dataarray,
            da_super_all,
        )

        transform_first = super_list[0].rio.transform()

        da_super_all.rio.set_spatial_dims(
            "x",
            "y",
            inplace=True
        )

        da_super_all.rio.write_crs(
            crs_wkt,
            inplace=True
        )

        da_super_all.rio.write_transform(
            transform_first,
            inplace=True
        )

    else:

        da_super_all = superresolve_single_time(
            da=dataarray_sub,
            crs_wkt=crs_wkt,
            transform=transform,
            model=model,
            device=device,
            bands_to_use=bands_to_use,
            model_band_order=model_band_order,
            old_res=old_res,
            new_res=new_res,
            nan_pixel_buffer=nan_pixel_buffer,
            edge_crop_px=edge_crop_px,
        )

    # =========================================================
    # Optional spectral indices
    # =========================================================

    if indices and len(indices) > 0:

        stac_idx = calculate_spectral_index(
            da_super_all,
            mission="s2",
            indices=indices,
        )

        if isinstance(stac_idx, xr.Dataset):

            stac_idx = stac_idx.to_array(
                dim="band"
            )

            stac_idx = stac_idx.assign_coords(
                band=list(stac_idx["band"].values)
            )

        elif isinstance(stac_idx, xr.DataArray):

            if "band" not in stac_idx.dims:

                for dim in stac_idx.dims:

                    if dim not in (
                        "time",
                        "x",
                        "y",
                    ):

                        stac_idx = stac_idx.rename(
                            {dim: "band"}
                        )

                        break

            if "band" not in stac_idx.dims:

                idx_name = stac_idx.name or "index"

                stac_idx = stac_idx.expand_dims(
                    band=[idx_name]
                )

        da_super_all = xr.concat(
            [
                da_super_all,
                stac_idx,
            ],
            dim="band",
        )

        da_super_all.rio.set_spatial_dims(
            "x",
            "y",
            inplace=True
        )

        da_super_all.rio.write_crs(
            crs_wkt,
            inplace=True
        )

        da_super_all.rio.write_transform(
            da_super_all.rio.transform(),
            inplace=True
        )

        da_super_all.attrs["indices"] = indices

    # =========================================================
    # Finalize and write NetCDF
    # =========================================================

    da_super_all.name = var_name

    ds_out = da_super_all.to_dataset(
        name=var_name
    )

    ds_out.rio.set_spatial_dims(
        "x",
        "y",
        inplace=True
    )

    ds_out.rio.write_crs(
        crs_wkt,
        inplace=True
    )

    ds_out.rio.write_transform(
        da_super_all.rio.transform(),
        inplace=True
    )

    ds_out[var_name].attrs.pop(
        "grid_mapping",
        None,
    )

    ds_out[var_name].encoding["grid_mapping"] = "spatial_ref"

    try:

        from rasterio.crs import CRS

        epsg = CRS.from_wkt(
            crs_wkt
        ).to_epsg()

        if epsg is not None:

            crs_epsg = f"EPSG:{epsg}"

            ds_out.attrs["crs"] = crs_epsg
            ds_out[var_name].attrs["crs"] = crs_epsg

    except Exception:
        pass

    if "spatial_ref" in ds_out.variables:

        transform_out = ds_out.rio.transform()

        ds_out["spatial_ref"].attrs["GeoTransform"] = (
            f"{transform_out.c} "
            f"{transform_out.a} "
            f"{transform_out.b} "
            f"{transform_out.f} "
            f"{transform_out.d} "
            f"{transform_out.e}"
        )

    transform_out = ds_out.rio.transform()

    transform9 = [
        float(transform_out.a),
        float(transform_out.b),
        float(transform_out.c),
        float(transform_out.d),
        float(transform_out.e),
        float(transform_out.f),
        0.0,
        0.0,
        1.0,
    ]

    ds_out.attrs["transform"] = transform9
    ds_out[var_name].attrs["transform"] = transform9

    ds_out.to_netcdf(output_path)

    print(
        "Data cube is super-resolved to 2.5 meters! "
        f"(SEN2SR, model_type={model_type_used})"
    )