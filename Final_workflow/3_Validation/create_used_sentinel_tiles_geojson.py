#!/usr/bin/env python3

"""
Create a GeoJSON showing the Sentinel-2 MGRS tiles used in the dataset.

The script reads train.csv, val.csv and test.csv, extracts the Sentinel-2
tile code from each patch filename and generates one polygon per unique tile.

Author: Victoria León
Project: SR4LC

"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
import csv
import json
import re

import mgrs
from pyproj import CRS, Transformer


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

DATASET_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
    "outputs/Segmentation/DataSet2"
)

OUTPUT_GEOJSON = DATASET_DIR / "sentinel2_tiles_used.geojson"

CSV_FILES = [
    DATASET_DIR / "train.csv",
    DATASET_DIR / "val.csv",
    DATASET_DIR / "test.csv",
]


# =============================================================================
# 3. TILE EXTRACTION
# =============================================================================

def extract_tile_code(image_path: str) -> str | None:
    """
    Extract a Sentinel-2 MGRS tile code such as T33TXF from a filename.
    """
    filename = Path(image_path).name

    match = re.search(r"_T(\d{2}[A-Z]{3})_", filename)

    if match is None:
        return None

    return match.group(1)


def collect_used_tiles() -> dict[str, int]:
    """
    Read all split CSV files and count patches associated with each tile.
    """
    tile_counts: dict[str, int] = {}

    for csv_path in CSV_FILES:
        with csv_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                tile_code = extract_tile_code(row["image"])

                if tile_code is None:
                    continue

                tile_counts[tile_code] = (
                    tile_counts.get(tile_code, 0) + 1
                )

    return tile_counts


# =============================================================================
# 4. MGRS TILE GEOMETRY
# =============================================================================

def tile_to_polygon(tile_code: str) -> list[list[float]]:
    """
    Convert a 100 km Sentinel-2 MGRS tile into a WGS84 polygon.

    Parameters
    ----------
    tile_code : str
        MGRS tile without the initial T, for example 33TXF.

    Returns
    -------
    list[list[float]]
        Closed polygon ring in longitude-latitude coordinates.
    """
    mgrs_converter = mgrs.MGRS()

    # Add zero coordinates to obtain the south-west corner of the MGRS square.
    southwest_mgrs = f"{tile_code}0000000000"

    latitude, longitude = mgrs_converter.toLatLon(
        southwest_mgrs
    )

    zone = int(tile_code[:2])
    latitude_band = tile_code[2]

    # Latitude bands N-X belong to the northern hemisphere.
    northern_hemisphere = latitude_band >= "N"

    utm_crs = CRS.from_dict({
        "proj": "utm",
        "zone": zone,
        "south": not northern_hemisphere,
        "datum": "WGS84",
    })

    to_utm = Transformer.from_crs(
        "EPSG:4326",
        utm_crs,
        always_xy=True,
    )

    to_wgs84 = Transformer.from_crs(
        utm_crs,
        "EPSG:4326",
        always_xy=True,
    )

    southwest_x, southwest_y = to_utm.transform(
        longitude,
        latitude,
    )

    # MGRS grid squares are 100 km × 100 km.
    corners_utm = [
        (southwest_x, southwest_y),
        (southwest_x + 100_000, southwest_y),
        (southwest_x + 100_000, southwest_y + 100_000),
        (southwest_x, southwest_y + 100_000),
        (southwest_x, southwest_y),
    ]

    polygon = []

    for x, y in corners_utm:
        lon, lat = to_wgs84.transform(x, y)
        polygon.append([lon, lat])

    return polygon


# =============================================================================
# 5. GEOJSON CREATION
# =============================================================================

def create_geojson(tile_counts: dict[str, int]) -> dict:
    """
    Create a GeoJSON FeatureCollection from the unique tile codes.
    """
    features = []

    for tile_code in sorted(tile_counts):
        polygon = tile_to_polygon(tile_code)

        feature = {
            "type": "Feature",
            "properties": {
                "tile": f"T{tile_code}",
                "patch_count": tile_counts[tile_code],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon],
            },
        }

        features.append(feature)

    return {
        "type": "FeatureCollection",
        "name": "sentinel2_tiles_used",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            },
        },
        "features": features,
    }


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    tile_counts = collect_used_tiles()

    print(f"Unique Sentinel-2 tiles found: {len(tile_counts)}")

    for tile_code, patch_count in sorted(tile_counts.items()):
        print(
            f"  T{tile_code}: {patch_count} patches"
        )

    geojson = create_geojson(tile_counts)

    with OUTPUT_GEOJSON.open("w") as file:
        json.dump(
            geojson,
            file,
            indent=2,
        )

    print(f"\nGeoJSON written to:\n{OUTPUT_GEOJSON}")


if __name__ == "__main__":
    main()