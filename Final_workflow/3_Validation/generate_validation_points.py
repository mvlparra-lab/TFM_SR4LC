#!/usr/bin/env python3

"""
Generate spatially balanced validation points for SR4LC.

Author: Victoria León
Project: SR4LC

Description:
Generate validation points within valid prediction pixels that also
overlap the Coastal Zones reference layer.

The reference sampling ratio is 250 points per 250,000 km². Because the
strict ratio produces too few points for the available scenes, the
reference sample size is multiplied by a configurable factor and a
minimum number of points is guaranteed for each raster.

Points are distributed using a systematic grid. One random valid pixel
is selected from each sampled grid cell.

Output attributes:
    - ID: Unique point identifier.
    - R_Class: Visual reference class, initially set to 0.
    - CZ_Class: Coastal Zones class from CODE_1_18.
    - SR_Class: Predicted class sampled from the raster.

Run this script from the QGIS Python Console editor.
"""

import math
import os
import random
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsSpatialIndex,
    QgsUnitTypes,
    QgsVectorFileWriter,
    QgsVectorLayer,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTIONS_DIR = (
    r"C:\Users\Josemi\OneDrive\Documentos\Master\Practicas"
    r"\Planetek\pkq003_SR4LC\QGIS\results - stage"
    r"\6_Validation\Predictions"
)

COASTAL_ZONES_PATH = (
    r"C:\Users\Josemi\OneDrive\Documentos\Master\Practicas"
    r"\Planetek\pkq003_SR4LC\QGIS\results - stage"
    r"\4_Test\RGBN\Nueva version\81672\Results"
    r"\CZ_2018\CZ_2018.shp"
)

OUTPUT_PATH = (
    r"C:\Users\Josemi\OneDrive\Documentos\Master\Practicas"
    r"\Planetek\pkq003_SR4LC\QGIS\results - stage"
    r"\6_Validation\Points\validation_points.shp"
)

CZ_CLASS_FIELD = "CODE_1_18"
PREDICTION_BAND = 1

VALID_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8}
PREDICTION_NODATA = 0

AREA_PER_POINT_KM2 = 1_000.0
SAMPLING_MULTIPLIER = 4
MIN_POINTS_PER_SCENE = 5

RANDOM_SEED = 42
COUNT_BLOCK_SIZE = 1024
MAX_GRID_REFINEMENTS = 4


# =============================================================================
# LAYER LOADING
# =============================================================================

def load_coastal_zones(path):
    """Load and validate the Coastal Zones layer."""
    layer = QgsVectorLayer(path, "Coastal_Zones", "ogr")

    if not layer.isValid():
        raise RuntimeError(
            f"Could not load the Coastal Zones layer:\n{path}"
        )

    field_names = [field.name() for field in layer.fields()]

    if CZ_CLASS_FIELD not in field_names:
        raise RuntimeError(
            f"Field '{CZ_CLASS_FIELD}' was not found in Coastal Zones."
        )

    return layer


def load_prediction_rasters(directory):
    """Load all valid GeoTIFF prediction rasters in a directory."""
    directory_path = Path(directory)

    if not directory_path.is_dir():
        raise RuntimeError(
            f"Prediction directory does not exist:\n{directory}"
        )

    raster_paths = sorted(
        list(directory_path.glob("*.tif"))
        + list(directory_path.glob("*.tiff"))
    )

    if not raster_paths:
        raise RuntimeError(
            f"No GeoTIFF files were found in:\n{directory}"
        )

    rasters = []

    for raster_path in raster_paths:
        layer = QgsRasterLayer(
            str(raster_path),
            raster_path.stem,
        )

        if not layer.isValid():
            print(f"WARNING: Invalid raster skipped: {raster_path.name}")
            continue

        if layer.crs().mapUnits() != QgsUnitTypes.DistanceMeters:
            raise RuntimeError(
                f"Raster '{raster_path.name}' is not in a metre-based CRS."
            )

        dataset = gdal.Open(str(raster_path), gdal.GA_ReadOnly)

        if dataset is None:
            print(f"WARNING: GDAL could not open: {raster_path.name}")
            continue

        rasters.append(
            {
                "name": raster_path.stem,
                "path": str(raster_path),
                "layer": layer,
                "dataset": dataset,
            }
        )

    if not rasters:
        raise RuntimeError("No valid prediction rasters were loaded.")

    return rasters


# =============================================================================
# COASTAL ZONES MASK
# =============================================================================

def build_cz_union_geometry(cz_layer):
    """Create one union geometry from all Coastal Zones features."""
    geometries = [
        feature.geometry()
        for feature in cz_layer.getFeatures()
        if feature.geometry() and not feature.geometry().isEmpty()
    ]

    if not geometries:
        raise RuntimeError(
            "No valid geometries were found in Coastal Zones."
        )

    union_geometry = QgsGeometry.unaryUnion(geometries)

    if union_geometry.isEmpty():
        raise RuntimeError(
            "Could not build the Coastal Zones union geometry."
        )

    return union_geometry


def transform_geometry(geometry, source_crs, target_crs):
    """Return a transformed copy of a QGIS geometry."""
    transformed = QgsGeometry(geometry)

    if source_crs != target_crs:
        coordinate_transform = QgsCoordinateTransform(
            source_crs,
            target_crs,
            QgsProject.instance(),
        )
        transformed.transform(coordinate_transform)

    return transformed


def create_cz_mask(dataset, raster_crs, cz_union, cz_crs):
    """
    Rasterize Coastal Zones into an in-memory mask aligned with a raster.

    This implementation uses gdal.RasterizeLayer, which is available in
    standard QGIS GDAL installations.

    Mask values:
        1 = inside Coastal Zones
        0 = outside Coastal Zones
    """
    transformed_geometry = transform_geometry(
        cz_union,
        cz_crs,
        raster_crs,
    )

    ogr_geometry = ogr.CreateGeometryFromWkb(
        bytes(transformed_geometry.asWkb())
    )

    if ogr_geometry is None:
        raise RuntimeError(
            "Could not convert the Coastal Zones geometry to OGR."
        )

    memory_driver = gdal.GetDriverByName("MEM")
    mask_dataset = memory_driver.Create(
        "",
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,
        gdal.GDT_Byte,
    )

    if mask_dataset is None:
        raise RuntimeError(
            "Could not create the in-memory Coastal Zones mask."
        )

    mask_dataset.SetGeoTransform(dataset.GetGeoTransform())
    mask_dataset.SetProjection(dataset.GetProjection())

    mask_band = mask_dataset.GetRasterBand(1)
    mask_band.Fill(0)
    mask_band.SetNoDataValue(0)

    ogr_driver = ogr.GetDriverByName("Memory")
    ogr_dataset = ogr_driver.CreateDataSource("cz_memory")

    if ogr_dataset is None:
        raise RuntimeError(
            "Could not create the in-memory OGR dataset."
        )

    spatial_reference = None
    projection_wkt = dataset.GetProjection()

    if projection_wkt:
        from osgeo import osr
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromWkt(projection_wkt)

    ogr_layer = ogr_dataset.CreateLayer(
        "coastal_zones",
        srs=spatial_reference,
        geom_type=ogr.wkbMultiPolygon,
    )

    if ogr_layer is None:
        raise RuntimeError(
            "Could not create the in-memory Coastal Zones layer."
        )

    feature_definition = ogr_layer.GetLayerDefn()
    ogr_feature = ogr.Feature(feature_definition)
    ogr_feature.SetGeometry(ogr_geometry)

    if ogr_layer.CreateFeature(ogr_feature) != 0:
        raise RuntimeError(
            "Could not add the Coastal Zones geometry to the memory layer."
        )

    ogr_feature = None

    error = gdal.RasterizeLayer(
        mask_dataset,
        [1],
        ogr_layer,
        burn_values=[1],
        options=["ALL_TOUCHED=FALSE"],
    )

    ogr_dataset = None

    if error != 0:
        raise RuntimeError("Could not rasterize Coastal Zones.")

    mask_band.FlushCache()

    return mask_dataset


# =============================================================================
# VALID AREA
# =============================================================================

def build_valid_mask(prediction_array, cz_mask_array, nodata_value):
    """Return the pixels that are valid in both datasets."""
    valid_prediction = np.isin(
        prediction_array,
        list(VALID_CLASSES),
    )

    if nodata_value is not None:
        valid_prediction &= prediction_array != nodata_value

    return valid_prediction & (cz_mask_array == 1)


def count_valid_pixels(dataset, mask_dataset):
    """Count valid prediction pixels inside Coastal Zones."""
    prediction_band = dataset.GetRasterBand(PREDICTION_BAND)
    mask_band = mask_dataset.GetRasterBand(1)

    nodata_value = prediction_band.GetNoDataValue()

    if nodata_value is None:
        nodata_value = PREDICTION_NODATA

    valid_pixels = 0

    for y_offset in range(0, dataset.RasterYSize, COUNT_BLOCK_SIZE):
        y_size = min(
            COUNT_BLOCK_SIZE,
            dataset.RasterYSize - y_offset,
        )

        for x_offset in range(0, dataset.RasterXSize, COUNT_BLOCK_SIZE):
            x_size = min(
                COUNT_BLOCK_SIZE,
                dataset.RasterXSize - x_offset,
            )

            prediction_array = prediction_band.ReadAsArray(
                x_offset,
                y_offset,
                x_size,
                y_size,
            )

            cz_mask_array = mask_band.ReadAsArray(
                x_offset,
                y_offset,
                x_size,
                y_size,
            )

            valid_mask = build_valid_mask(
                prediction_array,
                cz_mask_array,
                nodata_value,
            )

            valid_pixels += int(valid_mask.sum())

    return valid_pixels


def calculate_scene_areas(rasters, cz_layer, cz_union):
    """Calculate valid sampling area for each prediction raster."""
    scenes = []

    for raster_data in rasters:
        dataset = raster_data["dataset"]
        geotransform = dataset.GetGeoTransform()

        pixel_area_m2 = abs(
            geotransform[1] * geotransform[5]
        )

        mask_dataset = create_cz_mask(
            dataset,
            raster_data["layer"].crs(),
            cz_union,
            cz_layer.crs(),
        )

        valid_pixel_count = count_valid_pixels(
            dataset,
            mask_dataset,
        )

        area_km2 = (
            valid_pixel_count * pixel_area_m2 / 1_000_000.0
        )

        if valid_pixel_count == 0:
            print(
                f"WARNING: No valid prediction pixels found in "
                f"{raster_data['name']}."
            )
            continue

        scenes.append(
            {
                **raster_data,
                "mask_dataset": mask_dataset,
                "valid_pixel_count": valid_pixel_count,
                "area_km2": area_km2,
                "point_count": 0,
                "allocation_remainder": 0.0,
            }
        )

        print(
            f"{raster_data['name']}: "
            f"{valid_pixel_count:,} valid pixels, "
            f"{area_km2:.2f} km²"
        )

    if not scenes:
        raise RuntimeError(
            "No valid sampling area was found in the prediction rasters."
        )

    return scenes


# =============================================================================
# POINT ALLOCATION
# =============================================================================

def calculate_total_points(total_area_km2, number_of_scenes):
    """Calculate the final sample size."""
    reference_points = max(
        1,
        round(total_area_km2 / AREA_PER_POINT_KM2),
    )

    multiplied_points = (
        reference_points * SAMPLING_MULTIPLIER
    )

    minimum_points = (
        number_of_scenes * MIN_POINTS_PER_SCENE
    )

    final_points = max(
        multiplied_points,
        minimum_points,
    )

    return reference_points, multiplied_points, final_points


def allocate_points_by_area(scenes, total_points):
    """
    Allocate points by valid area using the largest-remainder method.
    """
    total_area = sum(
        scene["area_km2"]
        for scene in scenes
    )

    minimum_required = (
        MIN_POINTS_PER_SCENE * len(scenes)
    )

    remaining_points = total_points - minimum_required

    for scene in scenes:
        scene["point_count"] = MIN_POINTS_PER_SCENE
        scene["allocation_remainder"] = 0.0

    if remaining_points <= 0:
        return

    allocated_extra = 0

    for scene in scenes:
        exact_allocation = (
            remaining_points
            * scene["area_km2"]
            / total_area
        )

        integer_allocation = math.floor(exact_allocation)

        scene["point_count"] += integer_allocation
        scene["allocation_remainder"] = (
            exact_allocation - integer_allocation
        )

        allocated_extra += integer_allocation

    points_left = remaining_points - allocated_extra

    ordered_scenes = sorted(
        scenes,
        key=lambda scene: scene["allocation_remainder"],
        reverse=True,
    )

    for scene in ordered_scenes[:points_left]:
        scene["point_count"] += 1


# =============================================================================
# CLASS EXTRACTION
# =============================================================================

def build_cz_spatial_index(cz_layer):
    """Build a spatial index for Coastal Zones."""
    return QgsSpatialIndex(cz_layer.getFeatures())


def extract_cz_class(
    point,
    point_crs,
    cz_layer,
    cz_spatial_index,
):
    """Extract the Coastal Zones class at a point."""
    point_geometry = QgsGeometry.fromPointXY(point)

    transformed_point = transform_geometry(
        point_geometry,
        point_crs,
        cz_layer.crs(),
    )

    candidate_ids = cz_spatial_index.intersects(
        transformed_point.boundingBox()
    )

    for feature_id in candidate_ids:
        feature = cz_layer.getFeature(feature_id)

        if not feature.geometry().intersects(transformed_point):
            continue

        value = feature[CZ_CLASS_FIELD]

        if value is None:
            return None

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    return None


# =============================================================================
# SYSTEMATIC GRID SAMPLING
# =============================================================================

def calculate_grid_shape(width, height, target_cells):
    """Calculate a near-square grid adapted to raster dimensions."""
    aspect_ratio = width / height

    columns = max(
        1,
        math.ceil(math.sqrt(target_cells * aspect_ratio)),
    )

    rows = max(
        1,
        math.ceil(target_cells / columns),
    )

    return rows, columns


def pixel_center_to_map(geotransform, column, row):
    """Convert raster indices to map coordinates at pixel centre."""
    x_coordinate = (
        geotransform[0]
        + (column + 0.5) * geotransform[1]
        + (row + 0.5) * geotransform[2]
    )

    y_coordinate = (
        geotransform[3]
        + (column + 0.5) * geotransform[4]
        + (row + 0.5) * geotransform[5]
    )

    return QgsPointXY(x_coordinate, y_coordinate)


def sample_one_valid_pixel_from_cell(
    scene,
    x_start,
    y_start,
    x_end,
    y_end,
):
    """Select one random valid pixel from a grid cell."""
    x_size = x_end - x_start
    y_size = y_end - y_start

    if x_size <= 0 or y_size <= 0:
        return None

    prediction_band = scene["dataset"].GetRasterBand(
        PREDICTION_BAND
    )

    mask_band = scene["mask_dataset"].GetRasterBand(1)

    prediction_array = prediction_band.ReadAsArray(
        x_start,
        y_start,
        x_size,
        y_size,
    )

    cz_mask_array = mask_band.ReadAsArray(
        x_start,
        y_start,
        x_size,
        y_size,
    )

    nodata_value = prediction_band.GetNoDataValue()

    if nodata_value is None:
        nodata_value = PREDICTION_NODATA

    valid_mask = build_valid_mask(
        prediction_array,
        cz_mask_array,
        nodata_value,
    )

    valid_rows, valid_columns = np.where(valid_mask)

    if len(valid_rows) == 0:
        return None

    random_index = random.randrange(len(valid_rows))

    local_row = int(valid_rows[random_index])
    local_column = int(valid_columns[random_index])

    raster_row = y_start + local_row
    raster_column = x_start + local_column

    sr_class = int(
        round(float(prediction_array[local_row, local_column]))
    )

    point = pixel_center_to_map(
        scene["dataset"].GetGeoTransform(),
        raster_column,
        raster_row,
    )

    return point, sr_class


def generate_scene_samples(scene):
    """
    Generate spatially balanced samples using a systematic grid.

    The grid is progressively refined if too many cells contain no
    valid prediction pixels.
    """
    required_points = scene["point_count"]
    selected_samples = []
    used_pixels = set()

    for refinement in range(MAX_GRID_REFINEMENTS):
        target_cells = required_points * (2 ** refinement)

        rows, columns = calculate_grid_shape(
            scene["dataset"].RasterXSize,
            scene["dataset"].RasterYSize,
            target_cells,
        )

        row_edges = np.linspace(
            0,
            scene["dataset"].RasterYSize,
            rows + 1,
            dtype=int,
        )

        column_edges = np.linspace(
            0,
            scene["dataset"].RasterXSize,
            columns + 1,
            dtype=int,
        )

        cells = [
            (
                column_edges[column_index],
                row_edges[row_index],
                column_edges[column_index + 1],
                row_edges[row_index + 1],
            )
            for row_index in range(rows)
            for column_index in range(columns)
        ]

        random.shuffle(cells)

        for x_start, y_start, x_end, y_end in cells:
            sample = sample_one_valid_pixel_from_cell(
                scene,
                x_start,
                y_start,
                x_end,
                y_end,
            )

            if sample is None:
                continue

            point, sr_class = sample

            pixel_key = (
                round(point.x(), 6),
                round(point.y(), 6),
            )

            if pixel_key in used_pixels:
                continue

            selected_samples.append((point, sr_class))
            used_pixels.add(pixel_key)

            if len(selected_samples) == required_points:
                return selected_samples

    return selected_samples


# =============================================================================
# OUTPUT
# =============================================================================

def create_output_layer(output_crs):
    """Create the validation point layer."""
    output_layer = QgsVectorLayer(
        f"Point?crs={output_crs.authid()}",
        "Validation_Points",
        "memory",
    )

    provider = output_layer.dataProvider()

    fields = QgsFields()
    fields.append(QgsField("ID", QVariant.Int))
    fields.append(QgsField("R_Class", QVariant.Int))
    fields.append(QgsField("CZ_Class", QVariant.Int))
    fields.append(QgsField("SR_Class", QVariant.Int))

    provider.addAttributes(fields)
    output_layer.updateFields()

    return output_layer


def generate_validation_points(
    scenes,
    cz_layer,
    cz_spatial_index,
):
    """Generate points and populate all output attributes."""
    output_crs = scenes[0]["layer"].crs()
    output_layer = create_output_layer(output_crs)
    provider = output_layer.dataProvider()

    point_id = 1

    for scene in scenes:
        samples = generate_scene_samples(scene)

        for point, sr_class in samples:
            cz_class = extract_cz_class(
                point,
                scene["layer"].crs(),
                cz_layer,
                cz_spatial_index,
            )

            if cz_class is None:
                continue

            output_point_geometry = QgsGeometry.fromPointXY(point)

            if scene["layer"].crs() != output_crs:
                output_point_geometry = transform_geometry(
                    output_point_geometry,
                    scene["layer"].crs(),
                    output_crs,
                )

            feature = QgsFeature(output_layer.fields())
            feature.setGeometry(output_point_geometry)

            feature["ID"] = point_id
            feature["R_Class"] = 0
            feature["CZ_Class"] = cz_class
            feature["SR_Class"] = sr_class

            provider.addFeature(feature)
            point_id += 1

        print(
            f"{scene['name']}: "
            f"{len(samples)}/{scene['point_count']} points created"
        )

        if len(samples) < scene["point_count"]:
            print(
                f"WARNING: Could not create all requested points for "
                f"{scene['name']}."
            )

    output_layer.updateExtents()

    return output_layer


def save_output_layer(output_layer, output_path):
    """Save the output as an ESRI Shapefile and load it in QGIS."""
    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    if os.path.exists(output_path):
        QgsVectorFileWriter.deleteShapeFile(output_path)

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "UTF-8"

    result = QgsVectorFileWriter.writeAsVectorFormatV3(
        output_layer,
        output_path,
        QgsCoordinateTransformContext(),
        options,
    )

    if result[0] != QgsVectorFileWriter.NoError:
        raise RuntimeError(
            f"Could not save the validation points:\n{result}"
        )

    saved_layer = QgsVectorLayer(
        output_path,
        "Validation_Points",
        "ogr",
    )

    if not saved_layer.isValid():
        raise RuntimeError(
            "The output was created but could not be loaded in QGIS."
        )

    QgsProject.instance().addMapLayer(saved_layer)

    return saved_layer


# =============================================================================
# SUMMARY
# =============================================================================

def print_sampling_summary(
    scenes,
    total_area_km2,
    reference_points,
    multiplied_points,
    final_points,
):
    """Print the sampling design summary."""
    print("\n" + "=" * 72)
    print("VALIDATION SAMPLING SUMMARY")
    print("=" * 72)
    print(f"Valid rasters: {len(scenes)}")
    print(f"Total valid area: {total_area_km2:.2f} km²")
    print(f"Reference-ratio points: {reference_points}")
    print(
        f"Reference points x {SAMPLING_MULTIPLIER}: "
        f"{multiplied_points}"
    )
    print(
        f"Minimum scene representation: "
        f"{len(scenes) * MIN_POINTS_PER_SCENE}"
    )
    print(f"Final number of points: {final_points}")

    print("\nPoint allocation by raster:")

    for scene in scenes:
        print(
            f"  {scene['name']}: "
            f"{scene['area_km2']:.2f} km² -> "
            f"{scene['point_count']} points"
        )

    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete validation-point workflow."""
    random.seed(RANDOM_SEED)

    print("Loading Coastal Zones...")
    cz_layer = load_coastal_zones(
        COASTAL_ZONES_PATH
    )

    print("Loading prediction rasters...")
    rasters = load_prediction_rasters(
        PREDICTIONS_DIR
    )

    print(f"Prediction rasters loaded: {len(rasters)}")

    print("\nPreparing Coastal Zones geometry...")
    cz_union = build_cz_union_geometry(
        cz_layer
    )

    print("\nCalculating valid sampling areas...")
    scenes = calculate_scene_areas(
        rasters,
        cz_layer,
        cz_union,
    )

    total_area_km2 = sum(
        scene["area_km2"]
        for scene in scenes
    )

    (
        reference_points,
        multiplied_points,
        final_points,
    ) = calculate_total_points(
        total_area_km2,
        len(scenes),
    )

    allocate_points_by_area(
        scenes,
        final_points,
    )

    print_sampling_summary(
        scenes,
        total_area_km2,
        reference_points,
        multiplied_points,
        final_points,
    )

    cz_spatial_index = build_cz_spatial_index(
        cz_layer
    )

    print("\nGenerating validation points...")
    output_layer = generate_validation_points(
        scenes,
        cz_layer,
        cz_spatial_index,
    )

    print("\nSaving output...")
    saved_layer = save_output_layer(
        output_layer,
        OUTPUT_PATH,
    )

    print("\n" + "=" * 72)
    print("PROCESS COMPLETED")
    print("=" * 72)
    print(f"Validation points created: {saved_layer.featureCount()}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 72)


main()
