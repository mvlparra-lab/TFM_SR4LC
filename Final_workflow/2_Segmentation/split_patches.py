#!/usr/bin/env python3

"""
Create globally balanced Train / Validation / Test CSV splits from multiple
Dataset Builder outputs.

Author: Victoria León
Project: SR4LC

Objective
---------
For every land-cover class, distribute approximately one third of its pixels
to each split:

    Class 1 -> 1/3 Train, 1/3 Validation, 1/3 Test
    Class 2 -> 1/3 Train, 1/3 Validation, 1/3 Test
    ...
    Class 8 -> 1/3 Train, 1/3 Validation, 1/3 Test

The number of image-label patches does not have to be identical across the
three splits. Class balance is the primary objective.

When two assignments produce effectively the same class-balance score, the
priority is:

    Train -> Validation -> Test

This means Train may contain slightly more samples than Validation, and
Validation may contain slightly more than Test, without compromising the main
class-balancing objective.

Workflow
--------
1. Read valid image-label pairs from create_td_2 and create_td.
2. Load class statistics from Dataset Builder class_count_*.csv files.
3. Fall back to reading a label TIFF only when its CSV information is missing.
4. Merge both Dataset Builder outputs.
5. Remove duplicated patches, prioritizing create_td_2.
6. Assign patches globally according to per-class pixel targets.
7. Improve the assignment through pairwise sample swaps.
8. Write train.csv, val.csv and test.csv.
"""


# =============================================================================
# 1. IMPORTS
# =============================================================================

from datetime import datetime
from pathlib import Path
import csv
import random

import numpy as np
import rasterio


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Dataset Builder output folders.
#
# The order defines duplicate priority. Since create_td_2 is first, its version
# of a duplicated patch is retained.
SOURCE_DIRS = [
    Path(
        "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
        "outputs/Segmentation/create_td_2"
    ),
    Path(
        "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
        "outputs/Segmentation/create_td"
    ),
]

# Output folder for train.csv, val.csv and test.csv.
OUTPUT_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
    "outputs/Segmentation/DataSet2"
)

# Coastal Zone level 1 classes:
#   0 = nodata / background
#   1-8 = valid land-cover classes
NUM_CLASSES = 9
NODATA_CLASS = 0

# Reproducible randomization.
SEED = 42

# Split order also defines tie-breaking priority.
SPLIT_NAMES = ("train", "val", "test")

# Every class is divided into three equal targets.
TARGET_FRACTION = 1.0 / 3.0

# Refinement settings.
#
# The initial greedy assignment is followed by random pairwise swaps.
# A swap is accepted only if it improves the per-class balance.
REFINEMENT_PASSES = 5
SWAPS_PER_PAIR = 2500

# Numerical tolerance used when comparing nearly identical balance scores.
SCORE_TOLERANCE = 1e-12


# =============================================================================
# 3. CLASS COUNT UTILITIES
# =============================================================================

def get_label_class_counts(label_path: Path) -> np.ndarray:
    """
    Count pixels belonging to classes 0-8 inside one label TIFF.

    This is used only as a fallback when no valid class-count entry is available
    in the Dataset Builder CSV files.

    Parameters
    ----------
    label_path : Path
        Path to the label TIFF.

    Returns
    -------
    np.ndarray
        Array of length NUM_CLASSES containing the pixel count of each class.
    """
    with rasterio.open(label_path) as src:
        label_array = src.read(1)

    counts = np.bincount(
        label_array.ravel(),
        minlength=NUM_CLASSES,
    )

    # Ignore unexpected values above the configured class range.
    if len(counts) > NUM_CLASSES:
        counts = counts[:NUM_CLASSES]

    return counts.astype(np.int64)


def load_class_counts_from_csvs(
    ancillary_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Load patch-level class counts from Dataset Builder CSV files.

    Expected CSV columns:

        tile,0,1,2,3,4,5,6,7,8

    Empty, malformed or unreadable CSV files are ignored. Missing patch entries
    are later recovered by reading their label TIFFs.

    Parameters
    ----------
    ancillary_dir : Path
        Directory containing class_count_*.csv files.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from tile filename to class-count vector.
    """
    # -------------------------------------------------------------------------
    # Locate the class-count CSV files.
    # -------------------------------------------------------------------------
    csv_files = sorted(
        ancillary_dir.glob("class_count_*.csv")
    )

    print(
        f"  Class-count CSV files found: {len(csv_files)}",
        flush=True,
    )

    required_columns = [
        "tile",
        *[str(class_id) for class_id in range(NUM_CLASSES)],
    ]

    counts_by_tile: dict[str, np.ndarray] = {}

    valid_csvs = 0
    invalid_csvs = 0
    valid_rows = 0
    invalid_rows = 0

    # -------------------------------------------------------------------------
    # Read every CSV independently.
    # -------------------------------------------------------------------------
    for csv_path in csv_files:
        try:
            with csv_path.open(newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                if not reader.fieldnames:
                    print(
                        f"  [CSV fallback] Empty CSV: {csv_path.name}",
                        flush=True,
                    )
                    invalid_csvs += 1
                    continue

                if not all(
                    column in reader.fieldnames
                    for column in required_columns
                ):
                    print(
                        f"  [CSV fallback] Invalid header: {csv_path.name}",
                        flush=True,
                    )
                    invalid_csvs += 1
                    continue

                valid_csvs += 1

                for row in reader:
                    try:
                        tile_name = row["tile"].strip()

                        if not tile_name:
                            invalid_rows += 1
                            continue

                        counts = np.array(
                            [
                                int(float(row[str(class_id)]))
                                for class_id in range(NUM_CLASSES)
                            ],
                            dtype=np.int64,
                        )

                        counts_by_tile[tile_name] = counts
                        valid_rows += 1

                    except (TypeError, ValueError, KeyError):
                        invalid_rows += 1

        except Exception as exc:
            print(
                f"  [CSV fallback] Could not read "
                f"{csv_path.name}: {exc}",
                flush=True,
            )
            invalid_csvs += 1

    # -------------------------------------------------------------------------
    # Report CSV loading statistics.
    # -------------------------------------------------------------------------
    print(f"  Valid CSV files: {valid_csvs}", flush=True)
    print(f"  Invalid CSV files: {invalid_csvs}", flush=True)
    print(f"  Valid CSV rows: {valid_rows}", flush=True)
    print(f"  Invalid CSV rows: {invalid_rows}", flush=True)

    return counts_by_tile


# =============================================================================
# 4. DATASET COLLECTION
# =============================================================================

def collect_source_samples(
    source_dir: Path,
) -> list[dict]:
    """
    Collect valid image-label pairs from one Dataset Builder output.

    A patch is accepted only when:
      - the image TIFF exists;
      - the label TIFF exists;
      - class counts are available from CSV or TIFF;
      - at least one valid land-cover pixel is present.

    Parameters
    ----------
    source_dir : Path
        Dataset Builder output directory.

    Returns
    -------
    list[dict]
        Valid samples collected from this source.
    """
    print("\n" + "-" * 80, flush=True)
    print(f"Reading source: {source_dir}", flush=True)

    # -------------------------------------------------------------------------
    # Define the expected folder structure.
    # -------------------------------------------------------------------------
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels" / "CODE_1_18"
    ancillary_dir = labels_dir / "ancillary"

    # -------------------------------------------------------------------------
    # Validate the required directories.
    # -------------------------------------------------------------------------
    if not images_dir.exists():
        print(
            f"[SKIP] Missing images directory: {images_dir}",
            flush=True,
        )
        return []

    if not labels_dir.exists():
        print(
            f"[SKIP] Missing labels directory: {labels_dir}",
            flush=True,
        )
        return []

    # -------------------------------------------------------------------------
    # Load class-count information from CSV files.
    # -------------------------------------------------------------------------
    counts_by_tile = load_class_counts_from_csvs(
        ancillary_dir
    )

    # -------------------------------------------------------------------------
    # Load filenames into memory once.
    #
    # This avoids thousands of Path.exists() calls on Lightning Storage.
    # -------------------------------------------------------------------------
    image_files = {
        path.name: path
        for path in images_dir.glob("*.tif")
    }

    label_files = {
        path.name: path
        for path in labels_dir.glob("*.tif")
    }

    # Only filenames present in both folders are valid pairs.
    common_tiles = sorted(
        set(image_files) & set(label_files)
    )

    print(
        f"  Image TIFF files found: {len(image_files)}",
        flush=True,
    )
    print(
        f"  Label TIFF files found: {len(label_files)}",
        flush=True,
    )
    print(
        f"  Image-label pairs found: {len(common_tiles)}",
        flush=True,
    )

    source_samples: list[dict] = []

    stats = {
        "pairs_added": 0,
        "csv_counts": 0,
        "tiff_fallbacks": 0,
        "invalid_tiffs": 0,
        "nodata_only": 0,
    }

    # -------------------------------------------------------------------------
    # Process every valid image-label filename pair.
    # -------------------------------------------------------------------------
    for index, tile_name in enumerate(
        common_tiles,
        start=1,
    ):
        if index % 500 == 0:
            print(
                f"  [{source_dir.name}] "
                f"{index}/{len(common_tiles)} "
                f"({100 * index / len(common_tiles):.1f}%)",
                flush=True,
            )

        image_path = image_files[tile_name]
        label_path = label_files[tile_name]

        # ---------------------------------------------------------------------
        # Retrieve class statistics.
        #
        # Priority:
        #   1. Dataset Builder class_count CSV.
        #   2. Label TIFF fallback.
        # ---------------------------------------------------------------------
        counts = counts_by_tile.get(tile_name)

        if counts is not None:
            stats["csv_counts"] += 1
        else:
            try:
                counts = get_label_class_counts(label_path)
                stats["tiff_fallbacks"] += 1

            except Exception as exc:
                print(
                    f"  [SKIP] Invalid label TIFF:\n"
                    f"         {label_path}\n"
                    f"         {exc}",
                    flush=True,
                )
                stats["invalid_tiffs"] += 1
                continue

        # ---------------------------------------------------------------------
        # Ignore patches containing only nodata/background.
        # ---------------------------------------------------------------------
        valid_counts = counts.copy()
        valid_counts[NODATA_CLASS] = 0

        if valid_counts.sum() == 0:
            stats["nodata_only"] += 1
            continue

        # ---------------------------------------------------------------------
        # Store the valid sample.
        # ---------------------------------------------------------------------
        source_samples.append({
            "tile": tile_name,
            "image": str(image_path),
            "label": str(label_path),
            "counts": counts,
            "source": source_dir.name,
        })

        stats["pairs_added"] += 1

    # -------------------------------------------------------------------------
    # Report source-level statistics.
    # -------------------------------------------------------------------------
    print(f"  Valid pairs found: {stats['pairs_added']}", flush=True)
    print(f"  Counts from CSV: {stats['csv_counts']}", flush=True)
    print(f"  TIFF fallbacks: {stats['tiff_fallbacks']}", flush=True)
    print(f"  Invalid TIFFs: {stats['invalid_tiffs']}", flush=True)
    print(f"  Nodata-only patches: {stats['nodata_only']}", flush=True)

    return source_samples


# =============================================================================
# 5. DATASET MERGING
# =============================================================================

def collect_and_merge_samples() -> list[dict]:
    """
    Merge all source datasets and remove duplicated patches.

    Duplicate priority follows SOURCE_DIRS order. create_td_2 is listed first,
    so its version of a duplicate is retained.

    Returns
    -------
    list[dict]
        Unified list of unique valid image-label pairs.
    """
    all_source_samples: list[dict] = []

    # -------------------------------------------------------------------------
    # Collect every source independently.
    # -------------------------------------------------------------------------
    for source_dir in SOURCE_DIRS:
        source_samples = collect_source_samples(
            source_dir
        )
        all_source_samples.extend(
            source_samples
        )

    print("\n" + "=" * 80, flush=True)
    print("MERGING ALL DATASET SOURCES", flush=True)
    print("=" * 80, flush=True)

    print(
        f"Pairs before deduplication: {len(all_source_samples)}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Remove duplicated patch filenames.
    #
    # Keeping only one copy avoids duplicated training samples and data leakage
    # between Train, Validation and Test.
    # -------------------------------------------------------------------------
    unique_samples_by_tile: dict[str, dict] = {}
    duplicate_count = 0

    for sample in all_source_samples:
        tile_name = sample["tile"]

        if tile_name in unique_samples_by_tile:
            duplicate_count += 1
            continue

        unique_samples_by_tile[tile_name] = sample

    merged_samples = list(
        unique_samples_by_tile.values()
    )

    print(
        f"Duplicated patches removed: {duplicate_count}",
        flush=True,
    )
    print(
        f"Unique valid pairs after merge: {len(merged_samples)}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Report how many retained samples came from each source.
    # -------------------------------------------------------------------------
    source_counts: dict[str, int] = {}

    for sample in merged_samples:
        source_name = sample["source"]
        source_counts[source_name] = (
            source_counts.get(source_name, 0) + 1
        )

    print("\nPairs retained by source:", flush=True)

    for source_name, count in source_counts.items():
        print(f"  {source_name}: {count}", flush=True)

    return merged_samples


# =============================================================================
# 6. STRATIFICATION UTILITIES
# =============================================================================

def class_balance_score(
    split_counts: dict[str, np.ndarray],
    target_counts: np.ndarray,
) -> float:
    """
    Measure deviation from the one-third target for every class and split.

    The error is normalized independently for each class so that rare classes
    are not ignored merely because they contain fewer pixels.

    Parameters
    ----------
    split_counts : dict[str, np.ndarray]
        Current pixel counts per split and class.

    target_counts : np.ndarray
        Desired per-class counts for each split: global counts / 3.

    Returns
    -------
    float
        Normalized squared error. Lower is better.
    """
    score = 0.0

    # Ignore nodata class 0.
    target_valid = target_counts[1:]

    for split_name in SPLIT_NAMES:
        difference = (
            split_counts[split_name][1:]
            - target_valid
        )

        score += float(
            np.sum(
                (difference ** 2)
                / (target_valid ** 2 + 1.0)
            )
        )

    return score


def sample_priority(
    sample: dict,
    global_counts: np.ndarray,
) -> float:
    """
    Prioritize patches containing rare classes.

    Rare classes are harder to distribute evenly, so patches containing them
    are assigned before patches containing only common classes.
    """
    valid_counts = sample["counts"][1:].astype(np.float64)
    global_valid = global_counts[1:].astype(np.float64)

    presence = valid_counts > 0
    rarity_weights = 1.0 / (global_valid + 1.0)

    rarity_score = float(
        np.sum(rarity_weights[presence])
    )

    valid_pixel_count = float(
        valid_counts.sum()
    )

    return rarity_score, valid_pixel_count


def create_initial_split(
    samples: list[dict],
    target_counts: np.ndarray,
    global_counts: np.ndarray,
) -> tuple[dict[str, list[dict]], dict[str, np.ndarray]]:
    """
    Create the initial split using per-class one-third targets.

    No fixed sample count is imposed. For every patch, the algorithm simulates
    adding it to Train, Validation and Test, and selects the split producing the
    lowest total class-balance error.

    If two candidate scores are effectively equal, priority is:
        Train -> Validation -> Test
    """
    splits: dict[str, list[dict]] = {
        split_name: []
        for split_name in SPLIT_NAMES
    }

    split_counts: dict[str, np.ndarray] = {
        split_name: np.zeros(
            NUM_CLASSES,
            dtype=np.int64,
        )
        for split_name in SPLIT_NAMES
    }

    # -------------------------------------------------------------------------
    # Assign rare and information-rich patches first.
    # -------------------------------------------------------------------------
    samples.sort(
        key=lambda sample: sample_priority(
            sample,
            global_counts,
        ),
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # Greedy per-class assignment.
    # -------------------------------------------------------------------------
    for index, sample in enumerate(
        samples,
        start=1,
    ):
        if index % 500 == 0:
            print(
                f"  [assign] {index}/{len(samples)} "
                f"({100 * index / len(samples):.1f}%)",
                flush=True,
            )

        best_split: str | None = None
        best_score: float | None = None

        for split_name in SPLIT_NAMES:
            candidate_counts = {
                name: counts.copy()
                for name, counts in split_counts.items()
            }

            candidate_counts[split_name] += sample["counts"]

            score = class_balance_score(
                candidate_counts,
                target_counts,
            )

            if best_score is None:
                best_score = score
                best_split = split_name
                continue

            # Better class balance always wins.
            if score < best_score - SCORE_TOLERANCE:
                best_score = score
                best_split = split_name
                continue

            # When scores are effectively equal, the existing SPLIT_NAMES order
            # already gives priority to train, then val, then test.
            if abs(score - best_score) <= SCORE_TOLERANCE:
                continue

        if best_split is None:
            raise RuntimeError(
                "Could not assign sample to any split."
            )

        splits[best_split].append(sample)
        split_counts[best_split] += sample["counts"]

    return splits, split_counts


def refine_with_swaps(
    splits: dict[str, list[dict]],
    split_counts: dict[str, np.ndarray],
    target_counts: np.ndarray,
) -> None:
    """
    Improve class balance using pairwise sample swaps.

    Swapping one sample between two splits preserves their sample counts.
    A proposed swap is accepted only when it improves the global per-class
    one-third balance.
    """
    rng = random.Random(SEED)

    current_score = class_balance_score(
        split_counts,
        target_counts,
    )

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    print(
        f"  Initial balance score: {current_score:.10f}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Evaluate random swaps between every pair of splits.
    # -------------------------------------------------------------------------
    for pass_number in range(
        1,
        REFINEMENT_PASSES + 1,
    ):
        accepted_swaps = 0

        for split_a, split_b in split_pairs:
            if not splits[split_a] or not splits[split_b]:
                continue

            for _ in range(SWAPS_PER_PAIR):
                index_a = rng.randrange(len(splits[split_a]))
                index_b = rng.randrange(len(splits[split_b]))

                sample_a = splits[split_a][index_a]
                sample_b = splits[split_b][index_b]

                candidate_counts_a = (
                    split_counts[split_a]
                    - sample_a["counts"]
                    + sample_b["counts"]
                )

                candidate_counts_b = (
                    split_counts[split_b]
                    - sample_b["counts"]
                    + sample_a["counts"]
                )

                candidate_split_counts = {
                    name: counts.copy()
                    for name, counts in split_counts.items()
                }

                candidate_split_counts[split_a] = candidate_counts_a
                candidate_split_counts[split_b] = candidate_counts_b

                candidate_score = class_balance_score(
                    candidate_split_counts,
                    target_counts,
                )

                if candidate_score < current_score - SCORE_TOLERANCE:
                    splits[split_a][index_a] = sample_b
                    splits[split_b][index_b] = sample_a

                    split_counts[split_a] = candidate_counts_a
                    split_counts[split_b] = candidate_counts_b

                    current_score = candidate_score
                    accepted_swaps += 1

        print(
            f"  Refinement pass {pass_number}: "
            f"{accepted_swaps} swaps accepted, "
            f"score={current_score:.10f}",
            flush=True,
        )

        if accepted_swaps == 0:
            break


# =============================================================================
# 7. OUTPUT UTILITIES
# =============================================================================

def write_csv(
    csv_path: Path,
    samples: list[dict],
) -> None:
    """
    Write one split CSV containing image and label paths.
    """
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["image", "label"],
        )
        writer.writeheader()

        for sample in samples:
            writer.writerow({
                "image": sample["image"],
                "label": sample["label"],
            })


def print_class_allocation(
    split_counts: dict[str, np.ndarray],
    global_counts: np.ndarray,
) -> None:
    """
    Print the percentage of each global class assigned to every split.

    These are the percentages that should be close to one third:

        Class N -> Train %, Validation %, Test %
    """
    print("\nPer-class allocation across splits:", flush=True)

    for class_id in range(1, NUM_CLASSES):
        total = global_counts[class_id]

        print(f"\nClass {class_id}", flush=True)

        if total == 0:
            print("  No pixels found.", flush=True)
            continue

        for split_name in SPLIT_NAMES:
            count = split_counts[split_name][class_id]
            percentage = count / total * 100

            print(
                f"  {split_name}: "
                f"{count} pixels "
                f"({percentage:.2f}%)",
                flush=True,
            )


def print_split_summary(
    splits: dict[str, list[dict]],
) -> None:
    """
    Print the number of samples assigned to Train, Validation and Test.
    """
    print("\nSamples per split:", flush=True)

    for split_name in SPLIT_NAMES:
        print(
            f"  {split_name}: {len(splits[split_name])}",
            flush=True,
        )


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    """
    Create the final global Train / Validation / Test split.
    """
    # -------------------------------------------------------------------------
    # Start information.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("GLOBAL PER-CLASS STRATIFIED DATASET SPLIT", flush=True)
    print("=" * 80, flush=True)

    print(
        f"Started at: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )

    print("\nSource directories:", flush=True)

    for source_dir in SOURCE_DIRS:
        print(f"  - {source_dir}", flush=True)

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}",
        flush=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Collect and merge all valid samples.
    # -------------------------------------------------------------------------
    print(
        "\n[1/7] Collecting and merging all datasets...",
        flush=True,
    )

    samples = collect_and_merge_samples()

    if not samples:
        raise RuntimeError(
            "No valid image-label pairs were found."
        )

    # -------------------------------------------------------------------------
    # Shuffle reproducibly before sorting by sample priority.
    # -------------------------------------------------------------------------
    print(
        "\n[2/7] Shuffling merged samples...",
        flush=True,
    )

    random.seed(SEED)
    random.shuffle(samples)

    print(
        f"Total unique samples available: {len(samples)}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Compute global class totals.
    # -------------------------------------------------------------------------
    print(
        "\n[3/7] Computing global class totals...",
        flush=True,
    )

    global_counts = np.sum(
        [sample["counts"] for sample in samples],
        axis=0,
    )

    # -------------------------------------------------------------------------
    # Define one-third target counts for every valid class.
    # -------------------------------------------------------------------------
    print(
        "\n[4/7] Defining one-third targets for every class...",
        flush=True,
    )

    target_counts = (
        global_counts.astype(np.float64)
        * TARGET_FRACTION
    )

    for class_id in range(1, NUM_CLASSES):
        print(
            f"  class {class_id}: "
            f"global={global_counts[class_id]}, "
            f"target per split={target_counts[class_id]:.2f}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Create the initial per-class balanced split.
    # -------------------------------------------------------------------------
    print(
        "\n[5/7] Creating initial per-class balanced split...",
        flush=True,
    )

    splits, split_counts = create_initial_split(
        samples,
        target_counts,
        global_counts,
    )

    # -------------------------------------------------------------------------
    # Improve balance through pairwise swaps.
    # -------------------------------------------------------------------------
    print(
        "\n[6/7] Refining class balance with sample swaps...",
        flush=True,
    )

    refine_with_swaps(
        splits,
        split_counts,
        target_counts,
    )

    # -------------------------------------------------------------------------
    # Write the final CSV files.
    # -------------------------------------------------------------------------
    print(
        "\n[7/7] Writing output CSV files...",
        flush=True,
    )

    for split_name in SPLIT_NAMES:
        csv_path = OUTPUT_DIR / f"{split_name}.csv"

        write_csv(
            csv_path,
            splits[split_name],
        )

        print(
            f"  {split_name}.csv -> "
            f"{len(splits[split_name])} samples",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Final report.
    # -------------------------------------------------------------------------
    print("\nCSV generation completed.", flush=True)

    print(
        f"Finished at: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}",
        flush=True,
    )

    print_split_summary(splits)

    print_class_allocation(
        split_counts,
        global_counts,
    )


if __name__ == "__main__":
    main()
