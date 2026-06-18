"""
Create train/validation/test CSV split files from generated segmentation patches.

Author: Victoria León
Project: SR4LC

Workflow:
    - reads image-label patch pairs
    - analyzes the class distribution inside each label patch
    - splits patches into Train / Val / Test
    - tries to keep class distributions balanced across the three splits
    - writes train.csv, val.csv and test.csv

This script tries to keep the class distribution 
balanced across Train / Val / Test by looking at 
the pixel classes inside each label patch.
"""


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path
import csv
import random

import numpy as np
import rasterio


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Directory containing the generated image patches and label patches.
SOURCE_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/Segmentation/create_td"
)

# Directory where train.csv, val.csv and test.csv will be written.
OUTPUT_DIR = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/Segmentation/Italy_Fake/DataSet"
)

# Patch image directory.
IMAGES_DIR = SOURCE_DIR / "images"

# Level 1 Coastal Zone label directory.
LABELS_DIR = SOURCE_DIR / "labels" / "CODE_1_18"

# Coastal Zone level 1 contains:
#   0 = nodata / background
#   1-8 = valid land-cover classes
NUM_CLASSES = 9
NODATA_CLASS = 0

# Fixed random seed to make the split reproducible.
SEED = 42


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def get_label_class_counts(label_path):
    """
    Count how many pixels of each class are present in a label patch.

    Args:
        label_path (Path): path to the label TIFF.

    Returns:
        np.ndarray: array of length NUM_CLASSES.
                    Example:
                        counts[0] = number of nodata pixels
                        counts[1] = number of class 1 pixels
                        ...
                        counts[8] = number of class 8 pixels
    """
    with rasterio.open(label_path) as src:
        label_array = src.read(1)

    # Count pixels for each class.
    counts = np.bincount(label_array.ravel(), minlength=NUM_CLASSES)

    # Safety check in case unexpected class values appear.
    if len(counts) > NUM_CLASSES:
        counts = counts[:NUM_CLASSES]

    return counts.astype(np.int64)

def collect_samples():
    """
    Collect all valid image-label patch pairs.
    """
    samples = []

    label_files = sorted(LABELS_DIR.glob("*.tif"))

    print(f"Found {len(label_files)} label files", flush=True)

    for i, label_path in enumerate(label_files, start=1):

        if i % 50 == 0:
            print(
                f"[collect] {i}/{len(label_files)} "
                f"({100 * i / len(label_files):.1f}%)",
                flush=True,
            )

        image_path = IMAGES_DIR / label_path.name

        if not image_path.exists():
            continue

        counts = get_label_class_counts(label_path)

        samples.append({
            "image": str(image_path),
            "label": str(label_path),
            "counts": counts,
        })

    return samples


def write_csv(csv_path, samples):
    """
    Write a split CSV file.

    The output CSV contains only the columns required by the downstream
    optimization/training scripts:

        image,label

    Args:
        csv_path (Path): path of the CSV file to write.
        samples (list[dict]): samples assigned to the split.
    """
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "label"])
        writer.writeheader()

        for sample in samples:
            writer.writerow({
                "image": sample["image"],
                "label": sample["label"],
            })


def print_distribution(split_name, samples):
    """
    Print the class distribution of one split.

    This is useful to check whether Train / Val / Test have similar
    land-cover class distributions.

    Args:
        split_name (str): train, val or test.
        samples (list[dict]): samples assigned to the split.
    """
    total_counts = np.sum([sample["counts"] for sample in samples], axis=0)

    # Ignore nodata when reporting class percentages.
    valid_counts = total_counts.copy()
    valid_counts[NODATA_CLASS] = 0

    total_valid_pixels = valid_counts.sum()

    print(f"\n{split_name}")
    print(f"Samples: {len(samples)}")

    if total_valid_pixels == 0:
        print("No valid class pixels found.")
        return

    for class_id, count in enumerate(valid_counts):
        if class_id == NODATA_CLASS:
            continue

        percentage = count / total_valid_pixels * 100
        print(f"  class {class_id}: {count} pixels ({percentage:.2f}%)")


# =============================================================================
# 4. MAIN
# =============================================================================

from datetime import datetime


def main():
    """
    Create a stratified Train / Val / Test split.

    Strategy
    --------
    1. Read all valid image-label patch pairs.

    2. Count how many pixels of each class appear in every label patch.

    3. Compute the global class distribution of the full dataset.

    4. Split patches into Train / Val / Test with equal number of samples.

    5. Assign patches greedily:
        - for each patch, simulate adding it to train, val and test
        - choose the split whose class distribution remains closest
          to the expected target distribution

    6. Write:
        train.csv
        val.csv
        test.csv

    Notes
    -----
    This is a greedy approximation, not a perfect mathematical optimization.
    It is designed to be simple, reproducible and fast enough for large patch
    datasets.
    """

    # -------------------------------------------------------------------------
    # Start information.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("STRATIFIED DATASET SPLIT", flush=True)
    print("=" * 80, flush=True)
    print(
        f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )
    print(f"Images directory : {IMAGES_DIR}", flush=True)
    print(f"Labels directory : {LABELS_DIR}", flush=True)
    print(f"Output directory : {OUTPUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Collect all available samples.
    # -------------------------------------------------------------------------
    print("\n[1/6] Collecting image-label pairs...", flush=True)

    samples = collect_samples()

    if not samples:
        raise RuntimeError("No valid image-label pairs were found.")

    print(
        f"Collected {len(samples)} valid image-label pairs.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Shuffle samples.
    # -------------------------------------------------------------------------
    print("\n[2/6] Shuffling samples...", flush=True)

    random.seed(SEED)
    random.shuffle(samples)

    total_samples = len(samples)

    print(
        f"Total samples available: {total_samples}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Define split sizes.
    # -------------------------------------------------------------------------
    print("\n[3/6] Computing split sizes...", flush=True)

    split_sizes = {
        "train": total_samples // 3,
        "val": total_samples // 3,
        "test": total_samples - 2 * (total_samples // 3),
    }

    print("Target split sizes:", flush=True)
    print(f"  train : {split_sizes['train']}", flush=True)
    print(f"  val   : {split_sizes['val']}", flush=True)
    print(f"  test  : {split_sizes['test']}", flush=True)

    # -------------------------------------------------------------------------
    # Compute global class counts.
    # -------------------------------------------------------------------------
    print(
        "\n[4/6] Computing global class distributions...",
        flush=True,
    )

    total_counts = np.sum(
        [sample["counts"] for sample in samples],
        axis=0,
    )

    target_counts = {
        split_name: total_counts * (split_size / total_samples)
        for split_name, split_size in split_sizes.items()
    }

    print("Global class statistics computed.", flush=True)

    # -------------------------------------------------------------------------
    # Initialize split containers.
    # -------------------------------------------------------------------------
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    split_counts = {
        "train": np.zeros(NUM_CLASSES, dtype=np.int64),
        "val": np.zeros(NUM_CLASSES, dtype=np.int64),
        "test": np.zeros(NUM_CLASSES, dtype=np.int64),
    }

    # -------------------------------------------------------------------------
    # Sort by amount of valid pixels.
    # -------------------------------------------------------------------------
    print(
        "\n[5/6] Sorting patches by class information content...",
        flush=True,
    )

    samples.sort(
        key=lambda sample: sample["counts"][1:].sum(),
        reverse=True,
    )

    print(
        "Starting greedy stratified assignment...",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Greedy assignment.
    # -------------------------------------------------------------------------
    for idx, sample in enumerate(samples, start=1):

        if idx % 100 == 0:
            print(
                f"[assign] "
                f"{idx}/{len(samples)} "
                f"({100 * idx / len(samples):.1f}%)",
                flush=True,
            )

        best_split = None
        best_score = None

        for split_name in ["train", "val", "test"]:

            # Skip full splits.
            if len(splits[split_name]) >= split_sizes[split_name]:
                continue

            candidate_counts = (
                split_counts[split_name]
                + sample["counts"]
            )

            score = np.sum(
                (
                    (
                        candidate_counts[1:]
                        - target_counts[split_name][1:]
                    )
                    ** 2
                )
                / (target_counts[split_name][1:] + 1)
            )

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        if best_split is None:
            raise RuntimeError(
                "Could not assign sample to any split."
            )

        splits[best_split].append(sample)
        split_counts[best_split] += sample["counts"]

    print(
        "\nGreedy assignment completed successfully.",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Write CSV files.
    # -------------------------------------------------------------------------
    print("\n[6/6] Writing CSV files...", flush=True)

    for split_name, split_samples in splits.items():

        csv_path = OUTPUT_DIR / f"{split_name}.csv"

        write_csv(csv_path, split_samples)

        print(
            f"{split_name}.csv -> "
            f"{len(split_samples)} samples",
            flush=True,
        )

    print("\nCSV generation completed.", flush=True)

    print(
        f"\nFinished at: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Print final class distributions.
    # -------------------------------------------------------------------------
    print("\nClass distributions:", flush=True)

    for split_name, split_samples in splits.items():
        print_distribution(
            split_name,
            split_samples,
        )

if __name__ == "__main__":
    main()
