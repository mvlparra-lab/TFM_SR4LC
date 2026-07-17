#!/usr/bin/env python3

"""
Optimize the SR4LC segmentation dataset with LitData on Lightning Studio.

Original author: Planetek
Adapted by: Victoria León
Project: SR4LC

Overview
--------
This script reads the final train.csv, val.csv and test.csv files generated
for the SR4LC segmentation dataset, converts each image-label pair to tensors
using process_sample(), and writes optimized LitData chunks.

Unlike the original multi-country version, this script is adapted to a single
dataset and therefore creates the following output structure:

    <OUTPUT_ROOT>/
        train/
        val/
        test/

Expected input structure
------------------------
<SPLITS_ROOT>/
    train.csv
    val.csv
    test.csv

Each CSV must contain at least these columns:

    image
    label

Usage
-----
Dry-run:

    python optimize_vhr_dataset_lightning_single.py --dry-run

Full optimization:

    python optimize_vhr_dataset_lightning_single.py

Specific splits only:

    python optimize_vhr_dataset_lightning_single.py --splits train val

Optional worker override:

    OPTIMIZE_WORKERS=16 python optimize_vhr_dataset_lightning_single.py
"""


# =============================================================================
# 1. IMPORTS
# =============================================================================

import argparse
import faulthandler
import multiprocessing
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd
from litdata import optimize
import litdata.processing.data_processor as litdata_data_processor

# process_sample must remain in a separate importable module because LitData
# may spawn worker processes internally.
from process_sample import process_sample


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

# Folder containing train.csv, val.csv and test.csv.
SPLITS_ROOT = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
    "outputs/Segmentation/DataSet2"
)

# Folder where optimized LitData chunks will be written.
OUTPUT_ROOT = Path(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/"
    "outputs/Segmentation/optimize_td"
)

# LitData optimization settings.
CHUNK_BYTES = "64MB"
NUM_WORKERS = int(
    os.environ.get("OPTIMIZE_WORKERS", 8)
)

# Required CSV column names.
COL_IMAGE = "image"
COL_LABEL = "label"

# Default split order.
DEFAULT_SPLITS = ["train", "val", "test"]


# Disable LitData's initial filesystem size scan.
#
# LitData normally calls Path.resolve(), Path.exists() and os.path.getsize()
# for every input image and label before starting the workers. On Lightning
# Storage, each metadata query was taking approximately 12–18 seconds.
#
# Returning an equal weight for every sample bypasses that scan. Workers are
# therefore balanced by sample count rather than by total input file size.
def disable_litdata_initial_file_scan() -> None:
    """Replace LitData's slow per-file size scan with equal sample weights."""
    def equal_item_weights(
        items,
        base_path="",
    ):
        return [1] * len(items)

    litdata_data_processor._get_item_filesizes = equal_item_weights


# =============================================================================
# 3. DEBUG UTILITIES
# =============================================================================

DEBUG_HEARTBEAT_SECONDS = int(
    os.environ.get("DEBUG_HEARTBEAT_SECONDS", 60)
)


def log(message: str) -> None:
    """Print a timestamped message immediately."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def count_output_files(output_dir: Path) -> tuple[int, int]:
    """Return the number of files and total bytes currently written."""
    if not output_dir.exists():
        return 0, 0

    file_count = 0
    total_bytes = 0

    for path in output_dir.rglob("*"):
        if path.is_file():
            file_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    return file_count, total_bytes


def heartbeat(
    stop_event: threading.Event,
    output_dir: Path,
    started_at: float,
) -> None:
    """
    Periodically report that the main Python process is still alive.

    This does not prove that LitData workers are progressing, but it tells us
    whether control is still blocked inside optimize() and whether files are
    appearing in the output directory.
    """
    while not stop_event.wait(DEBUG_HEARTBEAT_SECONDS):
        elapsed = time.time() - started_at
        file_count, total_bytes = count_output_files(output_dir)

        log(
            "[heartbeat] optimize() has not returned yet | "
            f"elapsed={elapsed:.1f}s | "
            f"output_files={file_count} | "
            f"output_size={total_bytes / (1024 ** 2):.2f} MB | "
            f"active_threads={threading.active_count()}"
        )


def print_runtime_info() -> None:
    """Print basic runtime information useful for multiprocessing debugging."""
    try:
        import litdata
        litdata_version = getattr(litdata, "__version__", "unknown")
    except Exception:
        litdata_version = "unavailable"

    log(f"Python version       : {sys.version.split()[0]}")
    log(f"Platform             : {platform.platform()}")
    log(f"Executable           : {sys.executable}")
    log(f"Working directory    : {Path.cwd()}")
    log(f"CPU count            : {os.cpu_count()}")
    log(f"Multiprocessing mode : {multiprocessing.get_start_method(allow_none=True)}")
    log(f"LitData version      : {litdata_version}")


# =============================================================================
# 4. INPUT VALIDATION
# =============================================================================

def load_pairs(
    csv_path: Path,
) -> list[tuple[str, str]]:
    """
    Load image-label pairs from one split CSV.

    Empty rows are removed before returning the list.
    """
    dataframe = pd.read_csv(csv_path)

    missing_columns = [
        column
        for column in (COL_IMAGE, COL_LABEL)
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{csv_path}: missing columns {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe.dropna(
        subset=[COL_IMAGE, COL_LABEL]
    )

    dataframe = dataframe[
        dataframe[COL_IMAGE]
        .astype(str)
        .str.strip()
        != ""
    ]

    dataframe = dataframe[
        dataframe[COL_LABEL]
        .astype(str)
        .str.strip()
        != ""
    ]

    return list(
        zip(
            dataframe[COL_IMAGE].astype(str).tolist(),
            dataframe[COL_LABEL].astype(str).tolist(),
        )
    )


def probe_samples(
    pairs: list[tuple[str, str]],
    n: int = 3,
) -> bool:
    """
    Test the first samples of one split before multiprocessing starts.

    This catches path, TIFF, shape or dtype errors early and prints a full
    traceback when a sample fails.
    """
    n = min(n, len(pairs))

    print(
        f"  [probe] testing {n} sample(s) in-process ..."
    )

    all_ok = True

    for image_path, label_path in pairs[:n]:
        try:
            result = process_sample(
                (image_path, label_path)
            )

            shapes = {
                key: tuple(value.shape)
                for key, value in result.items()
            }

            dtypes = {
                key: str(value.dtype)
                for key, value in result.items()
            }

            print(
                f"  [probe] OK   "
                f"{Path(image_path).name}  "
                f"shapes={shapes}  "
                f"dtypes={dtypes}"
            )

        except Exception:
            print(
                f"  [probe] FAIL "
                f"{Path(image_path).name}"
            )

            traceback.print_exc()
            all_ok = False

    return all_ok


# =============================================================================
# 5. LITDATA OPTIMIZATION
# =============================================================================

def optimize_split(
    pairs: list[tuple[str, str]],
    output_dir: Path,
    chunk_bytes: str = CHUNK_BYTES,
    num_workers: int = NUM_WORKERS,
) -> None:
    """
    Optimize one dataset split with LitData and verbose debug monitoring.

    Existing output is overwritten.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_count_before, bytes_before = count_output_files(output_dir)

    log("About to call litdata.optimize()")
    log(f"  input samples      : {len(pairs)}")
    log(f"  output directory   : {output_dir.resolve()}")
    log(f"  existing files     : {file_count_before}")
    log(f"  existing size      : {bytes_before / (1024 ** 2):.2f} MB")
    log(f"  workers            : {num_workers}")
    log(f"  chunk size         : {chunk_bytes}")
    log(f"  first input image  : {pairs[0][0]}")
    log(f"  first input label  : {pairs[0][1]}")

    stop_event = threading.Event()
    started_at = time.time()

    monitor = threading.Thread(
        target=heartbeat,
        args=(stop_event, output_dir, started_at),
        daemon=True,
        name="litdata-debug-heartbeat",
    )
    monitor.start()

    try:
        log("Entering litdata.optimize() now")
        optimize(
            fn=process_sample,
            inputs=pairs,
            output_dir=str(output_dir),
            chunk_bytes=chunk_bytes,
            num_workers=num_workers,
            mode="overwrite",
        )
        log("litdata.optimize() returned normally")

    except BaseException:
        log("litdata.optimize() raised an exception")
        traceback.print_exc()
        raise

    finally:
        stop_event.set()
        monitor.join(timeout=2)

        elapsed = time.time() - started_at
        file_count_after, bytes_after = count_output_files(output_dir)

        log(
            "optimize() final state | "
            f"elapsed={elapsed:.1f}s | "
            f"output_files={file_count_after} | "
            f"output_size={bytes_after / (1024 ** 2):.2f} MB"
        )


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main(
    splits: list[str],
    dry_run: bool,
) -> None:
    """
    Probe and optionally optimize the requested dataset splits.
    """
    # -------------------------------------------------------------------------
    # Validate the input directory.
    # -------------------------------------------------------------------------
    if not SPLITS_ROOT.exists():
        sys.exit(
            f"ERROR: splits root not found: "
            f"{SPLITS_ROOT.resolve()}"
        )

    # -------------------------------------------------------------------------
    # Print the execution configuration.
    # -------------------------------------------------------------------------
    mode_label = (
        "DRY-RUN (probe only, no writing)"
        if dry_run
        else "FULL RUN"
    )

    print("=" * 70)
    print("SR4LC LITDATA OPTIMIZATION")
    print("=" * 70)
    print(f"Mode        : {mode_label}")
    print(f"Splits root : {SPLITS_ROOT.resolve()}")
    print(f"Output root : {OUTPUT_ROOT.resolve()}")
    print(f"Splits      : {splits}")
    print(f"Workers     : {NUM_WORKERS}")
    print(f"Chunk size  : {CHUNK_BYTES}")
    print()

    print_runtime_info()

    disable_litdata_initial_file_scan()
    log(
        "LitData initial file-size scan disabled; "
        "workers will be balanced by sample count"
    )
    log("Main workflow initialized")

    stats: list[dict] = []

    # -------------------------------------------------------------------------
    # Process each split independently.
    # -------------------------------------------------------------------------
    for split in splits:
        csv_path = SPLITS_ROOT / f"{split}.csv"

        if not csv_path.exists():
            print(
                f"⚠  {csv_path} not found — skipping"
            )

            stats.append({
                "split": split,
                "samples": 0,
                "status": "missing CSV",
            })

            continue

        log(f"Loading split CSV: {csv_path}")
        try:
            pairs = load_pairs(csv_path)
        except Exception as exc:
            print(
                f"✗  Failed to read {csv_path}: {exc}\n"
            )

            stats.append({
                "split": split,
                "samples": 0,
                "status": f"FAILED: {exc}",
            })

            continue

        if not pairs:
            print(
                f"⚠  {csv_path} is empty — skipping"
            )

            stats.append({
                "split": split,
                "samples": 0,
                "status": "empty CSV",
            })

            continue

        output_dir = OUTPUT_ROOT / split

        log(
            f"Loaded split '{split}' successfully with "
            f"{len(pairs)} valid image-label pairs"
        )

        print(
            f"→ {split} "
            f"({len(pairs)} samples) "
            f"→ {output_dir}"
        )

        # ---------------------------------------------------------------------
        # Always test a few samples before starting worker processes.
        # ---------------------------------------------------------------------
        log(f"Starting in-process probe for split '{split}'")
        probe_ok = probe_samples(
            pairs,
            n=3,
        )
        log(
            f"Probe finished for split '{split}' "
            f"with status={'OK' if probe_ok else 'FAILED'}"
        )

        if not probe_ok:
            stats.append({
                "split": split,
                "samples": len(pairs),
                "status": (
                    "FAILED: probe errors "
                    "(see traceback above)"
                ),
            })

            print(
                "  ✗  Probe failed — fix the errors "
                "above and run again.\n"
            )

            continue

        if dry_run:
            stats.append({
                "split": split,
                "samples": len(pairs),
                "status": "probe ok (dry-run)",
            })

            print("  ✓  Dry-run OK\n")
            continue

        # ---------------------------------------------------------------------
        # Run the full LitData optimization.
        # ---------------------------------------------------------------------
        try:
            optimize_split(
                pairs=pairs,
                output_dir=output_dir,
            )

            stats.append({
                "split": split,
                "samples": len(pairs),
                "status": "ok",
            })

            print(
                f"  ✓  {split} done "
                f"({len(pairs)} samples)\n"
            )

        except Exception as exc:
            stats.append({
                "split": split,
                "samples": len(pairs),
                "status": f"FAILED: {exc}",
            })

            print(
                f"  ✗  {split} FAILED: {exc}\n"
            )

    # -------------------------------------------------------------------------
    # Print the final execution summary.
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_dataframe = pd.DataFrame(stats)

    if not summary_dataframe.empty:
        print(
            summary_dataframe.to_string(
                index=False
            )
        )
    else:
        print("Nothing was processed.")


# =============================================================================
# 7. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Required for safe multiprocessing when LitData uses spawn.
    multiprocessing.freeze_support()

    # Print Python stack traces automatically if the process appears frozen.
    faulthandler.enable()
    faulthandler.dump_traceback_later(
        600,
        repeat=True,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Optimize the SR4LC segmentation dataset "
            "with LitData on Lightning Studio."
        )
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=DEFAULT_SPLITS,
        choices=DEFAULT_SPLITS,
        help=(
            "Dataset splits to process. "
            "Default: train val test."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Probe the first 3 samples of each split "
            "without writing optimized data."
        ),
    )

    args = parser.parse_args()

    start_time = time.time()

    main(
        splits=args.splits,
        dry_run=args.dry_run,
    )

    elapsed_time = time.time() - start_time

    print(
        f"\nTotal time: {elapsed_time:.1f}s"
    )
