#!/usr/bin/env python3

"""
evaluate_validation.py
----------------------

Calculates confusion matrices and Overall Accuracy for:

1. Coastal Zones classification
2. Super-resolution classification

Expected fields:
    R_Class  : manually interpreted reference class
    CZ_Class : Coastal Zones class
    SR_Class : super-resolution prediction

Outputs:
    - confusion_matrix_cz.csv
    - confusion_matrix_sr.csv
    - class_metrics_cz.csv
    - class_metrics_sr.csv
    - metrics_summary.csv
    - confusion_matrix_cz.png
    - confusion_matrix_sr.png

Project: SR4LC
Author: Victoria León Parra
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qgis.core import QgsVectorLayer


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

INPUT_FILE = Path(
    r"C:\Users\Josemi\OneDrive\Documentos\Master\Practicas\Planetek"
    r"\pkq003_SR4LC\QGIS\results - stage\6_Validation\Points"
    r"\validation_points.shp"
)

OUTPUT_DIR = Path(
    r"C:\Users\Josemi\OneDrive\Documentos\Master\Practicas\Planetek"
    r"\pkq003_SR4LC\QGIS\results - stage\6_Validation\Results"
)

REFERENCE_FIELD = "R_Class"
CZ_FIELD = "CZ_Class"
SR_FIELD = "SR_Class"

CLASS_LABELS = [1, 2, 3, 4, 5, 6, 7, 8]

CLASS_NAMES = [
    "1 - Urban",
    "2 - Cropland",
    "3 - Woodland and Forest",
    "4 - Grassland",
    "5 - Heathland and Shrub",
    "6 - Open Spaces",
    "7 - Wetlands",
    "8 - Water",
]


# -------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------

def read_validation_layer(input_file):
    """Reads the validation point layer using the QGIS API."""

    layer = QgsVectorLayer(
        str(input_file),
        "Validation points",
        "ogr",
    )

    if not layer.isValid():
        raise RuntimeError(
            f"Could not read the input layer:\n{input_file}"
        )

    return layer


def validate_fields(layer):
    """Checks that the required fields exist."""

    available_fields = [field.name() for field in layer.fields()]
    required_fields = [REFERENCE_FIELD, CZ_FIELD, SR_FIELD]

    missing_fields = [
        field for field in required_fields
        if field not in available_fields
    ]

    if missing_fields:
        raise ValueError(
            "Missing fields: "
            + ", ".join(missing_fields)
            + "\nAvailable fields: "
            + ", ".join(available_fields)
        )


def prepare_data(layer, prediction_field):
    """
    Extracts valid reference and prediction values from the point layer.
    """

    reference_values = []
    prediction_values = []
    excluded_points = 0

    for feature in layer.getFeatures():
        reference_value = feature[REFERENCE_FIELD]
        prediction_value = feature[prediction_field]

        try:
            reference_value = int(reference_value)
            prediction_value = int(prediction_value)
        except (TypeError, ValueError):
            excluded_points += 1
            continue

        if (
            reference_value not in CLASS_LABELS
            or prediction_value not in CLASS_LABELS
        ):
            excluded_points += 1
            continue

        reference_values.append(reference_value)
        prediction_values.append(prediction_value)

    if excluded_points > 0:
        print(
            f"Warning: {excluded_points} points were excluded "
            f"for {prediction_field}."
        )

    if not reference_values:
        raise ValueError(
            f"No valid points were found for {prediction_field}."
        )

    return (
        np.asarray(reference_values, dtype=int),
        np.asarray(prediction_values, dtype=int),
    )


def calculate_confusion_matrix(y_true, y_pred):
    """Calculates the confusion matrix."""

    matrix = np.zeros(
        (len(CLASS_LABELS), len(CLASS_LABELS)),
        dtype=int,
    )

    label_to_index = {
        label: index
        for index, label in enumerate(CLASS_LABELS)
    }

    for reference_value, prediction_value in zip(y_true, y_pred):
        row = label_to_index[reference_value]
        column = label_to_index[prediction_value]
        matrix[row, column] += 1

    return matrix


def calculate_class_metrics(matrix):
    """
    Calculates precision, recall and F1-score for each class.

    Rows represent reference classes.
    Columns represent predicted classes.
    """

    metrics = []

    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        support = matrix[index, :].sum()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = precision + recall

        f1_score = (
            2 * precision * recall / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        metrics.append(
            {
                "class_id": CLASS_LABELS[index],
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": int(support),
            }
        )

    return pd.DataFrame(metrics)


def save_confusion_matrix_csv(matrix, output_path):
    """Saves a confusion matrix as CSV."""

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_df.index.name = "Reference"
    matrix_df.columns.name = "Prediction"

    matrix_df.to_csv(
        output_path,
        encoding="utf-8-sig",
    )


def save_confusion_matrix_plot(matrix, title, output_path):
    """Creates a confusion matrix figure without sklearn."""

    figure, axis = plt.subplots(figsize=(12, 10))

    image = axis.imshow(matrix)

    axis.set_title(title, fontsize=15)
    axis.set_xlabel("Predicted class", fontsize=12)
    axis.set_ylabel("Reference class", fontsize=12)

    axis.set_xticks(np.arange(len(CLASS_NAMES)))
    axis.set_yticks(np.arange(len(CLASS_NAMES)))

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(CLASS_NAMES)

    maximum_value = matrix.max()
    threshold = maximum_value / 2 if maximum_value > 0 else 0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def evaluate_classification(
    layer,
    prediction_field,
    output_prefix,
    classification_name,
):
    """Evaluates one classification and saves its results."""

    y_true, y_pred = prepare_data(
        layer,
        prediction_field,
    )

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
    )

    correct_predictions = np.trace(matrix)
    total_points = matrix.sum()

    overall_accuracy = (
        correct_predictions / total_points
        if total_points > 0
        else 0.0
    )

    class_metrics = calculate_class_metrics(matrix)

    print("\nPer-class metrics")
    print("-----------------")
    print(
        class_metrics[
            [
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1_score",
                "support",
            ]
        ].round(3)
    )

    macro_precision = class_metrics["precision"].mean()
    macro_recall = class_metrics["recall"].mean()
    macro_f1 = class_metrics["f1_score"].mean()

    weighted_f1 = np.average(
        class_metrics["f1_score"],
        weights=class_metrics["support"],
    )

    save_confusion_matrix_csv(
        matrix,
        OUTPUT_DIR / f"confusion_matrix_{output_prefix}.csv",
    )

    save_confusion_matrix_plot(
        matrix,
        f"Confusion Matrix — {classification_name}",
        OUTPUT_DIR / f"confusion_matrix_{output_prefix}.png",
    )

    class_metrics.to_csv(
        OUTPUT_DIR / f"class_metrics_{output_prefix}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(classification_name)
    print("-" * len(classification_name))
    print(f"Valid points: {total_points}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    print(
        "\nMacro Precision: "
        f"{macro_precision:.3f}"
    )

    print(
        "Macro Recall: "
        f"{macro_recall:.3f}"
    )

    print(
        "Macro F1-score: "
        f"{macro_f1:.3f}"
    )

    print(
        "Weighted F1-score: "
        f"{weighted_f1:.3f}"
    )
    
    return {
        "classification": classification_name,
        "valid_points": int(total_points),
        "correct_predictions": int(correct_predictions),
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_percent": overall_accuracy * 100,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "weighted_f1_score": weighted_f1,
    }


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    """Runs the complete validation evaluation."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print(f"Reading validation points:\n{INPUT_FILE}")

    validation_layer = read_validation_layer(INPUT_FILE)

    print(
        f"Number of features: "
        f"{validation_layer.featureCount()}"
    )

    validate_fields(validation_layer)

    cz_metrics = evaluate_classification(
        layer=validation_layer,
        prediction_field=CZ_FIELD,
        output_prefix="cz",
        classification_name="Coastal Zones",
    )

    sr_metrics = evaluate_classification(
        layer=validation_layer,
        prediction_field=SR_FIELD,
        output_prefix="sr",
        classification_name="Super-resolution",
    )

    summary = pd.DataFrame(
        [cz_metrics, sr_metrics]
    )

    oa_difference = (
        sr_metrics["overall_accuracy_percent"]
        - cz_metrics["overall_accuracy_percent"]
    )

    summary["oa_difference_from_cz_pp"] = [
        0.0,
        oa_difference,
    ]

    summary.to_csv(
        OUTPUT_DIR / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Final comparison")
    print("----------------")
    print(
        "Coastal Zones OA: "
        f"{cz_metrics['overall_accuracy_percent']:.2f}%"
    )
    print(
        "Super-resolution OA: "
        f"{sr_metrics['overall_accuracy_percent']:.2f}%"
    )
    print(
        "Difference SR - CZ: "
        f"{oa_difference:+.2f} percentage points"
    )

    print()
    print(f"Results saved in:\n{OUTPUT_DIR}")


main()