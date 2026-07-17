# Evaluation of Super-Resolution Techniques for Sentinel-2 Land Cover Classification (SR4LC)

## Overview

This repository contains the scripts and documentation developed during my Master's Thesis:

**"Evaluation of Super-Resolution Techniques for Sentinel-2 Land Cover Classification"**

The work was carried out during an internship at **Planetek Italia** within the **SR4LC (Super Resolution for Land Cover Classification)** project.

The objective of the project is to evaluate the impact of image super-resolution on land cover classification by comparing the performance of semantic segmentation workflows using original Sentinel-2 imagery (10 m) and Sentinel-2 imagery super-resolved to 2.5 m using SEN2SR.

> **Note**
>
> Due to confidentiality restrictions, this repository only contains the scripts developed by the author. The core processing pipeline provided by Planetek Italia is proprietary and cannot be publicly distributed.
>
> To document the work carried out during the internship, the repository includes both the scripts used in the final workflow and earlier experimental versions developed throughout the project.

---

# Objectives

- Generate super-resolved Sentinel-2 imagery using SEN2SR.
- Build semantic segmentation datasets from Sentinel-2 imagery and Coastal Zones reference data.
- Train a UNet semantic segmentation model.
- Perform semantic segmentation inference on large Sentinel-2 scenes.
- Evaluate the classification results using confusion matrices and Overall Accuracy (OA).

---

# Study Area

The study focuses on coastal regions of Italy using:

- Sentinel-2 imagery
- Coastal Zones Land Cover/Land Use 2018 reference dataset

---

# Workflow

```
Sentinel-2 Composites
        │
        ▼
 Super-Resolution
    (SEN2SR)
        │
        ▼
 Dataset Creation
(Image + Labels)
        │
        ▼
 Train / Validation / Test Split
        │
        ▼
 Dataset Optimization
      (LitData)
        │
        ▼
 Semantic Segmentation
       (UNet)
        │
        ▼
     Inference
        │
        ▼
 Confusion Matrix
        │
        ▼
 Overall Accuracy
```

---

# Repository Organization

The repository is organized into two main folders to distinguish between the scripts used in the final methodology and those developed during earlier stages of the project.

## Final_Workflow

This folder contains the scripts corresponding to the final workflow presented in the Master's Thesis.

These scripts implement the methodology described in the dissertation, including:

- Super-resolution with SEN2SR
- Dataset preparation
- Dataset optimization
- Model training
- Large-image inference
- Validation

## Old_Workflow

This folder contains previous versions of the workflow, experimental implementations and debugging scripts developed throughout the project.

Many of these scripts were created while testing different approaches, troubleshooting issues or adapting the workflow to the Lightning AI environment. Although they are not part of the final methodology, they document the evolution of the project and the development process that led to the final workflow.

---

# Repository Structure

```
.
├── README.md
├── Final_Workflow/
│   ├── ...
├── Old_Workflow/
│   ├── ...
└── docs/
```

---

# Technologies

- Python
- PyTorch
- PyTorch Lightning
- Lightning AI
- LitData
- Rasterio
- GDAL
- QGIS
- Sentinel-2
- SEN2SR

---

# Author

**María Victoria León Parra**

Master's in GIS & Spatial Data Science

University of Girona (UNIGIS Girona)

---

# Acknowledgements

This work was developed during an internship at **Planetek Italia** within the **SR4LC (Super Resolution for Land Cover Classification)** project.