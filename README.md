# SR4LC – Evaluation of Super-Resolution Techniques for Sentinel-2 Land Cover Classification

## Overview

This repository contains the scripts and documentation developed during my Master's Thesis:

**"Evaluation of Super-Resolution Techniques for Sentinel-2 Land Cover Classification"**

The work was carried out during an internship at **Planetek Italia** within the **SR4LC (Super-Resolution for Land Cover Classification)** project.

---

## Abstract

**SR4LC** investigates whether deep learning-based super-resolution can improve semantic land cover classification from Sentinel-2 imagery. The proposed workflow combines image super-resolution using **SEN2SR**, semantic segmentation with **UNet**, and quantitative evaluation through confusion matrices and Overall Accuracy (OA). This repository documents the complete methodology developed during the project, from image preprocessing to model evaluation.

> **Note**
>
> Due to confidentiality restrictions, this repository only contains the scripts developed by the author.
>
> The core processing pipeline provided by **Planetek Italia** is proprietary and cannot be publicly distributed.
>
> To document the work carried out during the internship, the repository includes both the scripts used in the final workflow and earlier experimental versions developed throughout the project.

---

# Objectives

The main objectives of the project are:

- Generate super-resolved Sentinel-2 imagery using **SEN2SR**.
- Build semantic segmentation datasets from Sentinel-2 imagery and Coastal Zones reference data.
- Train a **UNet** semantic segmentation model.
- Perform semantic segmentation inference on large Sentinel-2 scenes.
- Evaluate the classification performance using confusion matrices and Overall Accuracy (OA).

---

# Study Area

The study focuses on coastal regions of Italy using:

- Sentinel-2 imagery
- Coastal Zones Land Cover/Land Use 2018 reference dataset

---

# Workflow

The complete workflow implemented during the project is summarized below.

> *(A workflow figure will be added here.)*

```
Sentinel-2 Composites
        │
        ▼
 Super-Resolution
     (SEN2SR)
        │
        ▼
 Dataset Generation
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
 Large Image Inference
        │
        ▼
      Validation
 (Confusion Matrix + OA)
```

---

# Repository Organization

The repository is organized into two main folders to distinguish between the scripts used in the final methodology and those developed during earlier stages of the project.

## Final_workflow

This folder contains the scripts corresponding to the final workflow presented in the Master's Thesis.

It includes:

- Super-resolution using SEN2SR
- Dataset generation
- Dataset optimization
- Semantic segmentation
- Large-image inference
- Validation

---

## Old_workflow

This folder contains previous versions of the workflow, experimental implementations and debugging scripts developed throughout the project.

Although these scripts are not part of the final methodology, they document the evolution of the project and the different approaches explored during development.

---

# Repository Structure

```
TFM_SR4LC/
│
├── README.md
├── Final_workflow/
│   ├── 1_SuperResolution/
│   ├── 2_Segmentation/
│   ├── 3_Validation/
│   ├── model/
│   └── requirements/
│
└── Old_workflow/
```

---

# Repository Contents

This repository contains:

- Python scripts
- Workflow documentation
- Model loading utilities
- Configuration files
- Validation scripts

This repository does **not** include:

- Sentinel-2 imagery
- Training datasets
- Trained model weights (`*.safetensor`)
- Proprietary Planetek Italia code

---

# Technologies

## Programming

- Python

## Deep Learning

- PyTorch
- PyTorch Lightning
- LitData

## Geospatial

- GDAL
- Rasterio
- QGIS

## Earth Observation

- Sentinel-2
- SEN2SR
- Lightning AI

---

# Author

**María Victoria León Parra**

Master's in GIS & Spatial Data Science

University of Girona (UNIGIS Girona)

---

# Acknowledgements

This work was developed during an internship at **Planetek Italia** as part of the **SR4LC (Super-Resolution for Land Cover Classification)** project.

The author gratefully acknowledges the support of the GeoAI team and all the guidance provided throughout the internship.

---

# Citation

If you use this repository in your research, please cite the corresponding Master's Thesis.

*A `CITATION.cff` file will be added in a future release.*
