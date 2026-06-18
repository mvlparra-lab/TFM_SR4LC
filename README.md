# Evaluation of Super-Resolution Techniques for Sentinel-2 Land Cover Classification

## Overview

This repository contains the workflows, scripts and documentation developed for the Master's Thesis:

**"Evaluation of Super-Resolution Techniques for Sentinel-2 Land Cover Classification"**

The study investigates whether super-resolution techniques can improve downstream land cover classification performance when applied to Sentinel-2 imagery. The work is conducted within the framework of the **SR4LC (Super Resolution for Land Cover Classification)** project developed at Planetek Italia.

The project compares land cover classification results obtained from:

* Original Sentinel-2 imagery (10 m)
* Super-resolved Sentinel-2 imagery (2.5 m)

using identical semantic segmentation workflows.

---

## Objectives

The main objectives of this work are:

* Generate super-resolved Sentinel-2 imagery using SEN2SR.
* Build land cover classification datasets from Sentinel-2 and Coastal Zones reference data.
* Train semantic segmentation models using both original and super-resolved imagery.
* Compare classification performance under identical experimental conditions.
* Assess the practical value of super-resolution for Earth Observation applications.

---

## Study Area

The study focuses on coastal regions of Italy using:

* Sentinel-2 imagery
* Coastal Zones Land Cover/Land Use 2018 reference dataset

---

## Workflow

```text
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
 Train / Val / Test Split
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
 Performance Evaluation
```

---

## Repository Structure

```text
.
├── docs/
├── notebooks/
├── scripts/
│   ├── super_resolution/
│   ├── dataset_creation/
│   ├── training/
│   └── evaluation/
├── workflows/
└── results/
```

---

## Main Components

### Super-Resolution

Implementation and evaluation of SEN2SR-based workflows for Sentinel-2 imagery enhancement.

### Dataset Generation

Creation of semantic segmentation datasets from Sentinel-2 composites and Coastal Zones reference labels.

### Semantic Segmentation

Training and evaluation of land cover classification models.

### Validation

Comparison between original and super-resolved workflows using classification metrics and visual assessment.

---

## Technologies

* Python
* PyTorch
* Lightning AI
* LitData
* QGIS
* Sentinel-2
* SEN2SR

---

## Author

**María Victoria León Parra**

MSc in GIS & Spatial Data Science

University of Girona (UNIGIS Girona)

---

## Acknowledgements

This work was developed during an internship at Planetek Italia within the SR4LC project.
