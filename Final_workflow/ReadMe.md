# Final Workflow

This folder contains the scripts corresponding to the final workflow used in the Master's Thesis.

These scripts represent the final implementation of the SR4LC methodology after the development and testing stages.

## Workflow

```
Sentinel-2 Images
        │
        ▼
Super-Resolution (SEN2SR)
        │
        ▼
Dataset Preparation
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Dataset Optimization (LitData)
        │
        ▼
Model Training (UNet)
        │
        ▼
Inference
        │
        ▼
Validation
```

## Contents

| Script | Description |
|---------|-------------|
| `super_resolution_rgbn.py` | Generates super-resolved Sentinel-2 RGBN images using SEN2SR. |
| `...` | ... |

> The scripts in this folder correspond to the methodology described in the Master's Thesis and are intended to reproduce the final processing workflow.

Some scripts depend on proprietary components developed by Planetek Italia and therefore may require additional software that is not included in this repository.