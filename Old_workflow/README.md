# SR4LC – Super-Resolution for Land Cover Classification

## Overview

This project explores the use of Sentinel-2 Super-Resolution (SR) techniques for improving Land Cover Classification workflows.

The project is being developed inside the Planetek GeoAI team activities.

---

## Project Structure

```text
SR4LC/
│
├── data/
├── outputs/
├── utils/
└── STAC4Cube/
    ├── Cube/
    ├── CloudMasking/
    ├── CoRegistration/
    └── SuperResolution/
        ├── SEN2SR/
        └── SEN2SRLite/
```

---

## Folder Description

- `data/`
  Input Sentinel-2 GeoTIFFs and test data

- `outputs/`
  Super-resolved outputs and intermediate patches

- `utils/`
  Shared helper functions for GeoTIFF handling, tiling, reconstruction, and preprocessing workflows

- `STAC4Cube/Cube/`
  Data cube generation and STAC-based workflows

- `STAC4Cube/CloudMasking/`
  Cloud masking workflows and preprocessing utilities

- `STAC4Cube/CoRegistration/`
  Image co-registration workflows

- `STAC4Cube/SuperResolution/SEN2SR/`
  SEN2SR resources and processing scripts

- `STAC4Cube/SuperResolution/SEN2SRLite/`
  SEN2SRLite resources and tiled inference workflows

---

## Technologies

- Python
- PyTorch
- SEN2SR / SEN2SRLite
- mlstac
- rasterio
- rioxarray
- QGIS
- Lightning AI