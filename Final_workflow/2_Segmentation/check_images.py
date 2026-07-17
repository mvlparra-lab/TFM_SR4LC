#!/usr/bin/env python3

"""
Debug script to inspect Sentinel-2 image values.

Reads a small window from the first Sentinel-2 images
and prints the minimum and maximum pixel values.

Author: Victoria León
Project: SR4LC
"""

import glob
import rasterio

files = sorted(glob.glob(
    "/teamspace/lightning_storage/pKq003_SR4LC_Data/outputs/S2_SEN2SR/RGBN/*.tif"
))

for f in files[:5]:
    with rasterio.open(f) as src:
        img = src.read(window=((0, 512), (0, 512)))
        print(f.split("/")[-1], img.min(), img.max())