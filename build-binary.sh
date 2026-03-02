#!/bin/bash
# Script to build a standalone binary using PyInstaller

set -e

echo "Installing PyInstaller..."
pip install pyinstaller

echo "Building binary..."
pyinstaller --onefile \
    --name lstm-anomaly \
    --add-data "config.yaml:." \
    --add-data "alerts:alerts" \
    --hidden-import numpy \
    --hidden-import pandas \
    --hidden-import torch \
    --hidden-import sklearn \
    --hidden-import yaml \
    --collect-all torch \
    --collect-all numpy \
    src/main.py

echo "Binary built successfully!"
echo "Location: dist/lstm-anomaly"
echo ""
echo "To run the binary:"
echo "  ./dist/lstm-anomaly --config config.yaml"
