#!/bin/bash

# Create conda environment for DFN
conda create -n dfn_filter python=3.10 -y
conda activate dfn_filter

# Install PyTorch with CUDA 12.6 support
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install required packages
pip install transformers
pip install Pillow
pip install pandas
pip install pyarrow
pip install webdataset
pip install tqdm
pip install accelerate
pip install open_clip_torch
pip install ftfy
pip install regex

echo "DFN environment setup complete!"