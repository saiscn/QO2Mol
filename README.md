# QO2Mol Dataset

<img alt="" src="https://img.shields.io/badge/license-CC_BY--NC--SA_4.0-blue" style="max-width: 100%;"> <img alt="" src="https://img.shields.io/badge/python->=3.10.0-blue" style="max-width: 100%;">

This repository contains the scripts for accessing QO2Mol dataset.

Data Access Link:
[Google Drive](https://drive.google.com/drive/folders/1-4FrnNrVBlL2RaBuXpalgNCk1q79VHtc?usp=drive_link)

- git clone this repo.
- download and put all `*.pkl` files under `./download/raw/` directory.


# Environment Preparation

>pip install -f requirements.txt

Note that `torch_geometric` may need to be installed separately follow the instruction on [PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/).

# Dataset Preprocess

Make sure current work directory is under the repository root
>pwd

Then run the data processing script. This should take a relatively long time (depending on Machine hardware specifications).
> sh process_data.sh

# Usage demo

We provide a usage demo that simply requires running the script
>sh scripts\dimenetpp\qm_benchmark\tr_baseline_e.sh

