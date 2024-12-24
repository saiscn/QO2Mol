# QO2Mol Dataset

<img alt="" src="https://img.shields.io/badge/license-CC_BY--NC--SA_4.0-blue" style="max-width: 100%;"> <img alt="" src="https://img.shields.io/badge/python->=3.11.0-blue" style="max-width: 100%;">

This repository contains the scripts for accessing QO2Mol dataset, the large-scale quantum chemistry dataset with 20 million conformers, designed for the research in molecular sciences under an open-source license.

# Download Files

Data files can be accessed at: [Google Drive](https://drive.google.com/drive/folders/1-4FrnNrVBlL2RaBuXpalgNCk1q79VHtc?usp=drive_link)

Note that the latest version now is `v1.1.0`.  
See details in `CHANGELOG.md`.

# Preparation
- git clone this repo.
- download and put all `*.pkl` files under `./download/raw/` directory. 


# Environment Preparation

>pip install -f requirements.txt

Note that `torch_geometric` may need to be installed separately follow the instruction on [PyG Documentation](https://pytorch-geometric.readthedocs.io/en/latest/).

# Dataset Preprocess

Make sure current work directory is under the repository root
>pwd

Then run the data processing script.
> sh process_data.sh

Take a cup of coffee☕️. This could take tens of minutes.

**Q&A**:

- All files are created and tested with `pytho==3.11.9` & `numpy==2.1.1`.
- If the repo is cloned into `/abc/QO2Mol`,  all `.pkl` files should be under `/abc/QO2Mol/download/raw/`.
- We offer MD5 hash values in `CHANGELOG.md` for download check.
- If `sh process_data.sh` cannot locate `qo2mol.py`, try absolute path with command `python /abc/QO2Mol/src/dataset/qo2mol/qo2mol.py`.
- Mac OS may have unexpected issues associated with the file system, Linux is strongly recommended.


# Usage demo

Make sure current work directory is under the repository root
>pwd

We provide a usage demo that simply requires running the script
>sh scripts\dimenetpp\qm_benchmark\tr_baseline_e.sh


# License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. The images or other third party material in this article are included in the article’s Creative Commons license, unless indicated otherwise in the credit line; if the material is not included under the Creative Commons license, users will need to obtain permission from the license holder to reproduce the material. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
