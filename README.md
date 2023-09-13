# Contrastive Learning for Cardiac Segmentation
An Improved Contrastive Learning Network for
Semi-supervised Multi-structure Segmentation
in Echocardiography
## Overview
This repository provides the method described in the paper
> Zhaowen Qiu, et al. "An Improved Contrastive Learning Network for
Semi-supervised Multi-structure Segmentation
in Echocardiography"
## Requirements
The repository is tested on Ubuntu 20.04.6 LTS, Python 3.8, PyTorch 1.13.0 and CUDA 12.0
```
pip install -r requirements.txt
```
## Description
The repository made two improvements to the paper [**Semi-supervised Semantic Segmentation with Directional Context-aware Consistency**](https://jiaya.me/papers/semiseg_cvpr21.pdf), building upon the work by replacing DeeplabV3+ with U-net and modifying the
structure of the projector. These changes aimed to tackle challenges in echocardiography, such as low
contrast, unclear boundaries, and incomplete cardiac structures.
# Related Repositories

This repository highly depends on the **CAC** repository at https://github.com/dvlab-research/Context-Aware-Consistency. We thank the authors of CAC for their great work and code.

Besides, we also borrow some codes from the following repositories.

- **Unet** at https://github.com/milesial/Pytorch-UNet/tree/master/unet.

## Usage
### Preprocessing of CAMUS dataset
1. Firstly, download the [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) Dataset.

2. Set the dir of your training set to 'data_dir' in the config file 'heart_cac_deeplabv3+_resnet50_1over8_datalist0.json'.

3. Set the dir of your validation set to 'val_loader' -> 'data_dir' in the config file 'heart_cac_deeplabv3+_resnet50_1over8_datalist0.json'.
