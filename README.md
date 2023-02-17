# SSD_SRGAN_PedestrianDetection

This repository implements [Distant Blurred Pedestrian Detection: Based on SSD Network with SRGAN Image Super Resolution](https://ieeexplore.ieee.org/abstract/document/9824948). The implementation is heavily influenced by the projects [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), and [SRGAN](https://github.com/tensorlayer/SRGAN).

## Installation
### Requirements
1. Python3.7
1. PyTorch 1.1.0
### Steps
```bash
git clone https://github.com/lufficc/SSD.git
# go to PyTorch official website to install the Torch 1.1.0 version
# both GPU or CPU version Torch is okey
```
### Setting Up Datasets
#### Pascal VOC
For Pascal VOC dataset, make the folder structure like this:
```
VOC_ROOT
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```
Where `VOC_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export VOC_ROOT="/path/to/voc_root"`.

## Test
```bash
python test.py
```
## Train
```bash
python train.py
```
## Evaluate
```bash
python eval.py
```
