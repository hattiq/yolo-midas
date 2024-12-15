# YOLO-MiDaS for Autonomous Navigation

This repository combines YOLOv3 for object detection and MiDaS for monocular depth estimation into a single model using a ResNeXt101 backbone. The idea is to use shared feature extractor for efficient real-time performance for autonomous navigation applications.

## **Model Architecture**
The model integrates:
- **[YOLOv3](https://arxiv.org/abs/1804.02767)** for detecting objects in images, with a focus on robust bounding box predictions.
- **[MiDaS](https://github.com/isl-org/MiDaS)** for generating dense monocular depth maps, which provide a pixel-wise depth estimation of the scene.


### **Key Features**
1. **Unified Backbone:** A single **[ResNeXt101](https://arxiv.org/abs/1611.05431) backbone** is used for feature extraction, shared by both YOLOv3 and MiDaS branches. This reduces computation overhead and enables faster inference.
Specifically uses **ResNeXt101_32x8d_wsl (Weakly Supervised Learning)** backbone, pre-trained on Instagram data. This backbone provides robust feature extraction for both object detection and depth estimation tasks.

2. **Task-Specific Heads:**
   - **YOLOv3 Head:** For object detection tasks such as bounding box regression, object classification, and confidence score prediction.
   - **MiDaS Head:** For predicting depth maps from the extracted features.











![structure](docs/assets/structure.PNG)

The model architecture change can be seen in `model/mde_net.py`

## Training

The model is trained on Construction Safety Gear Data which can be found here https://github.com/sarvan0506/EVA5-Vision-Squad/tree/Saravana/14_Construction_Safety_Dataset. If training need to done on custom datasets refer the data preparation steps mentioned in the page.

Place the data inside `data/customdata/custom.data` folder

`python3.6 train.py --data data/customdata/custom.data --batch 8 --cache --cfg cfg/mde.conf --epochs 50 --img-size 512`

Please refer the config file `cfg/mde.cfg` to change the network configuration, freeze different branches. The model is an extension of YOLOv3 and MiDaS networks. Most of the configurations can be understood if familiar with

1. https://github.com/ultralytics/yolov3
2. https://github.com/intel-isl/MiDaS

## Inference

Download the weights from https://drive.google.com/file/d/1LZoWaZbsD4gG4xgWQ4cW-ezyhmaHXV1O/view?usp=sharing and place it under `weights` folder

Place the images on which inference need to be run, inside `input` folder

`python3.6 detect.py --source input --conf-thres 0.1 --output output --weights weights/best.pt`

The inferred images will be stored inside `output` folder

## Inference Result Sample

![result](docs/assets/results.png)


### Refs
-  https://sarvan0506.medium.com/yolo-v3-and-midas-from-a-single-resnext101-backbone-8ba42948bf65
