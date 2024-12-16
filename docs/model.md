
# **Model Architecture**
The model integrates:
- **[YOLOv3](https://arxiv.org/abs/1804.02767)** for detecting objects in images, with a focus on robust bounding box predictions.
- **[MiDaS](https://github.com/isl-org/MiDaS)** for generating dense monocular depth maps, which provide a pixel-wise depth estimation of the scene.


## **Details**
1. **Unified Backbone:** A single **[ResNeXt101](https://arxiv.org/abs/1611.05431) backbone** is used for feature extraction, shared by both YOLOv3 and MiDaS branches. This reduces computation overhead and enables faster inference.
Specifically uses **ResNeXt101_32x8d_wsl (Weakly Supervised Learning)** backbone, pre-trained on Instagram data. This backbone provides robust feature extraction for both object detection and depth estimation tasks. 

2. **Task-Specific Heads:**
   - **YOLOv3 Head:** For object detection tasks such as bounding box regression, object classification, and confidence score prediction. In addition to Yolo regression layers, more convolutional layers are added to all three Yolo Heads for Small, Medium and Large objects.
   - **MiDaS Head:** For predicting depth maps from the extracted features.

## **Implementation Details**

Code is mostly adapted from these sources:
1. Originally almost all of the training and model code is from old version of [ultralytics/yolov3](https://github.com/ultralytics/yolov3). With alot of changes and additions from [CaptainEven/YOLOV4_MCMOT](https://github.com/CaptainEven/YOLOV4_MCMOT) and [ming71/yolov3-pytorch](https://github.com/ming71/yolov3-pytorch).
2. MiDas model implementation code is from the original implementation [isl-org/midas](https://github.com/isl-org/MiDaS).


![structure](assets/structure.PNG)

The model architecture change can be seen in:
1. `src/model/midas_blocks.py` for MiDas and ResNext blocks.
2. `src/model/yololayer.py` for Yolo layer.
3. `src/model/mde_net.py` for Combined architecture.


