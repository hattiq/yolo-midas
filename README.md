# <div align="center">YOLO-MiDaS for Autonomous Navigation</div>

This repository combines YOLOv3 for object detection and MiDaS for monocular depth estimation into a single model using a ResNeXt101 backbone. The idea is to use shared feature extractor for efficient real-time performance for autonomous navigation applications.

> The code in this repository is mostly adapted from [CaptainEven/YOLOV4_MCMOT](https://github.com/CaptainEven/YOLOV4_MCMOT) and old version of [ultralytics/yolov3](https://github.com/ultralytics/yolov3). MiDas model implementation code is from the original implementation [isl-org/midas](https://github.com/isl-org/MiDaS)

## <div align="center">Documentation</div>

### Install

Clone repo and install [requirements.txt](https://github.com/hattiq/yolo-midas/blob/master/requirements.txt) in a [**Python==3.6.0**](https://www.python.org/) environment.

```bash
git clone https://github.com/hattiq/yolo-midas  # clone
cd yolo-midas
pip install -r requirements.txt  # install
```

or use conda environment.

```bash
conda create -n yolo-midas python==3.6
conda activate yolo-midas
pip install -r requirements.txt
```


### Inference with detect.py

> Weights are not provided for the model, needs training first.

`src/detect.py` runs inference on images in the directory. Some sample images are provided at `data/sample`.

```bash
python src/detect.py --source data/sample --conf-thres 0.1 --output output --weights weights/best.pt
```

Outputs will be saved in `output` directory.

Sample output:

![result](docs/assets/results.png)


### Training

The model is trained on Construction Safety Gear Data which can be found here https://github.com/sarvan0506/EVA5-Vision-Squad/tree/Saravana/14_Construction_Safety_Dataset. If training need to done on custom datasets refer the data preparation steps mentioned in the page.

Place the data inside `data/customdata/custom.data` folder.
Please refer the config file `cfg/mde.cfg` to change the network configuration, freeze different branches. The model is an extension of YOLOv3 and MiDaS networks. Most of the configurations can be understood if familiar with yolo.

```bash
python src/train.py --data data/customdata/custom.data --batch 8 --cache --cfg cfg/mde.conf --epochs 50 --img-size 512
```

More details on training in [docs](docs/training.md).


## Further Documentation:
1. [Model Architecture](docs/model.md)
2. [Training Strategy](docs/training.md)


### Refs
-  https://sarvan0506.medium.com/yolo-v3-and-midas-from-a-single-resnext101-backbone-8ba42948bf65
