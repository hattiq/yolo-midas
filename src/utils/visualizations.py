import random
import cv2
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.utils import *

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_images(imgs, targets, paths=None, fname="images.png"):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs**0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(int(ns), int(ns), i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], ".-")
        plt.axis("off")
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(
                s[: min(len(s), 40)], fontdict={"size": 8}
            )  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def plot_results(
    start=0, stop=0, bucket="", id=()
):  # from utils.utils import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov3#training
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = [
        "GIoU",
        "Objectness",
        "Classification",
        "Precision",
        "Recall",
        "val GIoU",
        "val Objectness",
        "val Classification",
        "mAP@0.5",
        "F1",
    ]
    if bucket:
        os.system("rm -rf storage.googleapis.com")
        files = [
            "https://storage.googleapis.com/%s/results%g.txt" % (bucket, x)
            for x in id
        ]
    else:
        files = glob.glob("results*.txt") + glob.glob(
            "../../Downloads/results*.txt"
        )
    for f in sorted(files):
        try:
            results = np.loadtxt(
                f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2
            ).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                ax[i].plot(
                    x,
                    y,
                    marker=".",
                    label=Path(f).stem,
                    linewidth=2,
                    markersize=8,
                )
                ax[i].set_title(s[i])
                if i in [5, 6, 7]:  # share train and val loss y axes
                    ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print("Warning: Plotting error for %s, skipping file" % f)

    fig.tight_layout()
    ax[1].legend()
    fig.savefig("results.png", dpi=200)

