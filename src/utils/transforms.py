import math
import random

import cv2
import numpy as np


def letterbox(
    img,
    new_shape=(416, 416),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = (
            new_shape[0] / shape[1],
            new_shape[1] / shape[0],
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # print("shape", shape)
    # print("new_unpad", new_unpad)

    try:
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    except ValueError:
        # print("shape", shape)
        # print("new_unpad", new_unpad)
        pass

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def random_affine(
    img,
    dp_img,
    pln_img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    border=0,
):
    # torchvision.transforms.RandomAffine(
    #   degrees=(-10, 10),
    #   translate=(.1, .1),
    #   scale=(.9, 1.1),
    #   shear=(-10, 10)
    # )
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(
        angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s
    )

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(-translate, translate) * img.shape[0] + border
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(-translate, translate) * img.shape[1] + border
    )  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        random.uniform(-shear, shear) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        random.uniform(-shear, shear) * math.pi / 180
    )  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(
            img,
            M[:2],
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=(114, 114, 114),
        )
        # dp_img = cv2.warpAffine(dp_img, M[:2], dsize=(width, height),
        #               flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        # pln_img = cv2.warpAffine(pln_img, M[:2], dsize=(width, height),
        #               flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = (
            np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
            .reshape(4, n)
            .T
        )

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(
        #   abs(math.sin(radians)),
        #   abs(math.cos(radians))
        #   ) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate(
        #   (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        # ).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (
            targets[:, 4] - targets[:, 2]
        )
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets, dp_img, pln_img
