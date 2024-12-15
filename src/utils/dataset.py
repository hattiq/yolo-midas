import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from utils.transforms import letterbox, random_affine
from utils.utils import xywh2xyxy, xyxy2xywh

help_url = "https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data"
img_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".dng"]
vid_formats = [".mov", ".avi", ".mp4"]

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        img_size=416,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_labels=True,
        cache_images=False,
        single_cls=False,
    ):
        path = str(Path(path))  # os-agnostic
        # print("path:", path)
        assert os.path.isfile(path), "File not found %s. See %s" % (
            path,
            help_url,
        )
        with open(path, "r") as f:
            self.img_files = [
                x.replace("/", os.sep)
                for x in f.read().splitlines()  # os-agnostic
                if os.path.splitext(x)[-1].lower() in img_formats
            ]

        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = (
            self.augment and not self.rect
        )  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        self.label_files = [
            x.replace("images/", "labels/").replace(
                os.path.splitext(x)[-1], ".txt"
            )
            for x in self.img_files
        ]

        # Depth targets
        self.depth_files = [
            x.replace("images/", "depth_images/").replace(
                os.path.splitext(x)[-1], ".png"
            )
            for x in self.img_files
        ]

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Read image shapes (wh)
            sp = path.replace(".txt", ".shapes")  # shapefile path
            try:
                with open(sp, "r") as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, "Shapefile out of sync"
            except:
                s = [
                    exif_size(Image.open(f))
                    for f in tqdm(self.img_files, desc="Reading image shapes")
                ]
                np.savetxt(sp, s, fmt="%g")  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.depth_files = [self.depth_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]  # wh
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / 64.0).astype(np.int) * 64
            )

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.dpth_imgs = [None] * n
        self.labels = [None] * n
        if cache_labels or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * n
            extract_bounding_boxes = False
            create_datasubset = False
            pbar = tqdm(self.label_files, desc="Caching labels")
            nm, nf, ne, ns, nd = (
                0,
                0,
                0,
                0,
                0,
            )  # number missing, found, empty, datasubset, duplicate
            for i, file in enumerate(pbar):
                # print(i)
                # print(file)
                try:
                    with open(file, "r") as f:
                        l = np.array(
                            [x.split() for x in f.read().splitlines()],
                            dtype=np.float32,
                        )
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    # print('missing labels for image %s' % self.img_files[i])
                    # print('filename %s' % file)
                    continue

                if l.shape[0]:
                    assert l.shape[1] == 5, "> 5 label columns: %s" % file
                    assert (l >= 0).all(), "negative labels: %s" % file
                    assert (l[:, 1:] <= 1).all(), (
                        "non-normalized or out of bounds coordinate labels: %s"
                        % file
                    )
                    if (
                        np.unique(l, axis=0).shape[0] < l.shape[0]
                    ):  # duplicate rows
                        nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                    if single_cls:
                        l[:, 0] = 0  # force dataset into single-class mode
                    self.labels[i] = l
                    nf += 1  # file found
                    # print("found")

                    # Create subdataset (a smaller dataset)
                    if create_datasubset and ns < 1e4:
                        if ns == 0:
                            create_folder(path="./datasubset")
                            os.makedirs("./datasubset/images")
                        exclude_classes = 43
                        if exclude_classes not in l[:, 0]:
                            ns += 1
                            # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                            with open("./datasubset/images.txt", "a") as f:
                                f.write(self.img_files[i] + "\n")

                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w = img.shape[:2]
                        for j, x in enumerate(l):
                            f = "%s%sclassifier%s%g_%g_%s" % (
                                p.parent.parent,
                                os.sep,
                                os.sep,
                                x[0],
                                j,
                                p.name,
                            )
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(
                                    Path(f).parent
                                )  # make new output folder

                            b = x[1:] * [w, h, w, h]  # box
                            b[2:] = b[2:].max()  # rectangle to square
                            b[2:] = b[2:] * 1.3 + 30  # pad
                            b = (
                                xywh2xyxy(b.reshape(-1, 4))
                                .ravel()
                                .astype(np.int)
                            )

                            b[[0, 2]] = np.clip(
                                b[[0, 2]], 0, w
                            )  # clip boxes outside of image
                            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                            assert cv2.imwrite(
                                f, img[b[1] : b[3], b[0] : b[2]]
                            ), "Failure extracting classifier boxes"
                else:
                    ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                    # print('empty labels for image %s' % self.img_files[i])
                    # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

                pbar.desc = (
                    "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)"
                    % (nf, nm, ne, nd, n)
                )
            assert nf > 0, "No labels found in %s. See %s" % (
                os.path.dirname(file) + os.sep,
                help_url,
            )

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                # try:
                (
                    self.imgs[i],
                    self.img_hw0[i],
                    self.img_hw[i],
                    self.dpth_imgs[i],
                ) = self.load_image(
                    i
                )  # img, hw_original, hw_resized, depth_image
                gb += self.imgs[i].nbytes
                gb += self.dpth_imgs[i].nbytes
                pbar.desc = "Caching images (%.1fGB)" % (gb / 1e9)
            # except:
            #     pass

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image

            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except:
                    print("Corrupted image detected: %s" % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels, dp_img = load_mosaic(self, index)
            dp_img = cv2.resize(
                dp_img,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            shapes = None

        else:

            # Load image
            img, (h0, w0), (h, w), dp_img = self.load_image(index)

            # Letterbox
            shape = (
                self.batch_shapes[self.batch[index]]
                if self.rect
                else self.img_size
            )  # final letterboxed shape
            img, ratio, pad = letterbox(
                img, shape, auto=False, scaleFill=True, scaleup=self.augment
            )
            img = cv2.resize(
                img,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            dp_img = cv2.resize(
                dp_img,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x is not None and x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()

                labels[:, 1] = (
                    ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                )  # pad width
                labels[:, 2] = (
                    ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                )  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace

            if not self.mosaic:
                img, labels, dp_img = random_affine(
                    img,
                    dp_img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                )

            # Augment colorspace
            augment_hsv(
                img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"]
            )
            augment_hsv(
                dp_img,
                hgain=hyp["hsv_h"],
                sgain=hyp["hsv_s"],
                vgain=hyp["hsv_v"],
            )

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                # dp_img = np.fliplr(dp_img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        dp_img = dp_img[:, :, 0].reshape(
            1, dp_img.shape[0], dp_img.shape[1]
        )  # BGR to grey, to 1x416x416 depth
        dp_img = np.ascontiguousarray(dp_img)

        return (
            torch.from_numpy(img),
            labels_out,
            self.img_files[index],
            shapes,
            torch.from_numpy(dp_img),
        )

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, dp_img = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return (
            torch.stack(img, 0),
            torch.cat(label, 0),
            path,
            shapes,
            torch.stack(dp_img, 0),
        )

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            img_path = self.img_files[index]
            dpth_path = self.depth_files[index]
            img = cv2.imread(img_path)  # BGR
            dpth_img = cv2.imread(dpth_path)  # BGR

            assert img is not None, (
                "I cv2.imread(img_path)  # BGRmage Not Found " + img_path
            )
            assert dpth_img is not None, (
                "I cv2.imread(dpth_path)  # BGRmage Not Found " + dpth_path
            )

            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r < 1 or (
                self.augment and r != 1
            ):  # always resize down, only resize up if training with augmentation
                interp = (
                    cv2.INTER_AREA
                    if r < 1 and not self.augment
                    else cv2.INTER_LINEAR
                )
                img = cv2.resize(
                    img, (int(w0 * r), int(h0 * r)), interpolation=interp
                )

            return (
                img,
                (h0, w0),
                img.shape[:2],
                dpth_img,
            )  # img, hw_original, hw_resized, depth_image
        else:
            return (
                self.imgs[index],
                self.img_hw0[index],
                self.img_hw[index],
                self.dpth_imgs[index],
            )  # img, hw_original, hw_resized, depth_image


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    img_hsv = (
        (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x)
        .clip(None, 255)
        .astype(np.uint8)
    )
    np.clip(
        img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0]
    )  # inplace hue clip (0 - 179 deg)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    # print("image_size in load_mosaic", self.img_size)
    xc, yc = [
        int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)
    ]  # mosaic center x, y
    indices = [index] + [
        random.randint(0, len(self.labels) - 1) for _ in range(3)
    ]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w), dp_img = self.load_image(index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full(
                (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
            )  # base image with 4 tiles
            # dp_img4 = np.full((s * 2, s * 2, dp_img.shape[2]), 114, dtype=np.uint8)  # base depth image with 4 tiles
            x1a, y1a, x2a, y2a = (
                max(xc - w, 0),
                max(yc - h, 0),
                xc,
                yc,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                0,
                max(xc, w),
                min(y2a - y1a, h),
            )
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[
            y1b:y2b, x1b:x2b
        ]  # img4[ymin:ymax, xmin:xmax]
        # dp_img4[y1a:y2a, x1a:x2a] = dp_img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax] depth
        padw = x1a - x1b
        padh = y1a - y1b

        # Load labels
        label_path = self.label_files[index]
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, "r") as f:
                    x = np.array(
                        [x.split() for x in f.read().splitlines()],
                        dtype=np.float32,
                    )

            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            else:
                labels = np.zeros((0, 5), dtype=np.float32)
            labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(
            labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:]
        )  # use with random_affine

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4, dp_img = random_affine(
        img4,
        dp_img,
        labels4,
        degrees=self.hyp["degrees"] * 1,
        translate=self.hyp["translate"] * 1,
        scale=self.hyp["scale"] * 1,
        shear=self.hyp["shear"] * 1,
        border=-s // 2,
    )  # border to remove

    return img4, labels4, dp_img
