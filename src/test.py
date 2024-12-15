import argparse
import time
from torch.utils.data import DataLoader

from model.mde_net import *
from utils import *

def test(
    cfg,
    data,
    weights=None,
    batch_size=16,
    img_size=416,
    conf_thres=0.001,
    iou_thres=0.6,  # for nms
    save_json=False,
    single_cls=False,
    augment=False,
    model=None,
    dataloader=None,
):
    """
    Perform testing and evaluation of a YOLO-based object detection model.

    Args:
        cfg (str): Path to the model configuration file (e.g., .yaml or .cfg format).
        data (str): Path to the dataset configuration file (e.g., .yaml) defining training, validation, and test datasets.
        weights (str, optional): Path to the model weights file. Defaults to None, indicating the use of randomly initialized weights or a preloaded model.
        batch_size (int, optional): Batch size to use during testing. Defaults to 16.
        img_size (int, optional): Input image size (height and width) for testing. Defaults to 416.
        conf_thres (float, optional): Confidence threshold for object detection. Detections with scores below this threshold are discarded. Defaults to 0.001.
        iou_thres (float, optional): Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS). Defaults to 0.6.
        save_json (bool, optional): Whether to save test results in COCO JSON format for evaluation. Defaults to False.
        single_cls (bool, optional): If True, treats all classes as a single class for evaluation. Useful for binary or single-class datasets. Defaults to False.
        augment (bool, optional): If True, performs augmented inference (e.g., multi-scale testing). Defaults to False.
        model (torch.nn.Module, optional): Preloaded PyTorch model. If provided, bypasses loading from `cfg` and `weights`. Defaults to None.
        dataloader (torch.utils.data.DataLoader, optional): Preloaded DataLoader for testing. If provided, bypasses loading data from the `data` argument. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - metrics (tuple): A tuple with the following metrics:
                - mp (float): Mean precision across all classes.
                - mr (float): Mean recall across all classes.
                - map (float): Mean Average Precision (mAP) at IoU threshold 0.5.
                - mf1 (float): Mean F1 score across all classes.
                - loss (list): A list of average test losses, including:
                    - classification loss,
                    - objectness loss, and
                    - bounding box regression loss.
            - maps (list): Class-wise mAP values at IoU threshold 0.5 for each class.
    """






    yolo_props, freeze, alpha = parse_yolo_freeze_cfg(cfg)

    # Initialize/load model and set device
    if model is None:
        raise Exception("Model is None.")
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data["classes"])  # number of classes
    path = data["valid"]  # path to test images
    names = load_classes(data["names"])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(
        device
    )  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    __import__('ipdb').set_trace()
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(
            path, img_size, batch_size, rect=True, single_cls=opt.single_cls
        )
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min(
                [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
            ),
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    seen = 0

    model.eval()

    """
    model.training = False
    model.yolo1.training = False
    model.yolo2.training = False
    model.yolo3.training = False
    """

    _ = (
        model(torch.zeros((1, 3, img_size, img_size), device=device))
        if device.type != "cpu"
        else None
    )  # run once

    s = ("%20s" + "%10s" * 7) % (
        "Class",
        "Images",
        "Targets",
        " mse depth",
        "P",
        "R",
        "mAP@0.5",
        "F1",
    )
    # print(s)
    p, r, f1, mp, mr, map, mf1, t0, t1 = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes, dp_imgs) in enumerate(
        tqdm(dataloader, desc=s)
    ):
        imgs = (
            imgs.to(device).float() / 255.0
        )  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Plot images with bounding boxes
        f = "test_batch%g.png" % batch_i  # filename
        if batch_i < 1 and not os.path.exists(f):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            midas_out, yolo_inf_out, yolo_train_out = model(
                imgs, augment=augment
            )  # inference and training outputs

            """
            print("yolo_inf_out")
            print(yolo_inf_out.shape)
            print("yolo_out_0", yolo_train_out[1].shape)
            print("yolo_out_1", yolo_train_out[1].shape)
            print("yolo_out_2", yolo_train_out[2].shape)
            """

            t0 += time_synchronized() - t
            # Compute loss
            if hasattr(model, "hyp"):  # if model has loss hyperparameters
                loss += compute_loss(
                    yolo_train_out, targets, midas_out, dp_imgs, alpha, model
                )[1][
                    :4
                ]  # GIoU, obj, cls, ldepth
            # print("test loss", loss)
            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(
                yolo_inf_out, conf_thres=conf_thres, iou_thres=iou_thres
            )  # nms
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(
                        (
                            torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split("_")[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(
                    imgs[si].shape[1:], box, shapes[si][0], shapes[si][1]
                )  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                # for p, b in zip(pred.tolist(), box.tolist()):
                #     jdict.append(
                #         {
                #             "image_id": image_id,
                #             "category_id": coco91class[int(p[5])],
                #             "bbox": [round(x, 3) for x in b],
                #             "score": round(p[4], 5),
                #         }
                #     )

            # Assign all predictions as incorrect
            correct = torch.zeros(
                pred.shape[0], niou, dtype=torch.bool, device=device
            )
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (
                        (cls == tcls_tensor).nonzero().view(-1)
                    )  # prediction indices
                    pi = (
                        (cls == pred[:, 5]).nonzero().view(-1)
                    )  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                            1
                        )  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = (
                                    ious[j] > iouv
                                )  # iou_thres is 1xn
                                if (
                                    len(detected) == nl
                                ):  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            )

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = (
                p[:, 0],
                r[:, 0],
                ap.mean(1),
                ap[:, 0],
            )  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(
            stats[3].astype(np.int64), minlength=nc
        )  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%10.3g" * 7  # print format

    # ldepth = 1 - loss[3]/(batch_i+1)
    ldepth = loss[3] / (batch_i + 1)

    print(pf % ("all", seen, nt.sum(), ldepth, mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
            img_size,
            img_size,
            batch_size,
        )  # tuple
        print(
            "Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g"
            % t
        )

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument(
        "--cfg", type=str, default="cfg/yolov3-spp.cfg", help="*.cfg path"
    )
    parser.add_argument(
        "--data", type=str, default="data/coco2014.data", help="*.data path"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolov3-spp-ultralytics.pt",
        help="weights path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=416, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="object confidence threshold",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save a cocoapi-compatible JSON results file",
    )
    parser.add_argument(
        "--task", default="test", help="'test', 'study', 'benchmark'"
    )
    parser.add_argument(
        "--device", default="", help="device id (i.e. 0 or 0,1) or cpu"
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train as single-class dataset",
    )
    parser.add_argument(
        "--augment", action="store_true", help="augmented inference"
    )
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any(
        [x in opt.data for x in ["coco.data", "coco2014.data", "coco2017.data"]]
    )
    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == "test":  # (default) test normally
        test(
            opt.cfg,
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
        )

    elif opt.task == "benchmark":  # mAPs at 320-608 at conf 0.5 and 0.7
        y = []
        for i in [320, 416, 512, 608]:  # img-size
            for j in [0.5, 0.7]:  # iou-thres
                t = time.time()
                r = test(
                    opt.cfg,
                    opt.data,
                    opt.weights,
                    opt.batch_size,
                    i,
                    opt.conf_thres,
                    j,
                    opt.save_json,
                )[0]
                y.append(r + (time.time() - t,))
        np.savetxt(
            "benchmark.txt", y, fmt="%10.4g"
        )  # y = np.loadtxt('study.txt')

    elif opt.task == "study":  # Parameter study
        y = []
        x = np.arange(0.4, 0.9, 0.05)  # iou-thres
        for i in x:
            t = time.time()
            r = test(
                opt.cfg,
                opt.data,
                opt.weights,
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                i,
                opt.save_json,
            )[0]
            y.append(r + (time.time() - t,))
        np.savetxt("study.txt", y, fmt="%10.4g")  # y = np.loadtxt('study.txt')

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        y = np.stack(y, 0)
        ax[0].plot(x, y[:, 2], marker=".", label="mAP@0.5")
        ax[0].set_ylabel("mAP")
        ax[1].plot(x, y[:, 3], marker=".", label="mAP@0.5:0.95")
        ax[1].set_ylabel("mAP")
        ax[2].plot(x, y[:, -1], marker=".", label="time")
        ax[2].set_ylabel("time (s)")
        for i in range(3):
            ax[i].legend()
            ax[i].set_xlabel("iou_thr")
        fig.tight_layout()
        plt.savefig("study.jpg", dpi=200)
