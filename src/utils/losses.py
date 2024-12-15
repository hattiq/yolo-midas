import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import wh_iou, bbox_iou


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(
    p, targets, m, m_targets, alpha, model
):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lcls, lbox, lobj, ldepth, lpln = ft([0]), ft([0]), ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(p, targets, model)
    h = model.hyp  # hyperparameters
    red = "mean"  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h["cls_pw"]]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h["obj_pw"]]), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h["fl_gamma"]  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(
                ps[:, 0:2]
            )  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1e3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(
                pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True
            )  # giou computation
            lbox += (
                (1.0 - giou).sum() if red == "sum" else (1.0 - giou).mean()
            )  # giou loss
            tobj[b, a, gj, gi] = (
                1.0 - model.gr
            ) + model.gr * giou.detach().clamp(0).type(
                tobj.dtype
            )  # giou ratio

            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # lcls += CE(ps[:, 5:], tcls[i])  # CE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h["giou"]
    lobj *= h["obj"]
    lcls *= h["cls"]
    if red == "sum":
        bs = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng

    # Depth loss
    if m.shape != m_targets.shape:
        m_targets = F.interpolate(m_targets, (m.shape[2], m.shape[3]))

    # ldepth += (1 - pytorch_ssim.ssim(m.float().cuda(), m_targets.float().cuda()))
    ldepth += nn.MSELoss()(m.float().cuda(), m_targets.float().cuda())

    # print("ldepth", ldepth)
    # print("lbox", lbox)
    # print("lobj", lobj)
    # print("lcls", lcls)

    yolo_loss = alpha["yolo"] * (lbox + lobj + lcls)
    midas_loss = alpha["midas"] * ldepth

    loss = yolo_loss + midas_loss
    return loss, torch.cat((lbox, lobj, lcls, ldepth, loss)).detach()


def build_targets(p, targets, model):
    # targets = [image, class, x, y, w, h]

    nt = targets.shape[0]
    tcls, tbox, indices, av = [], [], [], []
    reject, use_all_anchors = True, True
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    # m = list(model.modules())[-1]
    # for i in range(m.nl):
    #    anchors = m.anchors[i]
    # multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate([model.yolo1, model.yolo2, model.yolo3]):
        # get number of grid points and anchor vec for this yolo layer
        # anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        anchors = j.anchor_vec
        nc = j.nc
        # iou of targets-anchors
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        t, a = targets * gain, []
        gwh = t[:, 4:6]
        if nt:
            iou = wh_iou(
                anchors, gwh
            )  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))

            if use_all_anchors:
                na = anchors.shape[0]  # number of anchors
                a = torch.arange(na).view(-1, 1).repeat(1, nt).view(-1)
                t = t.repeat(na, 1)
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = (
                    iou.view(-1) > model.hyp["iou_t"]
                )  # iou threshold hyperparameter
                t, a = t[j], a[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4]  # grid x, y
        gwh = t[:, 4:6]  # grid w, h
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))

        # Box
        gxy -= gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchors[a])  # anchor vec

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < nc, (
                "Model accepts %g classes labeled from 0-%g, however you labelled a class %g. "
                "See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data"
                % (nc, nc - 1, c.max())
            )

    return tcls, tbox, indices, av
