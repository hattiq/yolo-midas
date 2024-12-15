import torch
from torch import nn

ONNX_EXPORT = False

class YOLOLayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):

        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            # print("Creating grids...")
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng)

        # build xy offsets
        #print("self.training", self.training)
        if not self.training:
            #print("Creating grids...")
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):

        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            # print("Inside else creating grid")
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            #print("self.nx, self.ny", self.nx, self.ny)
            #print("nx, ny", nx, ny)
            if (self.nx, self.ny) != (nx, ny):
                #print("elif after ONNX_EXPORT Creating grids...")
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid = self.grid.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

    def save_weights(self, path='model.weights', cutoff=-1):
        # Converts a PyTorch model to Darket format (*.pt to *.weights)
        # Note: Does not work if model.fuse() is applied
        with open(path, 'wb') as f:
            # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
            self.version.tofile(f)  # (int32) version info: major, minor, revision
            self.seen.tofile(f)  # (int64) number of images seen during training

            # Iterate through layers
            for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
                if mdef['type'] == 'convolutional':
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if mdef['batch_normalize']:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.data.cpu().numpy().tofile(f)
                        bn_layer.running_var.data.cpu().numpy().tofile(f)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)

