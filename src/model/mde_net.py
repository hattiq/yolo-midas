import torch
from torch import nn
from model.midas_blocks import FeatureFusionBlock, Interpolate, _make_encoder
from model.yololayer import YOLOLayer
from utils.parse_config import *
from utils import torch_utils

ONNX_EXPORT = False

class MDENet(nn.Module):
    # ResNext+YOLOv3+MiDas object detection model

    def __init__(self, yolo_props, path=None, freeze={}, features=256, non_negative=True, img_size=(416, 416), verbose=False):
        
        super(MDENet, self).__init__()

        use_pretrained = True if path is None else False

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)
        
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )
        
        # YOLO head
        conv_output = (int(yolo_props["num_classes"]) + 5) * int((len(yolo_props["anchors"]) / 3))
        self.upsample1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # small objects
        self.yolo1_learner = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.yolo1_reduce = nn.Conv2d(1024, conv_output, kernel_size=1, stride=2, padding=1)
        self.yolo1 = YOLOLayer(yolo_props["anchors"][:3],
                               nc=int(yolo_props["num_classes"]),
                               img_size=img_size,
                               yolo_index=0,
                               layers=[],
                               stride=32)
        
        # medium objects
        self.yolo2_learner = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.yolo2_reduce = nn.Sequential(
            nn.Conv2d(512, conv_output, kernel_size=1, stride=2)
        )
        self.yolo2 = YOLOLayer(yolo_props["anchors"][3:6],
                               nc=int(yolo_props["num_classes"]),
                               img_size=img_size,
                               yolo_index=1,
                               layers=[],
                               stride=16)
        
        
        # large objects
        self.yolo3_learner = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        
        self.yolo3_reduce = nn.Sequential(
            nn.Conv2d(256, conv_output, kernel_size=1, stride=2)
        )
        self.yolo3 = YOLOLayer(yolo_props["anchors"][6:],
                               nc=int(yolo_props["num_classes"]),
                               img_size=img_size,
                               yolo_index=1,
                               layers=[],
                               stride=8)
        
        
        self.yolo = [self.upsample1, self.upsample2, self.yolo1_learner, self.yolo1,
                     self.yolo2_learner, self.yolo2, self.yolo3_learner, self.yolo3,
                     self.yolo1_reduce, self.yolo2_reduce, self.yolo3_reduce]
        
        if path:
            self.load(path)
        
        if freeze['resnet'] == "True":
            path = "freeze"
        if freeze['midas'] == "True":
            for param in self.scratch.parameters():
                param.requires_grad = False
        if freeze['yolo'] == "True":
            for mod in self.yolo:
                for param in mod.parameters():
                    param.requires_grad = False

        if path is not None:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_net(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y1 = []
            y2 = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                out = self.forward_net(xi)
                y1.append(out[0])
                y2.append(out[1])

            y2[1][..., :4] /= s[0]  # scale
            y2[1][..., 0] = img_size[1] - y2[1][..., 0]  # flip lr
            y2[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi
            
            #y1 = torch.cat(y1, 1)
            y2 = torch.cat(y2, 1)
            
            return y1, y2, None

    def forward_net(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)
            
        midas_out, yolo_out = self.run_batch(x)
        
        if self.training:  # train
            return midas_out, yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return midas_out, x, p
    
    def run_batch(self, x):
        
        # Pretrained resnext101
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        # Depth Detection
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        depth_out = self.scratch.output_conv(path_1)
        
        # Object Detection
        
        # small objects
        yolo1_out = self.yolo1(self.yolo1_reduce(self.yolo1_learner(layer_3)))
        
        # medium objects
        layer_3 = self.upsample1(layer_3)
        layer_2 = torch.cat([layer_3, layer_2], dim=1)
        layer_2 = self.yolo2_learner(layer_2)
        yolo2_out = self.yolo2(self.yolo2_reduce(layer_2))
        
        # large objects
        layer_2 = self.upsample2(layer_2)
        layer_1 = torch.cat([layer_1, layer_2], dim=1)
        layer_1 = self.yolo3_learner(layer_1)
        yolo3_out = self.yolo3(self.yolo3_reduce(layer_1))
        

        yolo_out = [yolo1_out, yolo2_out, yolo3_out]
        
        return depth_out, yolo_out

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)
    
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        print("path", path)
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
