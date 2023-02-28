"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

# from models.backbone.common import *
# from models.backbone.experimental import *
# from models.head.yolox_head import DetectYoloX
# from models.head.yolox_kp_head import DetectYoloXKeypoints
# from models.head.effidehead import EfficientDetect
from models.head.yolov5_head import Detect
# from models.head.tood import DetectTooD 
# from models.head.yolox_pss_head import DetectYoloXPss
# from models.head.nanodet_head import NanoDetHead
# from models.head.ppyoloe_head import PPYOLOEHead
# from utils.autoanchor import check_anchor_order
# from utils.general import check_yaml, make_divisible, print_args, set_logging
# from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info
   
from models.loss.loss import ComputeLoss
# from models.loss.yolox_loss import ComputeFastXLoss
# from models.head.yolov5_head import Detect
from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head
import logging
import torch.nn as nn
import torch
import math

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml'):  # model, input channels, number of classes
        super().__init__()
        self.cfg = cfg
    
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.head = build_head(cfg)
        # self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.names = cfg.Dataset.names # default names
        self.inplace = self.cfg.Model.inplace
        self.loss_fn = self.cfg.Loss.type
        if self.loss_fn is not None:
            self.loss_fn = eval(self.loss_fn) if isinstance(self.loss_fn, str) else None  # eval strings
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.head  # Detect()
        self.model_type = 'yolov5'
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # m.num_keypoints = self.num_keypoints
            m.stride = torch.Tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, cfg.Model.ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # LOGGER.info('Strides: %s' % m.stride.tolist())
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
    
    def _forward_once(self, x, profile=False, visualize=False):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
      
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.backbone.model[-1]  # Detect() module
        m = self.head  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # def _print_weights(self):
    #     for m in self.backbone.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.backbone.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward

        for m in self.neck.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        
        for layer in self.backbone.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
        for layer in self.neck.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    # def _apply(self, fn):
    #     # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
    #     self = super()._apply(fn)
    #     m = self.backbone[-1]  # Detect()
    #     if isinstance(m, Detect):
    #         m.stride = fn(m.stride)
    #         m.grid = list(map(fn, m.grid))
    #         if isinstance(m.anchor_grid, list):
    #             m.anchor_grid = list(map(fn, m.anchor_grid))
    #     return self


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--profile', action='store_true', help='profile model speed')
#     opt = parser.parse_args()
#     opt.cfg = check_yaml(opt.cfg)  # check YAML
#     print_args(FILE.stem, opt)
#     set_logging()
#     device = select_device(opt.device)

#     # Create model
#     model = Model(opt.cfg).to(device)
#     model.train()

#     # Profile
#     if opt.profile:
#         img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
#         y = model(img, profile=True)

#     # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
#     # from torch.utils.tensorboard import SummaryWriter
#     # tb_writer = SummaryWriter('.')
#     # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
#     # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
