"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from models.head.retina_head import RetinaDetect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.backbone.common import *
from models.backbone.experimental import *
from models.head.yolov5_head import Detect
from models.head.yolov7_head import IDetect
from models.head.yolov6_head import YoloV6Detect
from models.head.yolov8_head import YoloV8Detect
from models.head.yolox_head import YoloXDetect
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

from models.loss.loss import *
from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

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
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
        # Build strides, anchors
        self.check_head()
     
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def check_head(self):
        m = self.head  # Detect()
        self.model_type = 'yolov5'
        if isinstance(m, (Detect, RetinaDetect, IDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # m.num_keypoints = self.num_keypoints
            m.stride = torch.Tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.cfg.Model.ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.initialize_biases()  # only run once

        elif isinstance(m, (YoloXDetect, YoloV6Detect, YoloV8Detect)):
            m.inplace = self.inplace
            self.stride = torch.Tensor(m.stride)
            m.initialize_biases()  # only run once
            self.model_type = 'yolox'


    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
    
    def _forward_once(self, x, profile=False, visualize=False):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
      
        return x

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
                if isinstance(layer, QARepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
                if hasattr(layer, 'reparameterize'):
                    layer.reparameterize()
        for layer in self.neck.modules():
                if isinstance(layer, QARepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                if isinstance(layer, RepConv):
                    layer.fuse_repvgg_block()
                if hasattr(layer, 'reparameterize'):
                    layer.reparameterize()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
