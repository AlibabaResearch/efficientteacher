import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.backbone.common import *
from models.backbone.experimental import *
from models.head.yolox_head import DetectYoloX
from models.head.yolox_pss_head import DetectYoloXPss
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync
from models.loss.loss import *
from models.head.yolov5_head import Detect

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class YoloV7BackBone(nn.Module):
    def __init__(self, cfg):
        super(YoloV7BackBone, self).__init__()
        # self.version = version
        # gains = {'s': {'gd': 0.33, 'gw': 0.5},
        #          'm': {'gd': 0.67, 'gw': 0.75},
        #          'l': {'gd': 1, 'gw': 1},
        #          'x': {'gd': 1.33, 'gw': 1.25}}
        # self.gd = gains[self.version.lower()]['gd']  # depth gain
        # self.gw = gains[self.version.lower()]['gw']  # width gain
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        self.channels_out = {
            'stage0': 64,
            'stage1': 128,
            'stage2': 256,
            'stage3': 512,
            'stage4': 1024,
            'stage5': 1024,
        }
        self.re_channels_out()

        if cfg.Model.Backbone.activation == 'SiLU': 
            CONV_ACT = 'silu'
        elif cfg.Model.Backbone.activation == 'ReLU': 
            CONV_ACT = 'relu'
        elif cfg.Model.Backbone.activation == 'LeakyReLU': 
            CONV_ACT = 'lrelu'
        else:
            CONV_ACT = 'hard_swish'

        self.stage0 = PreConv(3, self.channels_out['stage0'], 1, 0.5, True, CONV_ACT)
        self.stage1 = PreConv(self.channels_out['stage0'], self.channels_out['stage1'], 1, 0.5, True, CONV_ACT)

        self.elan_0 = ELAN(self.channels_out['stage1'], self.channels_out['stage2'], self.get_depth(2), 0.5, False, False, CONV_ACT)
        self.elan_1 = ELAN(self.channels_out['stage2'], self.channels_out['stage3'], self.get_depth(2), 0.5, True, True, CONV_ACT)
        self.elan_2 = ELAN(self.channels_out['stage3'], self.channels_out['stage4'], self.get_depth(2), 0.5, True, True, CONV_ACT)
        self.elan_3 = ELAN(self.channels_out['stage4'], self.channels_out['stage5'], self.get_depth(2), 0.25, True, True, CONV_ACT)

        # print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.channels_out['stage1'],
                                                                    # self.channels_out['stage2'],
                                                                    # self.channels_out['stage3']))

    def forward(self, x):
        x0 = self.stage0(x) 
        x1 = self.stage1(x0) 

        x2 = self.elan_0(x1)
        x3 = self.elan_1(x2)
        x4 = self.elan_2(x3)
        x5 = self.elan_3(x4)
       
        return x3, x4, x5

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
