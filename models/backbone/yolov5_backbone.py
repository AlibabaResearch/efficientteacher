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
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from models.loss.loss import *
from models.head.yolov5_head import Detect

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class YoloV5BackBone(nn.Module):
    def __init__(self, cfg):
        super(YoloV5BackBone, self).__init__()
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        self.channels_out = {
            'stage1': 64,
            'stage2_1': 128,
            'stage2_2': 128,
            'stage3_1': 256,
            'stage3_2': 256,
            'stage4_1': 512,
            'stage4_2': 512,
            'stage5': 1024,
            'spp': 1024,
            'csp1': 1024,
            'conv1': 1024
        }
        self.re_channels_out()

        if cfg.Model.Backbone.activation == 'SiLU': 
            CONV_ACT = 'silu'
            C_ACT = 'silu'
        elif cfg.Model.Backbone.activation == 'ReLU': 
            CONV_ACT = 'relu'
            C_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
            C_ACT = 'relu_hswish'
        self.stage1 = Conv(3, self.channels_out['stage1'], 6, 2, 2, 1, CONV_ACT)

        # for latest yolov5, you can change BottleneckCSP to C3
        self.stage2_1 = Conv(self.channels_out['stage1'], self.channels_out['stage2_1'], 3, 2, None, 1, CONV_ACT)
        self.stage2_2 = C3(self.channels_out['stage2_1'], self.channels_out['stage2_2'], self.get_depth(3), True, 1, 0.5, C_ACT)
        self.stage3_1 = Conv(self.channels_out['stage2_2'], self.channels_out['stage3_1'], 3, 2, None, 1, CONV_ACT)
        self.stage3_2 = C3(self.channels_out['stage3_1'], self.channels_out['stage3_2'], self.get_depth(6), True, 1, 0.5, C_ACT)
        self.stage4_1 = Conv(self.channels_out['stage3_2'], self.channels_out['stage4_1'], 3, 2, None, 1, CONV_ACT)
        self.stage4_2 = C3(self.channels_out['stage4_1'], self.channels_out['stage4_2'], self.get_depth(9), True, 1, 0.5, C_ACT)
        self.stage5_1 = Conv(self.channels_out['stage4_2'], self.channels_out['stage5'], 3, 2, None, 1, CONV_ACT)
        self.stage5_2 = C3(self.channels_out['stage5'], self.channels_out['csp1'], self.get_depth(3), True, 1, 0.5, C_ACT)
        self.sppf = SPPF(self.channels_out['csp1'], self.channels_out['spp'], 5, CONV_ACT)
        # self.conv1 = Conv(self.channels_out['csp1'], self.channels_out['conv1'], 1, 1)
        self.out_shape = {'C3_size': self.channels_out['stage3_2'],
                          'C4_size': self.channels_out['stage4_2'],
                          'C5_size': self.channels_out['conv1']}
        # print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.channels_out['stage3_2'],
                                                                    # self.channels_out['stage4_2'],
                                                                    # self.channels_out['spp']))

    def forward(self, x):
        x1 = self.stage1(x) #0-P1/2
        x21 = self.stage2_1(x1) #1-P2/4
        x22 = self.stage2_2(x21)
        x31 = self.stage3_1(x22) #3-P3/8
        c3 = self.stage3_2(x31)
        x41 = self.stage4_1(c3) #5-P4/16
        c4 = self.stage4_2(x41)
        x51 = self.stage5_1(c4) #7-P5/32
        x5 = self.stage5_2(x51)

        sppf = self.sppf(x5)
        return c3, c4, sppf

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
