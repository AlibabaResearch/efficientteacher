import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

# from models.backbone.common import SPPF
# from models.backbone.experimental import *
# from models.head.yolox_head import DetectYoloX
# from models.head.yolox_pss_head import DetectYoloXPss
# from utils.autoanchor import check_anchor_order
# from utils.general import check_yaml, make_divisible, print_args, set_logging
# from utils.plots import feature_visualization
# from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    # select_device, time_sync
# from models.loss.loss import *
# from models.head.yolov5_head import Detect
import warnings
import logging
import torch.nn as nn
import torch
import math

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def get_activation(act=True):
    act_name = None
    if isinstance(act, str):
        if act == "silu":
            m = nn.SiLU()
        elif act == "relu":
            m = nn.ReLU(inplace=True)
            act_name = 'relu'
        elif act == "hard_swish":
            #m = nn.ReLU(inplace=True)
            m = nn.Hardswish(inplace=True)
            act_name = 'hard_swish'
        elif act == "lrelu":
            m = nn.LeakyReLU(0.1, inplace=True)
            act_name = 'leaky_relu'
        else:
            raise AttributeError("Unsupported act type: {}".format(act))
    else:
        m = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    return m, act_name

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act, self.act_name = get_activation(act=act)
        self.init_weights()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def init_weights(self):
        if self.act_name == "relu":
            # nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity="relu")
            pass

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion, act
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'silu':
            act = 'silu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        elif act == 'relu_hswish':
            act = 'relu'
            last_act = 'hard_swish'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=last_act)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut, act=act) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, act=True):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        if act == 'relu_silu':
            act = 'relu'
            last_act = 'silu'
        elif act == 'relu_lrelu':
            act = 'relu'
            last_act = 'lrelu'
        elif act == 'relu_hswish':
            act = 'relu'
            last_act = 'hard_swish'
        else:
            last_act = act
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=last_act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class YoloV5BackBone(nn.Module):
    def __init__(self, cfg):
        super(YoloV5BackBone, self).__init__()
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
        # print('x:', x)
        # print('x1:', x1)
        # import pdb
        # pdb.set_trace()
       
      
        return c3, c4, sppf

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
