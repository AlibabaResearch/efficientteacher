"""YOLOX-specific modules

"""

import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.backbone.common import *
from models.loss.loss import *
import torch.nn.functional as F
from torch.cuda.amp import autocast
from utils.general import make_divisible

# import shapely
# import shapely.geometry
import math

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# try:
#     # if error in importing polygon_inter_union_cuda, polygon_b_inter_union_cuda, please cd to ./iou_cuda and run "python setup.py install"
#     from polygon_inter_union_cuda import polygon_inter_union_cuda, polygon_b_inter_union_cuda
#     polygon_inter_union_cuda_enable = True
#     polygon_b_inter_union_cuda_enable = True
# except Exception as e:
#     print(f'Warning: "polygon_inter_union_cuda" and "polygon_b_inter_union_cuda" are not installed.')
#     print(f'The Exception is: {e}.')
#     polygon_inter_union_cuda_enable = False
#     polygon_b_inter_union_cuda_enable = False

LOGGER = logging.getLogger(__name__)

# YoloX官方代码
class YoloXDetect(nn.Module):
    stride = [8, 16, 32]
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self,
                 cfg
                 ):
        super().__init__()
        if isinstance(cfg.Model.anchors, (list, tuple)):
            self.n_anchors = len(cfg.Model.anchors)
        else:
            self.n_anchors = cfg.Model.anchors

        self.vino_export = False
        self.export = False
        # self.prune = False

        self.num_classes = cfg.Dataset.nc
        self.num_keypoints = cfg.Dataset.np
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple
        self.no = 5 + self.num_classes

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        self.channels = {
            'conv1': cfg.Model.Neck.out_channels[0],
            'conv2': cfg.Model.Neck.out_channels[1],
            'conv3': cfg.Model.Neck.out_channels[2],
            'dec': cfg.Model.Head.feat_channels,
        }

        self.re_channels_out()
        self.conv1_channel = self.channels['conv1']
        self.conv2_channel = self.channels['conv2']
        self.conv3_channel = self.channels['conv3']
        self.dec_channel = self.channels['dec']


        # self.cls0 = Conv(256, 256, 3, 1)
        self.cls0 = nn.ModuleList()
        self.reg0 = nn.ModuleList()
        self.cls1 = nn.ModuleList()
        self.reg1 = nn.ModuleList()
        self.cls2 = nn.ModuleList()
        self.reg2 = nn.ModuleList()

        if cfg.Model.Head.activation == 'SiLU': 
            CONV_ACT = 'silu'
        elif cfg.Model.Head.activation == 'ReLU': 
            CONV_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'

        num_decouple = cfg.Model.Head.num_decouple
        if num_decouple > 0: 
            self.num_decouple = self.get_depth(num_decouple)
        elif num_decouple == 0:
            self.num_decouple = 0
        # self.num_decouple = 2
        # print('num_decouple:', num_decouple)
        for i in range(self.num_decouple):
            self.cls0.append(Conv(self.dec_channel, self.dec_channel, 3, 1, act=CONV_ACT))
            self.reg0.append(Conv(self.dec_channel, self.dec_channel, 3, 1, act=CONV_ACT))
            self.cls1.append(Conv(self.dec_channel, self.dec_channel, 3, 1, act=CONV_ACT))
            self.reg1.append(Conv(self.dec_channel, self.dec_channel, 3, 1, act=CONV_ACT))
            self.cls2.append(Conv(self.dec_channel, self.dec_channel, 3, 1, act=CONV_ACT))
            self.reg2.append(Conv(self.dec_channel, self.dec_channel, 3, 1, act=CONV_ACT))

        # self.only_bbox = False 
        if self.num_decouple == 0:
            in_channels = [self.conv1_channel, self.conv1_channel, self.conv2_channel, self.conv2_channel, self.conv3_channel, self.conv3_channel]
        else:
            self.conv1 = Conv(self.conv1_channel, self.dec_channel, 1, 1, None, 1, CONV_ACT)
            self.conv2 = Conv(self.conv2_channel, self.dec_channel, 1, 1, None, 1, CONV_ACT)
            self.conv3 = Conv(self.conv3_channel, self.dec_channel, 1, 1, None, 1, CONV_ACT)
            in_channels = [self.dec_channel, self.dec_channel, self.dec_channel, self.dec_channel, self.dec_channel, self.dec_channel]
        cls_in_channels = in_channels[0::2]
        reg_in_channels = in_channels[1::2]
        for cls_in_channel, reg_in_channel in zip(cls_in_channels, reg_in_channels):
            cls_pred = nn.Conv2d(
                in_channels=cls_in_channel,
                out_channels=self.n_anchors * self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            reg_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            obj_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=self.n_anchors * 1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.cls_preds.append(cls_pred)
            self.reg_preds.append(reg_pred)
            self.obj_preds.append(obj_pred)

        self.nc = self.num_classes  # number of classes
        # self.no = self.num_classes + 5  # number of outputs per anchor
        self.nl = len(cls_in_channels)  # number of detection layers
        self.na = self.n_anchors  # number of anchors

        self.use_l1 = False
        self.grids = [torch.zeros(1)] * len(in_channels)  # 用于保存每层的每个网格的坐标
        self.prior_prob = cfg.Model.prior_prob
        self.inplace = cfg.Model.inplace  # use in-place ops (e.g. slice assignment)
        self.grid = [torch.zeros(1)] * self.nl


    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def initialize_biases(self):
        prior_prob = self.prior_prob

        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def post_process(self, outputs):
        if self.training:
            h, w = outputs[0][0].shape[2:4]
            h *= self.stride[0]
            w *= self.stride[0]
            x_shifts = []
            y_shifts = []
            expanded_strides = []
            newouts = []
            for i, stride_this_level in enumerate(self.stride):
                reg_output, obj_output, cls_output = outputs[i]
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, i, stride_this_level, reg_output.type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                        .fill_(stride_this_level)
                        .type_as(reg_output)
                )
                newouts.append(output)
            outputs = torch.cat(newouts, 1)
            x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
            y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
            expanded_strides = torch.cat(expanded_strides, 1)
            whwh = torch.Tensor([[w, h, w, h]]).type_as(outputs)
            return (outputs,None,x_shifts,y_shifts,expanded_strides,whwh)
        else:
            newouts = []
            for i, stride_this_level in enumerate(self.stride):
                reg_output, obj_output, cls_output = outputs[i]
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                newouts.append(output)
            outputs = newouts
            self.hw = [out.shape[-2:] for out in outputs]
            outputs = torch.cat(
                [out.flatten(start_dim=2) for out in outputs], dim=2
            ).permute(0, 2, 1)
            outputs = self.decode_outputs(outputs, dtype=outputs.type())
            return (outputs,)

    def forward_mobile(self, x):
         # outputs = []
        z = []
        d = x[0].device
        feature = []

        P3, P4, P5 = x 

        if self.num_decouple == 0:
            cls_xs_8 = P3
            reg_xs_8 = P3
            cls_xs_16 = P4
            reg_xs_16 = P4
            cls_xs_32 = P5
            reg_xs_32 = P5
        else:

            f_8 = self.conv1(P3)  # 1/8
            f_16 = self.conv2(P4)  # 1/16
            f_32 = self.conv3(P5)  # 1/32

            cls_xs_8 = f_8
            reg_xs_8 = f_8

            cls_xs_16 = f_16
            reg_xs_16 = f_16

            cls_xs_32 = f_32
            reg_xs_32 = f_32
            for i in range(self.num_decouple):
                cls_xs_8 = self.cls0[i](cls_xs_8)
                reg_xs_8 = self.reg0[i](reg_xs_8)

                cls_xs_16 = self.cls1[i](cls_xs_16)
                reg_xs_16 = self.reg1[i](reg_xs_16)

                cls_xs_32 = self.cls2[i](cls_xs_32)
                reg_xs_32 = self.reg2[i](reg_xs_32)

        cls_xs = [cls_xs_8, cls_xs_16, cls_xs_32]
        reg_xs = [reg_xs_8, reg_xs_16, reg_xs_32]
        for i, (stride_this_level, cls_x, reg_x) in enumerate(zip(self.stride, cls_xs, reg_xs)):

            cls_output = self.cls_preds[i](cls_x)  # [batch_size, num_classes, hsize, wsize]
            reg_output = self.reg_preds[i](reg_x)  # [batch_size, 4, hsize, wsize]
            obj_output = self.obj_preds[i](reg_x)  # [batch_size, 1, hsize, wsize]
            # z.append([reg_output, obj_output, cls_output])
            y = torch.cat([reg_output, obj_output, cls_output], 1)
            # out_conv = nn.Conv2d(self.no, self.no, kernel_size=1, stride=1, padding=0, groups=self.no, bias=False)
            # W_adj = nn.Parameter(torch.ones(self.no,self.no,1,1))
            # out_conv.state_dict()['weight'] = W_adj
            y = self.out_conv(y)
            z.append(y)
        return z

    def build_input(self, x):
        P3, P4, P5 = x 

        if self.num_decouple == 0:
            cls_xs_8 = P3
            reg_xs_8 = P3
            cls_xs_16 = P4
            reg_xs_16 = P4
            cls_xs_32 = P5
            reg_xs_32 = P5
        else:

            f_8 = self.conv1(P3)  # 1/8
            f_16 = self.conv2(P4)  # 1/16
            f_32 = self.conv3(P5)  # 1/32

            cls_xs_8 = f_8
            reg_xs_8 = f_8

            cls_xs_16 = f_16
            reg_xs_16 = f_16

            cls_xs_32 = f_32
            reg_xs_32 = f_32
            for i in range(self.num_decouple):
                cls_xs_8 = self.cls0[i](cls_xs_8)
                reg_xs_8 = self.reg0[i](reg_xs_8)

                cls_xs_16 = self.cls1[i](cls_xs_16)
                reg_xs_16 = self.reg1[i](reg_xs_16)

                cls_xs_32 = self.cls2[i](cls_xs_32)
                reg_xs_32 = self.reg2[i](reg_xs_32)

        cls_xs = [cls_xs_8, cls_xs_16, cls_xs_32]
        reg_xs = [reg_xs_8, reg_xs_16, reg_xs_32]
        return cls_xs, reg_xs

    def forward(self, x):
        # outputs = []
        z = []
        d = x[0].device
        feature = []
        cls_xs, reg_xs = self.build_input(x)
        for i, (stride_this_level, cls_x, reg_x) in enumerate(zip(self.stride, cls_xs, reg_xs)):

            cls_output = self.cls_preds[i](cls_x)  # [batch_size, num_classes, hsize, wsize]
            reg_output = self.reg_preds[i](reg_x)  # [batch_size, 4, hsize, wsize]
            obj_output = self.obj_preds[i](reg_x)  # [batch_size, 1, hsize, wsize]

            if self.training:
                # in_type = cls_xs[0].type()
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                bs, _, ny, nx = x[i].shape
                z.append(output.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())
            elif self.export:
                z.append([reg_output, obj_output, cls_output])
                # z.append(torch.cat([reg_output, obj_output, cls_output], 1))
            else:
                y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                bs, _, ny, nx = y.shape
                y = y.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                feature.append(output.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())
                if self.grid[i].shape[2:4] != y.shape[2:4]:
                    # d = self.stride.device
                    yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
                    self.grid[i] = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float().to(d)
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i] # wh
                else:
                    xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                    wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
            # outputs.append(output)
        # return z if self.training else torch.cat(z, 1)
        if self.export:
            return z
        return z if self.training else (torch.cat(z, 1), feature)

    # def forward(self, x):
    #     outputs = self._forward(x)

    #     if self.training or self.export:
    #         return outputs
    #     else:
    #         # for o in outputs:
    #         #     print(o.shape,o.sum())
    #         # print("*******")
    #         self.hw = [out.shape[-2:] for out in outputs]
    #         # print('self.hw:', self.hw)
    #         # [batch, n_anchors_all, 85]
    #         outputs = torch.cat(
    #             [out.flatten(start_dim=2) for out in outputs], dim=2
    #         ).permute(0, 2, 1)
    #         outputs = self.decode_outputs(outputs, dtype=x[0].type())
    #         return (outputs,)
    #         # return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # output[..., -5:-3] = (output[..., -5:-3] + grid) * stride
        # output[..., -3:-1] = torch.exp(output[..., -3:-1]) * stride
        # for i in [i for i in range(2, 8 + 1, 2)][::-1]:
        #         if i != 2:
        #             output[..., -1 * i - 1:-1 * (i - 2) - 1] = output[..., -1 * i - 1:-1 * (i - 2) - 1] * stride + output[..., :2]
        #         else:
        #             output[..., -3:-1] = output[..., -3:-1] * stride + output[..., :2]
        return output, grid



    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.stride):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

        return outputs
