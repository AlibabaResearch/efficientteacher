import torch
import torch.nn as nn
import math
from models.backbone.common import *
from utils.general import make_divisible
from module.nanodet_utils import generate_anchors, dist2bbox
import torch.nn.functional as F


class YoloV8Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, cfg):
        super().__init__()

        width_mul = cfg.Model.width_multiple
        # channels_list_backbone = cfg.Model.Backbone.out_channels
        channels_list_neck = cfg.Model.Neck.out_channels

        if isinstance(cfg.Model.anchors, (list, tuple)):
            num_anchors = len(cfg.Model.anchors)
        else:
            num_anchors = cfg.Model.anchors
        # num_repeats = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
        # channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_neck)]

        self.nc = cfg.Dataset.nc
        self.no = self.nc + 5  # number of outputs per anchor
        self.nl = cfg.Model.Neck.num_outs# number of detection layers
        # self.n_anchors = num_anchors
        self.num_keypoints = cfg.Dataset.np
        self.na = num_anchors
        # self.anchors = cfg.Model.anchors
        # self.register_buffer('anchors', torch.tensor(cfg.Model.anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        self.prune = False
        self.use_l1 = False
        self.export = False

         # Define criteria
        self.grids = [torch.zeros(1)] * len(cfg.Model.Head.in_channels)

        self.grid = [torch.zeros(1)] * self.nl 
        self.prior_prob = 1e-2
        self.inplace = cfg.Model.inplace
        # stride = [8, 16, 32]  # strides computed during build
        # self.stride = torch.tensor(stride)
        self.reg_max = cfg.Loss.reg_max
        self.use_dfl = cfg.Loss.use_dfl
        self.stride = torch.Tensor(cfg.Model.Head.strides)
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = cfg.Loss.grid_cell_offset
        self.grid_cell_size = cfg.Loss.grid_cell_size

        # Init decouple head
        # self.cls_convs = nn.ModuleList()
        # self.reg_convs = nn.ModuleList()
        # self.cls_preds = nn.ModuleList()
        # self.reg_preds = nn.ModuleList()
        # # self.obj_preds = nn.ModuleList()
        # self.stems = nn.ModuleList()
        # head_layers = tal_build_effidehead_layer(channels_list, num_anchors, self.nc, reg_max=self.reg_max)  # detection layer
        # assert head_layers is not None
        if cfg.Model.Head.activation == 'SiLU': 
            CONV_ACT = 'silu'
        elif cfg.Model.Head.activation == 'ReLU': 
            CONV_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
        ch = []
        for out_c in cfg.Model.Neck.out_channels:
            ch.append(int(out_c * cfg.Model.width_multiple))
        # Efficient decoupled head layers
        c2, c3 = max((16, ch[0] // 4, (self.reg_max + 1) * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3, 1, None, 1, act=CONV_ACT), Conv(c2, c2, 3, 1, None, 1, act=CONV_ACT), nn.Conv2d(c2, 4 * (self.reg_max + 1), 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3, 1, None, 1, act=CONV_ACT), Conv(c3, c3, 3, 1, None, 1, act=CONV_ACT), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        # for i in range(self.nl):
        #     idx = i*5
        #     # self.stems.append(head_layers[idx])
        #     self.cls_convs.append(head_layers[idx+1])
        #     self.reg_convs.append(head_layers[idx+2])
        #     self.cls_preds.append(head_layers[idx+3])
        #     self.reg_preds.append(head_layers[idx+4])
            # self.obj_preds.append(head_layers[idx+5])

    def initialize_biases(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.na, self.no, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.na * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
   
        return output, grid

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                # x[i] = self.stems[i](x[i])
                # cls_x = x[i]
                # reg_x = x[i]
                reg_output = self.cv2[i](x[i])
                cls_output = self.cv3[i](x[i])
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
            
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        elif self.export:
            # cls_score_list = []
            # reg_distri_list = []
            z = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                # x[i] = self.stems[i](x[i])
                reg_output = self.cv2[i](x[i])
                cls_output = self.cv3[i](x[i])
                # cls_x = x[i]
                # reg_x = x[i]
                # cls_feat = self.cls_convs[i](cls_x)
                # cls_output = self.cls_preds[i](cls_feat)
                # reg_feat = self.reg_convs[i](reg_x)
                # reg_output = self.reg_preds[i](reg_feat)
                ori_b, ori_c, ori_h, ori_w = reg_output.shape
                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))
                # reg_output = reg_output.permute(0, 3, 1, 2)
                reg_output = reg_output.reshape([-1, 4, ori_h, ori_w])

                cls_output = torch.sigmoid(cls_output)
                # cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                # reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                obj_output = torch.ones((b, 1, ori_h, ori_w), device=cls_output.device, dtype=cls_output.dtype)
                print(reg_output.shape, cls_output.shape)
                y = torch.cat([reg_output, obj_output, cls_output], 1)
                z.append(y)
            self.hw = [out.shape[-2:] for out in z]
            outputs = torch.cat( [out.flatten(start_dim=2) for out in z], dim=2).permute(0, 2, 1) 
            print('decode outputs', outputs.shape)
            z = self.decode_outputs(outputs, dtype=outputs.type())
            print(z)
            # cls_score_list = torch.cat(cls_score_list, axis=1)
            # reg_distri_list = torch.cat(reg_distri_list, axis=1)
            return z
        else:
            cls_score_list = []
            reg_distri_list = []
            cls_list = []
            reg_bbox_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True)

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                # x[i] = self.stems[i](x[i])
                # cls_x = x[i]
                # reg_x = x[i]
                # cls_feat = self.cls_convs[i](cls_x)
                # cls_output = self.cls_preds[i](cls_feat)
                # reg_feat = self.reg_convs[i](reg_x)
                # reg_output = self.reg_preds[i](reg_feat)
                # cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                # reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                reg_output = self.cv2[i](x[i])
                cls_output = self.cv3[i](x[i])
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                
                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))
                
                cls_output = torch.sigmoid(cls_output)
                cls_list.append(cls_output.reshape([b, self.nc, l]))
                reg_bbox_list.append(reg_output.reshape([b, 4, l]))
            
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)
            cls_list = torch.cat(cls_list, axis=-1).permute(0, 2, 1)
            reg_bbox_list = torch.cat(reg_bbox_list, axis=-1).permute(0, 2, 1)

            pred_bboxes = dist2bbox(reg_bbox_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            feature = (x, cls_score_list, reg_distri_list)
            return (torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_list
                ], axis=-1), feature)
    
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.stride):
            yv, xv = torch.meshgrid([torch.arange(hsize) + 0.5, torch.arange(wsize) + 0.5])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        # print('grids shape:', grids.shape)
        # print('outputs shape:', outputs.shape)
        lt, rb = torch.split(outputs[...,:4], 2, -1)
        x1y1 = grids - lt
        x2y2 = grids + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
        outputs[..., :4] = bbox * strides


        # outputs[..., :2] = (outputs[..., :2] + grids) * strides
        # outputs[..., 2:4] = (outputs[..., 2:4] + grids) * strides

        return outputs

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
