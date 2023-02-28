"""
Loss functions
"""

from cmath import isnan
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
import math
from torch.nn import functional as F
# from models.loss.iou_loss import IOUloss
from models.loss.loss import IOUloss
# from torch.cuda.amp import autocast
import numpy as np
from assigner import SimOTAAssigner
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class ComputeFastXLoss:
    # Compute losses
    def __init__(self, model, cfg):
        # super(ComputeFastXLoss, self).__init__()
        # device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        det = model.module.head if is_parallel(model) else model.head# Detect() module
        self.det = det
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_type = cfg.Loss.iou_type
        # self.iou_loss = IOUloss(iou_type=self.iou_type, reduction="none")
        self.iou_loss = IOUloss(iou_type=self.iou_type, reduction="none")
        self.num_classes = cfg.Dataset.nc
        self.strides = torch.tensor(cfg.Model.Head.strides)
        self.reg_weight = cfg.Loss.box_loss_weight
        self.obj_weight = cfg.Loss.obj_loss_weight
        self.cls_weight = cfg.Loss.cls_loss_weight
        self.n_anchors = len(cfg.Model.anchors)
        self.grids = [torch.zeros(1)] * len(cfg.Model.Head.in_channels)
        self.iou_obj = cfg.Loss.iou_obj
        self.formal_assigner = SimOTAAssigner(num_classes=self.num_classes, iou_weight=3.0,cls_weight=1.0, center_radius=2.5, iou_obj=self.iou_obj)
    
    def __call__(
        self,
        outputs,
        targets
    ):
        dtype = outputs[0].type()
        device = targets.device
        # print('targets type:', targets.type())
        loss_cls, loss_obj, loss_iou, loss_l1 = torch.zeros(1, device=device), torch.zeros(1, device=device), \
            torch.zeros(1, device=device), torch.zeros(1, device=device)

        outputs, outputs_origin, gt_bboxes_scale, xy_shifts, expanded_strides = self.get_outputs_and_grids(
            outputs, self.strides, dtype, device)

        with torch.cuda.amp.autocast(enabled=False):
            bbox_preds = outputs[:, :, :4].float()  # [batch, n_anchors_all, 4]
            bbox_preds_org = outputs_origin[:, :, :4].float()  # [batch, n_anchors_all, 4]
            obj_preds = outputs[:, :, 4].float().unsqueeze(-1)  # [batch, n_anchors_all, 1]
            cls_preds = outputs[:, :, 5:].float()  # [batch, n_anchors_all, n_cls]

        # targets
            batch_size = bbox_preds.shape[0]
            targets = self.preprocess(targets, batch_size, gt_bboxes_scale)

        # 添加assigner的实现
            cls_targets, reg_targets, obj_targets, l1_targets, fg_masks, num_fg, num_gts = self.formal_assigner(outputs.detach(),
                targets,
                bbox_preds.detach(),
                cls_preds.detach(),
                obj_preds.detach(),
                expanded_strides,
                xy_shifts)

        # loss
        # loss_iou += bbox_preds[i].mean() * 0
        # loss_cls += cls_preds.mean() * 0
        # loss_obj += obj_preds.mean() * 0
        # loss_l1 += bbox_preds_org.mean() * 0
            # for i, n in enumerate(num_fg): # batch_size, n_fg
                # fg_mask = fg_masks[i]
                # if n:
                #     loss_iou += (self.iou_loss(bbox_preds[i].view(-1, 4)[fg_mask], reg_targets[i])).mean()
                #     loss_l1 += (self.l1_loss(bbox_preds_org[i].view(-1, 4)[fg_mask], l1_targets[i])).mean()
                #     # loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets*1.0)).mean() * batch_size
                #     loss_cls += (self.bcewithlog_loss(cls_preds[i].view(-1, self.num_classes)[fg_mask], cls_targets[i])).mean()
                #     loss_obj += (self.bcewithlog_loss(obj_preds[i].view(-1, 1), obj_targets[i])).mean()
                # else:
                #     loss_iou += bbox_preds_org[i].mean() * 0
                #     loss_l1 += bbox_preds_org[i].mean() * 0
                #     loss_cls += cls_preds[i].mean() * 0
                #     loss_obj += obj_preds[i].mean() * 0
                # print('bbox preds org:', bbox_preds_org[i].mean())
                # print('bbox preds org:', bbox_preds[i].mean())
                # if torch.isnan(loss_iou).any():
                #     print('bbox_preds:', bbox_preds[i].type())
                #     print('bbox_preds:', bbox_preds[i])
                #     print('reg:', reg_targets)
                # print(num_fg)
            loss_iou += (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()/num_fg
            loss_l1 += (self.l1_loss(bbox_preds_org.view(-1, 4)[fg_masks], l1_targets)).sum()/num_fg
        # loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets*1.0)).mean() * batch_size
            loss_cls += (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()/num_fg
            loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()/num_fg
        # # if torch.isnan(loss_l1).any():
        # #     print('l1:', l1_targets)
        # #     print(num_fg)
        # if torch.isnan(loss_obj).any():
        #     print('obj:', obj_targets)
        #     print(num_fg)
        # if torch.isnan(loss_cls).any():
        #     print('cls:', cls_targets)
        #     print(num_fg)

        total_losses = self.reg_weight * loss_iou + loss_l1 + self.obj_weight * loss_obj + self.cls_weight*loss_cls
        # total_losses = total_losses * batch_size
        loss_dict = dict(loss_iou = self.reg_weight*loss_iou, loss_obj=self.obj_weight * loss_obj, loss_cls=self.cls_weight*loss_cls, loss=float(total_losses))
        return total_losses, loss_dict

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = batch_target

        return targets

    def decode_output(self, output, k, stride, dtype, device):
        grid = self.grids[k].to(device)
        batch_size = output.shape[0]
        hsize, wsize = output.shape[2:4]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(device)
            self.grids[k] = grid

        output = output.reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        output_origin = output.clone()
        grid = grid.view(1, -1, 2)

        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, output_origin, grid, hsize, wsize

    def get_outputs_and_grids(self, outputs, strides, dtype, device):
        xy_shifts = []
        expanded_strides = []
        outputs_new = []
        outputs_origin = []

        for k, output in enumerate(outputs):
            output, output_origin, grid, feat_h, feat_w = self.decode_output(
                output, k, strides[k], dtype, device)

            xy_shift = grid
            expanded_stride = torch.full((1, grid.shape[1], 1), strides[k], dtype=grid.dtype, device=grid.device)

            xy_shifts.append(xy_shift)
            expanded_strides.append(expanded_stride)
            outputs_new.append(output)
            outputs_origin.append(output_origin)

        xy_shifts = torch.cat(xy_shifts, 1)  # [1, n_anchors_all, 2]
        expanded_strides = torch.cat(expanded_strides, 1) # [1, n_anchors_all, 1]
        outputs_origin = torch.cat(outputs_origin, 1)
        outputs = torch.cat(outputs_new, 1)

        feat_h *= strides[-1]
        feat_w *= strides[-1]
        gt_bboxes_scale = torch.Tensor([[feat_w, feat_h, feat_w, feat_h]]).type_as(outputs)

        return outputs, outputs_origin, gt_bboxes_scale, xy_shifts, expanded_strides