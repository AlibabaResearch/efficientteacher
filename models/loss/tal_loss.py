#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.module.nanodet_utils import generate_anchors
from models.module.nanodet_utils import dist2bbox, xywh2xyxy
# from loss.yolox_loss import IOUloss
from models.assigner.yolo_atss_assigner import YOLOATSSAssigner
from models.assigner.tal_assigner import TaskAlignedAssigner
from utils.torch_utils import is_parallel
from models.loss.gfocal_loss import VarifocalLoss, BboxLoss

class ComputeTalLoss:
    '''Loss computation func.'''
    def __init__(self,
                 model,
                 cfg):
        # fpn_strides=[8, 16, 32]
        # grid_cell_size=5.0
        # grid_cell_offset=0.5
        # num_classes=80,
        # ori_img_size=640
        # use_dfl=True
        # reg_max=16
        # iou_type='siou'
        # num_classes = cfg.Dataset.nc
        # loss_weight={ 'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        # device = next(model.parameters()).device  # get model device
        self.epoch = 0
        self.det = model.module.head if is_parallel(model) else model.head# Detect() module
        self.fpn_strides = cfg.Model.Head.strides
        self.grid_cell_size = cfg.Loss.grid_cell_size
        self.grid_cell_offset = cfg.Loss.grid_cell_offset
        self.num_classes = cfg.Dataset.nc
        self.ori_img_size = cfg.Dataset.img_size

        # warmup_epoch=4
        self.warmup_epoch = cfg.hyp.warmup_epochs
        self.warmup_assigner = YOLOATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(top_k=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = cfg.Loss.use_dfl
        self.use_gfl = cfg.Loss.use_gfl
        self.reg_max = cfg.Loss.reg_max
        self.iou_type = cfg.Loss.iou_type
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bce = nn.BCELoss(reduction='none')
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = {'class': cfg.Loss.qfl_loss_weight, 'iou': cfg.Loss.box_loss_weight, 'dfl':cfg.Loss.dfl_loss_weight} 

    def __call__(
        self,
        outputs,
        targets
        # step_num
    ):

        with torch.cuda.amp.autocast(enabled=False):
            feats, pred_scores, pred_distri = outputs
            pred_scores = pred_scores.float()
            pred_distri = pred_distri.float()
            anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)

            assert pred_scores.type() == pred_distri.type()
            gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)
            batch_size = pred_scores.shape[0]

        # print('preprocess:', targets)
        # targets
            targets, num_gts =self.preprocess(targets, batch_size, gt_bboxes_scale)
            gt_labels = targets[:, :, :1]
            gt_bboxes = targets[:, :, 1:] #xyxy
            mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

            # pboxes
            anchor_points_s = anchor_points / stride_tensor
            pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy

            # if self.epoch < self.warmup_epoch:
            #     target_labels, target_bboxes, target_scores, fg_mask = \
            #         self.warmup_assigner(
            #             anchors,
            #             n_anchors_list,
            #             gt_labels,
            #             gt_bboxes,
            #             mask_gt,
            #             pred_bboxes.detach() * stride_tensor)
            # else:
                # if self.use_gfl:
            target_labels, target_bboxes, target_scores, fg_mask = \
                        self.formal_assigner(
                            pred_scores.detach().sigmoid(),
                            pred_bboxes.detach() * stride_tensor,
                            anchor_points,
                            gt_labels,
                            gt_bboxes,
                            mask_gt)

            #Dynamic release GPU memory
            # if step_num % 10 == 0:
            torch.cuda.empty_cache()

            # rescale bbox
            target_bboxes /= stride_tensor

            # cls loss
            if self.use_gfl:
                target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
                one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
                loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
               
            else:
                loss_cls = (F.binary_cross_entropy_with_logits(pred_scores.float(), target_scores.float(), reduction='none')).sum()
            target_scores_sum = max(target_scores.sum(), 1)
            if target_scores_sum > 0:
                loss_cls /= target_scores_sum
                # loss_cls = self.bce(pred_scores, target_scores).sum()# BCE

            # bbox loss
            loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        loss = torch.zeros(1, device=feats[0].device) + self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        loss_dict = dict(loss_iou = self.loss_weight['iou']*loss_iou, loss_dfl=self.loss_weight['dfl'] * loss_dfl,\
             loss_cls=self.loss_weight['class'] * loss_cls, loss=loss, num_fg=torch.sum(fg_mask)/max(num_gts, 1))
        return loss, loss_dict
        # return loss, \
        #     torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
        #                  (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
        #                  (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        num_gts = 0
        for l in targets_list:
            num_gts += len(l)
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets, num_gts

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)