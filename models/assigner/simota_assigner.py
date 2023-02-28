from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_bbox_iou(box1, box2, box_format='xywh'):
    """Calculate iou.
    This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
    """
    if box_format == 'xyxy':
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
        area_2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)

    elif box_format == 'xywh':
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, 2:] / 2),
            (box2[:, :2] - box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2] + box1[:, None, 2:] / 2),
            (box2[:, :2] + box2[:, 2:] / 2),
        )

        area_1 = torch.prod(box1[:, 2:], 1)
        area_2 = torch.prod(box2[:, 2:], 1)
    valid = (lt < rb).type(lt.type()).prod(dim=2)
    inter = torch.prod(rb - lt, 2) * valid
    return inter / (area_1[:, None] + area_2 - inter)

class SimOTAAssigner(nn.Module):
    def __init__(self,
                 num_classes=80,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 center_radius=2.5,
                 iou_obj=False,
                 top_k=10):
        super(SimOTAAssigner, self).__init__()
        self.num_classes = num_classes
        self.iou_weight = iou_weight
        self.center_radius = center_radius
        self.cls_weight = cls_weight
        self.iou_obj = iou_obj
        self.top_k = top_k

    def get_l1_target(self, l1_target, gt, stride, xy_shifts, eps=1e-8):
        l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
        l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
        return l1_target

    def forward(self,
                outputs,
                targets,
                bbox_preds,
                cls_preds,
                obj_preds,
                expanded_strides,
                xy_shifts):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        cls_targets, reg_targets, l1_targets, obj_targets, fg_masks = [], [], [], [], []
        num_gts = 0
        num_fg = 0
        batch_size = bbox_preds.shape[0]
        total_num_anchors = bbox_preds.shape[1]

        num_targets_list = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx in range(batch_size):
            num_gt = int(num_targets_list[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = bbox_preds.new_zeros((0, self.num_classes))
                reg_target = bbox_preds.new_zeros((0, 4))
                l1_target = bbox_preds.new_zeros((0, 4))
                obj_target = bbox_preds.new_zeros((total_num_anchors, 1))
                fg_mask = bbox_preds.new_zeros(total_num_anchors).bool()
                # num_fg.append(0)
            else:
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]
                gt_classes = targets[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        cls_preds_per_image,
                        obj_preds_per_image,
                        expanded_strides,
                        xy_shifts
                    )

                except RuntimeError:
                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    print("------------CPU Mode for This Batch-------------")

                    _gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
                    _gt_classes = gt_classes.cpu().float()
                    _bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
                    _cls_preds_per_image = cls_preds_per_image.cpu().float()
                    _obj_preds_per_image = obj_preds_per_image.cpu().float()

                    _expanded_strides = expanded_strides.cpu().float()
                    _xy_shifts = xy_shifts.cpu()

                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        num_gt,
                        total_num_anchors,
                        _gt_bboxes_per_image,
                        _gt_classes,
                        _bboxes_preds_per_image,
                        _cls_preds_per_image,
                        _obj_preds_per_image,
                        _expanded_strides,
                        _xy_shifts,
                    )

                    gt_matched_classes = gt_matched_classes.cuda()
                    fg_mask = fg_mask.cuda()
                    pred_ious_this_matching = pred_ious_this_matching.cuda()
                    matched_gt_inds = matched_gt_inds.cuda()

                torch.cuda.empty_cache()
                num_fg += num_fg_img
                if num_fg_img > 0:
                    if self.iou_obj:
                        cls_target = F.one_hot(
                            gt_matched_classes.to(torch.int64), self.num_classes
                            ) * 1.0
                        iou_mask = fg_mask * 1.0
                        iou_mask[fg_mask] = pred_ious_this_matching.float()
                        obj_target = iou_mask.unsqueeze(-1)
                        # obj_target = pred_ious_this_matching.unsqueeze(-1)
                    else:
                        cls_target = F.one_hot(
                            gt_matched_classes.to(torch.int64), self.num_classes
                            ) * pred_ious_this_matching.unsqueeze(-1)
                        obj_target = fg_mask.unsqueeze(-1)
                    reg_target = gt_bboxes_per_image[matched_gt_inds]

                    l1_target = self.get_l1_target(
                        bbox_preds.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        xy_shifts=xy_shifts[0][fg_mask],
                    )
                else:
                    cls_target = bbox_preds.new_zeros((0, self.num_classes))
                    reg_target = bbox_preds.new_zeros((0, 4))
                    l1_target = bbox_preds.new_zeros((0, 4))
                    obj_target = bbox_preds.new_zeros((total_num_anchors, 1))
                    fg_mask = bbox_preds.new_zeros(total_num_anchors).bool()

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target * 1.0)
            l1_targets.append(l1_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        l1_targets = torch.cat(l1_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        # num_fg = max(num_fg, 1)
        return cls_targets, reg_targets, obj_targets, l1_targets, fg_masks, num_fg, num_gts

    @torch.no_grad()
    def get_assignments(
            self,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            cls_preds_per_image,
            obj_preds_per_image,
            expanded_strides,
            xy_shifts
        ):

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            xy_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            # cost
            pair_wise_ious = pairwise_bbox_iou(gt_bboxes_per_image, bboxes_preds_per_image, box_format='xywh')
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

            gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
            )

            cls_preds_ = (
                cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
                * obj_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
            )

            if torch.isnan(cls_preds_).any() or torch.isnan(pair_wise_ious_loss).any():
                cls_preds_ = torch.zeros(cls_preds_.shape).to(cls_preds_.device)

            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_, obj_preds_

        cost = (
            self.cls_weight * pair_wise_cls_loss
            + self.iou_weight * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        xy_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        xy_shifts_per_image = xy_shifts[0] * expanded_strides_per_image
        xy_centers_per_image = (
            (xy_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1, 1)
        )  # [n_anchor, 2] -> [n_gt, n_anchor, 2]

        gt_bboxes_per_image_lt = (
            (gt_bboxes_per_image[:, 0:2] - 0.5 * gt_bboxes_per_image[:, 2:4])
            .unsqueeze(1)
            .repeat(1, total_num_anchors, 1)
        )
        gt_bboxes_per_image_rb = (
            (gt_bboxes_per_image[:, 0:2] + 0.5 * gt_bboxes_per_image[:, 2:4])
            .unsqueeze(1)
            .repeat(1, total_num_anchors, 1)
        )  # [n_gt, 2] -> [n_gt, n_anchor, 2]

        b_lt = xy_centers_per_image - gt_bboxes_per_image_lt
        b_rb = gt_bboxes_per_image_rb - xy_centers_per_image
        bbox_deltas = torch.cat([b_lt, b_rb], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # in fixed center
        gt_bboxes_per_image_lt = (gt_bboxes_per_image[:, 0:2]).unsqueeze(1).repeat(
            1, total_num_anchors, 1
        ) - self.center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_rb = (gt_bboxes_per_image[:, 0:2]).unsqueeze(1).repeat(
            1, total_num_anchors, 1
        ) + self.center_radius * expanded_strides_per_image.unsqueeze(0)

        c_lt = xy_centers_per_image - gt_bboxes_per_image_lt
        c_rb = gt_bboxes_per_image_rb - xy_centers_per_image
        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.top_k, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        # except:
        #     print('topk_ious:', topk_ious)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds