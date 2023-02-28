#Copyright (c) 2023, Alibaba Group
# EfficientTeacher by Alibaba Cloud 
"""
ssod utils
"""

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import random
from utils.general import clip_coords, xyxy2xywh, xywh2xyxy, xywhn2xyxy, non_max_suppression, box_iou
from utils.general import non_max_suppression_ssod
from utils.plots import plot_images_ssod, plot_images, plot_labels,  output_to_target_ssod
from utils.torch_utils import time_sync
import copy

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=dim)
    return output

class FairPseudoLabel:
    def __init__(self, cfg):
        self.nms_conf_thres = cfg.SSOD.nms_conf_thres
        self.nms_iou_thres = cfg.SSOD.nms_iou_thres
        self.debug = cfg.SSOD.debug
        self.multi_label = cfg.SSOD.multi_label
        self.names = cfg.Dataset.names
        self.num_points = cfg.Dataset.np

    def online_label_transform_with_image(self, img, targets, M, s, ud, lr, segments=(), border=(0, 0), perspective=0.0):
        if isinstance(img, torch.Tensor):
                img = img.cpu().float().numpy()
                img = img.transpose(1, 2, 0) * 255.0
                img = img.astype(np.uint8)
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        if ud == 1:
            img = np.flipud(img)
        if lr == 1:
            img = np.fliplr(img)
        img = torch.from_numpy(img.transpose(2, 0, 1)/255.0)
      

        # Transform label coordinates
        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))
            if use_segments:  # warp segments
                segments = resample_segments(segments)  # upsample
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                    # clip
                    new[i] = segment2box(xy, width, height)

            else:  # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets


    # def create_pseudo_label_online(self, out, target_imgs, M_s, target_imgs_ori, gt=None):
    #     n_img, _, height, width = target_imgs.shape  # batch size, channels, height, width
    #     lb = []
    #     target_out_targets_perspective = []
    #     invalid_target_shape = True

    #     out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres, multi_label=self.multi_label, labels=lb)
    #     out = [out_tensor.detach() for out_tensor in out]
    #     target_out_np = output_to_target_ssod(out)
    #     target_out_targets = torch.tensor(target_out_np)
    #     target_shape = target_out_targets.shape
    #     total_t1 = time_sync()
    #     # print('M:', M)
    #     if(target_shape[0] > 0 and target_shape[1] > 6):
    #         for i, img in enumerate(target_imgs):
             
    #             image_targets = target_out_targets[target_out_targets[:, 0] == i]
    #             if isinstance(image_targets, torch.Tensor):
    #                 image_targets = image_targets.cpu().numpy()
    #             image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
    #             M_select = M_s[M_s[:, 0] == i, :]  # image targets
    #             M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
    #             s = float(M_select[0][10])
    #             ud = int(M_select[0][11])
    #             lr = int(M_select[0][12])
    #             img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
               
    #             if 1:
    #                 image_targets = np.array(image_targets_random)
    #             else:
    #                 image_targets = np.array(image_targets[:, 1:])
    #             if image_targets.shape[0] != 0:
    #                 image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
    #                 image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])  # convert xyxy to xywh
    #                 image_targets[:, [3, 5]] /= height # normalized height 0-1
    #                 image_targets[:, [2, 4]] /= width # normalized width 0-1
    #                 if ud == 1:
    #                     image_targets[:, 3] = 1 - image_targets[:, 3]
    #                 if lr == 1:
    #                     image_targets[:, 2] = 1 - image_targets[:, 2]
    #                 target_out_targets_perspective.extend(image_targets.tolist())
    #         target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))
    #     # if self.RANK in [-1, 0]:
    #         # print('total time cost:', time_sync() - total_t1)

    #     if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
    #         invalid_target_shape = False
    #     return target_out_targets_perspective, target_imgs, invalid_target_shape
    
    def create_pseudo_label_on_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        # n_img, _, height, width = target_imgs.shape  # batch size, channels, height, width
        # target_out_targets = torch.tensor(target_out_np)
        # total_t1 = time_sync()
        invalid_target_shape = True
        target_out_targets_perspective = gt
        target_shape = target_out_targets_perspective.shape
        # print('M:', M)
        # img_list = []

        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
            invalid_target_shape = False
            if self.debug:
                if RANK in [-1 ,0]:
                    draw_image = plot_images(copy.deepcopy(target_imgs), target_out_targets_perspective, None, '/mnt/bowen/EfficientTeacher/unbias_teacher_pseudo_label.jpg')            
              
                # if 0:
                    # draw_image = plot_images(copy.deepcopy(target_imgs), target_out_targets_perspective, None, self.save_dir/'unbias_teacher_debug.jpg')            
                # else:
                # draw_image = plot_images(copy.deepcopy(target_imgs_ori), target_out_targets, None, '/mnt/bowen/EfficientTeacher/unbias_teacher_pseudo_label_ori.jpg')            
                # draw_image = plot_images(copy.deepcopy(target_imgs), gt, None, '/mnt/bowen/EfficientTeacher/unbias_teacher_gt.jpg')            
        return target_out_targets_perspective, target_imgs, invalid_target_shape


    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        n_img, _, height, width = target_imgs.shape  # batch size, channels, height, width
        lb = []
      
        target_out_targets_perspective = []
        invalid_target_shape = True
        out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres, \
                                       num_points=self.num_points, multi_label=self.multi_label, labels=lb)
        out = [out_tensor.detach() for out_tensor in out]
        target_out_np = output_to_target_ssod(out)
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape

        if(target_shape[0] > 0 and target_shape[1] > 6):
            for i, img in enumerate(target_imgs_ori):
                # image_targets = target_out_targets[target_out_targets[:, 0] == i]
                image_targets = target_out_np[target_out_np[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                M_select = M_s[M_s[:, 0] == i, :]  # image targets
                M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
               
                image_targets = np.array(image_targets_random)
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])  # convert xyxy to xywh
                    image_targets[:, [3, 5]] /= height # normalized height 0-1
                    image_targets[:, [2, 4]] /= width # normalized width 0-1
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))
        # img_list = torch.stack(img_list, 0)
        # if self.RANK in [-1, 0]:
            # print('total time cost:', time_sync() - total_t1)

        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
           invalid_target_shape = False
           if self.debug:
              if RANK in [-1 ,0]:
                draw_image = plot_images_ssod(copy.deepcopy(target_imgs), target_out_targets_perspective, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label.jpg', names=self.names)            
                draw_image = plot_images(copy.deepcopy(target_imgs), gt, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_gt.jpg', names=self.names)            
                    # raise 0

        return target_out_targets_perspective, invalid_target_shape


 #build pseudo label via pred from teacher model
    def create_pseudo_label_online_with_extra_teachers(self, out, extra_teacher_outs, target_imgs, M_s, \
     extra_teacher_class_idxs, RANK):
        n_img, _, height, width = target_imgs.shape  # batch size, channels, height, width
        lb = []
        target_out_targets_perspective = []
        invalid_target_shape = True

        # 提高阈值 再次筛选
        out = non_max_suppression(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres, labels=lb, multi_label=self.multi_label)
        out = [out_tensor.detach() for out_tensor in out]
        for teacher_idx, teacher_out in enumerate(extra_teacher_outs): #each teacher
            teacher_pseudo_out = non_max_suppression(teacher_out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres, labels=lb, multi_label=self.multi_label)
            for i, o in enumerate(out): #batch i
                pseudo_out_one_img = teacher_pseudo_out[i]
                if pseudo_out_one_img.shape[0] > 0 : #this img has bbox
                    for each in pseudo_out_one_img:
                        origin_class_idx = int(each[5].cpu().item() )
                        #print(teacher_idx, self.extra_teacher_class_idxs[teacher_idx] )
                        if origin_class_idx in extra_teacher_class_idxs[teacher_idx]:
                            each[5] = float(extra_teacher_class_idxs[teacher_idx][origin_class_idx]) #exchange class idx

                x = torch.cat([o, pseudo_out_one_img])
                #针对所有类同时进行nms，避免出现标注框中框现象
                c = x[:, 5:6] * 0
                boxes, scores = x[:, :4] + c, x[:, 4] 
                index = torchvision.ops.nms(boxes, scores, self.nms_iou_thres)  # NMS
                out[i] = x[index]

        target_out_np = output_to_target_ssod(out)
        target_out_targets = torch.tensor(target_out_np)
        target_shape = target_out_targets.shape
        if(target_shape[0] > 0 and target_shape[1] > 6):
            for i, img in enumerate(target_imgs):
                image_targets = target_out_targets[target_out_targets[:, 0] == i]
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()
                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])
                M_select = M_s[M_s[:, 0] == i, :]  # image targets
                M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
                s = float(M_select[0][10])
                ud = int(M_select[0][11])
                lr = int(M_select[0][12])
                # img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M[i][0], M[i][1])
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
                
                image_targets = np.array(image_targets_random)
                # target_imgs[i] = torch.from_numpy(img.transpose(2, 0, 1)/255.0).to(self.device)
                if image_targets.shape[0] != 0:
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])  # convert xyxy to xywh
                    image_targets[:, [3, 5]] /= height # normalized height 0-1
                    image_targets[:, [2, 4]] /= width # normalized width 0-1
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))
        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0:
            invalid_target_shape = False
            if self.debug:
               if RANK in [-1 ,0]:
                  draw_image = plot_images_ssod(copy.deepcopy(target_imgs), target_out_targets_perspective, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label.jpg', names=self.names)            
            # draw_image = plot_images(copy.deepcopy(target_imgs), target_out_targets_perspective, None, self.save_dir/'unbias_teacher_debug.jpg') 
        return target_out_targets_perspective, target_imgs, invalid_target_shape

## helper functions
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def online_random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    # targets = [cls, xyxy, score]
    # print(img.shape)

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets

def online_label_transform(img, targets, M, s, segments=(), border=(0, 0), perspective=0.0):
        height = img.shape[1] + border[0] * 2  # shape(h,w,c)
        width = img.shape[2] + border[1] * 2

        # Transform label coordinates
        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            new = np.zeros((n, 4))
            if use_segments:  # warp segments
                segments = resample_segments(segments)  # upsample
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                    # clip
                    new[i] = segment2box(xy, width, height)

            else:  # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets

def select_targets(targets, ignore_thres_low, ignore_thres_high):
        device = targets.device
        certain_targets = []
        uncertain_targets = []
        for t in targets:
            #伪标签得分大于相应类别的阈值,标记为正样本
            if t[6] >= ignore_thres_high[int(t[1])]:
                certain_targets.append(np.array(t.cpu()))
            # 伪标签在0.1到阈值去之前的，标记为忽略样本
            elif t[6] >= ignore_thres_low[int(t[1])]:
                uncertain_targets.append(np.array(t.cpu()))

        certain_targets = np.array(certain_targets).astype(np.float32)
        certain_targets= torch.from_numpy(certain_targets).contiguous()
        certain_targets = certain_targets.to(device)
        if certain_targets.shape[0] == 0:
            certain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_targets = np.array(uncertain_targets).astype(np.float32)
        uncertain_targets= torch.from_numpy(uncertain_targets).contiguous()
        uncertain_targets= uncertain_targets.to(device)
        if uncertain_targets.shape[0] == 0:
            uncertain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)
        return certain_targets, uncertain_targets

def check_pseudo_label_with_gt(detections, labels, iouv=torch.tensor([0.5]), ignore_thres_low=None, ignore_thres_high=None, batch_size=1):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    tp_num = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    fp_cls_num = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    fp_loc_num = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    gt = labels[:, 2:6]
    # print('labels shape:', labels.shape)
    if ignore_thres_low is not None:
        # print('thres_high:', ignore_thres_high)
        # print('thres_low:', ignore_thres_low)
        detections, uc_pseudo = select_targets(detections, ignore_thres_low, ignore_thres_high)
        # pseudo = detections[detections[:, 6] > ignore_thres]
        # print('pseudo:', detections)
        # print('uc_pseudo:', uc_pseudo)
        # detections = torch.cat((detections, uc_pseudo))
        detections = uc_pseudo
        pseudo = detections[:, 2:6]
        pseudo_label_num = pseudo.shape[0] / batch_size
        # pseudo_label_num = detections.shape[0]
        gt_label_num = gt.shape[0]/batch_size
    else:
        pseudo = detections[:, 2:6]
        pseudo_label_num = detections.shape[0]
        pseudo_label_num /= batch_size
        gt_label_num = gt.shape[0]/batch_size

    gt *= torch.tensor([640, 640] * 2)
    pseudo *= torch.tensor([640, 640] * 2)
    gt = xywh2xyxy(gt) + labels[:, 0:1] * torch.tensor([640, 640] * 2)    
    pseudo = xywh2xyxy(pseudo) + detections[:, 0][:, None] * torch.tensor([640, 640] * 2)
    # print(gt)
    # print(pseudo)
    # print(labels[:, 0:1][:,None].shape)
    iou = box_iou(gt, pseudo)  
    # print('iou:', iou)
    # print('iou shape', iou.shape)
    correct_class = labels[:, 1:2] == detections[:, 1]
    correct_image = labels[:, 0:1] == detections[:, 0]
    # print('correct_class:', correct_class)

    for i in range(len(iouv)):
        # print('correct_image:', correct_image)
        # print('correct_class:', correct_class)
        # print('iou:', (iou < iouv[i]) & (iou > torch.tensor(0.1)))
        tp_x = torch.where((iou >= iouv[i]) & correct_class & correct_image)  # IoU > threshold and classes match
        fp_cls_x = torch.where((iou >= iouv[i]) & ~correct_class & correct_image)  # IoU > threshold and classes match
        # fp_loc_x = torch.where((iou < iouv[i]) & (iou > torch.tensor(0.1)) & correct_class & correct_image)  # IoU > threshold and classes match
        fp_loc_x = torch.where((iou < iouv[i]) & (iou > torch.tensor(0.01)) & correct_image)  # IoU > threshold and classes match
        # print('fp_loc_x:', fp_loc_x.shape)
        # print('fp_cls_x:', fp_cls_x.shape)
        # print(iou.shape, correct_image.shape, correct_class.shape)
        # fp_both_x = torch.where((iou < iouv[i]) & correct_image & ~correct_class)  # IoU > threshold and classes match
        # print('x:', x)
        # print(torch.where(iou >= iouv[i]))
        if tp_x[0].shape[0]:
            matches = torch.cat((torch.stack(tp_x, 1), iou[tp_x[0], tp_x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if tp_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp_num[matches[:, 1].astype(int), i] = True
        if fp_cls_x[0].shape[0]:
            matches = torch.cat((torch.stack(fp_cls_x, 1), iou[fp_cls_x[0], fp_cls_x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if fp_cls_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            fp_cls_num[matches[:, 1].astype(int), i] = True
        if fp_loc_x[0].shape[0]:
            matches = torch.cat((torch.stack(fp_loc_x, 1), iou[fp_loc_x[0], fp_loc_x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if fp_loc_x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            fp_loc_num[matches[:, 1].astype(int), i] = True
    if detections.shape[0] == 0:
        tp_rate = 0
        fp_cls_rate = 0
        fp_loc_rate = 0
    else:
        tp_rate = np.sum(tp_num, 0) * 1.0/ detections.shape[0]
        fp_cls_rate = np.sum(fp_cls_num, 0) * 1.0/ detections.shape[0]
        fp_loc_rate = np.sum(fp_loc_num, 0) * 1.0/ detections.shape[0]
    # print('tp_rate:', tp_rate, tp_num)
    # print('fp_cls_rate:', fp_cls_rate, np.sum(fp_cls_num, 0))
    # print('fp_loc_rate:', fp_loc_rate, fp_loc_num, detections.shape[0])
    # iou_recall_rate = np.sum(correct, 0) * 1.0/ labels.shape[0]
    # print('correct:', np.sum(correct, 0))
    # if ignore_thres_low is not None: 
    # hit_rate = detections.shape[0] * 1.0 / labels.shape[0]
    # print('correct shape:', correct.shape)
    # print(np.sum(correct, 1))
    # print(detections.shape[0])
    return tp_rate, fp_cls_rate, fp_loc_rate, pseudo_label_num, gt_label_num

def check_pseudo_label(detections, ignore_thres_low=None, ignore_thres_high=None, batch_size=1):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # print('ignore_thres_high:', ignore_thres_high, ' ignore_thres_low:', ignore_thres_low)
    reliable_pseudo, uc_pseudo = select_targets(detections, ignore_thres_low, ignore_thres_high)
    reliable_num = reliable_pseudo.shape[0]/batch_size
    uncertain_num = uc_pseudo.shape[0]/batch_size
    denorm = reliable_num + uncertain_num
    if denorm == 0:
        precision_rate = 0
    else:
        precision_rate = reliable_num/ denorm
    if detections.shape[0] == 0:
        recall_rate = 0
    else:
        recall_rate = (reliable_num + uncertain_num) * batch_size / detections.shape[0]
    return precision_rate, recall_rate, reliable_num + uncertain_num, reliable_num

# if __name__ == '__main__':
    # label_match = LabelMatch(0.1, 80)