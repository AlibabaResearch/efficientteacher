#Copyright (c) 2023, Alibaba Group
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
# from scipy.optimize import linear_sum_assignment
import math
from torch.nn import functional as F
from torch.autograd import Variable,Function
import numpy as np
# from loss.yolox_loss import pairwise_bbox_iou
from utils.general import xywh2xyxy, box_iou
from assigner import YOLOAnchorAssigner
# from torch.cuda.amp import autocast

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

# for label match type semi spervised training
class ComputeStudentMatchLoss():
    # Compute losses
    def __init__(self, model, cfg):
        super(ComputeStudentMatchLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        #h = model.hyp  # hyperparameters
        autobalance = cfg.Loss.autobalance
        cls_pw = cfg.Loss.cls_pw
        obj_pw = cfg.Loss.obj_pw
        label_smoothing = cfg.Loss.label_smoothing

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))

        if cfg.SSOD.focal_loss > 0:
            BCEobj = FocalLoss(BCEobj)

        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        det = model.module.head if is_parallel(model) else model.head # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr,  self.autobalance = BCEcls, BCEobj, 1.0, autobalance
        self.box_w = cfg.SSOD.box_loss_weight
        self.obj_w = cfg.SSOD.obj_loss_weight
        # self.cls_w = cfg.SSOD.cls_loss_weight
        self.cls_w = cfg.SSOD.cls_loss_weight * cfg.Dataset.nc / 80. * 3. / det.nl
        self.anchor_t = cfg.Loss.anchor_t
        # self.ignore_thres = cfg.SSOD.ignore_thres
        self.ignore_thres_high = [cfg.SSOD.ignore_thres_high] * cfg.Dataset.nc
        self.ignore_thres_low = [cfg.SSOD.ignore_thres_low] * cfg.Dataset.nc
        self.uncertain_aug = cfg.SSOD.uncertain_aug
        self.use_ota = cfg.SSOD.use_ota
        self.ignore_obj = cfg.SSOD.ignore_obj
        self.pseudo_label_with_obj = cfg.SSOD.pseudo_label_with_obj
        self.pseudo_label_with_bbox = cfg.SSOD.pseudo_label_with_bbox
        self.pseudo_label_with_cls = cfg.SSOD.pseudo_label_with_cls
        self.num_keypoints = cfg.Dataset.np
        self.single_targets = False
        if not self.uncertain_aug:
            self.single_targets = True

        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))
        self.assigner = YOLOAnchorAssigner(self.na, self.nl, self.anchors, self.anchor_t, det.stride, \
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=self.use_ota)
    
    # def build_single_targets(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     targets = targets[:,:6]
    #     na, nt = self.na, targets.shape[0]  # number of anchors, targets
    #     tcls, tbox, indices, anch = [], [], [], []
    #     gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets
       
    #     for i in range(self.nl):
    #         anchors = self.anchors[i].to(targets.device)
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
    #             t = t[j]  # filter

    #             # Offsets
    #             # gxy = t[:, 2:4]  # grid xy
    #             # gxi = gain[[2, 3]] - gxy  # inverse
    #             # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             # j = torch.stack((torch.ones_like(j), j, k, l, m))

    #             # t = t.repeat((5, 1, 1))[j]
    #             # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #             offsets = 0
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         a = t[:, 6].long()  # anchor indices
    #         # a = t[:, 7].long()  # anchor indices
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    #         anch.append(anchors[a])  # anchors
    #         tcls.append(c)  # class

    #     return tcls, tbox, indices, anch 

    def select_targets(self, targets):
        '''
        targets: [batch, cls, x, y, x, y, conf, obj_conf, cls_conf]
        '''
        device = targets.device
        reliable_targets = []
        uncertain_targets = []
        uncertain_obj_targets = []
        uncertain_cls_targets = []
        for t in targets:
            #伪标签得分大于相应类别的阈值,标记为正样本
            t = np.array(t.cpu())
            # original logic
            if t[6] >= self.ignore_thres_high[int(t[1])]:
                    reliable_targets.append(t[:7])
            # if t[6] >= self.ignore_thres_high[int(t[1])]:
            #     # if t[7] >= self.ignore_thres_high[int(t[1])] and t[8] >= self.ignore_thres_high[int(t[1])]:
            #     if t[8] >= 0.99:
            #         reliable_targets.append(t[:7])
            #     else: #如果obj和cls中其中一个小于阈值, 标记为uncertain 
            #         if self.pseudo_label_with_obj:
            #             uncertain_targets.append(np.concatenate((t[:6], t[7:8])))
            #             if t[7] > 0.99:
            #                 uncertain_obj_targets.append(np.concatenate((t[:6], t[7:8])))
            #         else:
            #             uncertain_targets.append(t[:7])
            #伪标签低阈值和高阈值之间的，标记为不确定样本
            elif t[6] >= self.ignore_thres_low[int(t[1])]:
                if self.pseudo_label_with_obj:
                    uncertain_targets.append(np.concatenate((t[:6], t[7:8])))
                    #不确定样本里面obj特别高的，送出来修iou loss
                    if t[7] >= 0.99:
                        uncertain_obj_targets.append(np.concatenate((t[:6], t[7:8])))
                    #不确定样本里cls特别高的，送出来修cls loss
                    if t[8] >= 0.99:
                        uncertain_cls_targets.append(np.concatenate((t[:6], t[7:8])))
                else:
                    uncertain_targets.append(t[:7])

        reliable_targets = np.array(reliable_targets).astype(np.float32)
        reliable_targets= torch.from_numpy(reliable_targets).contiguous()
        reliable_targets = reliable_targets.to(device)
        if reliable_targets.shape[0] == 0:
            reliable_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_targets = np.array(uncertain_targets).astype(np.float32)
        uncertain_targets= torch.from_numpy(uncertain_targets).contiguous()
        uncertain_targets= uncertain_targets.to(device)
        if uncertain_targets.shape[0] == 0:
            uncertain_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_obj_targets = np.array(uncertain_obj_targets).astype(np.float32)
        uncertain_obj_targets= torch.from_numpy(uncertain_obj_targets).contiguous()
        uncertain_obj_targets= uncertain_obj_targets.to(device)
        if uncertain_obj_targets.shape[0] == 0:
            uncertain_obj_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)

        uncertain_cls_targets = np.array(uncertain_cls_targets).astype(np.float32)
        uncertain_cls_targets= torch.from_numpy(uncertain_cls_targets).contiguous()
        uncertain_cls_targets= uncertain_cls_targets.to(device)
        if uncertain_cls_targets.shape[0] == 0:
            uncertain_cls_targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])[False].to(device)
        return reliable_targets, uncertain_targets, uncertain_obj_targets, uncertain_cls_targets

    def default_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)        
        if targets.shape[1] > 6:
            certain_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            if self.uncertain_aug:
                tcls, tbox, indices, anchors  = self.assigner(p, certain_targets)  # targets
                _, _, uc_indices, _, uc_scores = self.assigner(p, uc_targets, with_pseudo_score=True)
                _, uc_tbox, uc_obj_indices, uc_anchors, _ = self.assigner(p, uc_obj_targets, with_pseudo_score=True)
                uc_tcls, _, uc_cls_indices, _, _ = self.assigner(p, uc_cls_targets, with_pseudo_score=True)
            else:
                tcls, tbox, indices, anchors  = self.assigner(p, certain_targets)  # targets
                _, _, uc_indices, _, uc_scores = self.assigner(p, uc_targets, with_pseudo_score=True)
                _, uc_tbox, uc_obj_indices, uc_anchors, _ = self.assigner(p, uc_obj_targets, with_pseudo_score=True)                
                uc_tcls, _, uc_cls_indices, _, _ = self.assigner(p, uc_cls_targets, with_pseudo_score=True)
        else:
            tcls, tbox, indices, anchors  = self.assigner(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            anchors_spec = anchors[i] 
            tbox_spec = tbox[i]
            n = b.shape[0]  # number of targets

            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox_spec, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # tobj[b, a, gj, gi] = 1.0

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            if targets.shape[1] > 6:
                #uncertain label cal obj loss
                uc_b, uc_a, uc_gj, uc_gi = uc_indices[i]
                n = uc_b.shape[0]
                if n:
                    if self.ignore_obj:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = -1 #ignore region set -1
                    else:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = uc_scores[i].type(tobj.dtype)  #ignore region set -1

                if self.pseudo_label_with_bbox:
                    #uncertain label cal iou loss
                    uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi = uc_obj_indices[i]
                    n = uc_obj_b.shape[0]
                    uc_tbox_spec = uc_tbox[i]
                    anchors_spec = uc_anchors[i]
                    if n:
                        uc_ps = pi[uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi]
                        pxy = uc_ps[:, :2].sigmoid() * 2. - 0.5
                        pwh = (uc_ps[:, 2:4].sigmoid() * 2) ** 2 * anchors_spec
                        pbox = torch.cat((pxy, pwh), 1)  # predicted box
                        iou = bbox_iou(pbox.T, uc_tbox_spec, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                        lbox += (1.0 - iou).mean()  # iou loss
                        # tobj[uc_obj_b, uc_obj_a, uc_obj_gj, uc_obj_gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                
                if self.pseudo_label_with_cls:
                    #uncertain label cal cls loss
                    uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi = uc_cls_indices[i]
                    n = uc_cls_b.shape[0]
                    if n:
                        uc_ps = pi[uc_cls_b, uc_cls_a, uc_cls_gj, uc_cls_gi]
                        if self.nc > 1:  # cls loss (only if multiple classes)
                            t = torch.full_like(uc_ps[:, 5:], self.cn, device=device)  # targets
                            t[range(n), uc_tcls[i]] = self.cp
                            lcls += self.BCEcls(uc_ps[:, 5:], t)  # BCE
            # filtering ignore region, only cal gradient on foreground and background
            valid_mask = tobj >= 0
            obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
            lobj += obji * self.balance[i]  # obj loss


        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        loss_dict = dict(ss_box = lbox, ss_obj = lobj, ss_cls = lcls)
        return loss * bs, loss_dict

    def __call__(self, p, targets):
        if self.use_ota == False:
            loss, loss_dict = self.default_loss(p, targets)
        else:
            loss, loss_dict = self.ota_loss(p, targets)
        return loss, loss_dict
    
    def ota_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if targets.shape[1] > 6:
            reliable_targets, uc_targets, uc_obj_targets, uc_cls_targets = self.select_targets(targets)
            bs, as_, gjs, gis, reliable_targets, anchors, tscores = self.assigner(p, reliable_targets, with_pseudo_scores=True)
            uc_bs, uc_as_, uc_gjs, uc_gis, uc_targets, uc_anchors, uc_tscores = self.assigner(p, uc_targets, with_pseudo_scores=True)
            pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    
            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    grid = torch.stack([gi, gj], dim=1)
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    #pxy = ps[:, :2].sigmoid() * 3. - 1.
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    selected_tbox = reliable_targets[i][:, 2:6] * pre_gen_gains[i]
                    selected_tbox[:, :2] -= grid
                    iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                    # Classification
                    selected_tcls = reliable_targets[i][:, 1].long()
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), selected_tcls] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                uc_b, uc_a, uc_gj, uc_gi = uc_bs[i], uc_as_[i], uc_gjs[i], uc_gis[i]
                n = uc_b.shape[0]
                if n:
                    if self.ignore_obj:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = -1
                    else:
                        tobj[uc_b, uc_a, uc_gj, uc_gi] = uc_tscores[i].type(tobj.dtype) 
                valid_mask = tobj >= 0
                obji = self.BCEobj(pi[..., 4][valid_mask], tobj[valid_mask])
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        else:
            bs, as_, gjs, gis, targets, anchors = self.assigner(p, targets)
            pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 
    
            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    grid = torch.stack([gi, gj], dim=1)
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    #pxy = ps[:, :2].sigmoid() * 3. - 1.
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                    selected_tbox[:, :2] -= grid
                    iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                    # Classification
                    selected_tcls = targets[i][:, 1].long()
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), selected_tcls] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                obji = self.BCEobj(pi[..., 4], tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        loss_dict = dict(ss_box = lbox, ss_obj = lobj, ss_cls = lcls)
        return loss * bs, loss_dict

    # def build_ota_targets(self, p, targets):
    #     #indices, anch = self.find_positive(p, targets)
    #     indices, anch= self.find_3_positive(p, targets)
    #     #indices, anch = self.find_4_positive(p, targets)
    #     #indices, anch = self.find_5_positive(p, targets)
    #     #indices, anch = self.find_9_positive(p, targets)

    #     matching_bs = [[] for pp in p]
    #     matching_as = [[] for pp in p]
    #     matching_gjs = [[] for pp in p]
    #     matching_gis = [[] for pp in p]
    #     matching_targets = [[] for pp in p]
    #     matching_anchs = [[] for pp in p]
    #     matching_tscores = [[] for pp in p]
        
    #     nl = len(p)    
    
    #     for batch_idx in range(p[0].shape[0]):
        
    #         b_idx = targets[:, 0]==batch_idx
    #         this_target = targets[b_idx]
    #         if this_target.shape[0] == 0:
    #             continue
                
    #         txywh = this_target[:, 2:6] * 640 #TODO
    #         txyxy = xywh2xyxy(txywh)

    #         pxyxys = []
    #         p_cls = []
    #         p_obj = []
    #         from_which_layer = []
    #         all_b = []
    #         all_a = []
    #         all_gj = []
    #         all_gi = []
    #         all_anch = []
            
    #         for i, pi in enumerate(p):
                
    #             b, a, gj, gi = indices[i]
    #             idx = (b == batch_idx)
    #             b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
    #             all_b.append(b)
    #             all_a.append(a)
    #             all_gj.append(gj)
    #             all_gi.append(gi)
    #             all_anch.append(anch[i][idx])
    #             from_which_layer.append(torch.ones(size=(len(b),)) * i)
                
    #             fg_pred = pi[b, a, gj, gi]                
    #             p_obj.append(fg_pred[:, 4:5])
    #             p_cls.append(fg_pred[:, 5:])
                
    #             grid = torch.stack([gi, gj], dim=1)
    #             pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
    #             #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
    #             pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
    #             pxywh = torch.cat([pxy, pwh], dim=-1)
    #             pxyxy = xywh2xyxy(pxywh)
    #             pxyxys.append(pxyxy)
            
    #         pxyxys = torch.cat(pxyxys, dim=0)
    #         if pxyxys.shape[0] == 0:
    #             continue
    #         p_obj = torch.cat(p_obj, dim=0)
    #         p_cls = torch.cat(p_cls, dim=0)
    #         from_which_layer = torch.cat(from_which_layer, dim=0)
    #         all_b = torch.cat(all_b, dim=0)
    #         all_a = torch.cat(all_a, dim=0)
    #         all_gj = torch.cat(all_gj, dim=0)
    #         all_gi = torch.cat(all_gi, dim=0)
    #         all_anch = torch.cat(all_anch, dim=0)
        
    #         pair_wise_iou = box_iou(txyxy, pxyxys)

    #         pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

    #         top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
    #         dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

    #         gt_cls_per_image = (
    #             F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
    #             .float()
    #             .unsqueeze(1)
    #             .repeat(1, pxyxys.shape[0], 1)
    #         )

    #         num_gt = this_target.shape[0]
    #         cls_preds_ = (
    #             p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
    #             * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
    #         )

    #         y = cls_preds_.sqrt_()
    #         pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
    #            torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
    #         ).sum(-1)
    #         del cls_preds_
        
    #         cost = (
    #             pair_wise_cls_loss
    #             + 3.0 * pair_wise_iou_loss
    #         )

    #         matching_matrix = torch.zeros_like(cost)

    #         for gt_idx in range(num_gt):
    #             _, pos_idx = torch.topk(
    #                 cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
    #             )
    #             matching_matrix[gt_idx][pos_idx] = 1.0

    #         del top_k, dynamic_ks
    #         anchor_matching_gt = matching_matrix.sum(0)
    #         if (anchor_matching_gt > 1).sum() > 0:
    #             _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
    #             matching_matrix[:, anchor_matching_gt > 1] *= 0.0
    #             matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
    #         fg_mask_inboxes = matching_matrix.sum(0) > 0.0
    #         matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
    #         from_which_layer = from_which_layer[fg_mask_inboxes]
    #         all_b = all_b[fg_mask_inboxes]
    #         all_a = all_a[fg_mask_inboxes]
    #         all_gj = all_gj[fg_mask_inboxes]
    #         all_gi = all_gi[fg_mask_inboxes]
    #         all_anch = all_anch[fg_mask_inboxes]
        
    #         this_target = this_target[matched_gt_inds]
        
    #         for i in range(nl):
    #             layer_idx = from_which_layer == i
    #             matching_bs[i].append(all_b[layer_idx])
    #             matching_as[i].append(all_a[layer_idx])
    #             matching_gjs[i].append(all_gj[layer_idx])
    #             matching_gis[i].append(all_gi[layer_idx])
    #             matching_targets[i].append(this_target[layer_idx])
    #             matching_anchs[i].append(all_anch[layer_idx])

    #     for i in range(nl):
    #         if matching_targets[i] != []:
    #             matching_bs[i] = torch.cat(matching_bs[i], dim=0)
    #             matching_as[i] = torch.cat(matching_as[i], dim=0)
    #             matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
    #             matching_gis[i] = torch.cat(matching_gis[i], dim=0)
    #             matching_targets[i] = torch.cat(matching_targets[i], dim=0)
    #             matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
    #         else:
    #             matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
    #             matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
    #             matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
    #             matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
    #             matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
    #             matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

    #     return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs



    # def find_3_positive_with_score(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     na, nt = self.na, targets.shape[0]  # number of anchors, targets
    #     indices, anch = [], []
    #     tscore = []
    #     gain = torch.ones(8, device=targets.device).long()  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets

    #     for i in range(self.nl):
    #         anchors = self.anchors[i]
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

    #         # Match targets to anchors
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
    #             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
    #             t = t[j]  # filter

    #             # Offsets
    #             gxy = t[:, 2:4]  # grid xy
    #             gxi = gain[[2, 3]] - gxy  # inverse
    #             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             j = torch.stack((torch.ones_like(j), j, k, l, m))
    #             t = t.repeat((5, 1, 1))[j]
    #             offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         a = t[:, 7].long()  # anchor indices
    #         score = t[:, 6].T
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         anch.append(anchors[a])  # anchors
    #         tscore.append(score)
    #         # print('score:', score)
    #         # print('anch:', anchors[a])
    #         # print('indice:', (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))

    #     return indices, anch, tscore

    # def find_3_positive(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     na, nt = self.na, targets.shape[0]  # number of anchors, targets
    #     indices, anch = [], []
    #     gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets

    #     for i in range(self.nl):
    #         anchors = self.anchors[i]
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

    #         # Match targets to anchors
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
    #             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
    #             t = t[j]  # filter

    #             # Offsets
    #             gxy = t[:, 2:4]  # grid xy
    #             gxi = gain[[2, 3]] - gxy  # inverse
    #             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             j = torch.stack((torch.ones_like(j), j, k, l, m))
    #             t = t.repeat((5, 1, 1))[j]
    #             offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         a = t[:, 6].long()  # anchor indices
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         anch.append(anchors[a])  # anchors

    #     return indices, anch

    # def build_uc_targets_aug(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     # targets = targets[:,:6]
    #     na, nt = self.na, targets.shape[0]  # number of anchors, targets
    #     tcls, tbox, indices, anch = [], [], [], []
    #     tscore = []
    #     gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets
       
    #     for i in range(self.nl):
    #         anchors = self.anchors[i].to(targets.device)
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
    #             t = t[j]  # filter

    #             # Offsets
    #             gxy = t[:, 2:4]  # grid xy
    #             gxi = gain[[2, 3]] - gxy  # inverse
    #             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             j = torch.stack((torch.ones_like(j), j, k, l, m))

    #             t = t.repeat((5, 1, 1))[j]
    #             offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #             # offsets = 0
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         # a = t[:, 6].long()  # anchor indices
    #         a = t[:, 7].long()  # anchor indices
    #         score = t[:,6].T
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    #         anch.append(anchors[a])  # anchors
    #         tcls.append(c)  # class
    #         tscore.append(score)

    #     return tcls, tbox, indices, anch , tscore

    # def build_uc_targets(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     # targets = targets[:,:6]
    #     na, nt = self.na, targets.shape[0]  # number of anchors, targets
    #     tcls, tbox, indices, anch = [], [], [], []
    #     tscore = []
    #     gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets
       
    #     for i in range(self.nl):
    #         anchors = self.anchors[i].to(targets.device)
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
    #             t = t[j]  # filter

    #             # Offsets
    #             # gxy = t[:, 2:4]  # grid xy
    #             # gxi = gain[[2, 3]] - gxy  # inverse
    #             # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             # j = torch.stack((torch.ones_like(j), j, k, l, m))

    #             # t = t.repeat((5, 1, 1))[j]
    #             # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #             offsets = 0
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         # a = t[:, 6].long()  # anchor indices
    #         a = t[:, 7].long()  # anchor indices
    #         score = t[:,6].T
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    #         anch.append(anchors[a])  # anchors
    #         tcls.append(c)  # class
    #         tscore.append(score)

    #     return tcls, tbox, indices, anch , tscore

    # def build_targets(self, p, targets):
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     targets = targets[:,:6]
    #     na, nt = self.na, targets.shape[0]  # number of anchors, targets
    #     tcls, tbox, indices, anch = [], [], [], []
    #     gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    #     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    #     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    #     g = 0.5  # bias
    #     off = torch.tensor([[0, 0],
    #                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
    #                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
    #                         ], device=targets.device).float() * g  # offsets
       
    #     for i in range(self.nl):
    #         anchors = self.anchors[i].to(targets.device)
    #         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
    #         t = targets * gain
    #         if nt:
    #             # Matches
    #             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
    #             j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
    #             t = t[j]  # filter

    #             # Offsets
    #             gxy = t[:, 2:4]  # grid xy
    #             gxi = gain[[2, 3]] - gxy  # inverse
    #             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
    #             l, m = ((gxi % 1. < g) & (gxi > 1.)).T
    #             j = torch.stack((torch.ones_like(j), j, k, l, m))

    #             t = t.repeat((5, 1, 1))[j]
    #             offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
    #         else:
    #             t = targets[0]
    #             offsets = 0

    #         # Define
    #         b, c = t[:, :2].long().T  # image, class
    #         gxy = t[:, 2:4]  # grid xy
    #         gwh = t[:, 4:6]  # grid wh
    #         gij = (gxy - offsets).long()
    #         gi, gj = gij.T  # grid xy indices

    #         # Append
    #         a = t[:, 6].long()  # anchor indices
    #         #score = t[:,6].T
    #         indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
    #         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
    #         anch.append(anchors[a])  # anchors
    #         tcls.append(c)  # class
    #         #tscore.append(score)

    #     return tcls, tbox, indices, anch  #, tscore