"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
from scipy.optimize import linear_sum_assignment
import math
from torch.nn import functional as F
from torch.autograd import Variable,Function

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class PssFocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, gamma=2, alpha=0.25):
        super(PssFocalLoss, self).__init__()
        # self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        # self.reduction = loss_fcn.reduction
        self.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        # loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability
        # print('pred', pred)
        # print('true', true)
        p = pred.clone()
        ce_loss = F.binary_cross_entropy(pred, true, reduction="none")
        loss = ce_loss

        # p_t = p * true + (1 - p) * (1 - true)
        # loss = ce_loss * ((1 - p_t) ** self.gamma)

        # if self.alpha >= 0:
        #     alpha_t = self.alpha * true + (1 - self.alpha) * (1 - true)
        #     loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, cfg):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters
        autobalance = cfg.Loss.autobalance
        cls_pw = cfg.Loss.cls_pw
        obj_pw = cfg.Loss.obj_pw
        label_smoothing = cfg.Loss.label_smoothing

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        # Focal loss
        g = cfg.Loss.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.head if is_parallel(model) else model.head # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0,  autobalance
        nl = det.nl
        nc = 1 if cfg.single_cls else cfg.Dataset.nc
        self.box_w = cfg.Loss.box*3.0/nl
        self.obj_w = cfg.Loss.obj
        self.cls_w = cfg.Loss.cls*nc / 80. * 3. / nl
        self.anchor_t = cfg.Loss.anchor_t

        self.LandMarkLoss = LandmarksLossYolov5(1.0)
        for k in 'na', 'nc', 'nl', 'num_keypoints', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, features = None):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors, tlandmarks, lmks_mask = self.build_targets_kps(p, targets, npoint=self.num_keypoints)  # targets
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        lmark = torch.zeros(1, device=device)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Landmark
                if self.num_keypoints >0:
                    plandmarks = ps[:,-self.num_keypoints*2:]
                    for idx in range(self.num_keypoints):    
                        plandmarks[:, (0+(2*idx)):(2+(2*idx))] = plandmarks[:, (0+(2*idx)):(2+(2*idx))] * anchors[i]
                    lmark = lmark + self.LandMarkLoss(plandmarks, tlandmarks[i], lmks_mask[i])

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

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
        
        # if features != None:
            # loss = lbox + lobj + lcls + 0 * (features[0].mean() + features[1].mean() + features[2].mean() )
        # else:
        loss = (lbox + lobj + lcls) 

        # self.loss_dict['box'] = lbox
        # self.loss_dict['obj'] = lobj
        # self.loss_dict['cls'] = lcls
        loss_dict = dict(box=lbox, obj=lobj, cls=lcls, loss=loss * bs)

        # return loss * bs, torch.cat((lbox, lobj, lcls)).detach()
        return loss * bs, loss_dict

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets[:,:6]
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def build_targets_kps(self, p, targets, npoint = 8):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
        gain = torch.ones(npoint*2+7,device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            gain_list = [3,2]*npoint                            # landmarks 10
            gain[6: 6+(npoint*2)] = torch.tensor(p[i].shape)[gain_list]  # xyxy gain
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                t = t[j]  # filter
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            a = t[:, npoint*2+6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            lks = t[:,6: (npoint*2+6)]
            lks_mask = torch.where(lks < 0, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))
            for idx in range(npoint):
                lks[:, [2*idx, 2*idx+1]] = (lks[:, [2*idx, 2*idx+1]] - gij)
            lks_mask_new = lks_mask
            lmks_mask.append(lks_mask_new)
            landmarks.append(lks)
        return tcls, tbox, indices, anch, landmarks, lmks_mask


# for domain adaptation training
class ComputeStudentLoss():
    # Compute losses
    def __init__(self, model, cfg):
        super(ComputeStudentLoss, self).__init__()
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
        self.cls_w = cfg.SSOD.cls_loss_weight
        self.anchor_t = cfg.Loss.anchor_t

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))


    def __call__(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)        
        tcls, tbox, indices, anchors  = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tmask = torch.zeros_like(pi[..., 0], device=device)  # target obj
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

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
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

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets[:,:6]
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
       
        for i in range(self.nl):
            anchors = self.anchors[i].to(targets.device)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))

                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            #score = t[:,6].T
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            #tscore.append(score)

        return tcls, tbox, indices, anch  #, tscore


class DomainFocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(DomainFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            #inputs = F.sigmoid(inputs)
            P = F.softmax(inputs, dim=1)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)
            # print(class_mask)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # batch_loss = -log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class TargetLoss():
    def __init__(self):
        self.fl = DomainFocalLoss(class_num=2)

    def __call__(self, feature):
        out_8 = feature[0]
        out_16 = feature[1]
        out_32 = feature[2]

        out_d_t_8 = out_8.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t_16 = out_16.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t_32 = out_32.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t = torch.cat((out_d_t_8, out_d_t_16, out_d_t_32), 0)
        # print('target_index:',target_index)

        # domain label
        domain_t = Variable(torch.ones(out_d_t.size(0)).long().cuda())
        dloss_t = 0.5 * self.fl(out_d_t, domain_t)
        # print('dloss_t:', dloss_t)
        return dloss_t


class DomainLoss():
    def __init__(self):
        self.fl = DomainFocalLoss(class_num=2)

    def __call__(self, feature):
        out_8 = feature[0]
        out_16 = feature[1]
        out_32 = feature[2]

        out_d_s_8 = out_8.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_s_16 = out_16.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_s_32 = out_32.permute(0, 2, 3, 1).reshape(-1, 2)
        # print('out_d_s_8:', out_d_s_8.shape)
        # print('out_d_s_16:', out_d_s_16.shape)
        # print('out_d_s_32:', out_d_s_32.shape)

        out_d_s = torch.cat((out_d_s_8, out_d_s_16, out_d_s_32), 0)
        # print('out_d_s:', out_d_s.shape)

        # domain label
        domain_s = Variable(torch.zeros(out_d_s.size(0)).long().cuda())
        # global alignment loss
        dloss_s = 0.5 * self.fl(out_d_s, domain_s)
        return dloss_s


class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel):
        mask = truel > 0
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 10e-14)

class LandmarksLossYolov5(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLossYolov5, self).__init__()
        self.loss_fcn = WingLoss()#nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 10e-14)

class RotateLandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(RotateLandmarksLoss, self).__init__()
        self.loss_fcn = nn.SmoothL1Loss()
        self.alpha = alpha

    def forward(self, pred, truel, iou=None):
        mask = truel > 0
        truel_left = torch.cat(((truel[:,6:8]), truel[:, :6]), axis=1)
        truel_right = torch.cat(((truel[:,2:]), truel[:, :2]), axis=1)
        loss_ori = self.loss_fcn(pred * mask, truel * mask)
        loss_left = self.loss_fcn(pred * mask, truel_left * mask)
        loss_right = self.loss_fcn(pred * mask, truel_right * mask)
        loss = torch.minimum(torch.minimum(loss_ori.sum(), loss_left.sum()), loss_right.sum())
        return loss / (torch.sum(mask) + 10e-14)

class JointBoneLoss(nn.Module):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i+1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j
        self.joint_num = joint_num

    def forward(self, joint_out, joint_gt):
        joint_out = joint_out.reshape(-1, self.joint_num, 2)
        joint_gt = joint_gt.reshape(-1, self.joint_num, 2)
        mask = joint_gt > 0
        joint_out = joint_out * mask
        joint_gt = joint_gt * mask
        J = torch.norm(joint_out[:,self.id_i,:] - joint_out[:,self.id_j,:], p=2, dim=-1, keepdim=False)
        Y = torch.norm(joint_gt[:,self.id_i,:] - joint_gt[:,self.id_j,:], p=2, dim=-1, keepdim=False)
        loss = torch.abs(J-Y)
        return loss.mean()

def smooth_l1_loss(pred, target, beta=1.0):
    '''
    Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
    Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)
    '''
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

def wing_loss(pred, target, omega=10.0, epsilon=2.0):
    '''
    Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
    Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)
    '''
    C = omega * (1.0 - math.log(1.0 + omega / epsilon))
    diff = torch.abs(pred - target)
    losses = torch.where(diff < omega, omega * torch.log(1.0 + diff / epsilon), diff - C)
    return losses

def hungarian_loss_quad(inputs, targets):
    quad_inputs  = inputs.reshape(-1, 4, 2)
    quad_targets = targets.reshape(-1, 4, 2)
    losses = torch.stack(
        [wing_loss(quad_inputs, quad_targets[:, i, :].unsqueeze(1).repeat(1, 4, 1)).sum(2) \
            for i in range(4)] , 1
            )
    indices = [linear_sum_assignment(loss.cpu().detach().numpy()) for loss in losses]
    match_loss = []
    for cnt, (row_ind,col_ind) in enumerate(indices):
        match_loss.append(losses[cnt, row_ind, col_ind])
    return torch.stack(match_loss).sum(1)

class HungarianLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', form='obb', loss_weight=1.0):
        super(HungarianLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.form = form

    def forward(self,
                pred,
                target
                # weight=None,
                # avg_factor=None,
                # reduction_override=None,
                ):
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        # reduction = (
            # reduction_override if reduction_override else self.reduction)
        # valid_idx = weight.nonzero()[:,0].unique()
        # if len(valid_idx) == 0:
            # return torch.tensor(0).float().cuda()
        # else:
        #     if self.form == 'obb':
        #         loss = hungarian_loss_obb(pred[valid_idx], target[valid_idx].float()) * self.loss_weight
        #     elif self.form == 'quad':
        loss = hungarian_loss_quad(pred, target.float()) * self.loss_weight
            # else:
                # raise NotImplementedError
        return loss.mean()
class WingLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', form='obb', loss_weight=1.0):
        super(WingLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.form = form
    
    def forward(self, pred, target):
        mask = target > 0
        losses = wing_loss(pred * mask, target.float() * mask)
        return losses.sum() / (torch.sum(mask) + 10e-14)
        # return losses.mean()

class WingLossYolov5(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLossYolov5, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class GWDLoss(nn.Module):
    def __init__(self):
        super(GWDLoss, self).__init__()

    def gt2gaussian(self, target):
        """Convert polygons to Gaussian distributions.
        Args:
            target (torch.Tensor): Polygons with shape (N, 4, 2).

        Returns:
            dict[str, torch.Tensor]: Gaussian distributions.
        """
        L = 3
        target = target.reshape(-1, 4, 2)
        center = torch.mean(target, dim=1)
        edge_1 = target[:, 1, :] - target[:, 0, :]
        edge_2 = target[:, 2, :] - target[:, 1, :]
        w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
        w_ = w.sqrt()
        h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
        diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
        cos_sin = edge_1 / w_
        neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
        R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)
        return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))
    
    def forward(self, pred, target, fun='log1p', tau=1.0):
        """Gaussian Wasserstein distance loss.

        Args:
            pred (torch.Tensor): Predicted bboxes.
            target (torch.Tensor): Corresponding gt bboxes.
            fun (str): The function applied to distance. Defaults to 'log1p'.
            tau (float): Defaults to 1.0.

        Returns:
            loss (torch.Tensor)
        """
        mu_p, sigma_p = self.gt2gaussian(pred)
        mu_t, sigma_t = self.gt2gaussian(target)

        mu_p = mu_p.reshape(-1, 2).float().to(target.device)
        mu_t = mu_t.reshape(-1, 2).float().to(target.device)
        sigma_p = sigma_p.reshape(-1, 2, 2).float().to(target.device)
        sigma_t = sigma_t.reshape(-1, 2, 2).float().to(target.device)
        # mu_p, sigma_p = pred
        # mu_t, sigma_t = target

        xy_distance = (mu_p - mu_t).square().sum(dim=-1)

        whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance + sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
        whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

        dis = xy_distance + whr_distance
        gwd_dis = dis.clamp(min=1e-6)

        if fun == 'sqrt':
            loss = 1 - 1 / (tau + torch.sqrt(gwd_dis))
        elif fun == 'log1p':
            loss = 1 - 1 / (tau + torch.log1p(gwd_dis))
        else:
            scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
            loss = torch.log1p(torch.sqrt(gwd_dis) / scale)
        return loss.mean()
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def gt2gaussian(self, target):
        """Convert polygons to Gaussian distributions.
        Args:
            target (torch.Tensor): Polygons with shape (N, 8).

        Returns:
            dict[str, torch.Tensor]: Gaussian distributions.
        """
        L = 3
        target = target.reshape(-1, 4, 2)
        center = torch.mean(target, dim=1)
        edge_1 = target[:, 1, :] - target[:, 0, :]
        edge_2 = target[:, 2, :] - target[:, 1, :]
        w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
        w_ = w.sqrt()
        h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
        diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
        cos_sin = edge_1 / w_
        neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
        R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)
        return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))
    
    def forward(self, pred, target, fun='log1p', tau=1.0):
        """Kullback-Leibler Divergence loss.

        Args:
            pred (torch.Tensor): Predicted bboxes.
            target (torch.Tensor): Corresponding gt bboxes.
            fun (str): The function applied to distance. Defaults to 'log1p'.
            tau (float): Defaults to 1.0.

        Returns:
            loss (torch.Tensor)
        """
        mu_p, sigma_p = self.gt2gaussian(pred)
        mu_t, sigma_t = self.gt2gaussian(target)
        # mu_p, sigma_p = pred
        # mu_t, sigma_t = target

        mu_p = mu_p.reshape(-1, 2).float().to(target.device)
        mu_t = mu_t.reshape(-1, 2).float().to(target.device)
        sigma_p = sigma_p.reshape(-1, 2, 2).float().to(target.device)
        sigma_t = sigma_t.reshape(-1, 2, 2).float().to(target.device)

        delta = (mu_p - mu_t).unsqueeze(-1)
        try:
            sigma_t_inv = torch.cholesky_inverse(sigma_t)
        except RuntimeError:
            print('sigma_t:', sigma_t, ' target:', target)
        term1 = delta.transpose(-1,
                            -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
        term2 = torch.diagonal( sigma_t_inv.matmul(sigma_p),
            dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) + \
            torch.log(torch.det(sigma_t) / torch.det(sigma_p)).reshape(-1, 1)
        dis = term1 + term2 - 2
        kl_dis = dis.clamp(min=1e-6)

        if fun == 'sqrt':
            kl_loss = 1 - 1 / (tau + torch.sqrt(kl_dis))
        else:
            kl_loss = 1 - 1 / (tau + torch.log1p(kl_dis))
        return kl_loss.mean()


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target, xyxy=False):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        if xyxy:
            tl = torch.max(pred[:, :2], target[:, :2])
            br = torch.min(pred[:, 2:], target[:, 2:])
            area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
            area_g = torch.prod(target[:, 2:] - target[:, :2], 1)
        else:
            tl = torch.max(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            br = torch.min(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_p = torch.prod(pred[:, 2:], 1)
            area_g = torch.prod(target[:, 2:], 1)

        hw = (br - tl).clamp(min=0)  # [rows, 2]
        area_i = torch.prod(hw, 1)

        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            if xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
                )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class ComputeXLoss:
    # Compute losses
    def __init__(self, model, cfg):
        super(ComputeXLoss, self).__init__()
        # device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        det = model.module.head if is_parallel(model) else model.head# Detect() module
        self.det = det
        # self.loss_dict = {'iou_loss' : 0, 'obj_loss' : 0, 'cls_loss' : 0, 'loss' : 0, 'num_fg': 0}

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        if (not self.det.training) or (len(p) == 0):
            return torch.zeros(1, device=device), torch.zeros(4, device=device)

        (loss,
         iou_loss,
         obj_loss,
         cls_loss,
         l1_loss,
         num_fg,
         kp_loss, 
         kp_obj_loss ) = self.det.get_losses(
            *p,
            targets,
            dtype=p[0].dtype,
        )
        # loss += 0 * (feature[0].mean() + feature[1].mean() + feature[2].mean()) 
        loss = torch.unsqueeze(loss, 0)
        num_fg = torch.tensor(num_fg).to(loss.device)

        loss_dict = dict(iou_loss = iou_loss, obj_loss = obj_loss, cls_loss = cls_loss, loss = loss, num_fg = num_fg)
        return loss, loss_dict

class ComputeNanoLoss:
    def __init__(self, model, cfg):
        super(ComputeNanoLoss, self).__init__()
        det = model.module.head if is_parallel(model) else model.head  # Detect() module
        self.det = det
        # self.loss_dict = {'loss_qfl':0, 'loss_bbox':0, 'loss_dfl':0}
    
    def __call__(self, p, targets):
        loss, loss_dict = self.det.get_losses(p, targets) 
        return loss, loss_dict
        

class ComputeKeyPointsLoss:
    # Compute losses
    def __init__(self, model, cfg):
        super(ComputeKeyPointsLoss, self).__init__()
        # device = next(model.parameters()).device  # get model device
        # h = model.hyp  # hyperparameters

        det = model.module.head if is_parallel(model) else model.head# Detect() module
        self.det = det

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        if (not self.det.training) or (len(p) == 0):
            return torch.zeros(1, device=device), torch.zeros(4, device=device)

        (loss,
         iou_loss,
         obj_loss,
         cls_loss,
         num_fg,
         lmk_num_fg,
         kp_loss, 
         kp_obj_loss ) = self.det.get_losses(
            *p,
            targets,
            dtype=p[0].dtype,
        )
        # loss += 0 * (feature[0].mean() + feature[1].mean() + feature[2].mean()) 
        loss = torch.unsqueeze(loss, 0)
        num_fg = torch.tensor(num_fg).to(loss.device)

        loss_dict = dict(iou_loss=iou_loss, obj_loss=obj_loss, cls_loss=cls_loss, n_fg=num_fg, lmk_n_fg=lmk_num_fg, kp_loss=kp_loss, kp_obj=kp_obj_loss, loss=loss)
        # self.loss_dict['iou_loss'] = iou_loss.detach()
        # self.loss_dict['obj_loss'] = obj_loss.detach()
        # self.loss_dict['cls_loss'] = cls_loss.detach()
        # self.loss_dict['loss'] = loss.detach()
        # self.loss_dict['num_fg'] = num_fg
        # self.loss_dict['kp_loss'] = kp_loss.detach()
        # self.loss_dict['kp_obj_loss'] = kp_obj_loss.detach()
        # loss_dict['lmk_num_fg'] = torch.tensor(lmk_num_fg).to(loss.device)
        # print('num_fg:', num_fg)
        return loss, loss_dict