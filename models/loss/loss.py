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
from assigner import YOLOAnchorAssigner

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
        self.single_targets = cfg.Loss.single_targets

        self.LandMarkLoss = LandmarksLossYolov5(1.0)
        for k in 'na', 'nc', 'nl', 'num_keypoints', 'anchors':
            setattr(self, k, getattr(det, k))
        self.ota = cfg.Loss.assigner_type == 'SimOTA'
        self.top_k = cfg.Loss.top_k
        self.assigner = YOLOAnchorAssigner(self.na, self.nl, self.anchors, self.anchor_t, det.stride, \
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=False)
        self.ota_assigner = YOLOAnchorAssigner(self.na, self.nl, self.anchors, self.anchor_t, det.stride, \
            self.nc, self.num_keypoints, single_targets=self.single_targets, ota=self.ota, top_k=self.top_k)

    def default_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if self.num_keypoints >0:
            tcls, tbox, indices, anchors, tlandmarks, lmks_mask = self.assigner(p, targets)  # targets
        else:
            tcls, tbox, indices, anchors = self.assigner(p, targets)  # targets
            # if self.single_targets:
            #     tcls, tbox, indices, anchors = self.build_single_targets(p, targets)  # targets
            # else:
            #     tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
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
        
        loss = (lbox + lobj + lcls) 

        loss_dict = dict(box=lbox, obj=lobj, cls=lcls, loss=loss * bs)

        # return loss * bs, torch.cat((lbox, lobj, lcls)).detach()
        return loss * bs, loss_dict
    
    def ota_loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, ota_targets, anchors = self.ota_assigner(p, targets)
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
                selected_tbox = ota_targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = ota_targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:5+self.nc], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5+self.nc], t)  # BCE

            obji = self.BCEobj(pi[...,-1], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        tcls, tbox, indices, anchors = self.assigner(p, targets)  # targets
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
                    t = torch.full_like(ps[:, 5:5+self.nc], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5+self.nc], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_w
        lobj *= self.obj_w
        lcls *= self.cls_w
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        loss_dict = dict(box=lbox, obj=lobj, cls=lcls, loss=loss * bs)
        # loss_dict = dict(box = lbox, obj = lobj, cls = lcls)
        return loss * bs, loss_dict

    def __call__(self, p, targets):  # predictions, targets, model
        if self.ota:
            return self.ota_loss(p, targets)
        else:
            return self.default_loss(p, targets)


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
    def __init__(self, reduction="none", iou_type="iou", xyxy=False):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xyxy = xyxy

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0] # pred:[N,4] target:[N,4]

        pred = pred.view(-1, 4).float()
        target = target.view(-1, 4).float()
        if self.xyxy:
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

        if self.iou_type == "iou":
            loss = 1 - iou ** 2
        elif self.iou_type == "giou":
            if self.xyxy:
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
        elif self.iou_type == "diou":
            if self.xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)  # 包围框的左上点
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)  # 包围框的右下点
                )
            convex_dis = torch.pow(c_br[:, 0]-c_tl[:, 0], 2) + torch.pow(c_br[:, 1]-c_tl[:, 1], 2) + 1e-7 # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0]-target[:, 0], 2) + torch.pow(pred[:, 1]-target[:,1], 2))  # center diagonal squared

            diou = iou - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)

        elif self.iou_type == "ciou":
            if self.xyxy:
                c_tl = torch.min(pred[:, :2], target[:, :2])
                c_br = torch.max(pred[:, 2:], target[:, 2:])
            else:
                c_tl = torch.min(
                    (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
                )
                c_br = torch.max(
                    (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
                )
            convex_dis = torch.pow(c_br[:, 0]-c_tl[:, 0], 2) + torch.pow(c_br[:, 1]-c_tl[:,1], 2) + 1e-7  # convex diagonal squared
            center_dis = (torch.pow(pred[:, 0]-target[:, 0], 2) + torch.pow(pred[:, 1]-target[:,1], 2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min = 1e-7)) - 
                                                torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min = 1e-7)), 2)

            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)
            
            ciou = iou - (center_dis / convex_dis + alpha * v)
            
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        elif self.iou_type == 'siou':
            box1 = pred.T
            box2 = target.T
            if self.xyxy:
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            else:
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7

            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + 1e-7
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + 1e-7
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
            loss = 1.0 - iou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # print('loss:', loss[0], loss.shape)
        # print('ciou:', ciou[0], ciou.shape)
        # print('iou:', iou.shape)

        return loss

class ComputeNanoLoss:
    def __init__(self, model, cfg):
        super(ComputeNanoLoss, self).__init__()
        det = model.module.head if is_parallel(model) else model.head  # Detect() module
        self.det = det
        # self.loss_dict = {'loss_qfl':0, 'loss_bbox':0, 'loss_dfl':0}
    
    def __call__(self, p, targets):
        loss, loss_dict = self.det.get_losses(p, targets) 
        return loss, loss_dict

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
        return loss, loss_dict