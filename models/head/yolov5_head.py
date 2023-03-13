import torch
import torch.nn as nn
import math
# from utils.autoanchor import check_anchor_order
from utils.general import check_version

class Detect(nn.Module):
    stride = None  # strides computed during build

    # def __init__(self, nc=80, anchors=(), ch=(), np=0):  # detection layer
    def __init__(self, cfg):  # detection layer
        super(Detect, self).__init__()
        self.nc = cfg.Dataset.nc  # number of classes
        self.num_keypoints = cfg.Dataset.np
        self.cur_imgsize = [cfg.Dataset.img_size,cfg.Dataset.img_size]
        anchors = cfg.Model.anchors
        # print('anchors:', anchors)
        ch = []
        for out_c in cfg.Model.Neck.out_channels:
            ch.append(int(out_c * cfg.Model.width_multiple))
        # print('ch:', ch)
        self.no = self.nc + self.num_keypoints + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.prune = False
        self.stride = cfg.Model.Head.strides
        # self.qat_export = False
        self.export = False
    
    def initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.backbone.model[-1]  # Detect() module
        # m = self.head  # Detect() module
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []  # inference output
        list_x = []
        for _ in x:
            list_x.append(_)
        x = list_x
        class_range = list(range(5 + self.nc))
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv

            bs, _, ny, nx = x[i].shape
            if hasattr(self, 'export') and self.export:
                # z.append(x[i])
                x[i] = x[i].view(bs, self.na, self.no, -1).permute(0, 1, 3, 2)  # .contiguous() #(bs, 3, 20*20, 85)
                # z.append((x[i], ny, nx))
                z.append(x[i])
                continue

            # bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid_old(nx, ny, i)

                y = torch.full_like(x[i], 0)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                z.append(y.view(bs, -1, self.no))

        # if hasattr(self, 'prune') and self.prune:
        #     return x
        # print(self.anchor_grid)
        if hasattr(self, 'export') and self.export:
            #return tuple(z) #for dp1000 export
            return z   # normal export

        return x if self.training  else (torch.cat(z, 1), x)

    def post_process(self, x):
        z = []  # inference output
        for i in range(self.nl):
            ny,nx = int(self.cur_imgsize[0]/self.stride[i]),int(self.cur_imgsize[1]/self.stride[i])
            bs, _, nyxnx, _ = x[i].shape  # x(bs, 3, 20*20, 85)
            x[i] = x[i].view(bs, self.na, ny, nx, self.no).contiguous()  # (bs, 3, 20, 20, 85)

            if not self.training:  # inference
                # if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid_old(nx, ny, i)

                y = torch.full_like(x[i], 0)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                class_range = list(range(5+self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i] # wh
                z.append(y.view(bs, -1, self.no))
        return x if self.training else [torch.cat(z, 1), x]

    def post_process_v2(self, x):
        z = []  
        self.grid[0], self.anchor_grid[0] = self._make_grid(80,80, 0)
        self.grid[1], self.anchor_grid[1] = self._make_grid(40,40, 1)
        self.grid[2], self.anchor_grid[2] = self._make_grid(20,20, 2)
        for i in range(self.nl):
            ny,nx = int(self.cur_imgsize[0]/self.stride[i]),int(self.cur_imgsize[1]/self.stride[i])
            bs, _, nyxnx, _ = x[i].shape  # x(bs, 3, 20*20, 85)
            x[i] = x[i].view(bs, self.na, ny, nx, self.no).contiguous()  # (bs, 3, 20, 20, 85)
            y = x[i].sigmoid()
            xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, -1, self.no))
        return (torch.cat(z, 1), x)

    # @staticmethod
    def _make_grid_old(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # print(self.anchors[i])
        # print(self.stride[i])
        # print((self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).shape)
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def _make_grid(self, nx=20, ny=20, i=0):
        na = 3
        anchors = torch.Tensor([[[ 1.25000,  1.62500],
                    [ 2.00000,  3.75000],
                    [ 4.12500,  2.87500]],

                   [[ 1.87500,  3.81250],
                    [ 3.87500,  2.81250],
                    [ 3.68750,  7.43750]],

                   [[ 3.62500,  2.81250],
                    [ 4.87500,  6.18750],
                    [11.65625, 10.18750]]])
        stride = torch.Tensor([ 8., 16., 32.])
        #print("self.na",self.na,self.anchors[i],self.stride[i])
        d = anchors[i].device
        t = anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (anchors[i] * stride[i]).view((1, na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid