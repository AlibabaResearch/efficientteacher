import torch
import torch.nn as nn
# from utils.autoanchor import check_anchor_order
# from utils.general import check_version

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

    def forward(self, x):
        z = []  # inference output
        x0, x1, x2 = x
        list_x = [x0, x1, x2]
        x = list_x
        class_range = list(range(5 + self.nc))
        for i in range(self.nl):
            # print(self.m[i])
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
            return z

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