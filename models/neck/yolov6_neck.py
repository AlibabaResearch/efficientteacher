import torch
from torch import nn
# from yolov6.layers.common import RepBlock, SimConv, Transpose
from models.backbone.common import LinearAddBlock, SimConv, Transpose
from models.backbone.common import RepVGGBlock, RepBlock, RealVGGBlock
from utils.general import make_divisible

class YoloV6Neck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(
        self,
        cfg
    ):
        super().__init__()

        depth_mul = cfg.Model.depth_multiple
        width_mul = cfg.Model.width_multiple
        channels_list_backbone = cfg.Model.Backbone.out_channels
        num_repeat_backbone = cfg.Model.Backbone.num_repeats
        num_repeat_neck = cfg.Model.Neck.num_repeats
        channels_list_neck = cfg.Model.Neck.out_channels
        num_repeats = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
        channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

        assert channels_list is not None
        assert num_repeats is not None

        if cfg.Model.RealVGGModel:
            block = RealVGGBlock
        elif cfg.Model.LinearAddModel:
            block = LinearAddBlock
        else:
            block = RepVGGBlock

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[5],
            block=block
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[6],
            block=block

        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[5] + channels_list[5],
            out_channels=channels_list[6],
            n=num_repeats[7],
            block=block
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[6] + channels_list[6],
            out_channels=channels_list[7],
            n=num_repeats[8],
            block=block
        )

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5]
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
            kernel_size=3,
            stride=2
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=2
        )

    def forward(self, input):

        (x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs
