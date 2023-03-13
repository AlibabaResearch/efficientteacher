import torch.nn as nn
from ..backbone.common import ELAN_NECK, Conv, Concat, SPPCSPC, MP, RepConv
from utils.general import make_divisible


class YoloV7Neck(nn.Module):
    """
    P3 --->  ELAN_NECK
    ^         |
    | concat  V
    P4 --->  ELAN_NECK
    ^         |
    | concat  V
    P5 --->  ELAN_NECK
    """

    # def __init__(self, input_p3=256, input_p4=512, input_p5=1024, output_p3=256, output_p4=512, output_p5=1024, version='S', act=''):
    def __init__(self, cfg):
        super(YoloV7Neck, self).__init__()
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        input_p3, input_p4, input_p5 = cfg.Model.Neck.in_channels
        output_p3, output_p4, output_p5 = cfg.Model.Neck.out_channels

        self.channels = {
            'input_p3': input_p3,
            'input_p4': input_p4,
            'input_p5': input_p5,
            'output_p3': output_p3,
            'output_p4': output_p4,
            'output_p5': output_p5,
        }
        self.re_channels_out()

        self.input_p3 = self.channels['input_p3']
        self.input_p4 = self.channels['input_p4']
        self.input_p5 = self.channels['input_p5']

        self.output_p3 = self.channels['output_p3']
        self.output_p4 = self.channels['output_p4']
        self.output_p5 = self.channels['output_p5']

        if cfg.Model.Neck.activation == 'SiLU':
            CONV_ACT = 'silu'
        elif cfg.Model.Neck.activation == 'ReLU':
            CONV_ACT = 'relu'
        elif cfg.Model.Neck.activation == 'LeakyReLU': 
            CONV_ACT = 'lrelu'
        else:
            CONV_ACT = 'hard_swish'

        self.c_0 = int(self.input_p5/2) #512
        self.c_1 = int(self.input_p5/4) #256
        self.c_2 = int(self.input_p5/8) #128
        self.c_3 = int(self.input_p5/16) #64

        # print('self channels:', self.input_p5)
        self.sppcspc = SPPCSPC(self.input_p5, self.c_0)
        self.conv1 = Conv(self.c_0, self.c_1, 1, 1, None, 1, CONV_ACT)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2 = Conv(self.input_p4, self.c_1, 1, 1, None, 1, CONV_ACT)

        self.elan_0 = ELAN_NECK(self.c_0, self.c_1, 3, 0.5, 0.5, CONV_ACT)

        self.conv10 = Conv(self.c_1, self.c_2, 1, 1, None, 1, CONV_ACT) #64
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv11 = Conv(self.c_0, self.c_2, 1, 1, None, 1, CONV_ACT) #66

        self.elan_1 = ELAN_NECK(self.c_1, self.c_2, 3, 0.5, 0.5, CONV_ACT)

        self.conv19 = Conv(self.c_2, self.c_2, 1, 1, None, 1, CONV_ACT) #77
        self.conv20 = Conv(self.c_2, self.c_2, 1, 1, None, 1, CONV_ACT) #78
        self.conv21 = Conv(self.c_2, self.c_2, 3, 2, None, 1, CONV_ACT) #79

        self.elan_2 = ELAN_NECK(self.c_0, self.c_1, 3, 0.5, 0.5, CONV_ACT)

        self.conv29 = Conv(self.c_1, self.c_1, 1, 1, None, 1, CONV_ACT) #90
        self.conv30 = Conv(self.c_1, self.c_1, 1, 1, None, 1, CONV_ACT) #91
        self.conv31 = Conv(self.c_1, self.c_1, 3, 2, None, 1, CONV_ACT) #92

        self.elan_3 = ELAN_NECK(self.input_p4, self.c_0, 3, 0.5, 0.5, CONV_ACT)

        self.repconv0 = RepConv(self.c_2, self.output_p3, 3, 1, None, 1, CONV_ACT)
        self.repconv1 = RepConv(self.c_1, self.output_p4, 3, 1, None, 1, CONV_ACT)
        self.repconv2 = RepConv(self.c_0, self.output_p5, 3, 1, None, 1, CONV_ACT)

        self.concat = Concat()
        self.mp = MP()

        # print("PAN input channel size: P3 {}, P4 {}, P5 {}".format(self.input_p3, self.input_p4, self.input_p5))
        # print("PAN output channel size: PP3 {}, PP4 {}, PP5 {}".format(self.output_p3, self.output_p4, self.output_p5))

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def forward(self, inputs):
        P3, P4, P5 = inputs
        x0 = self.sppcspc(P5) #51
        x1 = self.conv1(x0)
        x2 = self.upsample1(x1)
        x3 = self.conv2(P4) # 54
        x4 = self.concat([x3, x2])

        x12 = self.elan_0(x4)

        x13 = self.conv10(x12)

        x14 = self.upsample2(x13)
        x15 = self.conv11(P3)
        x16 = self.concat([x15, x14]) #67

        x24 = self.elan_1(x16)

        x25 = self.mp(x24)
        x26 = self.conv19(x25)
        x27 = self.conv20(x24) #78
        x28 = self.conv21(x27) #79
        x29 = self.concat([x28, x26, x12]) #80

        x37 = self.elan_2(x29)

        x38 = self.mp(x37)
        x39 = self.conv29(x38) #90
        x40 = self.conv30(x37) #91
        x41 = self.conv31(x40) #92
        x42 = self.concat([x41, x39, x0]) #93

        x50 = self.elan_3(x42)

        x51= self.repconv0(x24)
        x52= self.repconv1(x37)
        x53= self.repconv2(x50)

        return x51, x52, x53