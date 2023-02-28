import torch.nn as nn
from ..backbone.common import Conv, Concat, C2f
from utils.general import make_divisible


class YoloV8Neck(nn.Module):
    """
        This PAN refer to yolov8, there are many different versions of implementation, and the details will be different.
        默认的输出通道数设置成了yolov8L的输出通道数, 当backbone为YOLOV8时，会根据version对输出通道转为了YOLOv5 对应版本的输出。对于其他backbone，使用的默认值.


    P3 --->  PP3
    ^         |
    | concat  V
    P4 --->  PP4
    ^         |
    | concat  V
    P5 --->  PP5
    """

    # def __init__(self, input_p3=256, input_p4=512, input_p5=1024, output_p3=256, output_p4=512, output_p5=1024, version='S', act=''):
    def __init__(self, cfg):
        super(YoloV8Neck, self).__init__()
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

        self.input_p3 = self.channels['input_p3'] #256
        self.input_p4 = self.channels['input_p4'] #512
        self.input_p5 = self.channels['input_p5'] #768

        self.output_p3 = self.channels['output_p3'] #256
        self.output_p4 = self.channels['output_p4'] #512
        self.output_p5 = self.channels['output_p5'] #768

        if cfg.Model.Neck.activation == 'SiLU':
            CONV_ACT = 'silu'
            C_ACT = 'silu'
        elif cfg.Model.Neck.activation == 'ReLU':
            CONV_ACT = 'relu'
            C_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
            C_ACT = 'hard_swish'

        # print('self channels:', self.input_p5)
        # self.conv1 = Conv(self.input_p5, int(self.input_p5/2), 1, 1, None, 1, CONV_ACT)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest") #10
        self.C1 = C2f(self.input_p5 + self.input_p4, self.input_p4, self.get_depth(3), False, 1, 0.5, C_ACT) #12

        # self.conv2 = Conv(self.input_p4, self.input_p3, 1, 1, None, 1, CONV_ACT)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")#13
        self.C2 = C2f(self.input_p4 + self.input_p3, self.output_p3, self.get_depth(3), False, 1, 0.5, C_ACT) #15

        self.conv3 = Conv(self.output_p3 , self.output_p3, 3, 2, None, 1, CONV_ACT)
        self.C3 = C2f(self.output_p3 + self.input_p4, self.output_p4, self.get_depth(3), False, 1, 0.5, C_ACT) #20

        self.conv4 = Conv(self.output_p4, self.output_p4, 3, 2, None, 1, CONV_ACT)
        self.C4 = C2f(self.output_p4 + self.input_p5, self.output_p5, self.get_depth(3), False, 1, 0.5, C_ACT) #23

        self.concat = Concat()

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
        # print('P3:', P3.shape)
        # print('P4:', P4.shape)
        # print('P5:', P5.shape)
        x1 = self.upsample1(P5) #10
        x1 = self.concat([x1, P4]) #11
        x1 = self.C1(x1) # 12

        x2 = self.upsample2(x1) #13
        x2 = self.concat([x2, P3]) #14
        x2 = self.C2(x2) # 15

        x3 = self.conv3(x2) #16
        x3 = self.concat([x3, x1]) #17
        x3 = self.C3(x3) #18

        x4 = self.conv4(x3) #19
        x4 = self.concat([x4, P5]) #20
        x4 = self.C4(x4) #21
       
        return [x2, x3, x4]
