from torch import nn
from models.backbone.common import LinearAddBlock, RepVGGBlock, RepBlock, SimSPPF,RealVGGBlock, QARepVGGBlock
from utils.general import make_divisible


class YoloV6BackBone(nn.Module):
    """EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """

    def __init__(
        self,
        cfg
    ):
        super().__init__()

        in_channels = cfg.Model.Backbone.in_channels
        depth_mul = cfg.Model.depth_multiple
        width_mul = cfg.Model.width_multiple
        num_repeat_backbone = cfg.Model.Backbone.num_repeats
        channels_list_backbone = cfg.Model.Backbone.out_channels
        num_repeats = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone)]
        channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone)]

        assert channels_list is not None
        assert num_repeats is not None

        if cfg.Model.RealVGGModel:
            block = RealVGGBlock
        elif cfg.Model.QARepVGGModel:
            block = QARepVGGBlock
        elif cfg.Model.LinearAddModel:
            block = LinearAddBlock
        else:
            block = RepVGGBlock
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block

            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block

            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block

            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)
