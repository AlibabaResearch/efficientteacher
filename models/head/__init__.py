import copy


from .yolov5_head import Detect
from .yolov7_head import IDetect
from .yolov8_head import YoloV8Detect
from .yolox_head import YoloXDetect
from .yolov6_head import YoloV6Detect
# from .simple_conv_head import SimpleConvHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    # name = head_cfg.pop("name")
    name = head_cfg.Model.Head.name
    if name == "YoloX":
        return YoloXDetect(head_cfg)
    elif name == "YoloV5":
        return Detect(head_cfg)
    elif name == "YoloV6":
        return YoloV6Detect(head_cfg)
    elif name == "YoloV7":
        return IDetect(head_cfg)
    elif name == "YoloV8":
        return YoloV8Detect(head_cfg)
    else:
        raise NotImplementedError
