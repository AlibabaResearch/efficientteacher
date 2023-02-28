import copy

# from .gfl_head import GFLHead
# from .nanodet_head import NanoDetHead
# from .nanodet_plus_head import NanoDetPlusHead
from .yolov5_head import Detect
# from .yolox_kp_head import DetectYoloXKeypoints
# from .effidehead import EfficientDetect
# from .yolov7_head import IDetect
# from .yolox_head import DetectYoloX
# from .ppyoloe_head import PPYOLOEHead
# from .simple_conv_head import SimpleConvHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    # name = head_cfg.pop("name")
    name = head_cfg.Model.Head.name
    if name == "YoloV5":
        return Detect(head_cfg)
    else:
        raise NotImplementedError
