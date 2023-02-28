# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from .yolov5_neck import YoloV5Neck
from .yolov6_neck import YoloV6Neck
from .yolov7_neck import YoloV7Neck
from .yolov8_neck import YoloV8Neck


def build_neck(cfg,nas_arch=None,in_channels=None):
    fpn_cfg = copy.deepcopy(cfg)
    # name = fpn_cfg.pop("name")
    # name = "YoloV5"
    name = cfg.Model.Neck.name
    if name == "YoloV5":
        return YoloV5Neck(fpn_cfg)
    elif name == "YoloV6":
        return YoloV6Neck(fpn_cfg)
    elif name == "YoloV7":
        return YoloV7Neck(fpn_cfg)
    elif name == "YoloV8":
        return YoloV8Neck(fpn_cfg)
    # elif name == "YoloV5Ori":
        # return YoloV5Neck(fpn_cfg)
    else:
        raise NotImplementedError
