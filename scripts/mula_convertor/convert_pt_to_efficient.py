#Copyright (c) 2023, Alibaba Group
import torch
import sys
import os
from copy import deepcopy
from pathlib import Path
from models.detector.yolo import Model
from configs.defaults import get_cfg
import torch.nn as nn

file = Path(__file__).resolve()
parent, root, root1 = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(parent))


def convert_yolov5_to_efficient(pt_path='', cfg_path='', save_path='', map_path='map.txt'):
        with open(map_path, 'r') as f:
            content = f.readlines()
        content = [i.strip().split(' ') for i in content]
        match_dict = {}
        for i in content:
            value = i[0]
            key = i[1]
            match_dict[key] = value
        ckpt = torch.load(pt_path, map_location='cpu')

        yolov5_weight = ckpt['model'].state_dict()
        yolov5_keys = list(yolov5_weight.keys())

        new_yolov5s_weight = {}
        for yolov5_key in yolov5_keys:
            yolov5_key_prefix = '.'.join(yolov5_key.split('.')[:2])
            yolov5_key_suffix = '.'.join(yolov5_key.split('.')[2:])
            model_match_key = match_dict[yolov5_key_prefix] + '.' + yolov5_key_suffix
        # print('model_match_key:', model_match_key)
        # print('yolov5_key:', yolov5_key)
            new_yolov5s_weight[model_match_key] = yolov5_weight[yolov5_key] 

        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        model = Model(cfg)
        print('load weights from u-yolov5...')
        model.load_state_dict(new_yolov5s_weight,strict=False)

        ckpt['model'] = deepcopy(model.half())
        ckpt['ema'] = ckpt['model']
        ckpt['updates'] = 0
        torch.save(ckpt, save_path)
    


def convert_efficient_to_yolov5(efficient_path='',  yolov5_path='', save_path='', map_path='map_v5.txt'):
        with open(map_path, 'r') as f:
            content = f.readlines()
        content = [i.strip().split(' ') for i in content]
        match_dict = {}
        for i in content:
            value = i[1]
            key = i[0]
            match_dict[key] = value
        print('load weights from u-yolov5...')
        yolov5_ckpt = torch.load(yolov5_path, map_location='cpu')
        yolov5_model = yolov5_ckpt['model']
        print('load weights from EfficientTeacher...')
        ckpt = torch.load(efficient_path, map_location='cpu')
        efficient_weight = ckpt['model'].state_dict()
        efficient_keys = list(efficient_weight.keys())
        ori_yolov5_weight = {}
        print(match_dict)
        for efficient_key in efficient_keys:
            key_prefix = '.'.join(efficient_key.split('.')[:2])
            key_suffix = '.'.join(efficient_key.split('.')[2:])
            if 'anchors' in key_prefix:
                model_match_key = 'model.24.anchors'
                efficient_key = 'head.anchors'
            else:
                if key_prefix in ['det_8.conv1', 'det_8.conv2', 'det_16.conv1', 'det_16.conv2', 'det_32.conv1', 'det_32.conv2']:
                    continue
                model_match_key = match_dict[key_prefix] + '.' + key_suffix
            ori_yolov5_weight[model_match_key] = efficient_weight[efficient_key] 

        #special process for detect head
        ch = [efficient_weight['head.m.0.weight'].shape[1], efficient_weight['head.m.1.weight'].shape[1], efficient_weight['head.m.2.weight'].shape[1]]

        detect_head = yolov5_model.model[-1]
        detect_head.no = ckpt['model'].head.no
        detect_head.na = ckpt['model'].head.na
        detect_head.m = nn.ModuleList(nn.Conv2d(x, detect_head.no * detect_head.na, 1) for x in ch) 
        detect_head.nc = ckpt['model'].head.nc
        detect_head.anchors = ckpt['model'].head.anchors
        detect_head.names = ckpt['model'].names
        

        yolov5_model.load_state_dict(ori_yolov5_weight, strict=False)

        yolov5_ckpt['ema'] = deepcopy(yolov5_model)
        yolov5_ckpt['model'] = yolov5_ckpt['ema']
        # ckpt['updates'] = 0
        torch.save(yolov5_ckpt, save_path)

if __name__ == '__main__':
    # convert_yolov5_to_efficient( 'yolov5s.pt', 'efficientteacher/configs/sup/public/yolov5s_coco.yaml','efficient-yolov5s.pt')
    convert_efficient_to_yolov5('efficient-yolov5m.pt', yolov5_path='yolov5m.pt', save_path='test.pt')