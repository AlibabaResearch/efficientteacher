#Copyright (c) 2023, Alibaba Group
import torch
import sys
import os
from copy import deepcopy
from pathlib import Path
from models.detector.yolo import Model
from configs.defaults import get_cfg
file = Path(__file__).resolve()
parent, root, root1 = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(parent))


#model = attempt_load(weights, map_location=device)
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
        #cfg.merge_from_file('/mnt/bowen/EfficientTeacher/configs/public/yolov5s_coco.yaml')
        cfg.merge_from_file(cfg_path)
        model = Model(cfg)
        # mula_pt = save_path
        print('load weights from u-yolov5...')
        model.load_state_dict(new_yolov5s_weight,strict=False)

        ckpt['model'] = deepcopy(model)
        ckpt['ema'] = deepcopy(model)
        torch.save(ckpt, save_path)
    


def convert_efficient_to_yolov5(mula_path='',  yolov5_path='', save_path='', map_path='map.txt'):
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
        ckpt = torch.load(mula_path, map_location='cpu')
        mula_weight = ckpt['model'].state_dict()
        mula_keys = list(mula_weight.keys())
        # self.model.load_state_dict(self.new_yolov5s_weight,strict=False)
        ori_yolov5s_weight = {}
        print(match_dict)
        for mula_key in mula_keys:
            mula_key_prefix = '.'.join(mula_key.split('.')[:2])
            mula_key_suffix = '.'.join(mula_key.split('.')[2:])
            if 'anchors' in mula_key_prefix:
                continue
            if 'm' in mula_key_prefix:
                continue
            print('model_match_key:', mula_key_prefix)
            model_match_key = match_dict[mula_key_prefix] + '.' + mula_key_suffix
            # print('mula_key:', mula_weight[mula_key])
        # print('yolov5_key:', yolov5_key)
            ori_yolov5s_weight[model_match_key] = mula_weight[mula_key] 
        yolov5_model.load_state_dict(ori_yolov5s_weight, strict=False)

        ckpt['model'] = deepcopy(yolov5_model)
        ckpt['ema'] = deepcopy(yolov5_model)
        torch.save(ckpt, save_path)

if __name__ == '__main__':
    # convert_efficient_to_yolov5('/mnt/bowen/mula-yolov5s.pt', map_path='map.txt', yolov5_path='/mnt/bowen/yolov5s.pt', save_path='/mnt/bowen/test.pt')
    convert_yolov5_to_efficient( '/mnt/bowen/yolov5s.pt', 'mnt/bowen/EfficientTeacher/configs/public/yolov5s_coco.yaml','/mnt/bowen/efficient-yolov5s.pt')