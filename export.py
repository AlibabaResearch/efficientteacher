"""
Export a EfficientTeacher PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit
Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU
Usage:
    $ python path/to/export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...
Inference:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile
from models.backbone.common import Conv
from utils.downloads import attempt_download
from deploy.model_convert import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # EfficientTeacher root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#from models.backbone.experimental import attempt_load
from models.head.yolov5_head import Detect
# from models.head.yolox_kp_head import DetectYoloXKeypoints
from models.head.yolox_head import YoloXDetect
# from models.head.effidehead import EfficientDetect
from models.head.yolov6_head import YoloV6Detect
from models.head.yolov8_head import YoloV8Detect
from models.head.retina_head import RetinaDetect
#from utils.dataloaders import LoadImages
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
                           file_size, print_args, url2file)
from utils.torch_utils import select_device


def export_formats():
    # EfficientTeacher export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],
        ['Cambricon Magicmind', 'magicmind', '.magicmind', False, False], ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            #y.append(module(x, augment, profile, visualize)[0])
            y.append(module(x, augment, profile)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        #print(y)
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def load_model(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.detector.yolo import Detect, Model
    import models

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # OP change
    from utils.activations import Hardswish, SiLU
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, models.backbone.common.Conv): 
             if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
             elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    
    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # return ensemble


@torch.no_grad()
def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        train=False,  # model.train() mode
        keras=False,  # use Keras
        optimize=False,  # TorchScript: optimize for mobile
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=False,  # ONNX/TF: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        nms=False,  # TF: add NMS to model
        agnostic_nms=False,  # TF: add agnostic NMS to model
        topk_per_class=100,  # TF.js NMS: topk per class to keep
        topk_all=100,  # TF.js NMS: topk for all classes to keep
        iou_thres=0.45,  # TF.js NMS: IoU threshold
        conf_thres=0.25,  # TF.js NMS: confidence threshold
        da = False, #if model is from da training
        bgrd = False, #if convert 3 input channel to 4 input channel
        bgr2rgb= False,
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    # print(fmts)
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, magicmind = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = load_model(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    names =  model.names  # number of classes, class names
    if bgrd:
       bgr2rgbd_first_conv(model) 
    if bgr2rgb:
       bgr2rgb_first_conv(model)
    
    # append_yolox_output_depthwise_conv(model)
    #nc = model.nc

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    #assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    if bgrd:
        im = torch.zeros(batch_size, 4, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    else:
        im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
        if isinstance(m, RetinaDetect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
        elif isinstance(m, YoloXDetect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
        elif isinstance(m, YoloV6Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
        elif isinstance(m, YoloV8Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
    if da:
        model.forward = model.forward_export
        del model.det_8
        del model.det_16
        del model.det_32
        print(model)
    for _ in range(2):
        # if da:
        #     y, feat = model(im)
        # else:
            y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    #print(y)
    # shape = tuple(y[0].shape)  # model output shape
    # LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")


    # Exports
    f = [''] * 10  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:
        f[0] = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1] = export_engine(model, im, file, train, half, simplify, workspace, verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2] = export_onnx(model, im, file, opset, train, dynamic, simplify)
    if xml:  # OpenVINO
        f[3] = export_openvino(model, file, half)
    if coreml:
        _, f[4] = export_coreml(model, im, file, int8, half)

    # TensorFlow Exports
    if any((saved_model, pb, tflite, edgetpu, tfjs)):
        if int8 or edgetpu:  # TFLite --int8 bug https://github.com/ultralytics/yolov5/issues/5707
            check_requirements(('flatbuffers==1.12',))  # required before `import tensorflow`
        assert not tflite or not tfjs, 'TFLite and TF.js models must be exported separately, please pass only one type.'
        model, f[5] = export_saved_model(model.cpu(),
                                         im,
                                         file,
                                         dynamic,
                                         tf_nms=nms or agnostic_nms or tfjs,
                                         agnostic_nms=agnostic_nms or tfjs,
                                         topk_per_class=topk_per_class,
                                         topk_all=topk_all,
                                         iou_thres=iou_thres,
                                         conf_thres=conf_thres,
                                         keras=keras)
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6] = export_pb(model, file)
        if tflite or edgetpu:
            f[7] = export_tflite(model, im, file, int8=int8 or edgetpu, data=data, nms=nms, agnostic_nms=agnostic_nms)
        if edgetpu:
            f[8] = export_edgetpu(file)
        if tfjs:
            f[9] = export_tfjs(file)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = '--half' if half else ''  # --half FP16 inference arg
        LOGGER.info(f'\nExport complete ({time.time() - t:.2f}s)'
                    f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                    f"\nDetect:          python detect.py --weights {f[-1]} {h}"
                    f"\nValidate:        python val.py --weights {f[-1]} {h}"
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
                    f"\nVisualize:       https://netron.app")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--keras', action='store_true', help='TF: use Keras')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=9, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument('--da', action='store_true', help='if model is from da training, trigger it')
    parser.add_argument('--bgrd', action='store_true', help='conver 3 channel input to bgrd')
    parser.add_argument('--bgr2rgb', action='store_true', help='conver bgr input to rgb')
    parser.add_argument('--include',
                        nargs='+',
                        default=['torchscript', 'onnx'],
                        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
    opt = parser.parse_args()
    #print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)