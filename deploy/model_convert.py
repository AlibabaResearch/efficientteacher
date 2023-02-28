from code import compile_command
import datetime
import torch
import torch.nn as nn
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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#from models.backbone.experimental import attempt_load
# from deploy.load_model import load_model
from models.head.yolov5_head import Detect
#from utils.dataloaders import LoadImages
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_version, colorstr,
                           file_size, print_args, url2file)
from utils.torch_utils import select_device

def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    try:
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_onnx(model, im, file, opset, train, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    try:
        check_requirements(('onnx',))
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_openvino(model, file, half, prefix=colorstr('OpenVINO:')):
    # YOLOv5 OpenVINO export
    try:
        check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie

        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(file).replace('.pt', f'_openvino_model{os.sep}')

        cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
        subprocess.check_output(cmd.split())  # export
        with open(Path(f) / file.with_suffix('.yaml').name, 'w') as g:
            yaml.dump({'stride': int(max(model.stride)), 'names': model.names}, g)  # add metadata.yaml

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_coreml(model, im, file, int8, half, prefix=colorstr('CoreML:')):
    # YOLOv5 CoreML export
    try:
        check_requirements(('coremltools',))
        import coremltools as ct

        LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = file.with_suffix('.mlmodel')

        ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (32, None)
        if bits < 32:
            if platform.system() == 'Darwin':  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                print(f'{prefix} quantization only supported on macOS, skipping...')
        ct_model.save(f)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return ct_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None


def export_engine(model, im, file, train, half, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    try:
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        try:
            import tensorrt as trt
        except Exception:
            if platform.system() == 'Linux':
                check_requirements(('nvidia-tensorrt',), cmds=('-U --index-url https://pypi.ngc.nvidia.com',))
            import tensorrt as trt

        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, train, False, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
            export_onnx(model, im, file, 13, train, False, simplify)  # opset 13
        onnx = file.with_suffix('.onnx')

        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info(f'{prefix} Network Description:')
        for inp in inputs:
            LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {f}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix=colorstr('TensorFlow SavedModel:')):
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        from models.tf import TFDetect, TFModel

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(file).replace('.pt', '_saved_model')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW

        tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
        _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
        outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        keras_model.summary()
        if keras:
            keras_model.save(f, save_format='tf')
        else:
            spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
            m = tf.function(lambda x: keras_model(x))  # full model
            m = m.get_concrete_function(spec)
            frozen_func = convert_variables_to_constants_v2(m)
            tfm = tf.Module()
            tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x)[0], [spec])
            tfm.__call__(im)
            tf.saved_model.save(tfm,
                                f,
                                options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
                                if check_version(tf.__version__, '2.6') else tf.saved_model.SaveOptions())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return keras_model, f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')
        return None, None


def export_pb(keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLOv5 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    try:
        import tensorflow as tf

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW
        f = str(file).replace('.pt', '-fp16.tflite')

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            from models.tf import representative_dataset_gen
            dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative assets
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True
            f = str(file).replace('.pt', '-int8.tflite')
        if nms or agnostic_nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

        tflite_model = converter.convert()
        open(f, "wb").write(tflite_model)
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_edgetpu(file, prefix=colorstr('Edge TPU:')):
    # YOLOv5 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    try:
        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
        if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
            LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
            sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
            for c in (
                    'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                    'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                    'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
        f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model

        cmd = f"edgetpu_compiler -s -o {file.parent} {f_tfl}"
        subprocess.run(cmd.split(), check=True)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')


def export_tfjs(file, prefix=colorstr('TensorFlow.js:')):
    # YOLOv5 TensorFlow.js export
    try:
        check_requirements(('tensorflowjs',))
        import re

        import tensorflowjs as tfjs

        LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
        f = str(file).replace('.pt', '_web_model')  # js dir
        f_pb = file.with_suffix('.pb')  # *.pb path
        f_json = f'{f}/model.json'  # *.json path

        cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
              f'--output_node_names=Identity,Identity_1,Identity_2,Identity_3 {f_pb} {f}'
        subprocess.run(cmd.split())

        with open(f_json) as j:
            json = j.read()
        with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
            subst = re.sub(
                r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}}}', r'{"outputs": {"Identity": {"name": "Identity"}, '
                r'"Identity_1": {"name": "Identity_1"}, '
                r'"Identity_2": {"name": "Identity_2"}, '
                r'"Identity_3": {"name": "Identity_3"}}}', json)
            j.write(subst)

        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')

# TorchScript export
def pytorch2torchscript(model, img, opt ) :
    print('\n Starting TorchScript export with torch %s...' % torch.__version__)
    save_path = opt.save_dir + '/' + opt.export_model_name + '_' + str(datetime.date.today()) + '.torchscript.pt'
    ts = torch.jit.trace(model, img)
    ts.save(save_path)
    print('TorchScript export success, saved as %s' % save_path)
    return save_path

# ONNX export
def pytorch2onnx(model, img, opt):
    import onnx
    y = model(img)  # dry run
    print('\nStarting ONNX export with onnx {} opeset version {} ...'.format(onnx.__version__, opt.onnx_opset_version ))
    save_path = opt.save_dir + '/' + opt.export_model_name + '_' + str(datetime.date.today()) + '.onnx'
    torch.onnx.export(model, img, save_path, verbose=False, opset_version=opt.onnx_opset_version, input_names=['images'],
                        output_names=['classes', 'boxes'] if y is None else ['output'],
                        dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                    'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)

    onnx_model = onnx.load(save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    #print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % save_path)
    return save_path

# DP1000 export
def pytorch2dp1000(model, img, opt):
    from thirdparty.PytorchSimulatorForDP1000.simulator import DP1000Simulator
    import os
    print('\n Starting DP1000 export with torch ...')
    save_path = opt.save_dir + '/' + opt.export_model_name + '_' + str(datetime.date.today()) + '/'
    is_exist = os.path.exists(save_path)
    if is_exist:
        os.makedirs(save_path)
    net_int8 = DP1000Simulator()
    relay_func_json = "./relay_func.json"
    net_int8.quantify_to_json(model, opt.config, relay_func_json)
    net_int8.initialize_from_json(relay_func_json)
    net_int8.serialize_for_dp1000(relay_func_json, save_path )
    print('DP1000 export success, saved as %s' % save_path)



# openvivo export
def pytorch2openvivo(model, img, opt):
    import os
    print('\n Starting openvivo export with onnx ...')
    onnx_path = pytorch2onnx(model, img, opt)
    compile_command  = 'python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model {} --disable_omitting_optional --data_type=FP16 --scale={} --log_level DEBUG --reverse_input_channels'.format(
        onnx_path, 255 )
    print("openvivo compile cmd: \n\n\n", compile_command)
    os.system(compile_command)



# utils function for model transform
def rgb2bgr_first_conv(model, opt):
    with torch.no_grad():
        print("model_transform: rgb2bgr_first_conv ")
        for k, m in model.named_modules():
                if isinstance(m, nn.Conv2d):  # assign export-friendly activations
                    # m.in_channels = 3
                    print("first conv:", m)
                    conv_st_dict = m.state_dict()
                    W = conv_st_dict['weight'] 
                    fout,fin,kx,ky = W.shape   #
                    W_adj = nn.Parameter(torch.zeros(fout,m.in_channels,kx,ky) )
                    for i in range(W.shape[0]):
                        r = W[i][0].clone()
                        g = W[i][1].clone()
                        b = W[i][2].clone()
                        conv_st_dict['weight'][i][0] = b
                        conv_st_dict['weight'][i][1] = g
                        conv_st_dict['weight'][i][2] = r 
            
                    W_adj[:,:fin,:,:] = conv_st_dict['weight']
                    conv_st_dict['weight'] = W_adj
                    m.weight.data = W_adj
                    print(conv_st_dict['weight'].shape)
                    break


def bgr2rgb_first_conv(model, opt):
    with torch.no_grad():
        print("model_transform: bgr2rgb_first_conv ")
        for k, m in model.named_modules():
                if isinstance(m, nn.Conv2d):  # assign export-friendly activations
                    # m.in_channels = 3
                    print("first conv:", m)
                    conv_st_dict = m.state_dict()
                    W = conv_st_dict['weight'] 
                    fout,fin,kx,ky = W.shape   #
                    W_adj = nn.Parameter(torch.zeros(fout,m.in_channels,kx,ky) )
                    for i in range(W.shape[0]):
                        b = W[i][0].clone()
                        g = W[i][1].clone()
                        r = W[i][2].clone()
                        conv_st_dict['weight'][i][0] = r
                        conv_st_dict['weight'][i][1] = g
                        conv_st_dict['weight'][i][2] = b
            
                    W_adj[:,:fin,:,:] = conv_st_dict['weight']
                    conv_st_dict['weight'] = W_adj
                    m.weight.data = W_adj
                    print(conv_st_dict['weight'].shape)
                    break

def bgr2rgbd_first_conv(model, opt=None):
      for k, m in model.named_modules():
          if isinstance(m, nn.Conv2d):  # assign export-friendly activations
              m.in_channels = 4
              conv_st_dict = m.state_dict()
              W = conv_st_dict['weight']
              fout,fin,kx,ky = W.shape   # torch.Size([32, 3, 3, 3])
              W_adj = nn.Parameter(torch.zeros(fout,4,kx,ky) )
              for i in range(W.shape[0]):
                  r = W[i][0].clone()
                  g = W[i][1].clone()
                  b = W[i][2].clone()
                  # conv_st_dict['weight'][i][0] = b/255.0
                  # conv_st_dict['weight'][i][1] = g/255.0
                  # conv_st_dict['weight'][i][2] = r/255.0
 
                  conv_st_dict['weight'][i][0] = b/1.0
                  conv_st_dict['weight'][i][1] = g/1.0
                  conv_st_dict['weight'][i][2] = r/1.0
                  print('r:', r)
                  print('g:', g)
                  print('b:', b)
 
              W_adj[:,:fin,:,:] = conv_st_dict['weight']
              conv_st_dict['weight'] = W_adj
              m.weight.data = W_adj
              print(conv_st_dict['weight'].shape)
 
              break
 
      for k, m in model.named_modules():
          if isinstance(m, nn.Conv2d):  # assign export-friendly activations
              print("first conv:", m)
              conv_st_dict = m.state_dict()
              W = conv_st_dict['weight']
              print("W:", W.shape)
              break


def fuse_scale_first_conv(model, opt):
    with torch.no_grad():
        print("model_transform: fuse_scale_first_conv ")
        for k, m in model.named_modules():
                if isinstance(m, nn.Conv2d):  # assign export-friendly activations
                    # m.in_channels = 3
                    print("first conv:", m)
                    conv_st_dict = m.state_dict()
                    W = conv_st_dict['weight'] 
                    fout,fin,kx,ky = W.shape   #
                    W_adj = nn.Parameter(torch.zeros(fout,m.in_channels,kx,ky) )
                    for i in range(W.shape[0]):
                        b = W[i][0].clone()
                        g = W[i][1].clone()
                        r = W[i][2].clone()
                        conv_st_dict['weight'][i][0] = b / 255.0
                        conv_st_dict['weight'][i][1] = g / 255.0
                        conv_st_dict['weight'][i][2] = r / 255.0
            
                    W_adj[:,:fin,:,:] = conv_st_dict['weight']
                    conv_st_dict['weight'] = W_adj
                    m.weight.data = W_adj
                    print(conv_st_dict['weight'].shape)
                    break


