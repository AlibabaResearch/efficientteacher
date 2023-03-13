"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.backbone.common import *
from models.backbone.experimental import *
from models.head.yolox_head import YoloXDetect
from models.head.yolov5_head import Detect
from models.head.yolov7_head import IDetect
from models.head.yolov6_head import YoloV6Detect
from models.head.yolov8_head import YoloV8Detect
from models.head.retina_head import RetinaDetect
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync
from models.loss.loss import *
from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml'):  # model, input channels, number of classes
        super().__init__()
        self.cfg = cfg
      
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.head = build_head(cfg)
            
        # self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.names = cfg.Dataset.names # default names
        self.inplace = self.cfg.Model.inplace

        # Build strides, anchors
        self.model_type = 'yolov5'
        self.export = False

        self.det_8 = netD(cfg.Model.Neck.out_channels[0], cfg.Model.width_multiple)
        self.det_16 = netD(cfg.Model.Neck.out_channels[1], cfg.Model.width_multiple)
        self.det_32 = netD(cfg.Model.Neck.out_channels[2], cfg.Model.width_multiple)      

        self.check_head() 
        
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def check_head(self):
        m = self.head  # Detect()
        if isinstance(m, (Detect, RetinaDetect, IDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # m.num_keypoints = self.num_keypoints
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.cfg.Model.ch, s, s))[0] ] )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            m.initialize_biases()  # only run once
        elif isinstance(m, (YoloV6Detect, YoloV8Detect)):
            m.inplace = self.inplace
            self.stride = torch.tensor(m.stride)
            m.initialize_biases()  # only run once
            self.model_type = 'tal'
        elif isinstance(m, (YoloXDetect)):
            m.inplace = self.inplace
            self.stride = torch.tensor(m.stride)
            m.initialize_biases()  # only run once
            self.model_type = 'yolox'

    def forward(self, x, augment=False, profile=False, visualize=False):
        # if self.export is True:
        #     return self.forward_export(x)
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
    
    def forward_export(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)
        return out
    
    def _forward_once(self, x, profile=False, visualize=False):
        # y, dt = [], []  # outputs
        x = self.backbone(x)
        x = self.neck(x)
        out = self.head(x)

        f8, f16, f32 = x
        out_8  = self.det_8(GradReverse.apply(f8))
        out_16 = self.det_16(GradReverse.apply(f16))
        out_32 = self.det_32(GradReverse.apply(f32))
        feature = [out_8, out_16, out_32]
        # feature = [f8, f16, f32]

        return out, feature


    # def _print_weights(self):
    #     for m in self.backbone.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.backbone.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    # def _apply(self, fn):
    #     # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
    #     self = super()._apply(fn)
    #     m = self.backbone[-1]  # Detect()
    #     if isinstance(m, Detect):
    #         m.stride = fn(m.stride)
    #         m.grid = list(map(fn, m.grid))
    #         if isinstance(m.anchor_grid, list):
    #             m.anchor_grid = list(map(fn, m.anchor_grid))
    #     return self


## for domain adaptation training
class GradReverse(Function):
    def __init__(self):
        self.lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        result = x
        # ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        #pdb.set_trace()
        # return (grad_output * - self.lambd)
        # result, = ctx.saved_tensors
        return (- grad_output)

def conv1x1(in_planes, out_planes, stride):
  "1x1 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class YoloX_netD_8(nn.Module):
    def __init__(self, ratio, context=False):
        super(YoloX_netD_8, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        # self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        #print('net_8_before:', x.shape)
        x = self.conv2(x)
        #print('net_8_aftre:', x.shape)
        return x

class YoloX_netD_16(nn.Module):
    def __init__(self, ratio, context=False):
        super(YoloX_netD_16, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class YoloX_netD_32(nn.Module):
    def __init__(self, ratio, context=False):
        super(YoloX_netD_32, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        # self.conv1 = conv1x1(int(256 * self.ratio), int(256* self.ratio), stride=1)
        self.conv1 = conv1x1(int(256 * self.ratio), int(256* self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD(nn.Module):
    def __init__(self, channel, ratio, context=False):
        super(netD, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(channel * self.ratio), int(channel * self.ratio), stride=1)
        self.conv2 = conv1x1(int(channel * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        #print('net_8_before:', x.shape)
        x = self.conv2(x)
        #print('net_8_aftre:', x.shape)
        return x

class netD_8(nn.Module):
    def __init__(self, channel, ratio, context=False):
        super(netD, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(channel * self.ratio), int(channel * self.ratio), stride=1)
        self.conv2 = conv1x1(int(channel * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        #print('net_8_before:', x.shape)
        x = self.conv2(x)
        #print('net_8_aftre:', x.shape)
        return x

class netD_16(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_16, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(512 * self.ratio), int(512 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(512 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_32(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_32, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(1024 * self.ratio), int(1024* self.ratio), stride=1)
        self.conv2 = conv1x1(int(1024 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_res_8(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_res_8, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        #print('net_8_before:', x.shape)
        x = self.conv2(x)
        #print('net_8_aftre:', x.shape)
        return x
class netD_res_16(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_res_16, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class netD_res_32(nn.Module):
    def __init__(self, ratio, context=False):
        super(netD_res_32, self).__init__()
        self.ratio = ratio
        # print('ratio:', ratio)
        self.conv1 = conv1x1(int(256 * self.ratio), int(256 * self.ratio), stride=1)
        self.conv2 = conv1x1(int(256 * self.ratio), 2, stride=1)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        # batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
