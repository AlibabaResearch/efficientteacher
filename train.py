import os
import torch
import torch.distributed as dist
import torch.nn as nn
from trainer.trainer import Trainer
from trainer.ssod_trainer import SSODTrainer
from configs.defaults import get_cfg
from utils.general import increment_path, check_git_status, check_requirements, \
    print_args,  set_logging
from pathlib import Path
import logging
import os
import argparse
from utils.callbacks import Callbacks
from utils.torch_utils import select_device
import sys
from datetime import timedelta
import val

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # EfficientTeacher root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def setup(cfg):
    set_logging(RANK)
    if RANK in [-1, 0]:
        print_args(FILE.stem, cfg)
        # check_git_status()
        # check_requirements(exclude=['thop'])
 
    cfg.save_dir = str(increment_path(Path(cfg.project) / cfg.name, exist_ok=cfg.exist_ok))

    # DDP mode
    device = select_device(cfg.device, batch_size=cfg.Dataset.batch_size)
    if LOCAL_RANK != -1:
        timeout=timedelta(seconds=86400)
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert cfg.Dataset.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count {} VS {}'.format(cfg.Dataset.batch_size,WORLD_SIZE)
        # assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        # dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)  # distributed backend
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timeout, rank=RANK, world_size=WORLD_SIZE)
    return device

def main(opt, callbacks=Callbacks()):
    # Checks
        # dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    cfg = get_cfg()
    cfg.merge_from_file(opt.cfg)
    cfg.merge_from_list(opt.opts)

    device = setup(cfg)
    cfg.freeze()
    if cfg.SSOD.train_domain:
        trainer = SSODTrainer(cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE)
    else:
        trainer = Trainer(cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE)
        
    trainer.train(callbacks, val)
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        # dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)