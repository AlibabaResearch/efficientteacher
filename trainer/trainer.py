#Copyright (c) 2023, Alibaba Group
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --assets coco128.yaml --weights yolov5s.pt --img 640
"""

import logging
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, SGD, lr_scheduler
from tqdm import tqdm

# import val # for end-of-epoch mAP
from models.backbone.experimental import attempt_load
from models.detector.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, init_seeds, \
    strip_optimizer, check_img_size, check_suffix, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from models.loss.loss import ComputeLoss, ComputeNanoLoss
from models.loss.yolox_loss import ComputeFastXLoss
from utils.plots import plot_labels
from utils.torch_utils import ModelEMA, de_parallel, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.metrics import MetricMeter, fitness
from utils.loggers import Loggers
# from ..val import run # for end-of-epoch mAP
from contextlib import redirect_stdout
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE):
        self.cfg = cfg
        # is_coco = assets.endswith('coco.yaml') and nc == 80  # COCO dataset
        self.set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)
        self.opt_scales = None
        ckpt = self.build_model(cfg, device)
        self.build_optimizer(cfg,ckpt=ckpt)

        self.build_dataloader(cfg, callbacks)
       
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                f'Using {self.train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f'Starting training for {self.epochs} epochs...')
        # burn_epochs = cfg.hyp.burn_epochs

        self.build_ddp_model(cfg, device)
        self.device = device
        self.break_iter = -1
        self.break_epoch = -1
        # random_resize = cfg.hyp.random_resize
        # if random_resize is None:
            # multiscale_range = cfg.hyp.multiscale_range
            # if multiscale_range is not None:
                # min_size = int(self.imgsz / 32) - multiscale_range
                # max_size = int(self.imgsz / 32) + multiscale_range
                # random_resize = (min_size, max_size)
        # sz = None
    def build_dataloader(self, cfg, callbacks):
        # Image sizes
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        nl = self.model.head.nl  # number of detection layers (used for scaling hyp['obj'])
        self.imgsz = check_img_size(cfg.Dataset.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple
        print('self imgsz:', self.imgsz)

        # DP mode
        if self.cuda and self.RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.sync_bn and self.cuda and self.RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')
        # Trainloader
        self.train_loader, self.dataset = create_dataloader(self.data_dict['train'], self.imgsz, self.batch_size // self.WORLD_SIZE, gs, self.single_cls,
                                              hyp=cfg.hyp, augment=cfg.hyp.use_aug, cache=cfg.cache, rect=cfg.rect, rank=self.LOCAL_RANK,
                                              workers=cfg.Dataset.workers, prefix=colorstr('train: '),cfg=cfg)
        # for d in self.train_loader:
        #     print(len(d))
        #     assert 0
        # print('dataset labels:', dataset.labels)
        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())  # max label class
        self.nb = len(self.train_loader)  # number of batches
        assert mlc < self.nc, f'Label class {mlc} exceeds nc={self.nc} in {cfg.Dataset.data_name}. Possible class labels are 0-{self.nc - 1}'

        # Process 0
        if self.RANK in [-1, 0]:
            self.val_loader = create_dataloader(self.data_dict['val'] , self.imgsz, self.batch_size // self.WORLD_SIZE * 2, gs, self.single_cls,
                                       hyp=cfg.hyp, cache=None if self.noval else cfg.cache, rect=True, rank=-1,
                                       workers=cfg.Dataset.workers, pad=0.5,
                                       prefix=colorstr('val: '),cfg=cfg)[0]

            if not cfg.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if self.plots:
                    plot_labels(labels, self.names, self.save_dir)

                # Anchors
                if not cfg.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=cfg.hyp.anchor_t, imgsz=self.imgsz)
                self.model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')
        
        self.no_aug_epochs = cfg.hyp.no_aug_epochs


    def build_model(self, cfg, device):
        # Model
        check_suffix(cfg.weights, '.pt')  # check weights
        pretrained = cfg.weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(self.LOCAL_RANK):
                weights = attempt_download(cfg.weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            self.model = Model(cfg or ckpt['model'].yaml).to(device)  # create
            exclude = ['anchor'] if (cfg or cfg.Model.anchors) and not cfg.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            if cfg.prune_finetune:
                dynamic_load(self.model, csd,reinitialize=cfg.reinitial)
                if cfg.reinitial:
                    LOGGER.info("*** Reinitialize all")
                    self.model._initialize_biases()
                self.model.info()
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')  # report
        else:
            self.model = Model(cfg).to(device)  # create
            ckpt = None
        # Freeze
        freeze = [f'model.{x}.' for x in range(cfg.freeze_layer_num)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
        
          # EMA
        self.ema = ModelEMA(self.model) if self.RANK in [-1, 0] else None

        # Resume
        self.start_epoch = 0
        pretrained = cfg.weights.endswith('.pt') and not cfg.reinitial
        if pretrained:
        # Optimizer
            if ckpt['optimizer'] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    LOGGER.info('pretrain model with different type of optimizer')
                # best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                try:
                    self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                    self.ema.updates = ckpt['updates']
                except:
                    LOGGER.info('pretrain model with different type of ema')

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if cfg.resume:
                assert self.start_epoch > 0, f'{weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
                self.epochs += ckpt['epoch']  # finetune additional epochs

            # del ckpt, csd
        self.epoch = self.start_epoch
        self.model_type = self.model.model_type
        self.detect = self.model.head
        return ckpt

    def build_optimizer(self, cfg,optinit=True,weight_masks =None,ckpt=None):
        # Optimizer
        nbs = 64  # nominal batch size
        self.accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = cfg.hyp.weight_decay*self.batch_size * self.accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {weight_decay}")

        g_bnw, g_w, g_b = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g_b.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g_bnw.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g_w.append(v.weight)

        # TODO 合并optimizer的写法
        if not cfg.Model.RepOpt:
            if cfg.adam:
                # optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
                self.optimizer = AdamW(g_b, lr=cfg.hyp.lr0,betas=(cfg.hyp.momentum, 0.999))
            else:
                self.optimizer = SGD(g_b, lr=cfg.hyp.lr0, momentum=cfg.hyp.momentum, nesterov=True)
            self.optimizer.add_param_group({'params': g_w, 'weight_decay': weight_decay})  # add g1 with weight_decay
            self.optimizer.add_param_group({'params': g_bnw})  # add g2 (biases)
        else:
            from models.optimizers.RepOptimizer import RepVGGOptimizer
            assert cfg.Model.RepScale_weight
            if self.opt_scales is None:
                scales = torch.load(cfg.Model.RepScale_weight, map_location=self.device)
            else:
                scales = self.opt_scales
            assert not cfg.adam, "RepOptimizer Only Support SGD."
            params_groups = [
                {'params': g_bnw},
                {'params': g_w, 'weight_decay': weight_decay},
                {'params': g_b}
            ]

            reinit = False
            if cfg.weights=='' and optinit:
                reinit = True

            self.optimizer = RepVGGOptimizer(self.model,scales,cfg,reinit=reinit,device=self.device,params=params_groups,weight_masks=weight_masks)
        LOGGER.info(f"{colorstr('optimizer:')} {type(self.optimizer).__name__} with parameter groups "
                f"{len(g_w)} weight, {len(g_bnw)} weight (no decay), {len(g_b)} bias")
        del g_w, g_bnw, g_b

        # Scheduler
        if cfg.linear_lr:
            self.lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - cfg.hyp.lrf) + cfg.hyp.lrf  # linear
        else:
            self.lf = one_cycle(1, cfg.hyp.lrf, self.epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
        self.scheduler.last_epoch = self.epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        if ckpt is not None and 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            print("Load Optimizer statedict")
            self.optimizer.load_state_dict(ckpt['optimizer'])

    def set_env(self, cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks):
        self.save_dir, self.epochs, self.batch_size, weights, self.single_cls, data, self.noval, self.nosave  = \
        Path(cfg.save_dir), cfg.epochs, cfg.Dataset.batch_size, cfg.weights, cfg.single_cls, cfg.Dataset.data_name, \
        cfg.noval, cfg.nosave
        self.sync_bn = cfg.sync_bn
        self.save_period = cfg.save_period
        self.device = device

        self.warmup_epochs = cfg.hyp.warmup_epochs
        self.momentum = cfg.hyp.momentum
        self.warmup_momentum = cfg.hyp.warmup_momentum
        self.warmup_bias_lr = cfg.hyp.warmup_bias_lr
        # resume = cfg.resume

        self.LOCAL_RANK = LOCAL_RANK
        self.RANK = RANK
        self.WORLD_SIZE = WORLD_SIZE
        self.norm_scale = cfg.Dataset.norm_scale

        # Directories
        w = self.save_dir / 'weights'  # weights dir
        w.mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = w / 'last.pt', w / 'best.pt'

        with open(self.save_dir / 'opt.yaml', 'w') as f:
            with redirect_stdout(f): print(cfg.dump())

        # Loggers
        if RANK in [-1, 0]:
            loggers = Loggers(self.save_dir, weights, cfg, LOGGER)  # loggers instance
            if loggers.wandb:
                data_dict = loggers.wandb.data_dict
                if cfg.resume:
                    weights, epochs, hyp = cfg.weights, cfg.epochs, cfg.hyp
            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

        # Config
        self.plots = True # create plots
        self.cuda = device.type != 'cpu'
        init_seeds(1 + RANK)
        # with torch_distributed_zero_first(LOCAL_RANK):
            # self.data_dict = self.data_dict or check_dataset(assets)  # check if None
        # train_path, val_path = self.data_dict['train'], self.data_dict['val']
        self.data_dict = {}
        self.data_dict['train'] = cfg.Dataset.train
        self.data_dict['val'] = cfg.Dataset.val
        self.data_dict['nc'] = cfg.Dataset.nc
        self.data_dict['names'] = cfg.Dataset.names
        # val_path = cfg.Dataset.val
        self.nc = 1 if self.single_cls else int(self.data_dict['nc'])  # number of classes
        self.names = ['item'] if self.single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']  # class names
        assert len(self.names) == self.nc, f'{len(self.names)} names found for nc={self.nc} dataset in {data}'  # check

    def build_ddp_model(self, cfg, device):
        # loss_fn = self.model.loss_fn
         # DDP mode
        if self.cuda and self.RANK != -1:
            # print("Set DDP mode")
            self.model = DDP(self.model, device_ids=[self.LOCAL_RANK], output_device=self.LOCAL_RANK, find_unused_parameters=True)
      
        # cfg.hyp.label_smoothing = opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(device) * self.nc  # attach class weights
        self.model.names = self.names

        if cfg.Loss.type == 'ComputeLoss': 
            self.compute_loss = ComputeLoss(self.model, cfg)  # init loss class
        elif cfg.Loss.type == 'ComputeFastXLoss':
            self.compute_loss = ComputeFastXLoss(self.model, cfg)
        elif cfg.Loss.type == 'ComputeNanoLoss':
            self.compute_loss = ComputeNanoLoss(self.model, cfg)
        else:
            raise NotImplementedError

        is_distributed = is_parallel(self.model)
        if is_distributed:
            self.detect = self.model.module.head
        else:
            self.detect = self.model.head

    def before_train(self):
        return 0
    
    def build_train_logger(self):
        self.meter = MetricMeter()
        # loss_dict = self.compute_loss.loss_dict
        log_contents = ['Epoch', 'gpu_mem', 'labels', 'img_size']

        self.log_contents = log_contents
    
    def update_train_logger(self):
        for (imgs, targets, paths, _) in self.train_loader:  # batch -------------------------------------------------------------
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale  # uint8 to float32, 0-255 to 0.0-1.0
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                #TODO loss是否需要scale up需要进行讨论
            if self.RANK in [-1, 0]:
                for loss_key in loss_items.keys():
                    self.log_contents.append(loss_key)
            break
        LOGGER.info(('\n' + '%10s' * len(self.log_contents)) % tuple(self.log_contents))
    
    def before_epoch(self):
        self.model.train()
        self.build_train_logger()
        self.update_train_logger()

        if self.epoch == self.epochs - self.no_aug_epochs:
            LOGGER.info("--->No mosaic aug now!")
            self.dataset.mosaic = False  # close mosaic
            LOGGER.info("--->Add additional L1 loss now!")
            if self.model_type == 'yolox':
                self.detect.use_l1 = True

        self.meter = MetricMeter()

        if self.warmup_epochs > 0:
             self.nw = max(round(self.warmup_epochs * self.nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
             self.nw = min(self.nw, (self.epochs - self.start_epoch) / 2 * self.nb)
        else:
             self.nw = -1

        if self.RANK != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
    
    def update_optimizer(self, loss, ni):
        # Backward
        self.scaler.scale(loss).backward()
                
        self.accumulate = max(round(64 / self.batch_size), 1) 

        #warmup setting
        if ni <= self.nw:
            xi = [0, self.nw]
            self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [self.warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])
                # Optimize

        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = ni
    
    def train_in_epoch(self, callbacks):

        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            if i == self.break_iter:
                break
            ni = i + self.nb * self.epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale  # uint8 to float32, 0-255 to 0.0-1.0

            # Forward
            # with torch.autograd.set_detect_anomaly(True):
            with amp.autocast(enabled=self.cuda):
                pred = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                #TODO loss是否需要scale up需要进行讨论
                if self.RANK != -1:
                    loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode
                    # if opt.quad:
                    #     loss *= 4.

            self.update_optimizer(loss, ni)

            # Log
            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count= len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count+2)) % (
                        f'{self.epoch}/{self.epochs - 1}', mem, targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)
            # end batch ------------------------------------------------------------------------------------------------
            # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()

    def after_epoch(self, callbacks, val):
        if self.RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=self.epoch)
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (self.epoch + 1 == self.epochs)
            if not self.noval:  # Calculate mAP
                # val_da = self.cfg.DomainAdaptation.train_domain
                self.results, maps, _ = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=self.ema.ema,
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points = self.cfg.Dataset.np,
                                           val_kp=self.cfg.Dataset.val_kp)

                # Update best mAP
            fi = fitness(np.array(self.results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi
            log_vals = list(self.meter.get_avg())[:3] + list(self.results) + self.lr
            callbacks.run('on_fit_epoch_end', log_vals, self.epoch, self.best_fitness, fi)

            # Save model
            if (not self.nosave) or (final_epoch):  # if save
                ckpt = {'epoch': self.epoch,
                        'best_fitness': self.best_fitness,
                        'model': deepcopy(de_parallel(self.model)).half(),
                        'ema': deepcopy(self.ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id':  None}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
                    w = self.save_dir / 'weights'  # weights dir
                    torch.save(ckpt, w / f'epoch{self.epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', self.last, self.epoch, final_epoch, self.best_fitness, fi)
    
    def after_train(self, callbacks, val):
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, clss)
        if self.RANK in [-1, 0]:
            for f in self.last, self.best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is self.best:
                        LOGGER.info(f'\nValidating {f}...')
                        # val_da = self.cfg.DomainAdaptation.train_domain
                        results, _, _ = val.run(self.data_dict,
                                            batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                            imgsz=self.imgsz,
                                            model=attempt_load(f, self.device).half(),
                                            iou_thres=0.65,  # best pycocotools results at 0.65
                                            single_cls=self.single_cls,
                                            dataloader=self.val_loader,
                                            save_dir=self.save_dir,
                                            save_json=False,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=self.compute_loss,
                                            num_points=self.cfg.Dataset.np,
                                            val_kp=self.cfg.Dataset.val_kp)  # val best model with plots

            callbacks.run('on_train_end', self.last, self.best, self.plots, self.epoch)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        torch.cuda.empty_cache()
        return results

    
    def train(self, callbacks, val):
        # Start training
        t0 = time.time()
        self.last_opt_step = -1
        self.results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, clss)
        self.best_fitness = 0

        for self.epoch in range(self.start_epoch, self.epochs):  # epoch ------------------------------------------------------------------
            if self.epoch == self.break_epoch:
                break
            self.before_epoch()
            self.train_in_epoch(callbacks)
            self.after_epoch(callbacks, val)
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training -----------------------------------------------------------------------------------------------------
        results = self.after_train(callbacks, val)
        if self.RANK in [-1, 0]:
            LOGGER.info(f'\n{self.epoch - self.start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        return results