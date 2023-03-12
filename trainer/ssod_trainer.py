#Copyright (c) 2023, Alibaba Group
"""
Train an object detection model using domain adaptation  @ruiyang

"""
from cgitb import enable
from email.utils import encode_rfc2231
import json

from .trainer import Trainer
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
# import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tqdm import tqdm
from datetime import timedelta

# import val # for end-of-epoch mAP
from models.backbone.experimental import attempt_load
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from models.loss.loss import DomainLoss, TargetLoss
from models.loss import build_ssod_loss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first, is_parallel,time_sync, SemiSupModelEMA, CosineEMA
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
# from ..val import run # for end-of-epoch mAP
from utils.plots import plot_images, plot_labels, plot_results,  plot_images_debug, output_to_target
from utils.datasets_ssod import create_target_dataloader, augment_hsv, cutout
from utils.self_supervised_utils import FairPseudoLabel
from utils.labelmatch import LabelMatch 
from utils.self_supervised_utils import check_pseudo_label_with_gt, check_pseudo_label
from models.detector.yolo_ssod import Model
import torchvision
import copy

LOGGER = logging.getLogger(__name__)

class SSODTrainer(Trainer):
    def __init__(self, cfg, device, callbacks, LOCAL_RANK, RANK, WORLD_SIZE):
        self.cfg = cfg
        self.set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)

        self.build_model(cfg, device)
        self.build_optimizer(cfg)
        self.build_dataloader(cfg, callbacks)
       
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                f'Using {self.train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f'Starting training for {self.epochs} epochs...')
        # self.no_aug_epochs = cfg.hyp.no_aug_epochs
        # burn_epochs = cfg.hyp.burn_epochs
        if cfg.SSOD.pseudo_label_type == 'FairPseudoLabel':
            self.pseudo_label_creator = FairPseudoLabel(cfg)
        elif cfg.SSOD.pseudo_label_type == 'LabelMatch':
            self.pseudo_label_creator = LabelMatch(cfg, int(self.unlabeled_dataset.__len__()/self.WORLD_SIZE), self.label_num_per_image, cls_ratio_gt= self.cls_ratio_gt)

        self.build_ddp_model(cfg, device)
        self.device = device
    
    def set_env(self, cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks):
        super().set_env(cfg, device, LOCAL_RANK, RANK, WORLD_SIZE, callbacks)
        self.data_dict['target'] = cfg.Dataset.target
        self.target_with_gt = cfg.SSOD.ssod_hyp.with_gt
        self.break_epoch = -1
        self.epoch_adaptor = cfg.SSOD.epoch_adaptor
        self.da_loss_weights = cfg.SSOD.da_loss_weights
        self.cosine_ema = cfg.SSOD.cosine_ema
        self.fixed_accumulate = cfg.SSOD.fixed_accumulate
    
    def build_optimizer(self, cfg, optinit=True, weight_masks=None, ckpt=None):
        super().build_optimizer(cfg, optinit, weight_masks, ckpt)
        # Scheduler
        if cfg.SSOD.multi_step_lr:
            milestones = cfg.SSOD.milestones
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            self.scheduler.last_epoch = self.epoch - 1  # do not move
            print('self scheduler:', milestones)
            self.scaler = amp.GradScaler(enabled=self.cuda)

    def build_model(self, cfg, device):
        # use DomainAdpatationModel
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
                dynamic_load(self.model, csd)
                self.model.info()
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')  # report
        else:
            self.model = Model(cfg).to(device)  # create
        # Freeze
        freeze = [f'model.{x}.' for x in range(cfg.freeze_layer_num)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
        # EMA
        self.ema = ModelEMA(self.model)        
        if self.cfg.hyp.burn_epochs > 0:
            self.semi_ema = None
            # self.ema = ModelEMA(self.model, decay=self.cfg.SSOD.ema_rate)
        else:
            if self.cosine_ema:
                self.semi_ema = CosineEMA(self.ema.ema, decay_start=self.cfg.SSOD.ema_rate, total_epoch=self.epochs)
            else:
                self.semi_ema = SemiSupModelEMA(self.ema.ema, self.cfg.SSOD.ema_rate)

        # Resume
        self.start_epoch = 0
        pretrained = cfg.weights.endswith('.pt')
        if pretrained:
            if ckpt['optimizer'] is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    LOGGER.info('pretrain model with different type of optimizer')
                # best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
                self.ema.updates = ckpt['updates']
            
            if self.semi_ema and ckpt.get('ema'):
                self.semi_ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
            # EMA
            # if self.ema and ckpt.get('ema'):
            #     self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict(), strict=False)
            #     self.ema.updates = ckpt['updates']

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if cfg.resume:
                assert self.start_epoch > 0, f'{weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < self.start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd
        self.epoch = self.start_epoch
        # self.ema.update_decay(self.epoch, self.cfg.hyp.burn_epochs)
        self.model_type = self.model.model_type

        # load teacher model and extract class idx
        self.extra_teacher_models = []
        self.extra_teacher_class_idxs = []  
        if len(self.cfg.SSOD.extra_teachers) > 0 and len(self.cfg.SSOD.extra_teachers_class_names) > 0:
            assert(len(self.cfg.SSOD.extra_teachers) == len(self.cfg.SSOD.extra_teachers_class_names)  )
            for i, extra_teacher_path in enumerate(self.cfg.SSOD.extra_teachers):
                teacher_model = attempt_load(extra_teacher_path, map_location=device) 
                self.extra_teacher_models.append(teacher_model)
                if self.RANK in [-1 , 0]: 
                    print('load  {} teacher model and class...'.format(i))

                teacher_class_idx = {}  #{origin_class_idx : new_class_idx}
                assert len(self.cfg.SSOD.extra_teachers_class_names[i]) > 0
                if self.RANK in [-1 , 0]: 
                    print("origin name: {} current name: {}".format(teacher_model.names, self.cfg.Dataset.names) )
                for na in self.cfg.SSOD.extra_teachers_class_names[i]:
                    origin_idx = -1; curr_idx = -1
                    for idx, origin_name in enumerate(teacher_model.names):
                        if na == origin_name:
                            origin_idx = idx
                            break 
                    for idx, name  in enumerate(self.cfg.Dataset.names):
                        if na == name:
                            curr_idx = idx
                    if len(self.cfg.SSOD.extra_teachers_class_names[i]) == 1: #teacher model是通过single-cls 训练的,这里 将原来的class-idx改为0
                        if self.RANK in [-1 , 0]: 
                            print('single cls change ')
                        origin_idx = 0
                    teacher_class_idx[origin_idx] = curr_idx

                if self.RANK in [-1 , 0]: 
                    print('class_idx dic: ', teacher_class_idx)
                self.extra_teacher_class_idxs.append(teacher_class_idx)
                assert len(self.extra_teacher_class_idxs) == len(self.extra_teacher_models)
                assert len(self.extra_teacher_models) > 0

    def build_dataloader(self, cfg, callbacks):
        # Image sizes
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        # Model parameters
        self.imgsz = check_img_size(cfg.Dataset.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # DP mode
        if self.cuda and self.RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.sync_bn and self.cuda and self.RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        self.train_loader, self.dataset = create_dataloader(self.data_dict['train'], self.imgsz, self.batch_size // self.WORLD_SIZE, gs, self.single_cls,
                                              hyp=cfg.hyp, augment=True, cache=cfg.cache, rect=cfg.rect, rank=self.LOCAL_RANK,
                                              workers=cfg.Dataset.workers, prefix=colorstr('train: '), cfg=cfg)
        self.cls_ratio_gt = self.dataset.cls_ratio_gt 
        self.label_num_per_image = self.dataset.label_num_per_image
        # Trainloader for semi supervised training 
        self.unlabeled_dataloader, self.unlabeled_dataset = create_target_dataloader(self.data_dict['target'], self.imgsz, self.batch_size // self.WORLD_SIZE, gs, self.single_cls,
                                              hyp=cfg.hyp, augment=True, cache=cfg.cache, rect=cfg.rect, rank=self.LOCAL_RANK,
                                              workers=cfg.Dataset.workers, cfg=cfg, prefix=colorstr('target: '))

        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())  # max label class
        self.nb = len(self.train_loader)  # number of batches
        assert mlc < self.nc, f'Label class {mlc} exceeds nc={self.nc} in {cfg.Dataset.data_name}. Possible class labels are 0-{self.nc - 1}'

        # Process 0
        if self.RANK in [-1, 0]:
            self.val_loader = create_dataloader(self.data_dict['val'] , self.imgsz, self.batch_size // self.WORLD_SIZE * 2, gs, self.single_cls,
                                       hyp=cfg.hyp, cache=None if self.noval else cfg.cache, rect=True, rank=-1,
                                       workers=cfg.Dataset.workers, pad=0.5,
                                       prefix=colorstr('val: '), cfg=cfg)[0]

            if not cfg.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                if self.plots:
                    plot_labels(labels, self.names, self.save_dir)

                # Anchors
                if not cfg.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=cfg.Loss.anchor_t, imgsz=self.imgsz)
                self.model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')
        self.no_aug_epochs = cfg.hyp.no_aug_epochs


    def build_ddp_model(self, cfg, device):
        super().build_ddp_model(cfg, device)
        # if cfg.Loss.type == 'ComputeLoss': 
        self.compute_un_sup_loss = build_ssod_loss(self.model, cfg)
        self.domain_loss = DomainLoss()
        self.target_loss = TargetLoss()
    
    def update_train_logger(self):
        for (imgs, targets, paths, _) in self.train_loader:  # batch -------------------------------------------------------------
            imgs = imgs.to(self.device, non_blocking=True).float() / self.norm_scale  # uint8 to float32, 0-255 to 0.0-1.0
            with amp.autocast(enabled=self.cuda):
                pred, sup_feats = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                if self.model_type in ['yolox', 'tal']:
                    un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(pred, pred, targets.to(self.device))  
                else:
                    un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(pred, targets.to(self.device))  
            if self.RANK in [-1, 0]:
                for loss_key in loss_items.keys():
                    self.log_contents.append(loss_key)
            if (self.epoch >= self.cfg.hyp.burn_epochs):
                if self.RANK in [-1, 0]:
                    for loss_key in un_sup_loss_items.keys():
                        self.log_contents.append(loss_key)
            break
        if self.cfg.SSOD.train_domain == True and self.epoch >= self.cfg.hyp.burn_epochs:
            if self.RANK in [-1, 0]:
            # self.log_contents.append('hit_miss')
            # self.log_contents.append('hit_total')
                self.log_contents.append('tp')
                self.log_contents.append('fp_cls')
                self.log_contents.append('fp_loc')
                self.log_contents.append('pse_num')
                self.log_contents.append('gt_num')
        LOGGER.info(('\n' + '%10s' * len(self.log_contents)) % tuple(self.log_contents))
        
    
    def train_in_epoch(self, callbacks):
        if ( self.epoch < self.cfg.hyp.burn_epochs):
            if self.cfg.SSOD.with_da_loss:
                self.train_without_unlabeled_da(callbacks)
            else:
                self.train_without_unlabeled(callbacks)
            if self.RANK in [-1, 0]:
                print('burn_in_epoch: {}, cur_epoch: {}'.format(self.cfg.hyp.burn_epochs, self.epoch) )
        else:
            # if self.epoch == self.cfg.hyp.burn_epochs and self.cfg.hyp.burn_epochs > 0:
            if self.epoch == self.cfg.hyp.burn_epochs:
                msd = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()  # model state_dict
                for k, v in self.ema.ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        msd[k] = v
                    # if self.RANK in [-1, 0]:
                    #     print('ema:', v)
                    #     print('msd:', msd[k])
                if self.cosine_ema:
                    self.semi_ema = CosineEMA(self.ema.ema, decay_start=self.cfg.SSOD.ema_rate, total_epoch=self.epochs - self.cfg.hyp.burn_epochs)
                else:
                    self.semi_ema = SemiSupModelEMA(self.ema.ema, self.cfg.SSOD.ema_rate)
            self.train_with_unlabeled(callbacks)
    
    def after_epoch(self, callbacks, val):
        if self.cfg.SSOD.pseudo_label_type == 'LabelMatch' and self.epoch >= self.cfg.SSOD.dynamic_thres_epoch:
            self.pseudo_label_creator.update_epoch_cls_thr(self.epoch - self.start_epoch)
            self.compute_un_sup_loss.ignore_thres_high = self.pseudo_label_creator.cls_thr_high
            self.compute_un_sup_loss.ignore_thres_low = self.pseudo_label_creator.cls_thr_low
            # print(self.RANK,  self.pseudo_label_creator.cls_thr_high, self.pseudo_label_creator.cls_thr_low)
        if self.epoch >= self.cfg.hyp.burn_epochs:
            if self.model_type == 'tal':
                self.compute_un_sup_loss.cur_epoch = self.epoch - self.cfg.hyp.burn_epochs
            if self.cosine_ema:
                self.semi_ema.update_decay(self.epoch - self.cfg.hyp.burn_epochs)        
        if self.RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=self.epoch)
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (self.epoch + 1 == self.epochs)
            if not self.noval or final_epoch:  # Calculate mAP
                val_ssod = self.cfg.SSOD.train_domain
                # if (self.epoch >= self.cfg.hyp.burn_epochs):
                # if (1):
                self.results, maps, _, cls_thr = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=deepcopy(de_parallel(self.model)),
                                           conf_thres=self.cfg.val_conf_thres, 
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points=self.cfg.Dataset.np,
                                           val_ssod=val_ssod,
                                           val_kp=self.cfg.Dataset.val_kp)
                self.model.train()
                if (self.epoch >= self.cfg.hyp.burn_epochs):
                    self.results, maps, _, cls_thr = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=self.semi_ema.ema,
                                           conf_thres=self.cfg.val_conf_thres, 
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points = self.cfg.Dataset.np,
                                           val_ssod=val_ssod,
                                           val_kp=self.cfg.Dataset.val_kp)
                else:
                    self.results, maps, _, cls_thr = val.run(self.data_dict,
                                           batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                           imgsz=self.imgsz,
                                           model=self.ema.ema,
                                           conf_thres=self.cfg.val_conf_thres, 
                                           single_cls=self.single_cls,
                                           dataloader=self.val_loader,
                                           save_dir=self.save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=self.compute_loss,
                                           num_points = self.cfg.Dataset.np,
                                           val_ssod=val_ssod,
                                           val_kp=self.cfg.Dataset.val_kp)

            # Update best mAP
            fi = fitness(np.array(self.results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi
            log_vals = list(self.meter.get_avg())[:3] + list(self.results) + self.lr
            callbacks.run('on_fit_epoch_end', log_vals, self.epoch, self.best_fitness, fi)

            # Save model
            if (not self.nosave) or (final_epoch):  # if save
                if self.epoch >= self.cfg.hyp.burn_epochs:
                    ckpt = {'epoch': self.epoch,
                        'best_fitness': self.best_fitness,
                        'model': deepcopy(de_parallel(self.model)).half(),
                        'ema': deepcopy(self.semi_ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id':  None}
                else:
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

    def train_without_unlabeled(self, callbacks):
        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + self.nb * self.epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            # Forward
            #with torch.autograd.set_detect_anomaly(True):
            with amp.autocast(enabled=self.cuda):
                pred, sup_feats = self.model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size 

                if self.RANK != -1:
                    loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode
                    # print(self.WORLD_SIZE)
                    # print('scale loss:', loss_items)

                loss = loss + 0 * (sup_feats[0].mean() + sup_feats[1].mean() + sup_feats[2].mean())

            self.update_optimizer(loss, ni) 

            # Log
            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count= len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count+2)) % (
                    f'{self.epoch}/{self.epochs - 1}', mem,  targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)
        # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()  

    def update_optimizer(self, loss, ni):
        # Backward
        self.scaler.scale(loss).backward()
                
        if self.fixed_accumulate:
            self.accumulate = 1
        else:
            self.accumulate = max(round(64 / self.batch_size), 1) 

        #warmup setting
        if ni <= self.nw:
            xi = [0, self.nw]
            if self.fixed_accumulate:
                self.accumulate = max(1, np.interp(ni, xi, [1, 1]).round())
            else:
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
            self.ema.update(self.model)
            if self.semi_ema:
                self.semi_ema.update(self.ema.ema)
            self.last_opt_step = ni

    def train_without_unlabeled_da(self, callbacks):
        pbar = enumerate(self.train_loader)
        if self.RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar

        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + self.nb * self.epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            target_imgs, target_targets, target_paths, _, target_imgs_ori, target_M = next(self.unlabeled_dataloader.__iter__())
            target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0 
            total_imgs = torch.cat([imgs, target_imgs_ori], 0)
            n_img, _, _, _ = imgs.shape
            # Forward
            #with torch.autograd.set_detect_anomaly(True):
            with amp.autocast(enabled=self.cuda):
                total_pred, total_feature = self.model(total_imgs)  # forward

                sup_pred, sup_feature, un_sup_pred, un_sup_feature = self.split_predict_and_feature(total_pred, total_feature, n_img)
                loss, loss_items = self.compute_loss(sup_pred, targets.to(self.device))  # loss scaled by batch_size 
                d_loss = self.domain_loss(sup_feature)
                t_loss = self.target_loss(un_sup_feature) 

                loss = loss + d_loss * self.da_loss_weights + t_loss * self.da_loss_weights + 0 * un_sup_pred[0].mean() + 0 * un_sup_pred[1].mean() + 0 * un_sup_pred[2].mean()
                # else:
                    # loss = loss + 0 * d_loss + 0 * t_loss + 0 * un_sup_pred[0].mean() + 0 * un_sup_pred[1].mean() + 0 * un_sup_pred[2].mean()

                if self.RANK != -1:
                    loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode

            self.update_optimizer(loss, ni) 

            # Log
            if self.RANK in [-1, 0]:
                self.meter.update(loss_items)
                mloss_count= len(self.meter.meters.items())
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count+2)) % (
                    f'{self.epoch}/{self.epochs - 1}', mem,  targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
                callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)
        # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()  

    def after_train(self, callbacks, val):
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, clss)
        if self.RANK in [-1, 0]:
            for f in self.last, self.best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is self.best:
                        LOGGER.info(f'\nValidating {f}...')
                        # val_ssod = self.cfg.SSOD.train_domain
                        results, _, _, _ = val.run(self.data_dict,
                                            batch_size=self.batch_size // self.WORLD_SIZE * 2,
                                            imgsz=self.imgsz,
                                            model=attempt_load(f, self.device).half(),
                                            conf_thres=self.cfg.val_conf_thres, 
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
                                            val_ssod=self.cfg.SSOD.train_domain,
                                            val_kp=self.cfg.Dataset.val_kp)  # val best model with plots

            callbacks.run('on_train_end', self.last, self.best, self.plots, self.epoch)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
       
        torch.cuda.empty_cache()
        return results
    
    def split_predict_and_feature(self, total_pred, total_feature, n_img):
            sup_feature = [total_feature[0][:n_img, :, :, :], total_feature[1][:n_img, :, :, :], total_feature[2][:n_img, :, :, :]]
            un_sup_feature = [total_feature[0][n_img:, :, :, :], total_feature[1][n_img:, :, :, :], total_feature[2][n_img:, :, :, :]]
            if self.model_type == 'yolov5':
                sup_pred = [total_pred[0][:n_img, :, :, :, :], total_pred[1][:n_img, :, :, :, :], total_pred[2][:n_img, :, :, :, :]]
                un_sup_pred = [total_pred[0][n_img:, :, :, :, :], total_pred[1][n_img:, :, :, :, :], total_pred[2][n_img:, :, :, :, :]]
            elif self.model_type in ['yolox', 'yoloxkp']:
                sup_pred = [total_pred[0][:n_img, :, :], total_pred[1][:n_img, :, :], total_pred[2][:n_img, :, :]]
                un_sup_pred = [total_pred[0][n_img:, :, :], total_pred[1][n_img:, :, :], total_pred[2][n_img:, :, :]]
            elif self.model_type == 'tal':
                sup_pred = [[total_pred[0][0][:n_img, :, :, :], total_pred[0][1][:n_img, :, :, :], total_pred[0][2][:n_img, :, :, :]], total_pred[1][:n_img, :, :], total_pred[2][:n_img, :, :]]
                un_sup_pred = [[total_pred[0][0][n_img:, :, :, :], total_pred[0][1][n_img:, :, :, :], total_pred[0][2][n_img:, :, :, :]], total_pred[1][n_img:, :, :], total_pred[2][n_img:, :, :]]
            # elif self.model_type == 'yoloxkp':
            #     sup_pred = [total_pred[0][:n_img, :, :], total_pred[1], total_pred[2], total_pred[3], total_pred[4], total_pred[5]]
            #     un_sup_pred = [total_pred[0][n_img:, :, :], total_pred[1], total_pred[2], total_pred[3], total_pred[4], total_pred[5]]
            else:
                raise NotImplementedError
            return sup_pred, sup_feature, un_sup_pred, un_sup_feature
    
    def train_instance(self, imgs, targets, paths, unlabeled_imgs, unlabeled_imgs_ori, unlabeled_gt, unlabeled_M, ni, pbar, callbacks):
        n_img, _, _, _ = imgs.shape
        n_pse_img, _,_,_ = unlabeled_imgs.shape
        invalid_target_shape = True
        unlabeled_targets = torch.zeros(8)

        # Teacher Model Forward
        extra_teacher_outs = []
        with amp.autocast(enabled=self.cuda):
            #build pseudo label via pred from teacher model
            with torch.no_grad():
                if self.model_type in ['yolov5']:
                    (teacher_pred, train_out), teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                # elif self.model_type == 'tal':
                #     teacher_pred, teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                # elif self.model_type == 'yoloxkp':
                #     teacher_pred, teacher_feature = self.ema.ema(unlabeled_imgs_ori, augment=False)
                    # teacher_pred = torch.cat(outputs, 1)
                else:
                    raise NotImplementedError
                
                if len(self.extra_teacher_models) > 0 :
                    for teacher_model in self.extra_teacher_models:
                        teacher_out = teacher_model(unlabeled_imgs_ori)[0]
                        extra_teacher_outs.append(teacher_out)

        if len(self.extra_teacher_models) > 0 and len(extra_teacher_outs) > 0 :
            unlabeled_targets, unlabeled_imgs, invalid_target_shape = self.pseudo_label_creator.create_pseudo_label_online_with_extra_teachers(teacher_pred, extra_teacher_outs, copy.deepcopy(unlabeled_imgs), unlabeled_M, self.extra_teacher_class_idxs, self.RANK)
        elif len(self.extra_teacher_models) == 0 :
            if self.cfg.SSOD.pseudo_label_type == 'LabelMatch':
                self.pseudo_label_creator.update(targets, n_img, n_pse_img)
            unlabeled_targets, invalid_target_shape = self.pseudo_label_creator.create_pseudo_label_online_with_gt(teacher_pred, copy.deepcopy(unlabeled_imgs), unlabeled_M, copy.deepcopy(unlabeled_imgs_ori), unlabeled_gt, self.RANK)
            unlabeled_imgs = unlabeled_imgs.to(self.device)
        else:    
            raise NotImplementedError

        total_imgs = torch.cat([imgs, unlabeled_imgs], 0)
        
        with amp.autocast(enabled=self.cuda):
            total_pred, total_feature = self.model(total_imgs)  # forward
            sup_pred, sup_feature, un_sup_pred, un_sup_feature = self.split_predict_and_feature(total_pred, total_feature, n_img)
            sup_loss, sup_loss_items = self.compute_loss(sup_pred, targets.to(self.device)) 

            #计算domain adaptation部分loss
            d_loss = self.domain_loss(sup_feature)
            t_loss = self.target_loss(un_sup_feature) 
            if self.cfg.SSOD.with_da_loss:
                sup_loss = sup_loss + d_loss * self.da_loss_weights + t_loss * self.da_loss_weights
            else:
                sup_loss = sup_loss + d_loss * 0 + t_loss * 0
            # total_t2 = time_sync()
            if self.RANK != -1:
                sup_loss *= self.WORLD_SIZE  # gradient averaged between devices in DDP mode
            if( invalid_target_shape ): #伪标签生成质量没有达到要求之前不计算loss
                un_sup_loss = torch.zeros(1, device=self.device) 
                un_sup_loss_items = dict(ss_box=0, ss_obj=0, ss_cls=0)
                un_sup_loss = un_sup_loss * 0.0
            else:
                un_sup_loss, un_sup_loss_items = self.compute_un_sup_loss(un_sup_pred, unlabeled_targets.to(self.device))  
                # un_sup_loss = un_sup_loss * self.cfg.SSOD.teacher_loss_weight
            if self.RANK != -1:
                un_sup_loss *= self.WORLD_SIZE
        loss = sup_loss + un_sup_loss * self.cfg.SSOD.teacher_loss_weight
        # if self.cfg.SSOD.imitate_teacher:
            # loss += loss_imitate
        self.update_optimizer(loss, ni) 
        
        # Log
        if self.RANK in [-1, 0]:
            self.meter.update(sup_loss_items)
            self.meter.update(un_sup_loss_items)
            if invalid_target_shape: #no pseudo label created
                hit_rate = dict(tp=0, fp_cls=0, fp_loc=0, pse_num=0, gt_num=0)
                self.meter.update(hit_rate)
            else:
                if self.target_with_gt:
                    tp_rate, fp_cls_rate, fp_loc_rate, pse_num, gt_num = check_pseudo_label_with_gt(unlabeled_targets, unlabeled_gt, \
                        ignore_thres_low=self.compute_un_sup_loss.ignore_thres_low, ignore_thres_high=self.compute_un_sup_loss.ignore_thres_high, \
                        batch_size=self.batch_size // self.WORLD_SIZE)
                else:
                    tp_rate, fp_loc_rate, pse_num, gt_num = check_pseudo_label(unlabeled_targets, \
                        ignore_thres_low=self.compute_un_sup_loss.ignore_thres_low, ignore_thres_high=self.compute_un_sup_loss.ignore_thres_high, \
                            batch_size=self.batch_size // self.WORLD_SIZE)
                    fp_cls_rate = 0
                hit_rate = dict(tp=tp_rate, fp_cls=fp_cls_rate, fp_loc=fp_loc_rate, pse_num=pse_num, gt_num=gt_num)
                self.meter.update(hit_rate)


            mloss_count= len(self.meter.meters.items())
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * (mloss_count + 2 )) % (
                f'{self.epoch}/{self.epochs - 1}', mem, targets.shape[0], imgs.shape[-1], *self.meter.get_avg()))
            
            callbacks.run('on_train_batch_end', ni, self.model, imgs, targets, paths, self.plots, self.sync_bn, self.cfg.Dataset.np)

    def train_with_unlabeled(self, callbacks):
        # hit_rate = dict(p_rate=0, r_rate=0); 

        if self.epoch_adaptor:
            self.nb = len(self.unlabeled_dataloader)  # number of batches
            pbar = enumerate(self.unlabeled_dataloader)
            if self.RANK in [-1, 0]:
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i , (target_imgs, target_gt, target_paths, _, target_imgs_ori, target_M) in pbar:
                ni = i + self.nb * self.epoch  # number integrated batches (since train start)
                imgs, targets, paths, _ = next(self.train_loader.__iter__())
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs = target_imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0 
                self.train_instance(imgs, targets, paths, target_imgs, target_imgs_ori, target_gt, target_M, ni, pbar, callbacks)
        else:
            pbar = enumerate(self.train_loader)
            if self.RANK in [-1, 0]:
                pbar = tqdm(pbar, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i , (imgs, targets, paths, _) in pbar:
                ni = i + self.nb * self.epoch  # number integrated batches (since train start)
                target_imgs, target_gt, target_paths, _, target_imgs_ori, target_M= next(self.unlabeled_dataloader.__iter__())
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs = target_imgs.to(self.device, non_blocking=True).float() / 255.0 
                target_imgs_ori = target_imgs_ori.to(self.device, non_blocking=True).float() / 255.0 
                self.train_instance(imgs, targets, paths, target_imgs, target_imgs_ori, target_gt, target_M, ni, pbar, callbacks)
            
        # end batch ------------------------------------------------------------------------------------------------
        
        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()  