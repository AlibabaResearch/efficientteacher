# EfficientTeacher by Alibaba Cloud 
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import logging
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective, random_perspective_keypoints
from utils.general import check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, \
    xywh2xyxy, xywhn2xyxy, xyxy2xywhn, xyn2xy, xyn2xy_new
from utils.torch_utils import torch_distributed_zero_first
import math
from .autoaugment_utils import distort_image_with_autoaugment

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def car_license_cutout(image, labels, ori_labels):
    # 针对车辆标注进行数据增强，同时不影响车牌数据
    h, w = image.shape[:2]
    stride = 16
    mask = np.zeros((int(h/stride), int(w/stride)))
    for label in labels:
        if label[0] == 0:
            mask[int((label[4] - (label[4] - label[2]) / 3) / stride):int(label[4]/stride ),int(label[1]/stride):int(label[3]/stride)] = 255
    for label in labels:
        if label[0] == 1:
            mask[int(label[2]/stride):int(label[4]/stride), int(label[1]/stride):int(label[3]/stride)] = 0
    valid_coord = np.where(mask == 255)

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    # scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0825] * 8 + [0.05125] * 16  # image size fraction
    # scales = [0.25] * 2 + [0.125] * 4 + [0.0825] * 8 + [0.05125] * 16  # image size fraction
    scales = [0.05125] * 16 + [0.0825] * 8 + [0.125] * 4 + [0.25] * 1   # image size fraction

    #发现当前车辆前景不够多，直接不要进行复杂的数据增强
    mask_num = int(valid_coord[0].shape[0])
    # print(len(scales))
    # print(mask_num)
    if mask_num < 30:
        return labels, ori_labels
    bin = int(mask_num / len(scales))
    # print(bin)
    for i, s in enumerate(scales):
        # index = random.randint(0, int(valid_coord[0].shape[0] - 1))
        mask_h = random.randint(1, max(int(h * s), 1))
        mask_w = random.randint(1, max(int(w * s), 1))

        xmin = int(valid_coord[1][i * bin] * stride - mask_w/2)
        ymin = int(valid_coord[0][i * bin] * stride - mask_h/2)
        xmax = int(min(image.shape[1], xmin + mask_w))
        ymax = int(min(image.shape[0], ymin + mask_h))

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.80]  # remove >60% obscured labels
            ori_labels = ori_labels[ioa < 0.80]

    # cv2.imwrite('after.jpg', image)
    return labels, ori_labels

def cutout(image, labels, ori_labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    # 针对车辆标注进行数据增强，同时不影响车牌数据

    h, w = image.shape[:2]
    #labels = np.array(labels)
    #print(len(labels))
    bbox_roi_x_start = int(min(labels[:,1]))
    bbox_roi_y_start = int(min(labels[:,2]))
    bbox_roi_x_end   = int(max(labels[:,3]))
    bbox_roi_y_end   = int(max(labels[:,4]))
    h = bbox_roi_y_end - bbox_roi_y_start
    w = bbox_roi_x_end - bbox_roi_x_start

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0825] * 8 + [0.05125] * 16  # image size fraction
    #cv2.imwrite('before.jpg', image)
    for s in scales:
        mask_h = random.randint(1, max(int(h * s), 1))
        mask_w = random.randint(1, max(int(w * s), 1))
        #mask_h = random.randint(1, int(h * s))
        #mask_w = random.randint(1, int(w * s))

        ## box
        #xmin = max(0, random.randint(0, w) - mask_w // 2)
        #ymin = max(0, random.randint(0, h) - mask_h // 2)
        #xmax = min(w, xmin + mask_w)
        #ymax = min(h, ymin + mask_h)
        xmin = int(max(0, random.randint(bbox_roi_x_start, bbox_roi_x_end) - mask_w // 2))
        ymin = int(max(0, random.randint(bbox_roi_y_start, bbox_roi_y_end) - mask_h // 2))
        xmax = int(min(image.shape[1], xmin + mask_w))
        ymax = int(min(image.shape[0], ymin + mask_h))

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.80]  # remove >60% obscured labels
            ori_labels = ori_labels[ioa < 0.80]
    # cv2.imwrite('after.jpg', image)

    return labels, ori_labels

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

class DistributeBalancedBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, balance_type='class_balance', labels=None):
        self.labels = labels
        self.oridata = dataset
        self.dataset = dict()
        self.balanced_max = 0
        self.balance_type = balance_type
        # Save all the indices for all the classes
        # for idx in range(0, len(dataset)):
        #     label = self._get_label(dataset, idx)
        #     # print('label:', label)
        #     if label not in self.dataset:
        #         self.dataset[label] = list()
        #     self.dataset[label].append(idx)
        #     self.balanced_max = len(self.dataset[label]) \
        #         if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        for idx in range(0, len(dataset)):
            label_list = self._get_label(dataset, idx)
            if label_list is not list and type(label_list) is not np.ndarray:
                label = label_list
                if label not in self.dataset:
                    self.dataset[label] = list()
                self.dataset[label].append(idx)
                self.balanced_max = len(self.dataset[label]) \
                    if len(self.dataset[label]) > self.balanced_max else self.balanced_max
            else:
                for label in label_list:
                    if label not in self.dataset:
                        self.dataset[label] = list()
                    self.dataset[label].append(idx)
                    self.balanced_max = len(self.dataset[label]) \
                        if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        # Oversample the classes with fewer elements than the max
        # pdb.set_trace()
        #print('balanced_max:', self.balanced_max)
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1 + rank] * len(self.keys)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = True
        self.seed = 0
        self.balanced_list = range(self.balanced_max)
        # self.num_samples = math.ceil((self.balanced_max * len(self.keys) - self.num_replicas) / self.num_replicas)
        # self.total_size = self.num_samples * self.num_replicas
        # print('num_samples:', self.num_samples)
        # print('total_size:', self.total_size)
        # print('num_replicas:', self.num_replicas)
        # print('num_keys:', self.keys)

    def __iter__(self):
        if self.shuffle:
           g = torch.Generator()
           g.manual_seed(self.seed + self.epoch)
           self.random_indices = []
           for i in range(len(self.keys)):
                self.random_indices.append(torch.randperm(len(self.balanced_list), generator=g).tolist()) # type: ignore
        while self.indices[self.currentkey] < self.balanced_max - self.num_replicas:
            self.indices[self.currentkey] += self.num_replicas
            if self.shuffle:
                yield self.dataset[self.keys[self.currentkey]][self.random_indices[self.currentkey][self.indices[self.currentkey]]]
            else:
                yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1 + self.rank] * len(self.keys)

    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()

        else:
            # Trying guessing
            # img_id = dataset.images[idx]
            if self.balance_type == 'dir_balance':
                label_name = dataset.img_files[idx].split('/')[3]
            elif self.balance_type == 'class_balance':
                label_name = np.unique(dataset.labels[idx][:, 0])
            # print('label_name:', label_name)
            # file_name = dataset.coco.loadImgs(ids=[img_id])[0]['file_name']

            # return re.findall('(<[^>]*>|.)', dataset.labelnames[idx])[0]
            # print('file_name:', file_name)
            return label_name

    def __len__(self):
        return int(len(self.oridata)/ self.num_replicas)

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            print('label:', label)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        # pdb.set_trace()
        print('balanced_max:', self.balanced_max)
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

        self.seed = 0
        self.balanced_list = range(self.balanced_max)
        self.shuffle = True
    def __iter__(self):
        if(self.shuffle):
          g = torch.Generator()
          g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
          self.random_indices = []
          for i in range(len(self.keys)):
                self.random_indices.append(torch.randperm(len(self.balanced_list), generator=g).tolist()) # type: ignore
        while self.indices[self.currentkey] < self.balanced_max - 1:
             self.indices[self.currentkey] += 1
             if self.shuffle:
                 #g = torch.Generator()
                 #g.manual_seed(self.seed + self.epoch)
                 #random_indices = torch.randperm(len(self.balanced_list), generator=g).tolist()  # type: ignore
                 #print('indices:', self.random_indices[self.currentkey][self.indices[self.currentkey]])
                 #print('currentkey:',self.currentkey)
                 #print('keys:',self.keys)
                 yield self.dataset[self.keys[self.currentkey]][self.random_indices[self.currentkey][self.indices[self.currentkey]]]
             else:
                 #print('indices:', self.indices)
                 #print('currentkey:',self.currentkey)
                 #print('keys:',self.keys)
                 yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
             self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)

    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            # img_id = dataset.images[idx]
            file_name = dataset.img_files[idx]
            # file_name = dataset.coco.loadImgs(ids=[img_id])[0]['file_name']

            # return re.findall('(<[^>]*>|.)', dataset.labelnames[idx])[0]
            # print('file_name:', file_name)
            return file_name.split('/')[3]

    def __len__(self):
        return self.balanced_max*len(self.keys)

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='',cfg=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      cfg=cfg,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    if 'train' in prefix:
        print('world_size:', WORLD_SIZE)
        print('rank:', rank)
        if cfg.Dataset.sampler_type=='normal':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
        elif cfg.Dataset.sampler_type=='class_balance':
            print('use banlanced batch sampler')
            sampler = DistributeBalancedBatchSampler(dataset, WORLD_SIZE,  rank, 'class_balance') if rank != -1 else BalancedBatchSampler(dataset)
        elif cfg.Dataset.sampler_type=='dir_balance':
            sampler = DistributeBalancedBatchSampler(dataset, WORLD_SIZE, rank, 'dir_balance') if rank != -1 else BalancedBatchSampler(
                dataset)
        else:
            assert NotImplementedError

    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):

        if path.endswith('.txt'):
            with open(path,'r') as f:
                allinfo = f.readlines()
                allinfo = [a.strip() for a in allinfo]
            if len(allinfo[0].split(' '))==2:
                allinfo = [a.split(' ')[0] for a in allinfo]
            files = allinfo
        else:
            p = str(Path(path).resolve())  # os-agnostic absolute path
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # glob
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            else:
                raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    print('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    # sa, sb = os.sep + '/mnt/datasets/' + os.sep, os.sep + '/home/bowen/annotations/labels/lubiantingche_total_train/' + os.sep  # /images/, /labels/ substrings
    # sa, sb = '/mnt/bowen/', '/mnt/annotations/annotations/labels/lubiantingche_total_2in1/'
    # sa, sb = '/mnt/bowen/', '/mnt/bowen/'
    # sa, sb = '/mnt/datasets/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    # sa, sb = '/mnt/bowen/exp/coco/images/val2017/', '/mnt/bowen/exp/coco/labels/val2017/'
    sa, sb = 'images', 'labels'
    # sa, sb = '/mnt/datasets/', '/home/bowen/annotations/labels/lubiantingche_total_test_badcase_2021_02_24/'
    # sa, sb = '/AIDATA/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    # sa, sb = '/mnt/bowen/', '/mnt/annotations/annotations/labels/xiaokongbao_total_train/'
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, cfg=None, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        self.debug = False
        if cfg:
            self.num_points = cfg.Dataset.np
            self.cfg = cfg
            self.use_car_license_augment = cfg.Dataset.use_car_license_augment
            self.debug = cfg.Dataset.debug
        else:
            self.num_points = 0
            self.use_car_license_augment = False
        newpath = []
        path = path.split('||')
        for p in path:
            if '*' in p:
                dirpath,times = p.split('*')
                for t in range(int(times)):
                    newpath.append(dirpath)
            else:
                newpath.append(p)
        path = newpath
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = p.strip()
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            # print(len(f),"******")
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS+['txt']])
            # self.img_files = f
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            # assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
        if self.img_files[0].endswith('.txt') and len(self.img_files[0].split(' ')) == 2:
            self.label_files = [a.split(' ')[1] for a in self.img_files]
            self.img_files = [a.split(' ')[0] for a in self.img_files]

        else:
            # self.label_files = img2label_paths(self.hyp.image_dir, self.hyp.label_dir, self.img_files)  # labels
            self.label_files = img2label_paths(self.img_files)  # labels
        # Check cache
        # self.label_files = img2label_paths(self.img_files)  # labels
        # print(self.img_files, "|", self.label_files)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            if cfg and cfg.check_datacache:
                assert cache['version'] == self.cache_version  # same version
                assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        if self.num_points==0:
            self.labels = [l[:,:5] for l in self.labels]
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                #if segment:
                #    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                # if segment:
                    # self.segments[i][:, 0] = 0
        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
                print('self im cache dir:', self.im_cache_dir)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix), repeat(self.num_points))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp.mosaic
        # mosaic = False
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index, self.num_points)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp.mixup:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1), self.num_points))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                # print('labels shape:', labels.shape[-1])
                # pseudo keypoints label的所有点都是0
                if labels.shape[-1] == 5 and self.num_points > 0:
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    labels_lmk_pad = np.ones((labels.shape[0], self.num_points * 2)) * -1 # num_points==4 means (x,y) * 4
                    labels = np.concatenate((labels, labels_lmk_pad), 1)
                elif labels.shape[-1] > 5 and self.num_points > 0:
                    non_valid_index = (labels[:, 5:].sum(1) == 0)
                    non_valid_num = non_valid_index.sum(0)
                # else:
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    for n in range(self.num_points):
                        start_index = 5 + n * 2
                        end_index = 7 + n * 2
                        labels[:, start_index:end_index] = xyn2xy_new(labels[:, start_index:end_index], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    # labels[:, 7:9] = xyn2xy_new(labels[:, 7:9], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    # labels[:, 9:11] = xyn2xy_new(labels[:, 9:11], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    # labels[:, 11:13] = xyn2xy_new(labels[:, 11:13], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    labels[non_valid_index, 5:] = np.ones((non_valid_num, self.num_points * 2)) * -1
                elif labels.shape[-1] == 5 and self.num_points == 0:
                    labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])


            
            if self.augment:
            # Augment imagespace
            # if not mosaic:
                img, labels = random_perspective_keypoints(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'],
                                                 num_points=self.num_points)

        nl = len(labels)  # number of labels
        # print('before normal:', labels[:])
        ori_labels = labels.copy()
        if nl:
            if self.augment:
                if random.random() < hyp['cutout']:
                # if (0):
                #  if(len(labels) > 0):
                    # labels, ori_labels = cutout(img, labels, ori_labels)
                    labels, ori_labels = car_license_cutout(img, labels, ori_labels)

            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
            for n in range(5, 5 + self.num_points * 2):
                if n % 2 == 0:
                    labels[:, n] /= img.shape[0]
                else:
                    labels[:, n] /= img.shape[1]
            # labels[:, [5, 7, 9, 11]] /= img.shape[1]  # normalized landmark x 0-1
            # labels[:, [6, 8, 10, 12]] /= img.shape[0]  # normalized landmark y 0-1
        # print('after normal:', labels[:])
        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    for n in range(5, 5 + self.num_points * 2):
                        if n % 2 == 0:
                            labels[:, n] = 1 - labels[:, n]
                    # labels[:, 6] = 1 - labels[:, 6]
                    # labels[:, 8] = 1 - labels[:, 8]
                    # labels[:, 10] = 1 - labels[:, 10]
                    # labels[:, 12] = 1 - labels[:, 12]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    for n in range(5, 5 + self.num_points * 2):
                        if n % 2 != 0:
                            labels[:, n] = 1 - labels[:, n]
                    # labels[:, 5] = 1 - labels[:, 5]
                    # labels[:, 7] = 1 - labels[:, 7]
                    # labels[:, 9] = 1 - labels[:, 9]
                    # labels[:, 11] = 1 - labels[:, 11]
                    # TODO
                    if self.num_points == 4:
                        labels = np.concatenate([labels[:, :5], labels[:, 7:9], labels[:, 5:7], labels[:, 11:13], labels[:, 9:11]], axis=1)
                    elif self.num_points == 8:
                    # elif self.num_points == 8:
                        labels = np.concatenate([labels[:, :5], labels[:, 7:9], labels[:, 5:7], labels[:, 11:13], labels[:, 9:11], labels[:, 15:17], labels[:, 13:15], labels[:, 19:21], labels[:, 17:19]], axis=1)
                    # print('after:', labels)
            if self.num_points > 0:
                non_valid_index = ((ori_labels[:, 5:] == -1).sum(1) == self.num_points * 2)
                labels[non_valid_index, 5:] = ori_labels[non_valid_index, 5:]

        nl = len(labels)  # number of labels
        labels_out = torch.zeros((nl, 6 + self.num_points * 2))
        if nl:
            path = self.img_files[index]
            if self.use_car_license_augment:
                # dipan
                include_class = self.cfg.Dataset.chassis_id  # filter labels to include only these classes (optional)
                if len(include_class)>0:
                    include_class_array = np.array(include_class).reshape(1, -1)
                    j = (labels[:, 0:1] == include_class_array).any(1)
                    if sum(j) != 0:
                        labels[j, 5:13] = self.up_order_points_quadrangle_new(labels[j, 5:13])
                # chepai
                include_class = self.cfg.Dataset.plate_id  # filter labels to include only these classes (optional)
                if len(include_class)>0:
                    include_class_array = np.array(include_class).reshape(1, -1)
                    j = (labels[:, 0:1] == include_class_array).any(1)
                    if sum(j) != 0:
                        labels[j, 5:13] = self.order_points_quadrangle(labels[j, 5:13])
            labels[:, 1:] = labels[:, 1:].clip(0, 1.0)
            labels_out[:, 1:] = torch.from_numpy(labels)

            if self.debug: 
                if self.num_points > 0:
                    self.showlabels(img, labels[:, 1:5], labels[:, 5:5+ self.num_points *2 ], path)
                else:
                    self.showlabels(img, labels[:, 1:5], [], path)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def order_points_quadrangle(self, pts_sample):
        # 车牌标注点排序
        new_sample = []
        for pts in pts_sample:
            pts = np.reshape(pts,(4,2))
            # sort the points based on their x-coordinates
            xSorted = pts[np.argsort(pts[:, 0]), :]

            # grab the left-most and right-most points from the sorted
            # x-roodinate points
            leftMost = xSorted[:2, :]
            rightMost = xSorted[2:, :]

            # now, sort the left-most coordinates according to their
            # y-coordinates so we can grab the top-left and bottom-left
            # points, respectively
            if leftMost[0,1]!=leftMost[1,1]:
                leftMost=leftMost[np.argsort(leftMost[:,1]),:]
            else:
                leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
            # leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost

            if rightMost[0,1]!=rightMost[1,1]:
                rightMost=rightMost[np.argsort(rightMost[:,1]),:]
            else:
                rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
            (tr,br)=rightMost
            new_sample.append([tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]])
        return np.array(new_sample)

    def up_order_points_quadrangle_new(self, pts_sample):
        new_sample = []
        for pts in pts_sample:
            pts = np.reshape(pts,(4,2))
            center_pt_x = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4
            center_pt_y = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4
            d2s = []
            for i in range(pts.shape[0]):
        # print(i)
        # print(pts[i][0][0])
                vector_x = pts[i][0] - center_pt_x
                vector_y = pts[i][1] - center_pt_y
                # theta = np.arctan2(vector_y, vector_x) - np.pi - np.pi/18
                theta = np.arctan2(vector_y, vector_x)
        # theta = np.arctan2(vector_y, vector_x)
                # if theta < 0:
                    # theta += np.pi * 2
        # print(theta)
                d2s.append([pts[i], theta])

            d2s = sorted(d2s, key=lambda x:x[1])
            # order_pts = np.array([x[0] for x in d2s]).reshape(-1, 1, 2)
            tmp = [d2s[0][0][0], d2s[0][0][1], d2s[1][0][0], d2s[1][0][1], \
                d2s[2][0][0], d2s[2][0][1], d2s[3][0][0], d2s[3][0][1]]
            # new_sample.append([x[0].reshape(-1, 8) for x in d2s])
            # print('d2s:' , d2s , ' tmp:', tmp)
            new_sample.append(tmp)
        return np.array(new_sample)
        # return order_pts

    # 底盘标注点排序
    def up_order_points_quadrangle(self, pts_sample):
        new_sample = []
        for pts in pts_sample:
            pts = np.reshape(pts,(4,2))
            # sort the points based on their y-coordinates
            xSorted = pts[np.argsort(pts[:, 1]), :]
            # grab the left-most and right-most points from the sorted
            # x-roodinate points
            upMost = xSorted[:2, :]
            bottomMost = xSorted[2:, :]

            # now, sort the left-most coordinates according to their
            # y-coordinates so we can grab the top-left and bottom-left
            # points, respectively
            if upMost[0, 0] != upMost[1, 0]:
                upMost = upMost[np.argsort(upMost[:, 0]), :]
            else:
                upMost = upMost[np.argsort(upMost[:, 0])[::-1], :]
            # leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
            # leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, tr) = upMost

            if bottomMost[0, 0] != bottomMost[1, 0]:
                bottomMost = bottomMost[np.argsort(bottomMost[:, 0]), :]
            else:
                bottomMost = bottomMost[np.argsort(bottomMost[:, 0])[::-1], :]
            (bl, br) = bottomMost
            new_sample.append([tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]])
        return np.array(new_sample)
        # return np.array([[tl[0],tl[1]],[tr[0], tr[1]], [br[0], br[1]], [bl[0], bl[1]]])

    def showlabels(self, img, boxs, landmarks, path):
        img = img.astype(np.uint8)
        for box in boxs:
            x,y,w,h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            # print('xywh:', x,y,w,h)
            # print('img:', img)
            cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
        if landmarks != []:
          for landmark in landmarks:
            #cv2.circle(img,(60,60),30,(0,0,255))
            for i in range(8):
                cv2.circle(img, (int(landmark[2*i] * img.shape[1]), int(landmark[2*i+1]*img.shape[0])), 3 ,(0,0,255), -1)
            cv2.putText(img, "0", (int(landmark[0] * img.shape[1]), int(landmark[1]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "1", (int(landmark[2] * img.shape[1]), int(landmark[3]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "2", (int(landmark[4] * img.shape[1]), int(landmark[5]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "3", (int(landmark[6] * img.shape[1]), int(landmark[7]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "4", (int(landmark[8] * img.shape[1]), int(landmark[9]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "5", (int(landmark[10] * img.shape[1]), int(landmark[11]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "6", (int(landmark[12] * img.shape[1]), int(landmark[13]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
            cv2.putText(img, "7", (int(landmark[14] * img.shape[1]), int(landmark[15]*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 120/ (150), (0, 255, 0), round(240 / 150))
        cv2.imwrite('tmp/' + path.split('/')[-1],img)
        # assert 0

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, 'Image Not Found ' + path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            # im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST)
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index, num_points):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        # print('before labels:', labels)
        # print('num_points', num_points)
        if labels.size:
            if labels.shape[-1] == 5 and num_points > 0:
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                labels_lmk_pad = np.ones((labels.shape[0], num_points * 2)) * -1
                labels = np.concatenate((labels, labels_lmk_pad), 1)
            elif labels.shape[-1] > 5 and num_points > 0:
                non_valid_index = (labels[:, 5:].sum(1) == 0)
                non_valid_num = non_valid_index.sum(0)
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                for n in range(5, 5 + num_points * 2, 2):
                    labels[:, n:n+2] = xyn2xy_new(labels[:, n:n+2], w, h, padw, padh)
                labels[non_valid_index, 5:] = np.ones((non_valid_num, num_points * 2)) * -1
            elif labels.shape[-1] == 5 and num_points == 0:
                labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], w, h, padw, padh)
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        # print('labels:', labels)
        # assert 0
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:5], *segments4):
        # np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate
    # print('labels4:', labels4)

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp.copy_paste)
    img4, labels4 = random_perspective_keypoints(img4, labels4, segments4,
                                       degrees=self.hyp.degrees,
                                       translate=self.hyp.translate,
                                       scale=self.hyp.scale,
                                       shear=self.hyp.shear,
                                       perspective=self.hyp.perspective,
                                       num_points=num_points,
                                       border=self.mosaic_border)  # border to remove
    # print('labels4:', labels4)

    return img4, labels4


def load_mosaic9(self, index):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective_keypoints(img9, labels9, segments9,
                                       degrees=self.hyp.degrees,
                                       translate=self.hyp.translate,
                                       scale=self.hyp.scale,
                                       shear=self.hyp.shear,
                                       perspective=self.hyp.perspective,
                                       num_points=self.num_points,
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in IMG_FORMATS], [])  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix, num_points = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    length_with_points = 5 + num_points * 2
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    Image.open(im_file).save(im_file, format='JPEG', subsampling=0, quality=100)  # re-save image
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, 'r') as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                classes = []
                box_segments = []
                keypoints_segments = []
                if num_points > 0:
                    for x in l:
                        if len(x) == length_with_points:
                            classes.append(x[0])
                            box_segments.append(np.array(x[1:5], dtype=np.float32))
                            keypoints_segments.append(np.clip(np.array(x[5:length_with_points], dtype=np.float32), 0, 1.0))
                        else:
                            classes.append(x[0])
                            box_segments.append(np.array(x[1:5], dtype=np.float32))
                            keypoints_segments.append(np.array([0] * num_points * 2, dtype=np.float32))
                    if len(l):
                        classes = np.array(classes, dtype=np.float32)
                        box_segments = np.array(box_segments)
                        keypoints_segments = np.array(keypoints_segments)
                        l = np.concatenate((classes.reshape(-1, 1), box_segments, keypoints_segments), 1)
                else:
                    for x in l:
                        classes.append(x[0])
                        box_segments.append(np.array(x[1:5], dtype=np.float32))
                    if len(l):
                        classes = np.array(classes, dtype=np.float32)
                        box_segments = np.array(box_segments)
                        l = np.concatenate((classes.reshape(-1, 1), box_segments), 1)
                # for x in l:
                #     if len(x) == length_with_points and num_points > 0:
                #         classes.append(x[0])
                #         box_segments.append(np.array(x[1:5], dtype=np.float32))
                #         keypoints_segments.append(np.clip(np.array(x[5:length_with_points], dtype=np.float32), 0, 1.0))
                #     elif len(x) == 5 and num_points > 0:
                #         classes.append(x[0])
                #         box_segments.append(np.array(x[1:5], dtype=np.float32))
                #         keypoints_segments.append(np.array([0] * num_points * 2, dtype=np.float32))
                #     elif len(x) == 5 and num_points == 0:
                #         classes.append(x[0])
                #         box_segments.append(np.array(x[1:5], dtype=np.float32))
                # if len(l) and num_points > 0:
                #     classes = np.array(classes, dtype=np.float32)
                #     box_segments = np.array(box_segments)
                #     keypoints_segments = np.array(keypoints_segments)
                #     l = np.concatenate((classes.reshape(-1, 1), box_segments, keypoints_segments), 1)
                # elif len(l) and num_points == 0:
                #     classes = np.array(classes, dtype=np.float32)
                #     box_segments = np.array(box_segments)
                #     # keypoints_segments = np.array(keypoints_segments)
                #     l = np.concatenate((classes.reshape(-1, 1), box_segments), 1)
                

            nl = len(l)
            if nl:
                assert l.shape[1] == length_with_points, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    segments = np.unique(segments, axis=0)
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, length_with_points), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, length_with_points), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *[round(x, 4) for x in points]] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, quality=75)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file, 'r') as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats
