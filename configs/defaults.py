#Copyright (c) 2023, Alibaba Group
from .yacs import CfgNode as CN

#该部分尽量保持与yolov5一致
_C = CN()
_C.debug= False 
_C.do_test=False
_C.finetune=False
_C.device= ''
_C.ngpu=1 #训练时使用的gpu个数
_C.adam=False #是否使用Adam优化函数
_C.prune_finetune=False #是否根据一个已剪枝的模型进行finetune
_C.reinitial=False #读取的剪枝预训练模型是否重新初始化
_C.noautoanchor=True 
_C.project=''
_C.name='exp'
_C.epochs=300
_C.val_conf_thres = 0.001
# _C.batch_size=32
# _C.workers=8
_C.local_rank=-1
_C.save_period=-1
_C.weights=''
_C.freeze_layer_num = 0
_C.cache=False
_C.rect=False
_C.save_dir=''
_C.single_cls=False #是否使用单类别进行训练，将数据集中的类别id强制置0
_C.evolve=False
_C.noval=False
_C.nosave=False
_C.sync_bn=False
_C.resume=False
_C.exist_ok=False
_C.linear_lr=False
_C.check_datacache=False
_C.entity=None
_C.upload_dataset=False
_C.bbox_interval=-1
_C.artifact_alias='latest'
_C.find_unused_parameters=False #在新加入网络结构时用于检查有向无环图是否合理

#该部分尽量保持与yolov5一致，方便迁移调参
_C.hyp=CN()
_C.hyp.use_aug=True  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.hyp.lr0=0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.hyp.lrf=0.01  # final OneCycleLR learning rate (lr0 * lrf)
_C.hyp.momentum=0.937  # SGD momentum/Adam beta1
_C.hyp.weight_decay=0.0005  # optimizer weight decay 5e-4
_C.hyp.warmup_epochs=0  # warmup epochs (fractions ok)
_C.hyp.warmup_momentum=0.8  # warmup initial momentum
_C.hyp.warmup_bias_lr=0.1  # warmup initial bias lr

_C.hyp.hsv_h=0.5  # image HSV-Hue augmentation (fraction)
_C.hyp.hsv_s=0.5  # image HSV-Saturation augmentation (fraction)
_C.hyp.hsv_v=0.5  # image HSV-Value augmentation (fraction)
_C.hyp.degrees=0.0  # image rotation (+/- deg)
_C.hyp.translate=0.1  # image translation (+/- fraction)
_C.hyp.scale=0.5  # image scale (+/- gain)
_C.hyp.shear=0.0  # image shear (+/- deg)
_C.hyp.perspective=0.0  # image perspective (+/- fraction), range 0-0.001
_C.hyp.flipud=0.0  # image flip up-down (probability)
_C.hyp.fliplr=0.5  # image flip left-right (probability)
_C.hyp.mosaic=1.0  # image mosaic (probability)
_C.hyp.mixup=0.0  # image mixup (probability)
# _C.hyp.sparse_rate=0.01
_C.hyp.burn_epochs=1
_C.hyp.copy_paste=0.0  # segment copy-paste (probability)
_C.hyp.no_aug_epochs=0 # let dataloader close mosaic  
_C.hyp.cutout=0.0 


_C.Model=CN()
# _C.Model.noautoanchor=False 
_C.Model.weights=''
# _C.Model.dynamic_load = False

_C.Model.width_multiple = 1.0 #width ratio for s/m/l model
_C.Model.depth_multiple = 1.0 #depth ratio for s/m/l model
_C.Model.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
_C.Model.ch = 3 #输入通道数

_C.Model.Backbone = CN()
_C.Model.Backbone.name = 'darknet'
_C.Model.Backbone.stage_repeats = [4, 8, 4]
_C.Model.Backbone.output_layers = [6, 14, 18]
_C.Model.Backbone.model_size = '0.2x'
_C.Model.Backbone.activation = 'LeakyReLU'
_C.Model.Backbone.arch = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1] 
_C.Model.Backbone.first_input_channels = 1
_C.Model.Backbone.out_stages = [2, 3, 4]
_C.Model.Backbone.kernel_size = 3
_C.Model.Backbone.with_last_conv = False
_C.Model.Backbone.pretrain = False
_C.Model.Backbone.in_channels = 3
_C.Model.Backbone.out_channels = [64, 128, 256, 512, 1024]
_C.Model.Backbone.num_repeats = [1, 6, 12, 18, 6]
_C.Model.Backbone.lite_conv = False

_C.Model.Neck = CN()
_C.Model.Neck.name = 'darknet'
_C.Model.Neck.in_channels = [32, 64, 128]
_C.Model.Neck.out_channels = [32]
_C.Model.Neck.start_level = 0
_C.Model.Neck.end_level = -1
_C.Model.Neck.num_outs = 3
_C.Model.Neck.activation = 'ReLU'
_C.Model.Neck.interpolate = 'bilinear'
_C.Model.Neck.num_repeats = [12, 12, 12, 12]

_C.Model.Head = CN()
_C.Model.Head.name = 'darknet'
_C.Model.Head.share_cls_reg = False
_C.Model.Head.activation = 'SiLU'
_C.Model.Head.conv_type = 'DWConv'
_C.Model.Head.stacked_convs= 2
_C.Model.Head.octave_base_scale= 5
_C.Model.Head.feat_channels = 256
_C.Model.Head.strides = [8,16,32]
_C.Model.Head.in_channels = [128, 256, 512]
_C.Model.Head.num_decouple = 2

_C.Model.RepOpt = False
_C.Model.RepScale_weight = ''
_C.Model.RealVGGModel = False
_C.Model.LinearAddModel = False
_C.Model.QARepVGGModel = False

_C.Model.inplace=True
_C.Model.prior_prob = 0.01

_C.Loss = CN()

#YOLOv5使用的loss参数
_C.Loss.type = 'ComputeXLoss'# box loss gain
_C.Loss.box=0.05# box loss gain
_C.Loss.cls=0.5# cls loss gain
_C.Loss.cls_pw=1.0# cls BCELoss positive_weight
_C.Loss.obj=1.0 # obj loss gain (scale with pixels)
_C.Loss.obj_pw=1.0 # obj BCELoss positive_weight
_C.Loss.fl_gamma=0.0# focal loss gamma (efficientDet default gamma=1.5)
_C.Loss.autobalance= False
_C.Loss.label_smoothing=0.0
_C.Loss.anchor_t = 4.0 #标准yolov5计算正样本的阈值
_C.Loss.kp_loss_weight = 10.0  #关键点检测部分loss的权重
_C.Loss.static_assigner_epoch = 5
_C.Loss.single_targets=False #设置为True时 不使用multi positive采样

#nanodet使用的loss参数
_C.Loss.qfl_use_sigmoid=True 
_C.Loss.qfl_beta=2.0 #nanodet
_C.Loss.qfl_loss_weight=1.0
_C.Loss.dfl_loss_weight=0.25
_C.Loss.reg_max=7

#YOLOX使用的loss参数
_C.Loss.box_loss_weight=5.0
_C.Loss.obj_loss_weight=1.0
_C.Loss.cls_loss_weight=1.0
_C.Loss.iou_obj=False

#YOLOv6使用的loss参数
_C.Loss.use_dfl=True
_C.Loss.grid_cell_size=5.0
_C.Loss.grid_cell_offset=0.5
_C.Loss.iou_type='giou'
_C.Loss.use_gfl=False

#TAL使用的loss参数
_C.Loss.top_k=13
_C.Loss.assigner_type='TAL' #TAL/SimOTA/AnchorBased/ATSS
_C.Loss.embedding=64 #embedding长度


_C.Dataset=CN()
_C.Dataset.train=''  # 训练集数据地址
_C.Dataset.val=''  # 验证集数据地址，每训练完一个epoch在该数据集上进行验证
_C.Dataset.test=''  # 测试集数据地址
_C.Dataset.target = '' # 无标签数据集地址，用于进行半监督训练
_C.Dataset.img_path=''
_C.Dataset.label_path=''

_C.Dataset.batch_size=96 #训练时的batch图片数，8卡训练时每张卡获得96/8=12张图，以此类推
_C.Dataset.img_size=640 #训练时将图片resize的尺寸
_C.Dataset.rect=False #是否使用长方形进行训练
_C.Dataset.workers=16 #dataloader读图时使用的线程数，增加可一定程度加速训练
_C.Dataset.quad=False
# _C.Dataset.use_bgr=False
# _C.Dataset.lp_order=False
# _C.Dataset.label_length= 14
_C.Dataset.nc=80
_C.Dataset.np=0 #标注中关键点个数
_C.Dataset.num_ids=0 #如果不为0，那么标注中含有跟踪或者实例id
_C.Dataset.pseudo_ids=False #打开pseudo_ids后，会自动对没有id的标签按顺序添加标签
_C.Dataset.names=[] #训练集标签的名称
_C.Dataset.include_class=[] #只读取部分id的标签
_C.Dataset.data_name='default_name' #训练集的名称
_C.Dataset.sampler_type='normal' #训练数据的采样方法, normal|class_balance|dir_balance
_C.Dataset.norm_scale=255.0 #预处理数值 img/255
_C.Dataset.debug= False #开启后会将标注渲染到图片上保存本地
_C.Dataset.val_kp= False #验证时是否计算关键点的AP

#量化训练参数
_C.Qat=CN()
_C.Qat.use_qat = False  # True时重载QatTrainer
_C.Qat.quant_dir = False  # 用于保存ptq以及qat生成的模型、模型profile
_C.Qat.bitmode = 'int8'  # 默认为int8，可选int4，mix等等
_C.Qat.backend = 'tensorrt'  # backends
_C.Qat.use_defaultfuse = False  
_C.Qat.use_quant_sensitivity_analysis = True #默认开启量化敏感度分析, 开启后Qat.sensitive_num设置有效， 否则无效
_C.Qat.sensitive_num = -1  # 0代表全部量化，-1代表智能量化根据sensitive_relerror来判断，>0代表指定不量化的层数
_C.Qat.sensitive_relerror = 0.01  # -1 时启用
_C.Qat.sensitive_eval_batch = 30  # -1 时启用

#剪枝压缩训练参数
_C.Prune=CN()
_C.Prune.use_sparse = False
# 重要参数，建议在1e-2~1e-4之间，太高会导致模型在初始阶段被剪枝，此时模型结构还并不合理，太低会导致稀疏达不到预设剪枝率。
# 建议微调该值，使得收敛到目标准确率的50%以上时达到flops_target，剪枝。
_C.Prune.sparse_rate = 1e-3
_C.Prune.flops_target = 0.3
# 重要参数，每prune_freq进行一次模型检验，达到flops_target后进行剪枝，之后便关闭稀疏训练，每次剪枝率测试<1s
_C.Prune.prune_freq = 50
# 重要参数，nvidia推理建议使用8倍数
_C.Prune.channel_divide = 8
# 迭代式剪枝
_C.Prune.iterative_prune = False

# 暂时
_C.Prune.ft_reinit = False
_C.Prune.prune_finetune=False
_C.Prune.sr_type = ''
_C.Prune.update_sr = False

#蒸馏训练参数
_C.Distill=CN()
_C.Distill.use_distill=False
_C.Distill.dist_loss='l2'
_C.Distill.Tmodel=''
_C.Distill.temp=20
_C.Distill.giou=0.05
_C.Distill.dist=1.0
_C.Distill.boxloss=False
_C.Distill.objloss=False
_C.Distill.clsloss=False
_C.Distill.loss_type=''


#半监督训练参数
_C.SSOD = CN()
_C.SSOD.train_domain = False
_C.SSOD.extra_teachers = [ ]
_C.SSOD.extra_teachers_class_names = [ ]
_C.SSOD.conf_thres = 0.65
_C.SSOD.valid_thres = 0.55
_C.SSOD.nms_conf_thres = 0.3
_C.SSOD.nms_iou_thres = 0.6
_C.SSOD.teacher_loss_weight = 0.1 #关键参数，伪标签loss权重
_C.SSOD.cls_loss_weight= 0.0
_C.SSOD.obj_loss_weight= 1.0
_C.SSOD.box_loss_weight= 0.0
_C.SSOD.focal_loss= 0.0
_C.SSOD.loss_type='ComputeStudentLoss'
_C.SSOD.pseudo_label_type='FairPseudoLabel'
_C.SSOD.debug=False #load target datasets with gt
_C.SSOD.with_da_loss=False #turn on domain adaptation loss
_C.SSOD.da_loss_weights=0.1 #ratio of d_loss and t_loss
_C.SSOD.ema_rate= 0.999 #关键参数，对半监督学习影响较大
_C.SSOD.ignore_thres_high=0.3 #pseudo label assigner中的高阈值
_C.SSOD.ignore_thres_low=0.3 #pseudo label assigner中的低阈值
_C.SSOD.dynamic_thres_epoch=0
_C.SSOD.uncertain_aug=False #pseudo label assigner是否开启multi positive
_C.SSOD.use_ota= False
_C.SSOD.multi_label= False #生成伪标签时的nms是否做multi_label
_C.SSOD.ignore_obj = False #pseudo label assigner是忽略梯度还是使用软标签
# _C.SSOD.label_match_percent=0.2
_C.SSOD.resample_high_percent=0.0 #重采样时高阈值位于得分序列的位置
_C.SSOD.resample_low_percent=0.0 #重采样时低阈值位于得分序列的为孩子
_C.SSOD.multi_step_lr=False #是否使用指定迭代学习率下降策略
_C.SSOD.milestones=[10, 20] #在指定的epoch进行学习率下降，一般变为原来的0.1
_C.SSOD.pseudo_label_with_obj=False #pseudo label assigner生成伪标签时的得分是否转换为obj得分
_C.SSOD.pseudo_label_with_bbox=False #pseudo label assigner是否用uncertain label算bbox loss
_C.SSOD.pseudo_label_with_cls=False #pseudo label assigner是否用uncertain label算cls loss
_C.SSOD.epoch_adaptor=True #是否开启epoch_adaptor
_C.SSOD.teacher_ota_cost=False #是否使用teacher模型的输出来计算ota的cost
_C.SSOD.iou_type='giou'
_C.SSOD.cosine_ema=True #是否开启cosine ema方案
_C.SSOD.imitate_teacher=False #是否开启imitate方案
_C.SSOD.fixed_accumulate=False #开启时，关闭动态optimizer更新方案

_C.SSOD.ssod_hyp = CN()
_C.SSOD.ssod_hyp.mosaic=1.0
_C.SSOD.ssod_hyp.degrees=0.0  # image rotation (+/- deg)
_C.SSOD.ssod_hyp.translate=0.1  # image translation (+/- fraction)
_C.SSOD.ssod_hyp.scale=0.5  # image scale (+/- gain)
_C.SSOD.ssod_hyp.shear=0.0  # image shear (+/- deg)
_C.SSOD.ssod_hyp.flipud=0.0  # image flip up-down (probability)
_C.SSOD.ssod_hyp.fliplr=0.5  # image flip left-right (probability)
_C.SSOD.ssod_hyp.perspective=0.0
_C.SSOD.ssod_hyp.hsv_h=0.015  # image HSV-Hue augmentation (fraction)
_C.SSOD.ssod_hyp.hsv_s=0.7  # image HSV-Saturation augmentation (fraction)
_C.SSOD.ssod_hyp.hsv_v=0.4  # image HSV-Value augmentation (fraction)
_C.SSOD.ssod_hyp.with_gt=False #load target datasets with gt
_C.SSOD.ssod_hyp.cutout=0.9 #probability of cutout
_C.SSOD.ssod_hyp.autoaugment=0.9 #probability of auto augmentation utility 

# NAS
_C.NAS = CN()
_C.NAS.use_nas = False
_C.NAS.width_range = []
# 默认值很大
_C.NAS.params_target = [0,1e10]
_C.NAS.flops_target = [0,1e10]
_C.NAS.GEA = CN()
# GEA论文官方参数
_C.NAS.GEA.pop_size = 10
_C.NAS.GEA.sample_size = 3
_C.NAS.GEA.sample_epochs = 20
_C.NAS.GEA.sample_dataIter = -1
# 遗传搜索cycles个模型
_C.NAS.GEA.cycles = 100



def get_cfg():
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C.clone()