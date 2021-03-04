#!/usr/bin/python
import os
import sys
import time
import numpy as np
import copy
import scipy
import pickle
import builtins
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# import apex
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from config_modfeatures import configurations
from backbone import *
from head import *
from loss.loss import *
from dataset import *
from util.utils import *
from util.flops_counter import *
from optimizer.lr_scheduler import *
from optimizer.optimizer import *
#from torchprofile import profile_macs

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    


def main():
    cfg = configurations[1]
    
    GPU_AVAILABLE = torch.cuda.is_available()
    # GPU_AVAILABLE = False
            
    if GPU_AVAILABLE:
        device = torch.device('cuda:0')
        
        ngpus_per_node = len(cfg['GPU'])
        cfg['GPU'] = cfg['GPU'][0]
        world_size = cfg['WORLD_SIZE']
        cfg['WORLD_SIZE'] = ngpus_per_node * world_size
        cfg['GPU'] = 'cuda'

    else:
        ngpus_per_node = 1
        device = torch.device('cpu')
        cfg['GPU'] = 'cpu'
        
    val_dataset = get_val_data2(cfg['VAL_DATA_ROOT'], cfg['VAL_SET'], cfg['VAL_IN_LOWRES1'], cfg['VAL_IN_LOWRES2'])
    # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, val_dataset))

    SEED = cfg['SEED'] # random seed for reproduce results
    set_seed(SEED)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    if cfg['GPU']  != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    # cfg['RANK'] = cfg['RANK'] * ngpus_per_node + gpu
    # dist.init_process_group(backend=cfg['DIST_BACKEND'], init_method = cfg["DIST_URL"], world_size=cfg['WORLD_SIZE'], rank=cfg['RANK'])

    # Data loading code
    batch_size = int(cfg['BATCH_SIZE'])
    per_batch_size = int(batch_size / ngpus_per_node)
    #workers = int((cfg['NUM_WORKERS'] + ngpus_per_node - 1) / ngpus_per_node) # dataload threads
    workers = int(cfg['NUM_WORKERS'])
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    VAL_DATA_ROOT = cfg['VAL_DATA_ROOT']
    VAL_SET = cfg['VAL_SET']
    #RECORD_DIR = cfg['RECORD_DIR']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    DROP_LAST = cfg['DROP_LAST']
    OPTIMIZER = cfg['OPTIMIZER']
    LR_SCHEDULER = cfg['LR_SCHEDULER']
    LR_STEP_SIZE = cfg['LR_STEP_SIZE']
    LR_DECAY_EPOCH = cfg['LR_DECAY_EPOCH']
    LR_DECAT_GAMMA = cfg['LR_DECAT_GAMMA']
    LR_END = cfg['LR_END']
    WARMUP_EPOCH = cfg['WARMUP_EPOCH']
    WARMUP_LR = cfg['WARMUP_LR']
    NUM_EPOCH = cfg['NUM_EPOCH']
    USE_APEX = cfg['USE_APEX']
    EVAL_FREQ = cfg['EVAL_FREQ']
    SYNC_BN = cfg['SYNC_BN']
    RECORD_DIR = cfg['DATA_RECORD']


    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)
    transform_list = [transforms.RandomHorizontalFlip(),]
    if cfg['COLORJITTER']:
        transform_list.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
    if cfg['CUTOUT']:
        transform_list.append(Cutout())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean = RGB_MEAN,std = RGB_STD))

    # For lowres images: upscale x4 to 36x36 --> 144x144
    # if 'lowres' in cfg['DATA_RECORD']:
    #     transform_list.append(torch.nn.Upsample(cfg['INPUT_SIZE']))
    if cfg['RANDOM_ERASING']:
        transform_list.append(transforms.RandomErasing())
    train_transform = transforms.Compose(transform_list)
    if cfg['RANDAUGMENT']:
        train_transform.transforms.insert(0, RandAugment(n=cfg['RANDAUGMENT_N'], m=cfg['RANDAUGMENT_M']))

    
    print("=" * 60)
    print(train_transform)
    print("Train Transform Generated")
    print("=" * 60)
    dataset_train = FaceDataset2Sizes(DATA_ROOT, RECORD_DIR, train_transform, train_in_lowres1=cfg['VAL_IN_LOWRES1'], train_in_lowres2=cfg['VAL_IN_LOWRES2'])

    # dataset_train = MXFaceDataset(DATA_ROOT, train_transform)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=per_batch_size,
                                                shuffle = (train_sampler is None), num_workers=workers,
                                                # pin_memory=True, sampler=train_sampler, drop_last=DROP_LAST)
                                                pin_memory=True, drop_last=DROP_LAST)
    #SAMPLE_NUMS = dataset_train.get_sample_num_of_each_class()
    NUM_CLASS = len(train_loader.dataset.classes)
    # NUM_CLASS = train_loader.dataset.classes
    print("Number of Training Classes: {}".format(NUM_CLASS))

    

    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'MobileFaceNet': MobileFaceNet,
                     'ResNet_50': ResNet_50, 'ResNet_101': ResNet_101, 'ResNet_152': ResNet_152, 'IR_SE_18': IR_SE_18,
                     'IR_50': IR_50, 'IR_100': IR_100, 'IR_101': IR_101, 'IR_152': IR_152, 'IR_185': IR_185, 'IR_200': IR_200,
                     'IR_SE_50': IR_SE_50, 'IR_SE_100': IR_SE_100, 'IR_SE_101': IR_SE_101, 'IR_SE_152': IR_SE_152, 'IR_SE_185': IR_SE_185, 'IR_SE_200': IR_SE_200,
                     'AttentionNet_IR_56': AttentionNet_IR_56,'AttentionNet_IRSE_56': AttentionNet_IRSE_56,'AttentionNet_IR_92': AttentionNet_IR_92,'AttentionNet_IRSE_92': AttentionNet_IRSE_92,
                     'ResNeSt_50': resnest50, 'ResNeSt_101': resnest101, 'ResNeSt_100': resnest100,
                     'GhostNet': GhostNet, 'MobileNetV3': MobileNetV3, 'ProxylessNAS': proxylessnas, 'EfficientNet': efficientnet,
                     'DenseNet': densenet, 'ReXNetV1': ReXNetV1, 'MobileNeXt': MobileNeXt, 'MobileNetV2': MobileNetV2
                    } #'HRNet_W30': HRNet_W30, 'HRNet_W32': HRNet_W32, 'HRNet_W40': HRNet_W40, 'HRNet_W44': HRNet_W44, 'HRNet_W48': HRNet_W48, 'HRNet_W64': HRNet_W64

    BACKBONE_NAME1 = cfg['BACKBONE_NAME1']
    BACKBONE_NAME2 = cfg['BACKBONE_NAME2']
    # INPUT_SIZE = cfg['INPUT_SIZE']

    INPUT_SIZE1 = [32, 32] if cfg['VAL_IN_LOWRES1'] else [144, 144]
    INPUT_SIZE2 = [32, 32] if cfg['VAL_IN_LOWRES2'] else [144, 144]

    # assert INPUT_SIZE == [112, 112]
    backbone1 = BACKBONE_DICT[BACKBONE_NAME1](INPUT_SIZE1)
    backbone2 = BACKBONE_DICT[BACKBONE_NAME2](INPUT_SIZE2)
    # print("=" * 60)
    # print(backbone)
    # print("{} Backbone Generated".format(BACKBONE_NAME))
    # print("=" * 60)
    HEAD_DICT = {'Softmax': Softmax, 'ArcFace': ArcFace, 'Combined': Combined, 'CosFace': CosFace, 'SphereFace': SphereFace,
                 'Am_softmax': Am_softmax, 'CurricularFace': CurricularFace, 'ArcNegFace': ArcNegFace, 'SVX': SVXSoftmax, 
                 'AirFace': AirFace,'QAMFace': QAMFace, 'CircleLoss':CircleLoss
                }
    
    HEAD_NAME = cfg['HEAD_NAME']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    head = HEAD_DICT[HEAD_NAME](in_features = EMBEDDING_SIZE, out_features = NUM_CLASS)
    print("Params1: ", count_model_params(backbone1))
    print("Flops1:", count_model_flops(backbone1, input_res=INPUT_SIZE1))

    print("Params2: ", count_model_params(backbone2))
    print("Flops2:", count_model_flops(backbone2, input_res=INPUT_SIZE2))
    #backbone = backbone.eval()
    #summary(backbone, torch.randn(1, 3, 112, 112))
    #backbone = backbone.eval()
    #print("Flops: ", flops_to_string(2*float(profile_macs(backbone.eval(), torch.randn(1, 3, 112, 112)))))
    #backbone = backbone.train()
    print("=" * 60)
    print(head)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)


    # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    if BACKBONE_NAME1.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(backbone1)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone1)

    if GPU_AVAILABLE:
    
        torch.cuda.set_device(device)
        backbone1.cuda(device)
        backbone2.cuda(device)
        head.cuda(device)
    else:
    
        backbone1.to(device)
        backbone2.to(device)
        head.to(device)
    

    LR = cfg['LR'] # initial LR
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    params = [{'params': backbone_paras_wo_bn + list(head.parameters()), 'weight_decay': WEIGHT_DECAY},
            {'params': backbone_paras_only_bn}]
    if OPTIMIZER == 'sgd':
        optimizer = optim.SGD(params, lr=LR, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif OPTIMIZER == 'lookahead':
        base_optimizer = optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)
    elif OPTIMIZER == 'radam':
        optimizer = RAdam(params, lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif OPTIMIZER == 'ranger':
        optimizer = Ranger(params, lr=LR, alpha=0.5, k=6)
    elif OPTIMIZER == 'adamp':
        optimizer = AdamP(params, lr=LR, betas=(0.9, 0.999), weight_decay=1e-2)
    elif OPTIMIZER == 'sgdp':
        optimizer = SGDP(params, lr=LR, weight_decay=1e-5, momentum=0.9, nesterov=True)
    
    if LR_SCHEDULER == 'step':
        scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAT_GAMMA)
    elif LR_SCHEDULER == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=LR_DECAY_EPOCH, gamma=LR_DECAT_GAMMA)
    elif LR_SCHEDULER == 'cosine':
        scheduler = CosineWarmupLR(optimizer, batches=len(train_loader), epochs=NUM_EPOCH, base_lr=LR, target_lr=LR_END, warmup_epochs=WARMUP_EPOCH, warmup_lr=WARMUP_LR)

    print("=" * 60)
    print(optimizer)
    print("Optimizer Generated")
    print("=" * 60)

    # loss
    LOSS_NAME = cfg['LOSS_NAME']
    LOSS_DICT = {'Softmax'      : nn.CrossEntropyLoss(),
                 'LabelSmooth'  : LabelSmoothCrossEntropyLoss(classes=NUM_CLASS),
                 'Focal'        : FocalLoss(),
                 'HM'           : HardMining(),
                 'Softplus'     : nn.Softplus()}
    loss = LOSS_DICT[LOSS_NAME].to(device)
    print("=" * 60)
    print(loss)
    print("{} Loss Generated".format(loss))
    print("=" * 60)

    # optionally resume from a checkpoint
    BACKBONE_RESUME_ROOT1 = cfg['BACKBONE_RESUME_ROOT1'] # the root to resume training from a saved checkpoint
    BACKBONE_RESUME_ROOT2 = cfg['BACKBONE_RESUME_ROOT2'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    IS_RESUME = cfg['IS_RESUME']
    if IS_RESUME:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT1):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT1))
            loc = '{}:0'.format(cfg['GPU'])
            backbone1.load_state_dict(torch.load(BACKBONE_RESUME_ROOT1, map_location=loc))
            # if os.path.isfile(HEAD_RESUME_ROOT):
            #     print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            #     checkpoint = torch.load(HEAD_RESUME_ROOT, map_location=loc)
            #     cfg['START_EPOCH'] = checkpoint['EPOCH']
            #     head.load_state_dict(checkpoint['HEAD'])
            #     optimizer.load_state_dict(checkpoint['OPTIMIZER'])
            #     del(checkpoint)
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT1, HEAD_RESUME_ROOT))
        print("=" * 60)

        if os.path.isfile(BACKBONE_RESUME_ROOT2):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT2))
            loc = '{}:0'.format(cfg['GPU'])
            backbone2.load_state_dict(torch.load(BACKBONE_RESUME_ROOT2, map_location=loc))
            # if os.path.isfile(HEAD_RESUME_ROOT):
            #     print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            #     checkpoint = torch.load(HEAD_RESUME_ROOT, map_location=loc)
            #     cfg['START_EPOCH'] = checkpoint['EPOCH']
            #     head.load_state_dict(checkpoint['HEAD'])
            #     optimizer.load_state_dict(checkpoint['OPTIMIZER'])
            #     del(checkpoint)
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT2, HEAD_RESUME_ROOT))
        print("=" * 60)
    
    # ori_backbone = copy.deepcopy(backbone)
    # if SYNC_BN:
    #     backbone = apex.parallel.convert_syncbn_model(backbone)
    # if USE_APEX:
    #     [backbone, head], optimizer = amp.initialize([backbone, head], optimizer, opt_level='O2')
    #     backbone = DDP(backbone)
    #     head = DDP(head)
    # else:


    
    # Freeze backbone body: only output layer is trained
    for param in backbone1.body.parameters():
        param.requires_grad = False


    for param in backbone2.body.parameters():
        param.requires_grad = False


    backbone1 = torch.nn.DataParallel(backbone1, device_ids=[cfg['GPU']] if GPU_AVAILABLE else None)
    backbone2 = torch.nn.DataParallel(backbone2, device_ids=[cfg['GPU']] if GPU_AVAILABLE else None)
    head = torch.nn.DataParallel(head, device_ids=[cfg['GPU']] if GPU_AVAILABLE else None)
        # backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[cfg['GPU']])
        # head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[cfg['GPU']])
    
    # checkpoint and tensorboard dir
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status

    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)

    writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results
    # train
    
    for epoch in range(cfg['START_EPOCH'], cfg['NUM_EPOCH']):
        # train_sampler.set_epoch(epoch)
        if LR_SCHEDULER != 'cosine':
            scheduler.step()
        #train for one epoch
        DISP_FREQ = 100  # 100 batch
        batch = 0  # batch index
        # backbone.train()  # set to training mode
        head.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for inputs, labels in tqdm(iter(train_loader)):

            inputs1, inputs2 = inputs

            if LR_SCHEDULER == 'cosine':
                scheduler.step()
            # compute output
            start_time=time.time()
            
            if GPU_AVAILABLE and cfg['GPU'] == "cuda":
                inputs1 = inputs1.cuda(cfg['GPU'] if GPU_AVAILABLE else None, non_blocking=True)
                labels = labels.cuda(cfg['GPU'] if GPU_AVAILABLE else None, non_blocking=True)
                inputs2 = inputs2.cuda(cfg['GPU'] if GPU_AVAILABLE else None, non_blocking=True)
                
            if GPU_AVAILABLE:
                inputs1 = inputs1.cuda(cfg['GPU'], non_blocking=True)
                inputs2 = inputs2.cuda(cfg['GPU'], non_blocking=True)
                labels = labels.cuda(cfg['GPU'], non_blocking=True)

            if cfg['MIXUP']:
                    inputs1, labels_a, labels_b, lam = mixup_data(inputs1, labels, cfg['GPU'], cfg['MIXUP_PROB'], cfg['MIXUP_ALPHA'])
                    inputs1, labels_a, labels_b = map(Variable, (inputs1, labels_a, labels_b))
                    inputs2, labels_a, labels_b, lam = mixup_data(inputs2, labels, cfg['GPU'], cfg['MIXUP_PROB'], cfg['MIXUP_ALPHA'])
                    inputs2, labels_a, labels_b = map(Variable, (inputs2, labels_a, labels_b))
            elif cfg['CUTMIX']:
                    inputs1, labels_a, labels_b, lam = cutmix_data(inputs1, labels, cfg['GPU'], cfg['CUTMIX_PROB'], cfg['MIXUP_ALPHA'])
                    inputs1, labels_a, labels_b = map(Variable, (inputs1, labels_a, labels_b))
                    inputs2, labels_a, labels_b, lam = cutmix_data(inputs2, labels, cfg['GPU'], cfg['CUTMIX_PROB'], cfg['MIXUP_ALPHA'])
                    inputs2, labels_a, labels_b = map(Variable, (inputs2, labels_a, labels_b))
            
            
            features1 = backbone1(inputs1)
            features2 = backbone2(inputs2)

            features = features1 + features2

            # print(features1)
            # print(features2)
            # print(features)

            # sys.exit(0)

            outputs = head(features, labels)

            if cfg['MIXUP'] or cfg['CUTMIX']:
                lossx = mixup_criterion(loss, outputs, labels_a, labels_b, lam)
            else:
                lossx = loss(outputs, labels) if HEAD_NAME != 'CircleLoss' else loss(outputs).mean()
            end_time = time.time()
            duration = end_time - start_time
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("batch inference time", duration)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if USE_APEX:
                with amp.scale_loss(lossx, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                lossx.backward()
            optimizer.step()

            # measure accuracy and record loss
            print(outputs.data.shape)
            print(labels.shape)
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5)) if HEAD_NAME != 'CircleLoss' else accuracy(features.data, labels, topk = (1, 5))
            losses.update(lossx.data.item(), inputs[0].size(0))
            top1.update(prec1.data.item(), inputs[0].size(0))
            top5.update(prec5.data.item(), inputs[0].size(0))
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) or batch == 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                                'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                    epoch + 1, cfg['NUM_EPOCH'], batch + 1, len(train_loader), loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)

            # perform validation & save checkpoints per epoch
            # validation statistics per epoch (buffer for visualization)
            if (batch + 1) % EVAL_FREQ == 0:
                #lr = scheduler.get_last_lr()
                lr = optimizer.param_groups[0]['lr']
                print("Current lr", lr)
                print("=" * 60)
                print("Perform Evaluation on %s, and Save Checkpoints..."%(','.join([vs[2] for vs in val_dataset])))
                for vs in val_dataset:
                    # vs = [[hres1, hres2], hres_issame, 'idemia2']
                    acc, best_threshold, roc_curve = perform_val2(EMBEDDING_SIZE, per_batch_size, backbone1, backbone2, vs[0], vs[1])
                    x_coord = len(train_loader)*epoch + (batch+1)
                    buffer_val(writer, "%s"%(vs[2]), acc, best_threshold, roc_curve, x_coord)
                
                    print("Epoch {}/{}, Evaluation: {}, Acc: {}, Best_Threshold: {}".format(x_coord, NUM_EPOCH, vs[2], acc, best_threshold))
                print("=" * 60)

                # print("=" * 60)
                # print("Save Checkpoint...")
                # if cfg['RANK'] % ngpus_per_node == 0:
                #     '''
                #     if epoch+1==cfg['NUM_EPOCH']:
                #         torch.save(backbone.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, get_time())))
                #         save_dict = {'EPOCH': epoch+1,
                #                     'HEAD': head.module.state_dict(),
                #                     'OPTIMIZER': optimizer.state_dict()}
                #         torch.save(save_dict, os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, get_time())))
                #     '''
                #     if USE_APEX:
                #         ori_backbone = ori_backbone.half()
                #     ori_backbone.load_state_dict(backbone.module.state_dict())
                #     ori_backbone.eval()
                #     x = torch.randn(1,3,112,112).cuda()
                #     traced_cell = torch.jit.trace(ori_backbone, (x))
                #     torch.jit.save(traced_cell, os.path.join(MODEL_ROOT, "Epoch_{}_Time_{}_checkpoint.pth".format(epoch + 1, get_time())))
                    
            sys.stdout.flush()
            batch += 1 # batch index

        # All batch are finished: save epoch model and head
        print("=" * 60)
        print("Save Checkpoint...")
        save_dict = {'EPOCH': epoch+1,
                    'HEAD': head.module.state_dict(),
                    'OPTIMIZER': optimizer.state_dict()}
        torch.save(save_dict, os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, get_time())))
        torch.save(backbone1.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone1_{}_Epoch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME1, epoch + 1, get_time())))
        torch.save(backbone2.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone2_{}_Epoch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME2, epoch + 1, get_time())))

        epoch_loss = losses.avg
        epoch_acc = top1.avg
        print("=" * 60)
        print('Epoch: {}/{}\t''Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, cfg['NUM_EPOCH'], loss = losses, top1 = top1, top5 = top5))
        sys.stdout.flush()
        print("=" * 60)
        if cfg['RANK'] % ngpus_per_node == 0:
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            writer.add_scalar("Top1", top1.avg, epoch+1)
            writer.add_scalar("Top5", top5.avg, epoch+1)
        writer.flush()
        
    writer.close()


if __name__ == '__main__':
    main()