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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# import apex
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from config_eval import configurations
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


    VAL_DATA_ROOT = cfg['VAL_DATA_ROOT']
    VAL_SET = cfg['VAL_SET']
    VAL_IN_LOWRES = cfg['VAL_IN_LOWRES']
    INPUT_SIZE = [32, 32] if VAL_IN_LOWRES else [144, 144]
    
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    IS_RESUME = cfg['IS_RESUME']
    BACKBONE_NAME = cfg['BACKBONE_NAME']

    
    #workers = int((cfg['NUM_WORKERS'] + ngpus_per_node - 1) / ngpus_per_node) # dataload threads
    workers = int(cfg['NUM_WORKERS'])
    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    RECORD_DIR = cfg['DATA_RECORD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension

    # name to csv of features
    eval_name = cfg['CSV_NAME']
    
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

    batch_size = int(cfg['BATCH_SIZE'])
    per_batch_size = int(batch_size / ngpus_per_node)

    # load val_dataset    
    val_dataset = get_val_data(VAL_DATA_ROOT, VAL_SET, VAL_IN_LOWRES)

    ############################################################################

    DROP_LAST = cfg['DROP_LAST']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']

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

    dataset_eval = FaceDataset(DATA_ROOT, RECORD_DIR, train_transform, train_in_lowres=cfg['VAL_IN_LOWRES'])

    train_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=per_batch_size,
                                                shuffle = False, num_workers=workers,
                                                # pin_memory=True, sampler=train_sampler, drop_last=DROP_LAST)
                                                pin_memory=True, drop_last=DROP_LAST)
    #SAMPLE_NUMS = dataset_train.get_sample_num_of_each_class()
    NUM_CLASS = len(train_loader.dataset.classes)
    # NUM_CLASS = train_loader.dataset.classes
    print("Number of Training Classes: {}".format(NUM_CLASS))


    ############################################################################


    # Resume backbone from a checkpoint
    # INPUT_SIZE = cfg['INPUT_SIZE']

    BACKBONE_DICT = {'MobileFaceNet': MobileFaceNet,
                     'ResNet_50': ResNet_50, 'ResNet_101': ResNet_101, 'ResNet_152': ResNet_152, 'IR_SE_18': IR_SE_18,
                     'IR_50': IR_50, 'IR_100': IR_100, 'IR_101': IR_101, 'IR_152': IR_152, 'IR_185': IR_185, 'IR_200': IR_200,
                     'IR_SE_50': IR_SE_50, 'IR_SE_100': IR_SE_100, 'IR_SE_101': IR_SE_101, 'IR_SE_152': IR_SE_152, 'IR_SE_185': IR_SE_185, 'IR_SE_200': IR_SE_200,
                     'AttentionNet_IR_56': AttentionNet_IR_56,'AttentionNet_IRSE_56': AttentionNet_IRSE_56,'AttentionNet_IR_92': AttentionNet_IR_92,'AttentionNet_IRSE_92': AttentionNet_IRSE_92,
                     'ResNeSt_50': resnest50, 'ResNeSt_101': resnest101, 'ResNeSt_100': resnest100,
                     'GhostNet': GhostNet, 'MobileNetV3': MobileNetV3, 'ProxylessNAS': proxylessnas, 'EfficientNet': efficientnet,
                     'DenseNet': densenet, 'ReXNetV1': ReXNetV1, 'MobileNeXt': MobileNeXt, 'MobileNetV2': MobileNetV2
                    } #'HRNet_W30': HRNet_W30, 'HRNet_W32': HRNet_W32, 'HRNet_W40': HRNet_W40, 'HRNet_W44': HRNet_W44, 'HRNet_W48': HRNet_W48, 'HRNet_W64': HRNet_W64


    INPUT_SIZE = [32, 32] if cfg['VAL_IN_LOWRES'] else [144, 144]

    backbone = BACKBONE_DICT[BACKBONE_NAME](INPUT_SIZE)

    if IS_RESUME:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            loc = '{}:0'.format(cfg['GPU'])
            backbone.load_state_dict(torch.load(BACKBONE_RESUME_ROOT, map_location=loc))
        else:
            print("No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT))
        print("=" * 60)
    
    backbone = torch.nn.DataParallel(backbone, device_ids=[cfg['GPU']] if GPU_AVAILABLE else None)

    # print backbone 
    print("Params: ", count_model_params(backbone))
    print("Flops:", count_model_flops(backbone, input_res=INPUT_SIZE, device=device))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    ############################################################################

    backbone.eval()

    list_embeddings = []
    list_classes = []

    for inputs, labels in tqdm(iter(train_loader)):
            
            if GPU_AVAILABLE and cfg['GPU'] == "cuda":
                inputs = inputs.cuda(cfg['GPU'] if GPU_AVAILABLE else None, non_blocking=True)
                labels = labels.cuda(cfg['GPU'] if GPU_AVAILABLE else None, non_blocking=True)
                
            if GPU_AVAILABLE:
                inputs = inputs.cuda(cfg['GPU'], non_blocking=True)
                labels = labels.cuda(cfg['GPU'], non_blocking=True)

            features = backbone(inputs)

            list_embeddings.append(features.detach().cpu())
            list_classes.append(labels.detach().cpu())
            
            # print(features.detach().cpu().numpy().shape)
        
    embeddings_tensor = torch.cat(list_embeddings, dim=0)
    classes_tensor = torch.cat(list_classes, dim=0)
    print(embeddings_tensor.shape)
    print(classes_tensor.shape)

    os.makedirs("eval", exist_ok=True)
    

    np.savetxt(f"./eval/emb_{eval_name}.csv", embeddings_tensor.detach().cpu().numpy(), delimiter=",")
    np.savetxt(f"./eval/classes_{eval_name}.csv", classes_tensor.detach().cpu().numpy(), delimiter=",", fmt='%d')













    # ############################################################################



    # print("Perform Evaluation on %s, and Save Checkpoints..."%(','.join([vs[2] for vs in val_dataset])))

    # list_embeddings = []
    # i = 0
    # for vs in val_dataset[:10]:
    #     # print(i)
    #     embeddings = perform_val_backbone(EMBEDDING_SIZE, per_batch_size, backbone, vs[0], vs[1], samples=12000)
    #     # print(embeddings.shape)
    #     list_embeddings.append(embeddings)
    #     i+=1
        
    #     #x_coord = len(train_loader)*epoch + (batch+1)
    #     #buffer_val(writer, "%s"%(vs[2]), acc, best_threshold, roc_curve, x_coord)
                
    #     #print("Epoch {}/{}, Evaluation: {}, Acc: {}, Best_Threshold: {}".format(x_coord, NUM_EPOCH, vs[2], acc, best_threshold))
    #     #print("=" * 60)
    # embeddings_tensor = torch.cat(list_embeddings, dim=0)
    # print(embeddings_tensor.shape)

    # pca =  PCA(n_components=3)
    # embeddings_3d = pca.fit_transform(embeddings_tensor)
    # print(embeddings_3d.shape)

    # np.savetxt("emb_3d.csv", embeddings_3d, delimiter=",")
    # np.savetxt("emb.csv", embeddings_tensor.detach().cpu().numpy(), delimiter=",")

    # fig = plt.figure()
    # ax = Axes3D(fig)



    # ax.scatter(embeddings_3d[:,0], embeddings_3d[:,1], embeddings_3d[:,2])
    # plt.savefig("emb.png")

    


if __name__ == '__main__':
    main()
