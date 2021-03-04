import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize as torchResize

import torch.nn.functional as F

from .verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os
import sys
import time
import random 
# import cv2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def get_val_pair(path, name, to_lowres=False):
    # same as original
    
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    batch = np.array(carray)
    
    batch = batch[:,::-1,:,:]
    cropped = torch.tensor(batch.copy())
    
    batch = batch[:,:,:,::-1]
    flipped = torch.tensor(batch.copy())

    if to_lowres:
        resizer = torchResize((32, 32))
        cropped = resizer(cropped)
        flipped = resizer(flipped)
 
    issame = np.load('{}/{}_list.npy'.format(path, name))

    print("loading %s done"%(name), cropped.size())
    #sys.stdout.flush()

    return [cropped, flipped], issame

def get_val_pair2(path, name, to_lowres1=False, to_lowres2=False):
    # same as original
    
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    batch = np.array(carray)
    
    batch = batch[:,::-1,:,:]
    cropped = torch.tensor(batch.copy())
    
    batch = batch[:,:,:,::-1]
    flipped = torch.tensor(batch.copy())

    if to_lowres1:
        resizer1 = torchResize((32, 32))
        cropped1 = resizer1(cropped)
        flipped1 = resizer1(flipped)
    else:
        cropped1 = cropped
        flipped1 = flipped

    if to_lowres2:
        resizer2 = torchResize((32, 32))
        cropped2 = resizer2(cropped)
        flipped2 = resizer2(flipped)
    else:
        cropped2 = cropped
        flipped2 = flipped


 
    issame = np.load('{}/{}_list.npy'.format(path, name))

    # print("loading %s done"%(name), cropped.size())
    #sys.stdout.flush()

    return [[cropped1, flipped1], [cropped2, flipped2]], issame

def get_val_data2(data_path, data_set, to_lowres1=False, to_lowres2=False):
    [hres1, hres2], hres_issame = get_val_pair2(data_path, 'hres', to_lowres1=to_lowres1, to_lowres2=to_lowres2)  # le data_path est donné par le paramètre VAL_DATA_ROOT = 'proj_transverse_2020/hres_eval_pairs/'   dans le config.py
    return [[[hres1, hres2], hres_issame, 'idemia2']]
    # val_data = []


def get_val_data(data_path, data_set, to_lowres=False):
    hres, hres_issame = get_val_pair(data_path, 'hres', to_lowres)  # le data_path est donné par le paramètre VAL_DATA_ROOT = 'proj_transverse_2020/hres_eval_pairs/'   dans le config.py
    return [[hres, hres_issame, 'idemia']]
    # val_data = []
    # data_set =  data_set.strip().split(',')
    # for name in data_set:
    #     name =name.strip()
    #     vd, vd_issame = get_val_pair(data_path, name)
    #     val_data.append((vd, vd_issame, name))

    # return val_data

def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'resnet' in str(layer.__class__):
            continue
        if 'backbone' in str(layer.__class__):
            continue
        if 'resattnet' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

    # print(optimizer)


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)

def perform_val(embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, tta = True):
    backbone.eval() # switch to evaluation mode

    idx = 0
    shape = carray[0].shape
    embeddings = np.zeros([shape[0] , embedding_size])
    with torch.no_grad():
        while idx + batch_size <= shape[0]:
            if tta:
                cropped = carray[0][idx:idx + batch_size]
                flipped = carray[1][idx:idx + batch_size]
                emb_batch = backbone(cropped.cuda()).cpu() + backbone(flipped.cuda()).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = carray[0][idx:idx + batch_size]
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.cuda())).cpu()
            idx += batch_size
        if idx < shape[0]:
            if tta:
                cropped = carray[0][idx:,:,:,:]
                flipped = carray[1][idx:,:,:,:]
                emb_batch = backbone(cropped.cuda()).cpu() + backbone(flipped.cuda()).cpu()
                embeddings[idx:] = l2_norm(emb_batch)#[0:shape[0]-idx])
            else:
                ccropped = carray[0][idx:,:,:,:]
                embeddings[idx:] = l2_norm(backbone(ccropped.cuda())).cpu()
    tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)
    backbone.train()
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def perform_val2(embedding_size, batch_size, backbone1, backbone2, carray, issame, nrof_folds = 10, tta = True):
    #perform_val2(EMBEDDING_SIZE, per_batch_size, backbone1, backbone2, vs[0], vs[1])
    backbone1.eval() # switch to evaluation mode
    backbone2.eval() # switch to evaluation mode

    carray1 = carray[0]
    carray2 = carray[1]

    idx = 0
    shape = carray1[0].shape
    embeddings = np.zeros([shape[0] , embedding_size])
    with torch.no_grad():
        while idx + batch_size <= shape[0]:
            if tta:
                cropped1 = carray1[0][idx:idx + batch_size]
                flipped1 = carray1[1][idx:idx + batch_size]
                cropped2 = carray2[0][idx:idx + batch_size]
                flipped2 = carray2[1][idx:idx + batch_size]
                emb_batch1 = backbone1(cropped1.cuda()).cpu() + backbone1(flipped1.cuda()).cpu()
                emb_batch2 = backbone2(cropped2.cuda()).cpu() + backbone2(flipped2.cuda()).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch1+emb_batch2)
            else:
                ccropped1 = carray1[0][idx:idx + batch_size]
                ccropped2 = carray2[0][idx:idx + batch_size]
                embeddings[idx:idx + batch_size] = l2_norm(backbone1(ccropped1.cuda())+backbone2(ccropped2.cuda())).cpu()
            idx += batch_size
        if idx < shape[0]:
            if tta:
                cropped1 = carray1[0][idx:,:,:,:]
                flipped1 = carray1[1][idx:,:,:,:]
                cropped2 = carray2[0][idx:,:,:,:]
                flipped2 = carray2[1][idx:,:,:,:]
                emb_batch1 = backbone1(cropped1.cuda()).cpu() + backbone1(flipped1.cuda()).cpu()
                emb_batch2 = backbone2(cropped2.cuda()).cpu() + backbone2(flipped2.cuda()).cpu()
                embeddings[idx:] = l2_norm(emb_batch1+emb_batch2)#[0:shape[0]-idx])
            else:
                ccropped1 = carray2[0][idx:,:,:,:]
                ccropped1 = carray2[0][idx:,:,:,:]
                embeddings[idx:] = l2_norm(backbone1(ccropped1.cuda())+backbone2(ccropped2.cuda())).cpu()
    tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)
    backbone1.train()
    backbone2.train()
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def perform_val_backbone(embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, tta = False, samples=200):
    backbone.eval() # switch to evaluation mode

    idx = 0
    shape = carray[0].shape
    number_sample  = samples
    embeddings = np.zeros([number_sample , embedding_size])
    with torch.no_grad():
        while idx + batch_size <= number_sample:
            # print(idx)
            if tta:
                cropped = carray[0][idx:idx + batch_size]
                flipped = carray[1][idx:idx + batch_size]
                emb_batch = backbone(cropped).cpu() + backbone(flipped).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = carray[0][idx:idx + batch_size]
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped)).cpu()
                
            idx += batch_size
        if idx < number_sample:
            if tta:
                cropped = carray[0][idx:,:,:,:]
                flipped = carray[1][idx:,:,:,:]
                emb_batch = backbone(cropped).cpu() + backbone(flipped).cpu()
                embeddings[idx:] = l2_norm(emb_batch)#[0:shape[0]-idx])
            else:
                ccropped = carray[0][idx:,:,:,:]
                embeddings[idx:] = l2_norm(backbone(ccropped)).cpu()
    #tpr, fpr, accuracy, best_thresholds, bad_case = evaluate(embeddings, issame, nrof_folds)
    #buf = gen_plot(fpr, tpr)
    #roc_curve = Image.open(buf)
    #roc_curve_tensor = transforms.ToTensor()(roc_curve)
    #backbone.train()
    return torch.from_numpy(embeddings)
    #return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)
    
def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        print("lol")
        print(correct[:k].shape)
        sys.stdout.flush()
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
