import numpy as np
import torch
#import mxnet as mx
import copy

from PIL import Image
from torchvision.transforms import Resize as torchResize
# import cv2
from torch.utils.data import Dataset
import os
from collections import defaultdict
import numbers
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        classes, class_to_idx = self._find_classes(self.root_dir)
        samples, label_to_indexes = self._make_dataset(self.root_dir, class_to_idx)
        print('samples num', len(samples))
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.label_to_indexes = label_to_indexes
        self.classes = sorted(self.label_to_indexes.keys())
        print('class num', len(self.classes))

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def _make_dataset(self, root_dir, class_to_idx):
        root_dir = os.path.expanduser(root_dir)
        images = []
        label2index = defaultdict(list)
        image_index = 0
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    label2index[class_to_idx[target]].append(image_index)
                    image_index += 1

        return images, label2index

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

def read_samples_from_record(root_dir, record_dir, Train):
    samples = []
    classes = set()
    names = []
    label2index = defaultdict(list)
    with open(record_dir, "r") as f:
        for index, line in enumerate(f):
            line = line.split()
            if Train and len(line) < 2:
                print('Error, Label is missing')
                exit()
            elif len(line) == 1:
                image_dir = line[0]
                label = 0
            else:
                image_dir, label = line[0], line[1]
            label = int(label)
            names.append(image_dir)
            image_dir = os.path.join(root_dir, image_dir)
            samples.append((image_dir, label))
            classes.add(label)
            label2index[label].append(index)
    return samples, classes, names, label2index

class FaceDataset(Dataset):
    def __init__(self, root_dir, record_dir, transform, train_in_lowres=False, Train=True):
        super(FaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.train = Train
        self.imgs, self.classes, self.names, self.label_to_indexes = read_samples_from_record(root_dir, record_dir, Train=Train)
        print("Number of Sampels:{} Number of Classes: {}".format(len(self.imgs), len(self.classes)))

        self.isLowres = 'lowres' in record_dir
        self.isHighres = 'highres' in record_dir

        if self.isHighres and train_in_lowres:
            self.resizer = torchResize((32, 32))
        
        elif self.isHighres and not train_in_lowres:
            self.resizer = None

        elif self.isLowres and not train_in_lowres:
            self.resizer = torchResize((144, 144))
        
        elif self.isLowres and train_in_lowres:
            self.resizer = torchResize((32, 32))
        
        # Uncomment to do highres144 -> downscale to 32 -> upscale to 144
        # self.resizer = torchResize((32, 32))
        # self.resizer2 = torchResize((144, 144))

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = Image.open(path)
        sample = sample.convert("RGB")
        #using opencv
        #sample = cv2.imread(path, cv2.IMREAD_COLOR)
        #sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        #sample = Image.fromarray(sample)
        if self.resizer is not None:
            sample = self.resizer(sample)
            # sample = self.resizer2(sample)
    
        if self.transform is not None:
            sample = self.transform(sample)
        # For lowres images: upscale x4 to 36x36 --> 144x144

        if self.train:
            return sample, target
        else:
            return sample, target, self.names[index]

    def __len__(self):
        return len(self.imgs)

    def get_sample_num_of_each_class(self):
        sample_num = []
        for label in self.classes:
            sample_num.append(len(self.label_to_indexes[label]))
        return sample_num

class FaceDataset2Sizes(Dataset):
    def __init__(self, root_dir, record_dir, transform, train_in_lowres1=False, train_in_lowres2=False, Train=True):
        super(FaceDataset2Sizes, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.train = Train
        self.imgs, self.classes, self.names, self.label_to_indexes = read_samples_from_record(root_dir, record_dir, Train=Train)
        print("Number of Sampels:{} Number of Classes: {}".format(len(self.imgs), len(self.classes)))

        self.isLowres = 'lowres' in record_dir
        self.isHighres = 'highres' in record_dir

        self.transforms1 = copy.deepcopy(self.transform) if self.transform is not None else None
        self.transforms2 = copy.deepcopy(self.transform) if self.transform is not None else None

        if self.isHighres and train_in_lowres1:
            self.resizer1 = torchResize((32, 32))
        
        elif self.isHighres and not train_in_lowres1:
            self.resizer1 = None

        elif self.isLowres and not train_in_lowres1:
            self.resizer1 = torchResize((144, 144))
        
        elif self.isLowres and train_in_lowres1:
            self.resizer1 = torchResize((32, 32))




        if self.isHighres and train_in_lowres2:
            self.resizer2 = torchResize((32, 32))
        
        elif self.isHighres and not train_in_lowres2:
            self.resizer2 = None

        elif self.isLowres and not train_in_lowres2:
            self.resizer2 = torchResize((144, 144))
        
        elif self.isLowres and train_in_lowres2:
            self.resizer2 = torchResize((32, 32))
        
        

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = Image.open(path)
        sample = sample.convert("RGB")
        #using opencv
        #sample = cv2.imread(path, cv2.IMREAD_COLOR)
        #sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        #sample = Image.fromarray(sample)
        if self.resizer1 is not None:
            sample1 = self.resizer1(sample)
        else:
            sample1 = sample

        if self.resizer2 is not None:
            sample2 = self.resizer2(sample)
        else:
            sample2 = sample
    
        if self.transforms1 is not None:
            sample1 = self.transforms1(sample1)
        if self.transforms2 is not None:
            sample2 = self.transforms2(sample2)
        # For lowres images: upscale x4 to 36x36 --> 144x144

        if self.train:
            return [sample1, sample2], target
        else:
            return [sample1, sample2], target, self.names[index]

    def __len__(self):
        return len(self.imgs)

    def get_sample_num_of_each_class(self):
        sample_num = []
        for label in self.classes:
            sample_num.append(len(self.label_to_indexes[label]))
        return sample_num

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform, Train=True):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.train = Train
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag>0:
            print('header0 label', header.label)
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = list(range(1, int(header.label[0])))
        else:
            self.imgidx = list(self.imgrec.keys)
        self.classes = self.header0[1]-self.header0[0]
        print("Number of Samples: {} Number of Classes: {}".format(len(self.imgidx), self.classes))
    
    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy() #RGB
        if self.transform is not None:
            PIL_sample = Image.fromarray(sample)
            PIL_sample = self.transform(PIL_sample)
        if self.train:
            return PIL_sample, label
    def __len__(self):
        return len(self.imgidx)
        
