# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/11 0011, matt '


import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tfs

from PIL import Image
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))


def read_mul_img(train="train"):
    if train == "train":
        path = os.path.join(cur_path, "train.txt")
    elif train == "val":
        path = os.path.join(cur_path, "val.txt")
    else:
        path = os.path.join(cur_path, "test.txt")

    f = open(path, 'r')
    result = list()
    for line in open(path):
        line = f.readline().split(" ")
        result.append(line)
    f.close()

    np.random.shuffle(result)
    pre_image_path = [os.path.join(cur_path, "rgb_data", r[0]) for r in result]
    next_image_path = [os.path.join(cur_path, "rgb_data", r[1]) for r in result]
    optical_label_path = [os.path.join(cur_path, "rgb_data", r[2]) for r in result]
    label = [(1 if r[3] == "1" else 0) for r in result]
    # print("pre", len(pre_image_path))
    # print("next", len(next_image_path))
    # print("optical", len(optical_label_path))
    # print("label", label)
    return pre_image_path, next_image_path, optical_label_path, label


def read_normal_image(train="train"):
    if train == "train":
        path = os.path.join(cur_path, "train.txt")
    elif train == "val":
        path = os.path.join(cur_path, "val.txt")
    else:
        path = os.path.join(cur_path, "test.txt")

    f = open(path, 'r')
    result = list()
    for line in open(path):
        line = f.readline().split(" ")
        result.append(line)
    f.close()
    np.random.shuffle(result)

    image_path = [os.path.join(cur_path, "rgb_data", r[0]) for r in result]
    label = [(1 if r[3] == "1" else 0) for r in result]

    return image_path,  label


def mul_transforms(pre, next, optical, label):
    im_aug = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pre = im_aug(pre)
    next = im_aug(next)
    img = torch.cat((pre, next))
    optical = tfs.Resize((96, 96))(optical)
    optical = im_aug(optical)
    label = np.array(label)
    label = torch.from_numpy(label)
    return img, optical, label


def normal_transforms(img, label):
    im_aug = tfs.Compose([
        tfs.Resize((48, 48)),
        tfs.RandomHorizontalFlip(),
        tfs.RandomRotation(10),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = im_aug(img)
    label = np.array(label)
    label = torch.from_numpy(label)
    return img, label


class Mul_Smoke_dataset(Dataset):
    def __init__(self, train="train", transforms=None):
        self.transforms = transforms
        self.pre_image_path, self.next_image_path, self.optical_label_path, self.label = read_mul_img(train)
        print("read " + str(len(self.pre_image_path)) + " images")

    def __getitem__(self, item):
        pre_image_path = self.pre_image_path[item]
        next_image_path = self.next_image_path[item]
        optical_label_path = self.optical_label_path[item]
        label = self.label[item]

        pre_image = Image.open(pre_image_path).convert('RGB')
        next_image = Image.open(next_image_path).convert('RGB')
        optical_label = Image.open(optical_label_path).convert("RGB")

        return self.transforms(pre_image, next_image, optical_label, label)

    def __len__(self):
        return len(self.next_image_path)


class Normal_Smoke_dataset(Dataset):
    def __init__(self, train="train", transforms=None):
        self.transforms = transforms
        self.image_path, self.label = read_normal_image(train)

        print("read " + str(len(self.image_path)) + " images")

    def __getitem__(self, item):
        image_path = self.image_path[item]
        label = self.label[item]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image, label)

    def __len__(self):
        return len(self.image_path)


def get_dataset(arg, train="train"):
    if arg.net == "Mul_ConvNet":
        dataset = Mul_Smoke_dataset(train, mul_transforms)
        data_loader = DataLoader(dataset, arg.train_batch_size if train else arg.test_batch_size, shuffle=True,
                                 num_workers=arg.num_worker)
        return data_loader
    elif arg.net == "DNCNN":
        dataset = Normal_Smoke_dataset(train, normal_transforms)
        data_loader = DataLoader(dataset, arg.train_batch_size if train else arg.test_batch_size, shuffle=True,
                                 num_workers=arg.num_worker)
        return data_loader


if __name__ == "__main__":
    print(read_normal_image(train="train"))



