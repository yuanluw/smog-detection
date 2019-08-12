# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/12 0012, matt '

import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import *
from model import DNCNN

cur_path = os.path.abspath(os.path.dirname(__file__))


def inference(arg):
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    net = DNCNN()
    if arg.mul_gpu:
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(cur_path, "pre_train", str("DNCNN" + "_.pkl"))))
    net = net.cuda()
    net = net.eval()

    test_data = get_dataset(arg, "test")
    for im, label in test_data:
        with torch.no_grad():
            im = Variable(im.cuda())
            label = Variable(label.cuda())

        out = net(im)
        _, pred_label = out.max(1)
        smoke_index = (label == 1)
        no_smoke_index = (label == 0)

        total = out.shape[0]
        print(label)
        print(pred_label)
        t_TP = (pred_label[smoke_index] == 1).sum().item()
        t_FN = (pred_label[smoke_index] == 0).sum().item()
        t_TN = (pred_label[no_smoke_index] == 0).sum().item()
        t_FP = (pred_label[no_smoke_index] == 1).sum().item()
        print("TP %d FN %d TN %d FP %d" % (t_TP, t_FN, t_TN, t_FP))
        TP += t_TP
        FN += t_FN
        TN += t_TN
        FP += t_FP

    print("TP %d FN %d TN %d FP %d" % (TP, FN, TN, FP))
    DR = TP/(TP+FN)
    FAR = FP/(FP+TN)
    AR = (TP+TN)/(TP+TN+FP+FN)
    print("DR %f FAR %f AR %f" % (DR, FAR, AR))


def run(arg):
    inference(arg)

