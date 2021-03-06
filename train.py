# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/11 0011, matt '



import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datetime import datetime
import os

from model import Mul_ConvNet
from utils import *
from dataset import *

cur_path = os.path.abspath(os.path.dirname(__file__))


def train(net, train_data, val_data, optimizer, criterion, arg):

    net = net.cuda()
    best_acc = 0
    best_state_dict = 0
    classification_criterion = criterion[0]
    print(classification_criterion)
    mse_criterion = criterion[1]
    print(mse_criterion)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=arg.gamma, )
    if arg.use_visdom:
        viz = Display_board(env_name="smoke_detection_train")
        train_acc_win = viz.add_Line_windows(name="train_acc")
        train_loss_win = viz.add_Line_windows(name="train_loss")
        val_acc_win = viz.add_Line_windows(name="val_acc")
        val_loss_win = viz.add_Line_windows(name="val_loss")
        train_y_axis = 0
        val_y_axis = 0
    print("start training: ", datetime.now())

    for epoch in range(arg.epochs):

        # train stage
        train_loss = 0.0
        train_acc = 0.0
        net = net.train()
        i = 0
        if arg.use_visdom is not True:
            prev_time = datetime.now()
        for im, mask, label in train_data:
            i += 1  # train number
            im = Variable(im.cuda())
            mask = Variable(mask.cuda())
            label = Variable(label.cuda())
            out1, out2 = net(im)
            classification_loss = classification_criterion(out1, label)
            mse_loss = mse_criterion(out2, mask)
            loss = classification_loss + 0.012*mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            cur_acc = get_acc(out1, label)

            train_loss += cur_loss
            train_acc += cur_acc

            # visualize curve
            if arg.use_visdom:
                train_y_axis += 1
                viz.update_line(w=train_acc_win, Y=cur_acc, X=train_y_axis)
                viz.update_line(w=train_loss_win, Y=cur_loss, X=train_y_axis)
            else:
                now_time = datetime.now()
                time_str = count_time(prev_time, now_time)
                print("train: current (%d/%d) batch loss is %f acc is %f time is %s" % (i, len(train_data), cur_loss,
                                                                                        cur_acc, time_str))
                prev_time = now_time

        print("train: the (%d/%d) epochs acc: %f loss: %f, cur time: %s" % (epoch, arg.epochs, train_acc/len(
            train_data), train_loss/len(train_data), str(datetime.now())))

        # val stage
        if val_data is not None:
            val_loss = 0.0
            val_acc = 0.0
            net = net.eval()
            j = 0
            if arg.use_visdom is not True:
                prev_time = datetime.now()
            for im, mask, label in val_data:
                j += 1
                with torch.no_grad():
                    im = Variable(im.cuda())
                    mask = Variable(mask.cuda())
                    label = Variable(label.cuda())
                out1, out2 = net(im)
                classification_loss = classification_criterion(out1, label)
                mse_loss = mse_criterion(out2, mask)
                loss = classification_loss + 0.1 * mse_loss

                cur_acc = get_acc(out1, label)
                cur_loss = loss.item()

                val_acc += cur_acc
                val_loss += cur_loss

                # visualize curve
                if arg.use_visdom:
                    val_y_axis += 1
                    viz.update_line(w=val_acc_win, Y=cur_acc, X=val_y_axis)
                    viz.update_line(w=val_loss_win, Y=cur_loss, X=val_y_axis)
                else:
                    now_time = datetime.now()
                    time_str = count_time(prev_time, now_time)
                    print(
                        "val: current (%d/%d) batch loss is %f acc is %f time is %s" % (j, len(val_data), cur_loss,
                                                                                          cur_acc, time_str))
                    prev_time = now_time

            print("val: the (%d/%d) epochs acc: %f loss: %f, cur time: %s" % (epoch, arg.epochs, val_acc / len(
                val_data), val_loss / len(val_data), str(datetime.now())))
            if best_acc < val_acc / len(val_data):
                best_acc = val_acc / len(val_data)
                best_state_dict = net.state_dict()
        scheduler.step()
    print("end time: ", datetime.now())
    if os.path.exists(os.path.join(cur_path, "pre_train", str("Mul_ConvNet" + "_.pkl"))):
        os.remove(os.path.join(cur_path, "pre_train", str("Mul_ConvNet" + "_.pkl")))
    torch.save(best_state_dict, os.path.join(cur_path, "pre_train", str("Mul_ConvNet" + "_.pkl")))


def run(arg):
    print("lr %f, epoch_num %d, decay_rate %f pre_train %d gamma %f" % (arg.lr, arg.epochs, arg.decay,
                                                                        arg.pre_train, arg.gamma))

    train_data = get_dataset(arg, train="train")
    val_data = get_dataset(arg, train="val")

    net = Mul_ConvNet()

    if arg.mul_gpu:
        net = nn.DataParallel(net)

    if arg.pre_train:
        net.load_state_dict(torch.load(os.path.join(cur_path, "pre_train", str("Mul_ConvNet" + "_.pkl"))))

    optimizer = optim.Adam(net.parameters(), lr=arg.lr, weight_decay=arg.decay)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    train(net, train_data, val_data, optimizer, (criterion1, criterion2), arg)

