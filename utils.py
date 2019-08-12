# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/11 0011, matt '


from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt


class Display_board:
    def __init__(self, port=8097, viz=None, env_name=None):
        if viz is None:
            self.viz = Visdom(port=port, env=env_name)
        else:
            self.viz = viz

    def add_Line_windows(self, name, X=0, Y=0):

        w = self.viz.line(X=np.array([X]), Y=np.array([Y]), opts=dict(title=name))
        return w

    def update_line(self, w, X, Y):
        self.viz.line(X=np.array([X]), Y=np.array([Y]), win=w, update="append")

    def show_image(self, image):
        plt.imshow(image)
        self.viz.matplot(plt)


def count_time(prev_time, cur_time):
    h, reminder = divmod((cur_time-prev_time).seconds, 3600)
    m, s = divmod(reminder, 60)
    time_str = "time %02d:%02d:%02d" %(h, m, s)
    return time_str


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct/total