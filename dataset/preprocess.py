# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/11 0011, matt '


import os
import numpy as np
import cv2 as cv

cur_path = os.path.abspath(os.path.dirname(__file__))


def generater_dataset(file_path, file_name, txt_fname, index):
    read_str = ""
    rgb_index = 1
    optical_flow_index = 1
    f = open(txt_fname, 'a')
    print(f)
    print(txt_fname)
    cap = cv.VideoCapture(os.path.join(file_path, file_name + ".avi"))
    ret, frame1 = cap.read()
    # frame1 = cv.resize(frame1, (227, 227))
    # cv.imwrite(os.path.join(cur_path, "rgb_data", file_name + "_" + str(rgb_index)+".jpg"), frame1)
    read_str = file_name + "_" + str(rgb_index)+".jpg" + " "
    # hsv = np.zeros_like(frame1)
    # hsv[..., 2] = 255
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    while 1:
        ret, frame2 = cap.read()
        if frame2 is None:
            break
        rgb_index += 1
        # frame2 = cv.resize(frame2, (227, 227))
        # cv.imwrite(os.path.join(cur_path, "rgb_data", file_name + "_" + str(rgb_index) + ".jpg"), frame2)
        read_str += (file_name + "_" + str(rgb_index) + ".jpg" + " ")
        # frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        # flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 1] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # cv.imwrite(os.path.join(cur_path, "optical_data", file_name + "_" + str(optical_flow_index) + ".jpg"), rgb)
        read_str += (file_name + "_" + str(optical_flow_index) + ".jpg") + " "
        if index < 11:
            read_str += "1" + " "
        else:
            read_str += "0" + " "
        print(read_str)
        f.write(read_str+'\n')
        read_str = file_name + "_" + str(rgb_index) + ".jpg" + " "

        optical_flow_index += 1
        # frame1 = frame2
        # cv.imshow("rgb", rgb)
        # cv.waitKey(32)

    # cv.destroyAllWindows()


def run():
    file_path = os.path.join(cur_path, "trainData")

    txt_fname = os.path.join(cur_path, "train.txt")

    for i in range(1, 16):
        file_name = "train-" + str(i)
        generater_dataset(file_path, file_name, txt_fname, i)


def generater_val():
    path = os.path.join(cur_path, "train.txt")
    f = open(path, 'r')
    result = list()
    for line in open(path):
        result.append(line)
    f.close()
    print(result)
    np.random.shuffle(result)
    print(result)
    count = int(len(result)*0.3)
    val = result[:count]
    train = result[count:]
    os.remove(path)
    f = open(path, 'a')
    f.writelines(train)
    f.close()
    f = open(os.path.join(cur_path, "val.txt"), 'a')
    f.writelines(val)
    f.close()


if __name__ == "__main__":
    generater_val()

