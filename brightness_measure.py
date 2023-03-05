from cv2 import cv2 as cv

import numpy as np

import copy

import random

import math

from PIL import Image

from PIL import ImageStat  # 亮度库函数


def brightness1(im_file):  # 平均像素亮度
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.rms[0]


def brightness2(im_file):  # RMS像素亮度
    im = Image.open(im_file)
    stat = ImageStat.Stat(im)
    r, g, b = stat.mean
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def brightness3(im_file):  # 平均像素，然后转换为感知亮度
    im = Image.open(im_file).convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def brightness4(im_file):  # 像素的均方根，然后转换为感知亮度
    im = Image.open(im_file)
    stat = ImageStat.Stat(im)
    gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
          for r, g, b in im.getdata())
    return sum(gs) / stat.count[0]


im_file1 = './photos/darker.png'  # 在这里修改要判断的图片或者图片索引
avarage_brightness = (brightness2(im_file1) + brightness3(im_file1) + brightness4(im_file1)) / 3
brightness=brightness1(im_file1)
if avarage_brightness <= 85 or brightness < 100 :
     print("This is a picture of night")
     print(avarage_brightness)
     print(brightness)

else:
     print("This is a picture of day")
     print(avarage_brightness)
     print(brightness)


#通过极为简陋的亮度检测结合经验公式和数据判断白天和黑夜，处理了约二十张图片，结果相对准确；但是对于一些特殊情况，例如
#白天的窗口，夜晚的室内灯光无法判断；
# 测试过对比度，并没有显著区别，如果能够训练应该能够捕获更多特征，需要进一步学习(正解必然是深度学习，提取图像多个数据结合经验数据进行分析)
#希望能学到这类方法