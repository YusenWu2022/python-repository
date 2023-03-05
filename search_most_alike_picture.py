
import numpy as np
from cv2 import cv2 as cv2

from matplotlib import pyplot as plt


def createHistRGB(image):
    # 创建 rgb 三通道直方图
    h, w, c = image.shape  # 创建一个的初始矩阵

    rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)  # 初始化
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 创建三通道直方图
            index = int(b / bsize) * 16 * 16 + \
                int(g / bsize) * 16 + int(r / bsize)
            # 该处形成的矩阵即为直方图矩阵
            rgbhist[int(index), 0] += 1
    plt.ylim([0, 10000])
    plt.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.3)
    return rgbhist


def histCompare(image1, image2):
    # 比较直方图
    # # 创建第一幅图的rgb三通道直方图
    hist1 = createHistRGB(image1)   # 创建第二幅图的rgb三通道直方图
    hist2 = createHistRGB(image2)   # 进行三种方式的直方图比较

    match = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return match

src2=np.array([1,2,3,4,5,6])
src1 = cv2.imread("./standard.jpg")  # 这里改成标准图片库的地址

maxlike = 0
src2[1] = cv2.imread("./example1.jpg")  # 这里改成对比图片的地址
src2[2] = cv2.imread("./example2.jpg")
src2[3] = cv2.imread("./example3.jpg")
src2[4] = cv2.imread("./example4.jpg")
src2[5] = cv2.imread("./example5.jpg")
src2[6] = cv2.imread("./example6.jpg") 
for i in range(1, 6):
  #逐一读取待比较图片

 
  looklike = histCompare(src1, src2[i])#得到各张图片的相似度并比较得到最相似的图片作为结果
  if maxlike < looklike    :
     maxlike = looklike
     maxnum = i
print(i)  #输出相似度最高的图片
cv2.waitKey()  # 防止异常退出


input(0)
#可以进一步找最为近似的结果