import seaborn as sns
from cv2 import cv2
# 系统相关函数
import sys
img = cv2.imread('./lighter.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)
# 把矩阵转换成一维向量，可加速统计
flat_img = gray.flatten()
# hist -> histogram 直方图，也就是按值统计像素
p = sns.histplot(flat_img)
f = p.get_figure()
# 把直方图存到这个文件里 (PDF支持矢量图，亦可选择其它图像格式）
f.savefig('./tablelight.png')
f.clf()  # 清空画布方面后面的代码绘
