# 用于改变图片大小
import numpy as np
from cv2 import cv2 as cv2
from PIL import Image
img = Image.open("D:\\python\\pythontest\\photos\\xieli_content.jpg")
img = img.resize((512, 512), Image.ANTIALIAS)
img.show()
img.save("D:\\python\\pythontest\\photos\\xieli.jpg")
