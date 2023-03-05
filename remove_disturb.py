import numpy as np
from cv2 import cv2 as cv2

img = cv2.imread('D:/python/pythontest/photos/from.jpg')

img_0 = cv2.blur(img, ksize=(15, 15))
img_1 = cv2.GaussianBlur(img, ksize=(15, 15), sigmaX=0)
img_2 = cv2.bilateralFilter(img, 15, sigmaSpace=75, sigmaColor=75)
cv2.imshow('image_0', img_0)
cv2.imshow('image_1', img_1)
cv2.imshow('image_2', img_2)
cv2.imshow('image', img)

cv2.waitKey()
