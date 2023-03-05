from cv2 import cv2
import numpy as np
img = cv2.imread('./lighter3.png')
print(img.shape)
print(img.dtype)
norm_img = img.astype(np.float64) / 255.
light_norm_img = np.power(norm_img, 0.4)
light_img = light_norm_img * 255.
cv2.imwrite('./lighter.png', light_img)
