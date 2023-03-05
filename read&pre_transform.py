#下面是遍历文件,获得文件活动地址的程序:所得到的file就是可以读取的（图片）文件地址
import os
for file in os.listdir("文件夹路径"):
    print(os.path.join("文件夹路径",file))#“join“就是把两段地址名字连接起来
'''
import os
for root,dirs,files in os.walk("文件夹")：
    for file in files:
        print(os.path.join(root,file))
'''
'''
import glob
for file in glob.glob(os.path.join("data","*")):
    print(file)
'''
'''
from pathlib import Path
for file in Path('./data').glob('*'):
    print(file)
'''
'''
#注：后缀名判断方法：
file.endswith(".jpg)
os.path.splitext(file)[-1]==".jpg"
file.split(".")[-1]=="jpg"


#不建议在 __init__ 内读图片并存成 list 或 tensor,建议在 __getitem__ 内读图
'''






#从上面获取的图片路径读图的方法：

from PIL import Image
img==Image.open(filename/path).convert("RGB")

'''
from cv2 import cv2
img=cv2.imread(filename/path)
#cv2读取进来的是GBR格式，需要转换成RGB格式:
    #1
        b,g,r=cv2.split(img)
        img=cv2.merge[r,g,b]
    #2
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #3
        img=img[...,::-1]
        img=img[...,[2,1,0]]
        img=img[:,:,(2,1,0)]
'''
'''
import torchvision
img=torchvision.io.read_image(filename/path)
#这个可以直接转换成tensor输出
'''
'''
import skimage.io as io
img=io.imread(filename/path)
'''
'''
#结合numpy和cv2,numpy和skimage:
import numpy as np
import skimage.io as io
img=np.array(io.ImageCollection(filename/path))

import numpy as np
from cv2 import cv2 as cv2
img=imdecode(np.fromfile(filename,dtype=np.float64,cv2.TMREAD_COLOR))
'''





#图像预处理和img-->tensor转换：

#用PIL或cv2读取的需要转换为tensor：
import torchvision.transforms as transforms
transform=transforms.Compose([
    transforms.toTensor(),transforms,Normalize(mean=[0.485,0.456,0.406],std=[0.226,0.224,0.225])])
tensor=transform(img)
#注：可以不读lable直接返回

'''
#使用cv2还可以这样：
tensor=torch.from.numpy(img.transpose((2,0,1))).float().div(255)#自带了色彩通道转换
mean=tensor.new_tensor([0.485,0.4560.406]).view(-1,1,1)
std=tensor.new_tensor([0.229,0.224,0.225]).view(-1,1,1)
tensor=(tensor-mean)/std
'''


    


    