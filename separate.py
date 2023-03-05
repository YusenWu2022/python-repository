from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
import os
import os.path as osp

file_path = 'D:\python_cv'

base_name = osp.basename(file_path)  # vicon_03301_03

color_dir = osp.join(file_path, 'Color')  # color文件夹
if not osp.exists(color_dir):
    os.makedirs(color_dir)

result_folder = 'D:\python_cv'
output_folder = osp.join(result_folder, 'result')

if not os.path.exists(output_folder):  # 就是在输出结果的文件夹下创建一个名为vicon_03301_03的文件夹
    os.makedirs(output_folder)


def human_segment(net, path, nc=21):
    print("segmenting...")
    img = Image.open(path)
    trf = T.Compose([T.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
                     T.Normalize(mean=[0.485, 0.456, 0.406],  # 把tensor正则化，Normalized_image=(image-mean)/std
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)  # 返回一个新的张量，对输入的制定位置插入维度 1
    out = net(inp)['out']
    image = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                           128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # 每个像素对应的类别赋予相应的颜色
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # 这个就是语义分割的彩色图
    rgb = np.stack([r, g, b], axis=2)  # 堆栈

    save_image = osp.join(output_folder, osp.basename(path))
    plt.imsave(save_image, rgb)


dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

'''
i=0
for filename in os.listdir(color_dir):  # 包含想要划分的图像的文件夹
    image_dir = osp.join(color_dir, filename)
    human_segment(dlab, image_dir)
    i+=1
    print("success:{}".format(i))
    '''

human_segment(dlab, 'D:\python_cv\Color\cycle.png')
