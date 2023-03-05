from cv2 import cv2 
import torchvision.transforms as transforms
import torch
from PIL import Image
import os  # 写标签

# 智能读取一次，创建一次文件，否则遍历时会出错（无文件）
path = "D:\\python\\pythontest\\png_form_images"  # 图片集路径
classes = [i for i in os.listdir(path)]
files = os.listdir(path)
train = open("D:\\python\\pythontest\\train.txt", 'w')
val = open("D:\\python\\pythontest\\val.txt", 'w')
for i in classes:
    s = 0
    for imgname in os.listdir(os.path.join(path, i)):

        # if s % 7 != 0:  # 7：1划分训练集测试集
        name = os.path.join(path, i) + '\\' + imgname + ' ' + \
            str(classes.index(i)) + '\n'  # 我是win10,是\\,ubuntu注意！
        train.write(name)
'''
        else:
            name = os.path.join(path, i) + '\\' + imgname + \
                ' ' + str(classes.index(i)) + '\n'
            val.write(name)
        s += 1
'''
val.close()
train.close()


class MyDataset(torch.utils.data.Dataset):  # 创类：MyDataset,继承torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 打开txt，读取内容
        imgs = []
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除本行string字符串末尾的指定字符
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            # 把txt里的内容读入imgs列表保存，words[0]是图片信息，words[1]是label
            imgs.append((words[0], str(words[1])))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path
        img = Image.open(fn).convert('RGB')  # from PIL import Image

        if self.transform is not None:  # 是否进行transform
            img = self.transform(img)
        return img, label  # return回哪些内容，在训练时循环读取每个batch，就能获得哪些内容

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.imgs)


# 标准化、图片变换
mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]
train_transforms = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)])

train_data = MyDataset(
    datatxt='D:\\python\\pythontest\\train.txt', transform=train_transforms)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=64, shuffle=True)




# 训练时:
for data, label in train_loader:
    pass



'''
from PIL import Image
from torchvision import transforms

img = Image.open('1.jpg')  # [H,W,C] [0,255] RGB
# img.show()
# tf=transforms.ToTensor()
# pic=tf(img)   # 单个操作

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])		# 组合操作
img = transform(img)  # [C,H,W] [0,1] RGB
'''