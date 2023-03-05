import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from cv2 import cv2 as cv2
import glob
from PIL import Image

'''
预先准备数据npy:
import numpy as np
A = np.load('D:\\python_storage\\mnist.npz')
train_data=A['x_train']
train_label=A['y_train']
test_data=A['x_test']
test_label=A['y_test']
np.save("D:\\python\\mnist_data\\MNIST\\raw\\train_data.npy",train_data )
np.save("D:\\python\\mnist_data\\MNIST\\raw\\train_label.npy",train_label )
np.save("D:\\python\\mnist_data\\MNIST\\raw\\test_data.npy",test_data )
np.save("D:\\python\\mnist_data\\MNIST\\raw\\test_label.npy",test_label )
'''


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, data, label):
        super(CIFAR10Dataset, self).__init__()  # 调用父类的构造函数
        self.transform = transform  # 设置对象属性 transform
        self.images = data  # 假设data的shape为 (图片数, 32, 32, 3)
        # 假设data的数据类型是 np.float32，值域 [0,1]
        self.labels = label  # 假设label的shape为 (图片数, )
        # PyTorch 会在计算交叉熵时自动转为 one-hot 编码
        # 相当于重载 [] 运算符

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
# 实现了这个方法之后，对于一一个这个类的对象，可以用 len(obj) 来获取长度

    def __len__(self):
        return len(self.images)


        # 定义 transform：包括两个顺序步骤
# 1 把numpy数组转化为pytorch张量
# 2 归一化到 [-0.5, 0.5]，有利于 ReLU
'''
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 假设你把数据都做好了


# 注意：从自定义的npy文件中方读取的图片数据尺寸不一样，不能进行x.view操作，需要解决！！！！
train_data = np.load(
    'D:\\python\\mnist_data\\MNIST\\raw\\train_data.npy', allow_pickle=True)
# train_data = np.load('train_data.npy')
train_label = np.load(
    'D:\\python\\mnist_data\\MNIST\\raw\\train_label.npy', allow_pickle=True)
test_data = np.load(
    'D:\\python\\mnist_data\\MNIST\\raw\\test_data.npy', allow_pickle=True)
test_label = np.load(
    'D:\\python\\mnist_data\\MNIST\\raw\\test_label.npy', allow_pickle=True)
# trainset是一个CIFAR10Dataset实例，可以用下标索引
# 下标索引会返回一个sample的data和label
trainset = CIFAR10Dataset(
    transform=transform, data=train_data, label=train_label)
# PyTorch 提供的dataloader可以让你方便地控制batchsize和shuffle
# 以及提供了异步接口
# 如果出现异步问题，设置num_workers=0
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=0
)
testset = CIFAR10Dataset(transform=transform, data=test_data, label=test_label)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=0
)

# 继承一个nn.Module，实现了构造函数和forward方法，就是一个网络模型


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
# 2维卷积，输入通道3，输出通道6，卷积核大小5x5
# 还有其它参数可以设置 (stride, padding)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
# fc fully connected，全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 4 * 4)
        # x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# net = Net().cuda() #改成这个把网络放到GPU上
# 交叉熵，PyTorch默认自带SoftMax
criterion = nn.CrossEntropyLoss()
# Stochastic Gradient Descent
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',)

running_loss = 0.0
for epoch in range(6):
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据，trainloader 每次会产生一个元组
        # 也就是 Dataset 返回的 data 和 label
        inputs, labels = data
# 但是 因为用DataLoader包装了，因此PyTorch会自动帮组成batch
# data.shape: (batchsize, 3, 32, 32)
# label.shape: (batchsize, 10)
# 如果使用GPU，要通过
        # inputs = inputs.cuda() #把数据放到GPU上
# 同样的也要
        # labels = labels.cuda()
        optimizer.zero_grad()
# 前传
        outputs = net(inputs)
# 计算 loss
        loss = criterion(outputs, labels.long())
        # running_loss += loss.item()
        running_loss += loss
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
# 反传

        loss.backward()
# 更新
        optimizer.step()
# 存模型
    
PATH = 'D:\\python\\pythontest\\model_save\\net_nn_mnist.pth'
torch.save(net.state_dict(), PATH)

net = Net()
# 加载之前训好的模型参数
net.load_state_dict(torch.load(
    'D:\\python\\pythontest\\model_save\\net_nn_mnist.pth'))
cnt = 0
correct = 0
'''
# 这部分不要反传
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
# torch.max 执行在outputs的维度1上
# 每次会输出两项，即max和argmax
        _, predicted = torch.max(outputs, 1)
# 一个简便的方法标出正确的预测
        c = (predicted == labels).sum()
        correct += c
        cnt += data.shape[0]
acc = correct / cnt
print(acc)
'''
#评估函数
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images, labels = data
        outputs = net(images)
# torch.max 执行在outputs的维度1上
# 每次会输出两项，即max和argmax
        _, predicted = torch.max(outputs, 1)
# 一个简便的方法标出正确的预测
        if predicted!=labels:
            correct=correct+1
        cnt =cnt+1
acc = correct / cnt
print(acc)

'''
re:
实验目的
1.编写构建2层神经网络及传入，迭代，反传，输出模型，载入，测试的过程
2.对照所给示例代码和操作理解神经网络的一般流程，类比考虑风格化所用神经网络结构和使用操作
实验设备：CPU
实验过程和结果
1.在学习率0.001，epoch=10时达到了99.1%的准确率accuracy，相较之前用过的非线性svm方法的98.5%有所提升。注意到，若把学习率调高到0.01量级，
准确率将以难以接受的速度大幅下降，结合资料推测应该是lr过大导致的model梯度无法收敛.
2.每次计算总用时（按0.001计算）大约10~15分钟，调大lr应该能够减小时间，但精度会下降
3.另外，过程中还保存了epoch=3和6，10时候的model
其中在6处达到了令人较为满意的98.3%的准确率，而3处仅有90%左右的准确率。或许适当选取epoch也能够在较短时间内得到相当的结果
遇到的问题
1.CPU真的没法跑神经网络，发烫，CPU占用达到100%算了几乎半小时。。。。qwq得去弄GPU了；又发现AMD显卡不支持pytorch和cuda.....只能租服务器。
2.能够拿到的只有MNIST的npz格式，需要load后变成map一样的文件夹，依据关键词key读取其中的x_train,y_train,x_test,y_test.npy文件
才能用于写好的.npy读取（当然，直接读取npz的子文件内容也可以直接用于训练数据）
3.MNIST是单通道，而课件里拿到的范例是三通道，所以需要在网络的卷积层和全连接层里改成单通道输入（或者也可以用unsqueeze函数扩充成三通道，
但是感觉过程会比较复杂)。
4.nn.Linear(16 * 5 * 5, 120)以及reshape函数需要根据数据的尺寸改成对应的（16*4*4，120），否则不匹配
'''
