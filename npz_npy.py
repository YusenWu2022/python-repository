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

