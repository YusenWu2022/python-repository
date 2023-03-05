import pickle
import gzip
import numpy as np
from sklearn import svm
import time
from sklearn.preprocessing import scale


# TIME函数库用于输出时间供比较训练结果
# 纳入全部函数库
def load_data():
    """
    返回包含训练数据、验证数据、测试数据的元组的模式识别数据
    训练数据包含50，000张图片，测试数据和验证数据都只包含10,000张图片
    """
    f = gzip.open('D:/python/pythontest/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)
 # 从‘pkl.gz'中读取训练和测试数据


def svm_baseline():
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    # 输出开始时间
    training_data, validation_data, test_data = load_data()
    #training_data = scale(training_data[0])
    # 从load_data中读取传递训练模型的参数（默认的参数）
    clf = svm.SVC(C=1000.0, kernel='rbf', gamma=0.03, verbose=1, max_iter=2000)
    #clf = svm.SVC(C=8.0, kernel='rbf',cache_size=8000,probability=False)
    #clf = svm.LinearSVC(dual=True, tol=0.0001,C=100.0, verbose=1, max_iter=10000)
    # 两种可能svc函数供选择，事实证明非线性的svm.SVC训练效果更好
    # 非线性模型训练，调大C加快速率
    clf.fit(training_data[0], training_data[1])

    # data[0]是图形集，data[1]是标签集，实现训练
    # test
    # 测试集测试预测结果
    predictions = [int(a) for a in clf.predict(test_data[0])]
    #for i in range(1, 100):
        #print(predictions[i])
    # 可以输出预期结果
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("%s of %s test values correct." % (num_correct, len(test_data[1])))
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    # 输出结束时间


if __name__ == "__main__":
    svm_baseline()
# 主函数
