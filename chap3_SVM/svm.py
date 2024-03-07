# python: 3.5.2
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import math
import os

def drawLine(dataRed, dataBlue, w, b, title='Linear Classifier'):
    # points
    plt.plot(dataRed[:, 0], dataRed[:, 1], 'ro')
    plt.plot(dataBlue[:, 0], dataBlue[:, 1], 'b*')

    # line
    maxX1 = math.ceil(max(max(dataRed[:,0]), max(dataBlue[:,0])))
    minX1 = math.floor(min(min(dataRed[:, 0]), min(dataBlue[:, 0])))
    x0 = np.linspace(minX1, maxX1, 200)
    w0 = w[0][0]
    w1 = w[0][1]
    x1 = (-b - w0 * x0)/w1

    # labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

    plt.plot(x0, x1)
    plt.show()
 

def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self, learningRate=1e-7, normLambda=0.01):
        self.w = np.random.rand(1,2)
        self.b = np.random.rand(1,1)
        self.lambdaV = normLambda
        self.learningRate = learningRate

    def sgd(self, dataTrain):
        predY = np.dot(dataTrain[:, :2], np.transpose(self.w)) + self.b

        # squared loss calculate
        ySubt = np.subtract(predY, dataTrain[:, 2].reshape(len(dataTrain), 1)) 
        self.loss = np.power(np.linalg.norm(ySubt, axis=0),2) + self.lambdaV * np.power(np.linalg.norm(self.w, axis=1),2)

        # calculate gradient and sgd training
        wgrad = np.dot(np.transpose(2 * ySubt), dataTrain[:, :2]) + 2 * self.lambdaV * self.w
        bgrad = np.sum(2 * ySubt, axis=0)
        self.w = self.w - self.learningRate * wgrad
        self.b = self.b - self.learningRate * bgrad

    def train(self, data_train, epoch=100):
        """
        训练模型。
        """
        for i in range(epoch):
            self.sgd(data_train)
            print("The ", i, " th epoch, Loss is ", self.loss)
    

    def predict(self, x):
        """
        预测标签。
        """
        xi = x[:,0]
        xj = x[:,1]
        if xi.ndim == 1 or xi.shape[1] == 1:
            x = np.concatenate((xi.reshape(len(xi), 1), xj.reshape(len(xj), 1)), axis=1)
        
        predY = np.dot(x, np.transpose(self.w)) + self.b
        predY[predY > 0] = 1
        predY[predY < 0] = -1
        return np.ravel(predY)



if __name__ == '__main__':
    # 获得文件所在绝对路径
    root_dir = os.path.split(os.path.realpath(__file__))[0]
    # 载入数据，实际实用时将x替换为具体名称
    train_file = root_dir+'\\data\\train_linear.txt'
    test_file = root_dir+'\\data\\test_linear.txt'
    print(test_file)
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_train_pred = svm.predict(x_train)  # 预测标签
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
    drawLine(x_test[:100], x_test[100:], svm.w, svm.b[0][0], title='Linear Classifier')
