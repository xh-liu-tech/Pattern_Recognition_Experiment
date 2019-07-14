# coding=utf-8

import numpy as np

# 样本数据
omega1 = np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0,2.7], [4.1, 2.8],
                   [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]])
omega2 = np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9],
                   [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]])
omega3 = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2],
                   [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9,5.1]])
omega4 = np.array([[-2.0, 8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
                   [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])

# 每一列表示一个样本
# 每一类前 8 个样本为训练样本
X = np.c_[omega1.T[:, :8], omega2.T[:, :8], omega3.T[:, :8], omega4.T[:, :8]]
# 每一类后 2 个为测试样本
test = np.c_[omega1.T[:, 8:], omega2.T[:, 8:], omega3.T[:, 8:], omega4.T[:, 8:]]
# 测试样本的标签
test_label = np.array([1, 1, 2, 2, 3, 3, 4, 4])

# 构造回归值，共 4 类，4 * 8 = 32 个训练样本
Y = np.zeros([4, 4 * 8])
Y[0, :8] = 1
Y[1, 8:16] = 1
Y[2, 16:24] = 1
Y[3, 24:32] = 1

# MSE Muti-class
X_hat = np.r_['0, 2', X, np.ones(X.shape[1])]
W_hat = np.dot(np.linalg.inv(np.dot(X_hat, X_hat.T) + 1e-6 * np.eye(len(X_hat))), np.dot(X_hat, Y.T))

# 测试
test_hat = np.r_['0, 2', test, np.ones(test.shape[1])]
g = np.dot(W_hat.T, test_hat)
predictions = []
for i in range(test_hat.shape[1]):
    predictions.append(np.argmax(g[:, i]) + 1)

print('分类正确率：%.2f%%' % ((sum(predictions == test_label) / len(test_label)) * 100))

# 运行结果：
# 分类正确率：75.00%