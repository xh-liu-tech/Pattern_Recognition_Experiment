# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

def batch_perception(x, y, a, eta):
    y = -y # 规范化
    y = np.r_[x, y] # 样本集合
    Y = y[np.dot(a, y.T) <= 0] # 错分样本集合
    i = 0
    while len(Y) > 0:
        sum_Y = Y.sum(axis = 0)
        a += eta * sum_Y
        Y = y[np.dot(a, y.T) <= 0] # 错分样本集合
        i += 1
    
    print('The number of iterations required for convergence: %d' % i)
    print('The weight vector a: %s^T' % a)

# 样本数据
omega1 = np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0,2.7], [4.1, 2.8],
                   [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]])
omega2 = np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9],
                   [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]])
omega3 = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2],
                   [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9,5.1]])
omega4 = np.array([[-2.0, 8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
                   [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])

n = len(omega1)
ones = np.ones(n)

# 化为齐次坐标表示
omega1 = np.c_[omega1, ones]
omega2 = np.c_[omega2, ones]
omega3 = np.c_[omega3, ones]
omega4 = np.c_[omega4, ones]

eta = 0.5

# ω1 & ω2
a = np.zeros(len(omega1[0]))
batch_perception(omega1, omega2, a, eta)

# 绘制分类结果
# 样本
plt.scatter(omega1[:, 0], omega1[:, 1], color = 'r', label = 'omega1')
plt.scatter(omega2[:, 0], omega2[:, 1], color = 'g', label = 'omega2')
plt.legend(loc='lower right')
# 决策面
x = np.linspace(-4, 9, 20)
y = (-a[2] - a[0] * x) / a[1]
plt.plot(x, y)
plt.show()

# ω3 & ω2
a = np.zeros(len(omega1[0]))
batch_perception(omega3, omega2, a, eta)

# 绘制分类结果
# 样本
plt.scatter(omega2[:, 0], omega2[:, 1], color = 'g', label = 'omega2')
plt.scatter(omega3[:, 0], omega3[:, 1], color = 'r', label = 'omega3')
plt.legend(loc='lower right')
# 决策面
x = np.linspace(-6, 9, 20)
y = (-a[2] - a[0] * x) / a[1]
plt.plot(x, y)
plt.show()

# 运行结果：
# The number of iterations required for convergence: 23
# The weight vector a: [-15.2   17.05  17.  ]^T
# (图片)
# The number of iterations required for convergence: 16
# The weight vector a: [-20.7  24.3   9.5]^T
# (图片)