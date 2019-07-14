# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

def ho_kashyap(Y, a, b, eta, b_min, k_max):
    k = 0
    while k < k_max:
        e = np.dot(Y, a) - b
        J_s = np.sum(e ** 2) # 训练误差
        e_plus = 0.5 * (e + abs(e))
        b += 2.0 * eta * e_plus
        a = np.dot(np.linalg.pinv(Y), b)

        if len(e[abs(e) > b_min]) == 0: # abs(e) <= b_min
            print('权向量 a: %s^T' % a)
            print('向量 b: %s^T' % b)
            print('迭代次数: %d' % k)
            print('训练误差: %.2f' % J_s)
            return a, b
        k += 1
    
    print('No solution found!')
    print('权向量 a: %s^T' % a)
    print('向量 b: %s^T' % b)
    print('迭代次数: %d' % k)
    print('训练误差: %.2f' % J_s)
    return a, b

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

# ω1 & ω3
a = np.zeros(len(omega1[0]))
Y = np.r_[omega1, -omega3]
b = np.array([0.1] * len(Y))
a, b = ho_kashyap(Y, a, b, 0.5, 0.01, 10000)

# 绘制分类结果
# 样本
plt.scatter(omega1[:, 0], omega1[:, 1], color = 'r', label = 'omega1')
plt.scatter(omega3[:, 0], omega3[:, 1], color = 'g', label = 'omega3')
plt.legend(loc='lower right')
# 决策面
x = np.linspace(-5.5, 7.5, 20)
y = (-a[2] - a[0] * x) / a[1]
plt.plot(x, y)
plt.show()

# ω2 & ω4
a = np.zeros(len(omega2[0]))
Y = np.r_[omega2, -omega4]
b = np.array([0.1] * len(Y))
a, b = ho_kashyap(Y, a, b, 0.5, 0.01, 10000)

# 绘制分类结果
# 样本
plt.scatter(omega2[:, 0], omega2[:, 1], color = 'r', label = 'omega2')
plt.scatter(omega4[:, 0], omega4[:, 1], color = 'g', label = 'omega4')
plt.legend(loc='lower right')
# 决策面
x = np.linspace(-9.5, 9.5, 20)
y = (-a[2] - a[0] * x) / a[1]
plt.plot(x, y)
plt.show()

# 运行结果
# No solution found!
# 权向量 a: [ 0.03453682 -0.02514952  0.0481429 ]^T
# 向量 b: [ 0.1         0.10443171  0.1         0.1         0.11932522  0.1         0.1
  # 0.1         0.1         0.1         0.1         0.15338952  0.1         0.1
  # 0.14533334  0.1         0.22520933  0.17896645  0.16823413  0.1       ]^T
# 迭代次数: 10000
# 训练误差: 0.08
# (图片)
# No solution found!
# 权向量 a: [ 0.03652388 -0.00054699  0.02169241]^T
# 向量 b: [ 0.27871458  0.1         0.18604987  0.25091766  0.17405342  0.1
  # 0.11153769  0.11633877  0.32646912  0.1726437   0.1         0.30347951
  # 0.12749603  0.28701018  0.2208296   0.1         0.16821929  0.29256458
  # 0.23232129  0.26705256]^T
# 迭代次数: 10000
# 训练误差: 0.03
# (图片)