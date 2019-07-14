# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import math, random, csv

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class Node(object): # 神经网络的结点
    def __init__(self, f = None):
        self.f = f
        self.net_val = None
        
    def calc_f_net_val(self, x, w = None): # 计算激励后的 net 值
        if self.f == 'sigmoid': # 输出层
            self.net_val = np.dot(np.array(x), np.array(w))
            return  sigmoid(self.net_val) # 激励函数为 sigmoid 函数
        elif self.f == 'tanh': # 隐含层
            self.net_val = np.dot(np.array(x), np.array(w))
            return math.tanh(self.net_val) # 激励函数为双曲正切函数
        else: # 输入层
            return x
            
class NeuralNetwork(object): # 三层神经网络，采用随机更新算法
    def __init__(self, layers, eta = 0.5, theta = 0.5):
        self.input = [Node() for _ in range(layers[0])] # 输入层有 layers[0] 个结点
        self.hidden = [Node('tanh') for _ in range(layers[1])] # 隐含层有 layers[1] 个结点
        self.output = [Node('sigmoid') for _ in range(layers[2])] # 输出层有 layers[2] 个结点
        self.eta = eta # 学习率
        self.theta = theta # 终止条件
        
        # 随机初始化权重
        self.w_ih = np.array([random.uniform(-2, 2) for _ in range(layers[0] * layers[1])])
        self.w_hj = np.array([random.uniform(-2, 2) for _ in range(layers[2] * layers[1])])
        self.w_ih = self.w_ih.reshape(layers[0], layers[1])
        self.w_hj = self.w_hj.reshape(layers[1], layers[2])
        
        # 每层各结点的值
        self.input_val = None
        self.hidden_val = None
        self.output_val = None
        
        # 记录目标函数的变化
        self.record_e = []
        
    def __calc(self, x):
        self.input_val = np.array(x) # 输入第 k 个样本
                
        # 计算隐含层结点的值
        self.hidden_val = np.array([self.hidden[h].calc_f_net_val(self.input_val, self.w_ih[:, h]) for h in range(len(self.hidden))])
        
        # 计算输出层结点的值
        self.output_val = np.array([self.output[j].calc_f_net_val(self.hidden_val, self.w_hj[:, j]) for j in range(len(self.output))])
    
    def __total_err(self, x, t):
        sum = 0.0
        self.err_samples = [] # 被错分的样本
        for k in range(len(x)):
            self.__calc(x[k])
            sum += np.linalg.norm(self.output_val - np.array(t[k]))
            if self.output_val.argmax() != t[k].argmax():
                self.err_samples.append(x[k])
        print(sum, len(self.err_samples))
        return sum

    def stochastic_bp(self, x, t):
        while True:
            k = random.randint(0, len(x) - 1) # 随机选择第 k 个样本进行更新
            self.__calc(x[k])
            
            delta_j = [] # 输出层结点收集到的误差信号
            for j in range(len(self.output)):
                net_j = self.output[j].net_val
                delta_j.append(sigmoid(net_j) * (1 - sigmoid(net_j)) * (t[k][j] - self.output_val[j]))
            delta_j = np.array(delta_j)
            
            # 更新隐含层到输出层的权重
            for h in range(len(self.hidden)):
                for j in range(len(self.output)):
                    self.w_hj[h, j] += self.eta * delta_j[j] * self.hidden_val[h]
            
            delta_h = [] # 隐含层结点收集到的误差信号
            for h in range(len(self.hidden)):
                net_h = self.hidden[h].net_val
                delta_h.append((1 - math.tanh(net_h) ** 2) * np.dot(self.w_hj[h, :], delta_j))
            delta_h = np.array(delta_h)
            
            # 更新输入层到隐含层的权重
            for i in range(len(self.input)):
                for h in range(len(self.hidden)):
                    self.w_ih[i, h] += self.eta * delta_h[h] * self.input_val[i]
        
            self.record_e.append(self.__total_err(x, t)) # 计算并记录误差
            
            if len(self.record_e) > 10: # 由于单样本更新不够稳定，采用连续十次更新的变化判断是否收敛
                is_converged = True
                for i in range(10):
                    if abs(self.record_e[len(self.record_e) - i - 1] - self.record_e[len(self.record_e) - i - 2]) >= self.theta:
                        is_converged = False
                        break
            else: # 未迭代够十次
                is_converged = False
            
            # 判断是否满足终止条件
            if is_converged:
                print('达到终止条件')
                print('迭代次数：%d' % len(self.record_e))
                print('被错分的样本数：%d' % len(self.err_samples))
                plt.plot(range(len(self.record_e)), self.record_e)
                plt.xlabel("Iteration")
                plt.ylabel("Error")
                plt.show()
                break

                
if __name__ == '__main__':
    csv_reader = csv.reader(open('data.csv', 'r'))
    x = [] # 样本
    t = [] # 标识
    for line in csv_reader:
        x.append([1, float(line[0]), float(line[1]), float(line[2])]) # bias & data
        target = [0, 0, 0]
        target[int(line[3]) - 1] = 1
        t.append(target)
    x = np.array(x)
    t = np.array(t)
    
    nn = NeuralNetwork(layers = [4, 15, 3], eta = 0.5, theta = 1e-4)
    nn.stochastic_bp(x, t)