# coding=utf-8

# Python 3.6.3 win_amd64
# 依赖库：numpy、sklearn、scipy
# MNIST数据文件夹位于当前目录下

import data_adapter as da
import numpy as np
from sklearn.decomposition import PCA
import math
import time

# 主成分分析，对特征降维
def pca_feature(train_imgs, test_imgs):
    pca = PCA(n_components = 50) # 保留 50 个特征
    train_imgs = pca.fit_transform(train_imgs)
    test_imgs = pca.transform(test_imgs)
    
    print('PCA 保留的特征数：%d' % pca.n_components_)
    print('主成分占方差比：%.2f%%' % (sum(pca.explained_variance_ratio_) * 100))
    
    return train_imgs, test_imgs
    
# 计算先验概率
def get_priors(train_labels, label_list):
    length = len(train_labels)
    priors = np.zeros(len(label_list))
    for label in label_list:
        label_num = np.sum(label == train_labels) # 当前标签总数
        priors[label] = label_num / length
    return priors

# 计算均值
def get_means(train_imgs, train_labels, label_list):
    means = np.zeros([len(label_list), train_imgs.shape[1]])
    for label in label_list:
        means[label] = np.mean(train_imgs[train_labels == label], axis = 0)
    return means

# 分别计算每个类别的协方差矩阵
def get_covs(train_imgs, train_labels, means, label_list):
    covs = np.zeros([len(label_list), train_imgs.shape[1], train_imgs.shape[1]])
    for label in label_list:
        covs[label] = np.cov(train_imgs[train_labels == label], rowvar = False)
    return covs
    
if __name__ == '__main__':
    # 加载数据
    train_imgs = da.img_loader('MNIST/train-images.idx3-ubyte')
    train_labels = da.label_loader('MNIST/train-labels.idx1-ubyte')
    test_imgs = da.img_loader('MNIST/t10k-images.idx3-ubyte')
    test_labels = da.label_loader('MNIST/t10k-labels.idx1-ubyte')
    
    print('----------------------------------------------------')
    
    # 进行主成分分析，消除冗余特征，避免协方差矩阵奇异
    train_imgs, test_imgs = pca_feature(train_imgs, test_imgs)
    
    print('----------------------------------------------------')
    
    label_list = np.unique(train_labels)
    
    priors = get_priors(train_labels, label_list)
    means = get_means(train_imgs, train_labels, label_list)
    covs = get_covs(train_imgs, train_labels, means, label_list)
    cov = np.cov(train_imgs, rowvar = False)

    # LDF
    g = np.zeros([test_imgs.shape[0], len(label_list)])
    success_count = 0 # LDF 方法分类正确的次数
    success_01_count = 0 # LDF 方法对类别 0 和 1 分类正确的次数
    test01_total = sum(test_labels == 0) + sum(test_labels == 1) # 测试集中类别 0 和 1 的总数
    
    start = time.clock() # 计时开始
    
    for label in label_list:
        # 计算 LDF 参数
        w = np.dot(np.linalg.inv(cov), means[label])
        w0 = -0.5 * np.dot(np.dot(means[label].T, np.linalg.inv(cov)), means[label]) + math.log(priors[label])
        
        for x in range(test_imgs.shape[0]):
            g[x][label] = np.dot(w.T, test_imgs[x]) + w0 # 判别函数
        
    for i in range(test_imgs.shape[0]):
        if np.argmax(g[i]) == test_labels[i]: # LDF 方法分类正确
            success_count += 1
            if test_labels[i] == 0 or test_labels[i] == 1: # LDF 方法对类别 0 和 1 分类正确
                success_01_count += 1
    
    interval = time.clock() - start # 计时结束，计算时间间隔
    
    print('LDF 用时：%.3f s' % interval)
    print('LDF 0 & 1 二分类的正确率为：%.2f%%' % (float(success_01_count / test01_total) * 100))
    print('LDF十分类的正确率为：%.2f%%' % (float(success_count / test_imgs.shape[0]) * 100))
    
    print('----------------------------------------------------')
    
    # QDF
    success_count = 0 # QDF 方法分类正确的次数
    success_01_count = 0 # QDF 方法对类别 0 和 1 分类正确的次数
    
    start = time.clock() # 计时开始
    
    for label in label_list:
        # 计算 QDF 参数
        W = -0.5 * np.linalg.inv(covs[label])
        w = np.dot(np.linalg.inv(covs[label]), means[label])
        w0 = -0.5 * np.dot(np.dot(means[label].T, np.linalg.inv(covs[label])), means[label]) + math.log(priors[label]) - 0.5 * math.log(np.linalg.det(covs[label]))
        
        for x in range(test_imgs.shape[0]):
            g[x][label] = np.dot(np.dot(test_imgs[x].T, W), test_imgs[x]) + np.dot(w.T, test_imgs[x]) + w0 # 判别函数
    
    for i in range(test_imgs.shape[0]):
        if np.argmax(g[i]) == test_labels[i]: # QDF 方法分类正确
            success_count += 1
            if test_labels[i] == 0 or test_labels[i] == 1: # QDF 方法对类别 0 和 1 分类正确
                success_01_count += 1
    
    interval = time.clock() - start # 计时结束，计算时间间隔
    
    print('QDF 用时：%.3f s' % interval)
    print('QDF 0 & 1 二分类的正确率为：%.2f%%' % (float(success_01_count / test01_total) * 100))
    print('QDF十分类的正确率为：%.2f%%' % (float(success_count / test_imgs.shape[0]) * 100))
    
# 运行结果：
# Load MNIST/train-images.idx3-ubyte succeeded...
# Load MNIST/train-labels.idx1-ubyte succeeded...
# Load MNIST/t10k-images.idx3-ubyte succeeded...
# Load MNIST/t10k-labels.idx1-ubyte succeeded...
# ----------------------------------------------------
# PCA 保留的特征数：50
# 主成分占方差比：82.46%
# ----------------------------------------------------
# LDF 用时：0.315 s
# LDF 0 & 1 二分类的正确率为：93.66%
# LDF十分类的正确率为：85.82%
# ----------------------------------------------------
# QDF 用时：0.706 s
# QDF 0 & 1 二分类的正确率为：97.78%
# QDF十分类的正确率为：96.35%