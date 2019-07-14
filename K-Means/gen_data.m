clear all;clc;close all;

%% 生成1000个数据点，存储在矩阵X中
Sigma = [1, 0; 0, 1];
mu1 = [1, -1];
x1 = mvnrnd(mu1, Sigma, 200);
mu2 = [5.5, -4.5];
x2 = mvnrnd(mu2, Sigma, 200);
mu3 = [1, 4];
x3 = mvnrnd(mu3, Sigma, 200);
mu4 = [6, 4.5];
x4 = mvnrnd(mu4, Sigma, 200);
mu5 = [9, 0.0];
x5 = mvnrnd(mu5, Sigma, 200);
% obtain the 1000 data points to be clustered
X = [x1; x2; x3; x4; x5];
% 各真实分布的均值向量分别为mu1, mu2, mu3, mu4, mu5
mu = [mu1;mu2;mu3;mu4;mu5];
% 保存数据
save k_means_data.mat;