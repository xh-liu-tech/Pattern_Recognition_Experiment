clear all;clc;close all;

%% ����1000�����ݵ㣬�洢�ھ���X��
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
% ����ʵ�ֲ��ľ�ֵ�����ֱ�Ϊmu1, mu2, mu3, mu4, mu5
mu = [mu1;mu2;mu3;mu4;mu5];
% ��������
save k_means_data.mat;