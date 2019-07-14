clear all;clc;close all;

%% 读取数据
load('data.txt');

%% 设置参数
k = 2; % 聚类个数
k_neighbor = 10; % 选取的最近邻样本点数
w_sigma = 1; % 点对亲和性（即边权值）计算的参数
n = size(data, 1); % 样本总数

%% Normalized Spectral Clustering - Algorithm 3 (Ng 算法）
% 计算亲和度矩阵
W = zeros(n, n);
kdtree = createns(data, 'NSMethod', 'kdtree'); % 建立使用欧氏距离的kd树
for i = 1 : n
    [neighbor, dist] = knnsearch(kdtree, data(i, :), 'K', k_neighbor + 1); % 查找k近邻，返回结果为升序
    for j = 2 : size(neighbor, 2) % 第一个点为本身
        W(i, neighbor(j)) = exp(-dist(j) / (2 * w_sigma^2)); % 采用高斯函数计算点对亲和性
    end
end

% 保证亲和度矩阵是对称矩阵
W = (W' + W) / 2;

% 计算度矩阵和拉普拉斯矩阵
D = diag(sum(W, 2));
L = D - W;
D = D^(-0.5);
L = D * L * D + 1e-3 * eye(n); % avoid singular

% 计算前 k 个最小的特征向量
[U, ~] = eigs(L, k, 'SM');

% 归一化
for i = 1 : n
    U(i, :) = U(i, :) / norm(U(i, :));
end

% K-Means聚类
idx = kmeans(U, k);

%% 统计结果
class_1 = idx(1:100);
class_2 = idx(101:200);
correct_class_1_count = length(find(class_1 == mode(class_1)));
if mode(class_1) == mode(class_2)
    class_2 = find(class_2 ~= mode(class_2));
    correct_class_2_count = length(find(class_2 == mode(class_2)));
else
    correct_class_2_count = length(find(class_2 == mode(class_2)));
end
accuracy = (correct_class_1_count + correct_class_2_count) / n;
fprintf('Accuracy: %f\n', accuracy);

title(['k\_neighbor = ' num2str(k_neighbor) '     w\_sigma = ' num2str(w_sigma) '     Accuracy: ' num2str(accuracy)]);
hold on;
xlabel('X');
ylabel('Y');

for i = 1 : k
    idx_class = find(idx == i);
    plot(data(idx_class, 1), data(idx_class, 2), 'x', 'LineWidth', 1, 'Display', ['CLASS : ' num2str(i)]);
end

legend('show');