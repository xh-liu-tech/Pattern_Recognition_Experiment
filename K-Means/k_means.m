clear all;clc;close all;

%% 读取数据
load k_means_data.mat;

% Show the data point
plot(x1(:,1), x1(:,2), 'r.'); hold on;
plot(x2(:,1), x2(:,2), 'b.');
plot(x3(:,1), x3(:,2), 'k.');
plot(x4(:,1), x4(:,2), 'g.');
plot(x5(:,1), x5(:,2), 'm.');

%% K-Means聚类
c = 5; % 聚类个数
i = 0; % 迭代次数
n = size(X, 1); % 样本总数
cluster_center = X(unidrnd(n, c, 1), :); % 随机选择五个点初始化聚类中心
fprintf('初始聚类中心：\n');
for i = 1 : c
    fprintf('(%f, %f)\n', cluster_center(i, 1), cluster_center(i, 2));
end
% 画出聚类中心
plot(cluster_center(1, 1), cluster_center(1, 2), 'bx');
plot(cluster_center(2, 1), cluster_center(2, 2), 'bx');
plot(cluster_center(3, 1), cluster_center(3, 2), 'bx');
plot(cluster_center(4, 1), cluster_center(4, 2), 'bx');
plot(cluster_center(5, 1), cluster_center(5, 2), 'bx');
while true
    not_change = true; % 聚类中心是否不变
    [~, nearest_center] = min(pdist2(cluster_center, X)); % 计算每个样本最近的聚类中心
    for j = 1 : c
        new_cluster_center = mean(X(nearest_center == j, :)); % 计算新的聚类中心
        if not_change && norm(new_cluster_center - cluster_center(j, :)) > 1e-8 % 聚类中心改变
            not_change = false;
        end
        cluster_center(j, :) = new_cluster_center;
    end 
    if not_change
        break;
    end
    i = i +  1;
end

%% 聚类结果
figure
plot(X(nearest_center == 1, 1), X(nearest_center == 1, 2), 'r.'); hold on;
plot(X(nearest_center == 2, 1), X(nearest_center == 2, 2), 'b.');
plot(X(nearest_center == 3, 1), X(nearest_center == 3, 2), 'k.');
plot(X(nearest_center == 4, 1), X(nearest_center == 4, 2), 'g.');
plot(X(nearest_center == 5, 1), X(nearest_center == 5, 2), 'm.');
% 画出聚类中心
plot(cluster_center(1, 1), cluster_center(1, 2), 'kx');
plot(cluster_center(2, 1), cluster_center(2, 2), 'rx');
plot(cluster_center(3, 1), cluster_center(3, 2), 'rx');
plot(cluster_center(4, 1), cluster_center(4, 2), 'kx');
plot(cluster_center(5, 1), cluster_center(5, 2), 'kx');

fprintf('迭代次数：%d\n', i);

fprintf('最终聚类中心：\n');

for i = 1 : c
    fprintf('第 %d 个聚类中心的坐标：(%f, %f)\t样本数：%d\n', i, cluster_center(i, 1), cluster_center(i, 2), size(find(nearest_center == i), 2));
end

[center_dist, ~] = min(pdist2(mu, cluster_center));
center_dist_var = var(center_dist);
fprintf('均方误差：%f\n', center_dist_var);