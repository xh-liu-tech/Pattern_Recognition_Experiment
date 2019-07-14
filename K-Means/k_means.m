clear all;clc;close all;

%% ��ȡ����
load k_means_data.mat;

% Show the data point
plot(x1(:,1), x1(:,2), 'r.'); hold on;
plot(x2(:,1), x2(:,2), 'b.');
plot(x3(:,1), x3(:,2), 'k.');
plot(x4(:,1), x4(:,2), 'g.');
plot(x5(:,1), x5(:,2), 'm.');

%% K-Means����
c = 5; % �������
i = 0; % ��������
n = size(X, 1); % ��������
cluster_center = X(unidrnd(n, c, 1), :); % ���ѡ��������ʼ����������
fprintf('��ʼ�������ģ�\n');
for i = 1 : c
    fprintf('(%f, %f)\n', cluster_center(i, 1), cluster_center(i, 2));
end
% ������������
plot(cluster_center(1, 1), cluster_center(1, 2), 'bx');
plot(cluster_center(2, 1), cluster_center(2, 2), 'bx');
plot(cluster_center(3, 1), cluster_center(3, 2), 'bx');
plot(cluster_center(4, 1), cluster_center(4, 2), 'bx');
plot(cluster_center(5, 1), cluster_center(5, 2), 'bx');
while true
    not_change = true; % ���������Ƿ񲻱�
    [~, nearest_center] = min(pdist2(cluster_center, X)); % ����ÿ����������ľ�������
    for j = 1 : c
        new_cluster_center = mean(X(nearest_center == j, :)); % �����µľ�������
        if not_change && norm(new_cluster_center - cluster_center(j, :)) > 1e-8 % �������ĸı�
            not_change = false;
        end
        cluster_center(j, :) = new_cluster_center;
    end 
    if not_change
        break;
    end
    i = i +  1;
end

%% ������
figure
plot(X(nearest_center == 1, 1), X(nearest_center == 1, 2), 'r.'); hold on;
plot(X(nearest_center == 2, 1), X(nearest_center == 2, 2), 'b.');
plot(X(nearest_center == 3, 1), X(nearest_center == 3, 2), 'k.');
plot(X(nearest_center == 4, 1), X(nearest_center == 4, 2), 'g.');
plot(X(nearest_center == 5, 1), X(nearest_center == 5, 2), 'm.');
% ������������
plot(cluster_center(1, 1), cluster_center(1, 2), 'kx');
plot(cluster_center(2, 1), cluster_center(2, 2), 'rx');
plot(cluster_center(3, 1), cluster_center(3, 2), 'rx');
plot(cluster_center(4, 1), cluster_center(4, 2), 'kx');
plot(cluster_center(5, 1), cluster_center(5, 2), 'kx');

fprintf('����������%d\n', i);

fprintf('���վ������ģ�\n');

for i = 1 : c
    fprintf('�� %d ���������ĵ����꣺(%f, %f)\t��������%d\n', i, cluster_center(i, 1), cluster_center(i, 2), size(find(nearest_center == i), 2));
end

[center_dist, ~] = min(pdist2(mu, cluster_center));
center_dist_var = var(center_dist);
fprintf('������%f\n', center_dist_var);