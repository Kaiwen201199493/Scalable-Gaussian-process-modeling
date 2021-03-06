clearvars;clc;close all;
addpath('dace');
fun_name = 'Ellipsoid';
d = 100;
lower_bound = -5.12*ones(1,d);
upper_bound = 5.12*ones(1,d);

 time =zeros(1,10);
 RMSE = zeros(1,10);
 dim = 50;
 k = 50;
for i = 1:10
tic%保存当前时间
% sampling
 num_sample = 1000;
    sample_x = lhsdesign(num_sample,d).*(upper_bound - lower_bound)+lower_bound;%lhsdesign拉丁超立方采样,代码中是一个2(行) x 100(列)的矩阵;*符号是对矩阵;.*符号是对数组
    sample_y = feval(fun_name,sample_x);
    test = lle(sample_x',k,dim);
    num_train = 100;
if size(test',1) > num_train
    sample_x_select = zeros(num_train,dim);
    sample_x_temp = test';
    a = size(sample_x,1);
    indx = randi(size(test',1));
    sample_x_select(1,:) = sample_x_temp(indx,:);
    sample_x_temp(indx,:) = [];
    for ii = 1 : num_train - 1
        %[~,idx] = max(min(pdist2(sample_x_temp,sample_x_select(1:ii,:)),[],2));
        aa = pdist2(sample_x_temp,sample_x_select(1:ii,:));
        bb = min(aa,[],2);
        [~,idx] = max(bb);
        sample_x_select(ii+1,:) = sample_x_temp(idx,:);
        sample_x_temp(idx,:) = [];
    end
end

    
    sample_y_select = feval(fun_name,sample_x_select);
    GP_model = dacefit(sample_x_select,sample_y_select, 'regpoly0', 'corrgauss', 1*ones(1,dim), 0.001*ones(1,dim), 1000*ones(1,dim));
    
 test_sample = 1000;
     test_x = lhsdesign(test_sample,d).*(upper_bound - lower_bound)+lower_bound;
     test_points = lle(test_x',k,dim);
     test_y = feval(fun_name,test_x);
     
 test_predition = predictor(test_points',GP_model);

 RMSe=sqrt(sum((test_predition- test_y).^2)/test_sample);
 toc; 
 RMSE(i)=RMSe;
 time(i)=toc;
end
