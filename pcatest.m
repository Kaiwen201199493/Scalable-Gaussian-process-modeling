clearvars;clc;close all;
addpath('dace');%引用文件夹
fun_name = 'Rosenbrock';
d = 100;
lower_bound = -2.048*ones(1,d);%下限
upper_bound = 2.048*ones(1,d);%上限

time =zeros(1,10);
RMSE = zeros(1,10);
dim = 50;
for i = 1:1
tic%保存当前时间
% sampling
 num_sample = 1000;%采样的数量
    sample_x = lhsdesign(num_sample,d).*(upper_bound - lower_bound)+lower_bound;%lhsdesign拉丁超立方采样,代码中是一个2(行) x 100(列)的矩阵;*符号是对矩阵;.*符号是对数组
    %sample_y = feval(fun_name,sample_x);
    [coeff,score,latent,tsquare] = pca(sample_x);
    new_sample_x=score(:,1:dim);
%     a = sample_x*coeff
    sample_y = feval(fun_name,sample_x);
    GP_model = dacefit(new_sample_x,sample_y, 'regpoly0', 'corrgauss', 1*ones(1,dim), 0.001*ones(1,dim), 1000*ones(1,dim));
    
 test_sample = 1000;%测试点的数量
     test_x = lhsdesign(test_sample,d).*(upper_bound - lower_bound)+lower_bound;
     new_test_x = test_x*coeff(:,1:dim);
     test_y = feval(fun_name,test_x);
%     
 test_predition = predictor(new_test_x,GP_model);
% 
 RMSe=sqrt(sum((test_predition- test_y).^2)/test_sample);
 toc%记录程序完成时间
RMSE(i)=RMSe;
time(i)=toc;
end

%save('PCA.mat','time','RMSE')

