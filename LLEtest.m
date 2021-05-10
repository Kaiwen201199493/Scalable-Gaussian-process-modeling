clearvars;clc;close all;
addpath('dace');%引用文件夹
fun_name = 'Griewank';
d = 100;
k = 50;
dim = 5;
lower_bound = -600*ones(1,d);%下限
upper_bound = 600*ones(1,d);%上限

time =zeros(1,10);
RMSE = zeros(1,10);
for i = 1:10
tic%保存当前时间
% sampling
 num_sample = 1000; %采样的数量
    sample_x = lhsdesign(num_sample,d).*(upper_bound - lower_bound)+lower_bound;
     test = lle(sample_x',k,dim);
     a = test';
    sample_y = feval(fun_name,sample_x);
    GP_model = dacefit(a,sample_y, 'regpoly0', 'corrgauss', 1*ones(1,dim), 0.001*ones(1,dim), 1000*ones(1,dim));
    
 test_sample = 1000;%测试点的数量
     test_x = lhsdesign(test_sample,d).*(upper_bound - lower_bound)+lower_bound;
     test_points = lle(test_x',k,dim);
     b = test_points';
     test_y = feval(fun_name,test_x);  
test_predition = predictor(b ,GP_model);
RMSe=sqrt(sum((test_predition- test_y).^2)/test_sample);
 toc%记录程序完成时间  
 RMSE(i)=RMSe;
 time(i)=toc;
 end

% save('t.mat','time','RMSE')