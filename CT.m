clearvars;clc;close all;
addpath('dace');
fun_name = 'Ellipsoid';
d = 100;
lower_bound = -5.12*ones(1,d);%下限
upper_bound = 5.12*ones(1,d);%上限

time =zeros(1,10);
RMSE = zeros(1,10);
for i = 1:10
tic%保存当前时间
% sampling
 num_sample = 1000 ; %采样的数量
    sample_x = lhsdesign(num_sample,d).*(upper_bound - lower_bound)+lower_bound;
    sample_y = feval(fun_name,sample_x);
    GP_model = dacefit(sample_x,sample_y, 'regpoly0', 'corrgauss', 1*ones(1,d), 0.001*ones(1,d), 1000*ones(1,d));
    
 test_sample = 1000;%测试点的数量
     test_x = lhsdesign(test_sample,d).*(upper_bound - lower_bound)+lower_bound;
     test_y = feval(fun_name,test_x);  
 test_predition = predictor(test_x,GP_model);
 RMSe=sqrt(sum((test_predition- test_y).^2)/test_sample);
 toc%记录程序完成时间  
 RMSE(i)=RMSe;
 time(i)=toc;
 end

% save('t.mat','time','RMSE')

