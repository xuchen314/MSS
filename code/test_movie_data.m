% This script is for the paper: 
% A Unified Convex Surrogate for the Schatten-p Norm. Chen Xu, Zhouchen Lin, Hongbin Zha. AAAI-17
% If you have any questions, feel free contract Chen Xu (xuen@pku.edu.cn)
% 
% Copyright: Peking University

addpath('data')
addpath('Utilities')  % It is better to mex the files in this fold first
dataset = 2;
namelist = {'moive-100K','moive-1M','moive-10M','Netflix'};
%You can download the full dataset from https://drive.google.com/open?id=0B244f561lTfqazF4U0MyZjNfc2c
fname = namelist{dataset};
load(fname,'M');
if size(M,1) > size(M,2)
    M = M';
end
normB = sqrt(sum(M.*M));
zerocolidx = find(normB==0);
if ~isempty(zerocolidx)
    fullcol = setdiff(1:size(M,2), zerocolidx);
    M = M(:,fullcol);
end
clear fullcol

% normB2 = sqrt(sum(M.*M,2));
% zerorowsidx = find(normB2==0);
% if ~isempty(zerorowsidx)
%     fullrow = setdiff(1:size(M,1), zerorowsidx);
%     M = M(fullrow,:);
% end
% clear fullrow

[m,n] = size(M);
[I_all,J_all,data_all] = find(M);
fprintf('Size is %d * %d, Sample Rate is %g \n', m, n, length(I_all)/m/n)
clear M


rng(1,'twister');

train_rate = 0.80;  
trian_sample = randperm(length(I_all),round(train_rate*length(I_all)));
trian_sample = sort(trian_sample);
test_sample = setdiff(1:length(I_all), trian_sample);
I_train = I_all(trian_sample);       I_test = I_all(test_sample);             clear I_all 
J_train = J_all(trian_sample);       J_test = J_all(test_sample);             clear J_all 
data_train = data_all(trian_sample); data_test = data_all(test_sample);       clear data_all
clear trian_sample test_sample
L=length(I_train);

p_i=[1,1,1,1];        %1/p=  \sum_{i=1}^{I} 1/p_i
k=10;                 %Estimated Rank
for i=1:length(p_i)
     fprintf('p_%1d=%1d, ',i,p_i(i))
end
fprintf('\n')

X = cell(1,length(p_i));
X{1} = rand(m,k);  
X{end} = rand(k,n);  
for i = 2:length(p_i)-1
    X{i} = eye(k);
end

clear opts
opts.X = X; 
opts.I_test = I_test; opts.J_test = J_test; opts.data_test = data_test;

opts.usemex = 0;              % Whether to use the mex the files in Utilities to accelerate computation and handle large scale matrixes
opts.show_progress = true;    % Whether show the progress  
opts.show_interval = 100;   
opts.eta = 0.1;               % Muliply eta <1 to get a locally  Lipchitz constants for better performance
opts.lambda = 200;            % 500 also works for the movie-data
opts.p_i = p_i;
opts.maxit = 2000;            % Max Iteration  
opts.tol = 1e-7;        
opts.maxtime = 8e3;           % Sometimes terminating early is good for testing. 
opts.gate_upt = 2/5;          % The gate indicating which factor to be updated
siz.m = m; siz.n = n;  siz.k = k;
[X,Out]=f_multix(data_train,I_train,J_train,siz,opts);

clear opts 
