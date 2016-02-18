% This script is an example to do cross_validation on 3K data, the data is
% divided into 6 folds. I created a specific parameter space for M to test 
% M in that space and get the best M with minimal error rate.

clear all;
rand('state',0);
data_3k = load('train_3k_mnist');
labels = data_3k.labels;

X = reshape(data_3k.imgs,400,3000);

% get the random indices for the 6 partions
indices = crossvalind('Kfold', X(400,1:3000),6);

% Set parameters
W=5; % filter size
Q=8; % pooling size
% W = [2 3 4 5 6 7 8]
% Q = [4 5 6 7 8 9 10]
pooling_step = 2; % number of pixels between pooling points
num_for_each = 10;
% num_for_each = [5 6 7 8 9 10 11 12 13];
M=[3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000];  %number of hidden units

min_err = 100;
best_M = M(1);

for j = 1:length(M)
    err = 0;
    for k = 1:6 % 6-fold cross validation
        test = (indices==k);
        train = ~test;
        train_imgs = X(:,train);
        train_labels = labels(train);
        test_imgs=X(:,test);
        test_labels = labels(test);
        err = err + cross_train(train_imgs, train_labels, test_imgs, test_labels, W, Q, pooling_step, num_for_each, M(j));
    end
    err = err/6 % compute the average err
    if err < min_err
        min_err = err;
        best_M = M(j);
    end
end


