% This script will load the trained models and do classifcation on test
% images, the err value of the three models will be returned

clear all;
test_10k = load('test_10k_mnist');
load('model_3k');
load('model_12k');
load('model_60k');
[err_60k, err_12k, err_3k] = mnist_test_10k(test_10k.imgs, test_10k.labels, model_60k, model_12k, model_3k)
