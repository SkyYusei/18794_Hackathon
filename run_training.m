% This script loads the three datasets and train the corresponding models,
% the trained models will be saved into mat file for testing.

clear all;
rand('state',0);

% load the training set
data_3k = load('train_3k_mnist');
data_12k = load('train_12k_mnist');
data_60k = load('train_60k_mnist');

% train three models and save them into files
model_3k = mnist_train_3k(data_3k.imgs, data_3k.labels);
save('model_3k.mat','model_3k'); clear data_3k model_3k

model_12k = mnist_train_12k(data_12k.imgs, data_12k.labels);
save('model_12k.mat','model_12k'); clear data_12k model_12k

model_60k = mnist_train_60k(data_60k.imgs, data_60k.labels);
save('model_60k.mat','model_60k'); clear data_60k model_60k



