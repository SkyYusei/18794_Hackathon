function [ error_rate_test ] = get_err( X_test, labels, model )
% This function computes error rate on the test images with given model

% read parameters from the model
test_size = 10000;
image_size = 20;
W = model.W;
Q = model.Q;
max_feature = model.max_feature;
pooling_step = model.pooling_step;
Filters = model.Filters;
W_in = model.W_in;
W_output = model.W_output;

% get and scale the features of test images
Features = get_conv_features(W,Q,pooling_step,image_size,test_size,X_test,Filters);clear X_test
Features = sqrt(Features/max_feature);

Y_predicted_test = W_output * ((W_in*[Features;ones(1,test_size)]).^2);
% get output layer response and then classify it
[MaxVal,ClassificationID_test] = max(Y_predicted_test); 

% compute the error rate
error_rate_test = 100*(length(find(ClassificationID_test-1-labels'~=0))/test_size)

end

