function [ err ] = cross_train( train_imgs, train_labels, test_imgs, test_labels, W, Q, pooling_step, num_for_each, M)
% A helper function for cross_validation

X = train_imgs;
X_test = test_imgs;
train_size = size(train_labels,1);
test_size = size(test_labels,1);
image_size = 20;
num_of_classes = 10;
label_matrix = zeros(train_size, num_of_classes);
for i = 1:train_size
    label_matrix(i, train_labels(i)+1)=1;
end

% Get filters
Filters = get_filters(W, image_size, num_of_classes, num_for_each, X, train_labels);

% Get features
Features = get_conv_features(W, Q, pooling_step, image_size, train_size, X, Filters);clear X
TestFeatures = get_conv_features(W,Q,pooling_step,image_size,test_size,X_test,Filters);clear X_test

% train the shallow CNN
% begin by ensuring training data is in the interval [0,1] and boost the smaller values to help make more Gaussian
max_feature = max(max(Features));
Features = sqrt(Features/max_feature);
TestFeatures = sqrt(TestFeatures/max_feature);

% train the output layer
[Y_predicted_train,W_in,W_output] = shallowcnn_train(size(Features,1),Features,label_matrix,train_size,train_labels,M);

Y_predicted_test = W_output * ((W_in*[TestFeatures;ones(1,test_size)]).^2);
[MaxVal,ClassificationID_test] = max(Y_predicted_test); %get output layer response and then classify it
err = 100*(length(find(ClassificationID_test-1-test_labels'~=0))/test_size);


end

