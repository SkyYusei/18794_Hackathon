function [Features] = get_conv_features(W, Q, pooling_step, image_size, train_size, X, Filters) 
% Extract the features from the input training images
% The implementation is based on the paper at http://arxiv.org/abs/1503.04596

pool_rows = W+5:pooling_step:image_size+Q-1-5; %index into the full size: ImageSize+W+Q-2,ImageSize+W+Q-2
pool_cols = W+5:pooling_step:image_size+Q-1-5;

%define some dimensions
num_of_filters = size(Filters,1);
full_size = image_size+W-1;
 
%get pooling matrix
pool_indices = zeros(image_size+W+Q-2,image_size+W+Q-2);
pool_indices(pool_rows,pool_cols)=1;
num_of_features_per_filter = length(find(pool_indices(:))==1);

pooling_filter = ones(Q,Q)/Q^2;
P0 = convmtx2(pooling_filter,full_size,full_size); 
P = P0(pool_indices(:)==1,:);
 
%get features
Features = zeros(num_of_filters*num_of_features_per_filter,train_size);
for i = 1:num_of_filters
    %get convolution matrix
    F = convmtx2(squeeze(Filters(i,:,:)),image_size,image_size);
    
    %get features
    Features((i-1)*num_of_features_per_filter+1:i*num_of_features_per_filter,:) = sqrt(P*((F*X).^2));
end