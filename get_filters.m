function [Filters] = get_filters(W, image_size, num_of_classes, num_for_each, X, labels) 
% This function returns the filters extracted from the center patches of 
% the training images

num_of_filters = num_of_classes*num_for_each;
Filters = zeros(num_of_filters,W,W);
count = 1;

for i = 1:num_for_each
    for current_class = 1:num_of_classes
        current_class = find(labels==current_class-1);
        X0 = reshape(X(:,current_class(i)),[image_size,image_size]); %get an image from class i
        X1 = X0(round((image_size-W)/2)+1:round((image_size-W)/2)+W,round((image_size-W)/2)+1:round((image_size-W)/2)+W); %extract a patch from the centre of the image
        Filters(count,:,:) = X1-mean(X1(:));
        count = count + 1;
    end
end