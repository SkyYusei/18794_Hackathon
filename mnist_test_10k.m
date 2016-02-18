function [ err_60k, err_12k, err_3k ] = mnist_test_10k( imgs, labels, model_60k, model_12k, model_3k )
% This fucntion computes the error rates on test images with three
% different models

X_test = reshape(imgs, 400, 10000);
[err_60k, error_indices_60k] = get_err(X_test, labels, model_60k);
[err_12k, error_indices_12k] = get_err(X_test, labels, model_12k);
[err_3k, error_indices_3k] = get_err(X_test, labels, model_3k);
save('error_indices_3k.mat','error_indices_3k');
save('error_indices_12k.mat', 'error_indices_12k');
save('error_indices_60k.mat', 'error_indices_60k');
end

