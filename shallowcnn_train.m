function [Y_predicted_train,W_in,W_output] = shallowcnn_train(feature_len, Features, label_matrix, train_size, labels, M)
% This function will train the shallow CNN to get the weight matrix for
% output which is W_output

W_in = zeros(M,feature_len);
biases = zeros(M,1);

% Uses the "Constrained weights" method in the paper.
for i = 1:M
    W_row_norm = 0;
    indices = ones(2,1);
    while labels(indices(1)) == labels(indices(2)) ||  W_row_norm < eps
        indices = randperm(train_size,2);
        F_diff = Features(:,indices(1))-Features(:,indices(2));
        W_row = F_diff-mean(F_diff);
        W_row_norm  = sqrt(sum(W_row.^2));
    end
    W_in(i,:) = W_row/W_row_norm;
    biases(i) = 0.5*(Features(:,indices(1))+Features(:,indices(2)))'*W_row/W_row_norm;
end

%to implement biases, set an extra input dimension to 1, and put the biases in the input weights matrix
Features = [Features;ones(1,train_size)];
W_in = [W_in biases];

% train the W_output using ridge regression
A = (W_in*Features).^2; % get hidden layer activations
Z = A*A'/train_size;
U = (A*label_matrix)'/train_size;
ridge = 0.5*min(diag(Z))*size(label_matrix,2)^2/M^2; % compute the ridge parameters according to the paper
W_output = U/(Z+ridge*eye(M)); % find output weights by solving the for regularised least mean square weights

%test with trained-on data
Y_predicted_train = W_output*A;