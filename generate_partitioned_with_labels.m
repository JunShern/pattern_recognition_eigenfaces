function [training_data, test_data, l_train, l_test] = generate_partitioned_with_labels(split)
% generate_partitioned Separates data into 70% training_data and 30%
% test_data

load face.mat;

N = size(X,2);
data = vertcat(1:N, X);         % Labelling the original positions, in case want to do random partiioning of data
data = data(:, randperm(N));    % Random permutation of the integers, just the columns

% Partition data
training_data = data(:, 1:(split/10)*N);
test_data = data(:, (split/10)*N+1:end);

% Remove indices
training_indices = training_data(1, :);
test_indices = test_data(1, :);
training_data = training_data(2:end, :);
test_data = test_data(2:end, :);

% Always select labels using indices from file
l_train = l(training_indices);
l_test = l(test_indices);

end