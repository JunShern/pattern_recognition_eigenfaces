function [training_data, test_data, l_train, l_test] = generate_partitioned_by_class(split) 
% generate_partitioned Separates data into 70% training_data and 30%
% test_data

load face.mat;


data = vertcat(l, X);         % Labelling the original positions, in case want to do random partioning of data

training_data = zeros(size(data,1), 1);
test_data = zeros(size(data,1), 1);
for class = 1:size(data,2)
    class_data = data(:, data(1,:)==class);
    N = size(class_data,2);
    class_data = class_data(:, randperm(N));    % Random permutation of the integers, just the columns

    % Partition data
    training_data = horzcat(training_data, class_data(:, 1:(split/10)*N));
    test_data = horzcat(test_data, class_data(:, (split/10)*N+1:end));
end
% Removing preallocated column of zeros
training_data = training_data(:,2:end);
test_data = test_data(:,2:end);

% Remove indices
l_train = training_data(1, :);
l_test = test_data(1, :);
training_data = training_data(2:end, :);
test_data = test_data(2:end, :);

end

