%clear;

% Use raw image
split = 9;
%[training_data, test_data, l_train, l_test] = generate_partitioned_with_labels(9);

% Scaled data
N = size(training_data,2);
raw_concat = horzcat(training_data, test_data);
raw_scaled = zscore(raw_concat, 0, 2);
training_scaled = raw_scaled(:, 1:N);
test_scaled = raw_scaled(:, N+1:size(raw_scaled,2));

% PCA data
% load atapca.mat;
[pca_training_data, pca_test_data] = get_pca(training_data, test_data);
% pca_training_data = faces_training';
% pca_test_data = faces_test';

% Scaled PCA data
N_pca = size(training_data,2);
pca_concat = horzcat(pca_training_data, pca_test_data);
pca_scaled_1to1 = zscore(pca_concat, 0, 2);
pca_training_scaled_1to1 = pca_scaled_1to1(:, 1:N_pca);
pca_test_scaled_1to1 = pca_scaled_1to1(:, N_pca+1:size(pca_scaled_1to1,2));

pca_scaled_1toR = zscore(pca_concat, 0, 1);
pca_training_scaled_1toR = pca_scaled_1toR(:, 1:N_pca);
pca_test_scaled_1toR = pca_scaled_1toR(:, N_pca+1:size(pca_scaled_1toR,2));

%% ONE-TO-ONE
kernel_parameters = '-t 0';

% Basic
err_1v1_unscaled = svm_one_to_one(l_train, l_test, training_data, test_data, kernel_parameters);

% Scaled
err_1v1_scaled = svm_one_to_one(l_train, l_test, training_scaled, test_scaled, kernel_parameters);

% PCA unscaled
err_1v1_pca_unscaled = svm_one_to_one(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters);

% PCA scaled
err_1v1_pca_scaled = svm_one_to_one(l_train, l_test, pca_training_scaled_1to1, pca_test_scaled_1to1, kernel_parameters);

%% ONE-TO-REST

C = 10:10:100;

kernel_parameters = sprintf('-t 0 -b 1 -c %i -g %i', cost, gamma);

err_1vR_unscaled = svm_one_to_rest(l_train, l_test, training_data, test_data, kernel_parameters);

err_1vR_scaled = svm_one_to_rest(l_train, l_test, training_scaled, test_scaled, kernel_parameters);

err_1vR_pca_unscaled = svm_one_to_rest(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters);

err_1vR_pca_scaled = svm_one_to_rest(l_train, l_test, pca_training_scaled_1toR, pca_test_scaled_1toR, kernel_parameters);
