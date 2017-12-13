clear;

% Use raw image
split = 9;
[training_data, test_data, l_train, l_test] = generate_partitioned_with_labels(split);
%load pca.mat;

% Scaled data
N = size(training_data,2);
raw_concat = horzcat(training_data, test_data);
raw_scaled = zscore(raw_concat, 0, 2);
training_scaled = raw_scaled(:, 1:N);
test_scaled = raw_scaled(:, N+1:size(raw_scaled,2));

% PCA data
% load atapca.mat;
% [pca_training_data, pca_test_data] = get_pca(training_data, test_data);
% pca_training_data = faces_training';
% pca_test_data = faces_test';

% Scaled PCA data
% N_pca = size(training_data,2);
% pca_concat = horzcat(pca_training_data, pca_test_data);
% pca_scaled_1to1 = zscore(pca_concat, 0, 2);
% pca_training_scaled_1to1 = pca_scaled_1to1(:, 1:N_pca);
% pca_test_scaled_1to1 = pca_scaled_1to1(:, N_pca+1:size(pca_scaled_1to1,2));
% 
% pca_scaled_1toR = zscore(pca_concat, 0, 1);
% pca_training_scaled_1toR = pca_scaled_1toR(:, 1:N_pca);
% pca_test_scaled_1toR =  pca_scaled_1toR(:, N_pca+1:size(pca_scaled_1toR,2));

%% Cross validation - LINEAR KERNEL

C = 2.^linspace(-20,5,21);
% K-fold cross-validation 
K = 10;
indices = crossvalind('Kfold',l_train,K);

err_linear_1v1_scaled_array = zeros(1,size(C,2));
err_linear_1vR_scaled_array = zeros(1,size(C,2));

for i=1:size(C,2) % Param search loop
    acc_cross_val_err_1v1 = 0; % Accumulate errors
    acc_cross_val_err_1vR = 0; % Accumulate errors
    for j=1:K % Cross validation loop
        sprintf('Round i = %d, j = %d, C = %f', i, j, C(i));
        % Prep data
        cv_train_data = training_scaled(:, indices~=j);
        cv_test_data = training_scaled(:, indices==j);
        cv_train_labels = l_train(:, indices~=j);
        cv_test_labels = l_train(:, indices==j);
        
        % Run SVMs and get errors
        kernel_parameters = sprintf('-t 0 -c %f -q', C(1,i));
        [err_1v1, ~, ~, ~, ~] = svm_one_to_one(cv_train_labels, cv_test_labels, cv_train_data, cv_test_data, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1v1)', 'tmp');
        acc_cross_val_err_1v1 = acc_cross_val_err_1v1 + err_1v1;
        
        kernel_parameters = sprintf('-t 0 -b 1 -c %f -q', C(1,i));
        [err_1vR, ~, ~, ~, ~] = svm_one_to_rest(cv_train_labels, cv_test_labels, cv_train_data, cv_test_data, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1vR)', 'tmp');
        acc_cross_val_err_1vR = acc_cross_val_err_1vR + err_1vR;
    end
    % Average to get cross-val error
    err_linear_1v1_scaled_array(1,i) = acc_cross_val_err_1v1 / K;
    err_linear_1vR_scaled_array(1,i) = acc_cross_val_err_1vR / K;
    
    % Print status
    sprintf('Completed round i = %d, C = %f', i, j, C(1,i));
    sprintf('1v1 cross-val error: %f', err_linear_1v1_scaled_array(1,i));
    err_linear_1v1_scaled_array
    sprintf('1vR cross-val error: %f', err_linear_1vR_scaled_array(1,i));
    err_linear_1vR_scaled_array
end

save('crossval_linear.mat','err_linear_1v1_scaled_array','err_linear_1vR_scaled_array');

%% Plot results
load('crossval_linear.mat');
C = 2.^linspace(-20,5,21);
figure
plot(log(C), err_linear_1v1_scaled_array)
title('Cross Validation Error against C for 1v1')
xlabel('log2(C)')
ylabel('cross-validation error')
saveas(gcf,'crossValErr_linear_1v1.png')

figure
plot(log(C), err_linear_1vR_scaled_array)
title('Cross Validation Error against C for 1vR')
xlabel('log2(C)')
ylabel('cross-validation error')
saveas(gcf,'crossValErr_linear_1vR.png')

%% Cross validation - RBF KERNEL

C = 2.^linspace(-10,10,8);
gamma = 2.^linspace(-10,10,8);
% K-fold cross-validation 
K = 10;
indices = crossvalind('Kfold',l_train,K);

err_rbf_1v1_scaled_array = zeros(size(gamma,2),size(C,2));
err_rbf_1vR_scaled_array = zeros(size(gamma,2),size(C,2));

for i=1:size(gamma,2) % Param search loops
    for j=1:size(C,2) 
        
        acc_cross_val_err_1v1 = 0; % Accumulate errors
        acc_cross_val_err_1vR = 0; % Accumulate errors
        for k=1:K % Cross validation loop
            % Prep data
            cv_train_data = training_scaled(:, indices~=k);
            cv_test_data = training_scaled(:, indices==k);
            cv_train_labels = l_train(:, indices~=k);
            cv_test_labels = l_train(:, indices==k);

            % Run SVMs and get errors
            kernel_parameters = sprintf('-t 0 -c %f -g %f -q', C(1,j), gamma(1,i));
            [err_1v1, ~, ~, ~, ~] = svm_one_to_one(cv_train_labels, cv_test_labels, cv_train_data, cv_test_data, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1v1)', 'tmp');
            acc_cross_val_err_1v1 = acc_cross_val_err_1v1 + err_1v1;

            kernel_parameters = sprintf('-t 0 -c %f -g %f -b 1 -q', C(1,j), gamma(1,i));
            [err_1vR, ~, ~, ~, ~] = svm_one_to_rest(cv_train_labels, cv_test_labels, cv_train_data, cv_test_data, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1vR)', 'tmp');
            acc_cross_val_err_1vR = acc_cross_val_err_1vR + err_1vR;
        end
        % Average to get cross-val error
        err_rbf_1v1_scaled_array(i,j) = acc_cross_val_err_1v1 / K;
        err_rbf_1vR_scaled_array(i,j) = acc_cross_val_err_1vR / K;

        % Print status
        err_rbf_1v1_scaled_array
        err_rbf_1vR_scaled_array
    end
end

save('crossval_rbf.mat','err_rbf_1v1_scaled_array','err_rbf_1vR_scaled_array');

%% Plot results
load('crossval_rbf.mat');
% heatmap(log(C), log(gamma), err_rbf_1v1_scaled_array)
% heatmap(log(C), log(gamma), err_rbf_1vR_scaled_array)
