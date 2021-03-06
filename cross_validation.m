% Use raw image
split = 9;
[training_data, test_data, l_train, l_test] = generate_partitioned_by_class(split);
save('crossval_dataset.mat','training_data','test_data','l_train','l_test');

clear;

load crossval_dataset.mat;
load pca.mat;

% Scaled data
N = size(training_data,2);
raw_concat = horzcat(training_data, test_data);
raw_scaled = zscore(raw_concat, 0, 2);
training_scaled = raw_scaled(:, 1:N);
test_scaled = raw_scaled(:, N+1:size(raw_scaled,2));

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

figure('position', [0 0 1280 800]);
plot(split_arr/10, errors, 'linewidth', 3)
plot(log(C), err_linear_1v1_scaled_array)
title('Cross Validation Error against C for 1v1')
xlabel('log2(C)')
ylabel('cross-validation error')
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize',32);
saveas(gcf,'crossValErr_linear_1v1.png')

figure('position', [0 0 1280 800]);
plot(split_arr/10, errors, 'linewidth', 3)
plot(log(C), err_linear_1vR_scaled_array)
title('Cross Validation Error against C for 1vR')
xlabel('log2(C)')
ylabel('cross-validation error')
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize',32);
saveas(gcf,'crossValErr_linear_1vR.png')

%% Cross validation - RBF KERNEL

C = 2.^linspace(-10,20,10);
gamma = 2.^linspace(-20,10,10);

% Parallel computation
%C = C(:,1:4);
%C = C(:,5:end);
%save('crossval_dataset.mat','training_data','test_data','l_train','l_test','C','gamma');

% K-fold cross-validation 
K = 10;
indices = crossvalind('Kfold',l_train,K);
K = 2; % Speed things up

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
            kernel_parameters = sprintf('-t 2 -c %f -g %f -q', C(1,j), gamma(1,i));
            [err_1v1, ~, ~, ~, ~] = svm_one_to_one(cv_train_labels, cv_test_labels, cv_train_data, cv_test_data, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1v1)', 'tmp');
            acc_cross_val_err_1v1 = acc_cross_val_err_1v1 + err_1v1;

            kernel_parameters = sprintf('-t 2 -c %f -g %f -b 1 -q', C(1,j), gamma(1,i));
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

save('crossval_rbf_final.mat','err_rbf_1v1_scaled_array','err_rbf_1vR_scaled_array','C','gamma');

%% Plot results
load('crossval_rbf.mat');
%errs = horzcat(err_rbf_1v1_scaled_array1, err_rbf_1v1_scaled_array2);

%heatmap(log(C), log(gamma), err_rbf_1v1_scaled_array)
%heatmap(log(C), log(gamma), err_rbf_1vR_scaled_array)
imagesc(err_rbf_1v1_scaled_array);
h = colorbar;
ylabel(h, 'Error');

imagesc(err_rbf_1vR_scaled_array);
h = colorbar;
ylabel(h, 'Error');

