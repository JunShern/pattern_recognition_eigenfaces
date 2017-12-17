clear;

split_arr = 5:1:9;
errors_1v1 = zeros(1,size(split_arr,2));
errors_1vR = zeros(1,size(split_arr,2));
errors_1v1_PCA = zeros(1,size(split_arr,2));
errors_1vR_PCA = zeros(1,size(split_arr,2));


for i = 1:size(split_arr,2)
    split = split_arr(i);
    
    [training_data, test_data, l_train, l_test] = generate_partitioned_by_class(split);

    % Scaled data
    N = size(training_data,2);
    raw_concat = horzcat(training_data, test_data);
    raw_scaled = zscore(raw_concat, 0, 2);
    training_scaled = raw_scaled(:, 1:N);
    test_scaled = raw_scaled(:, N+1:size(raw_scaled,2));

    %M = 468;
    % load atapca.mat;
    % Need to do get_pca without M input
    [pca_training_data, pca_test_data] = get_pca(training_data, test_data);
    
    kernel_parameters = sprintf('-t 0 -b 1');
    [err_1v1_scaled, ~, ~, ~, ~] = svm_one_to_one(l_train, l_test, training_scaled, test_scaled, kernel_parameters, '~', '~');
    [err_1vR_scaled, ~, ~, ~, ~] = svm_one_to_rest(l_train, l_test, training_scaled, test_scaled, kernel_parameters, '~', '~');
    [err_1v1_pca_unscaled, ~, ~, ~, ~] = svm_one_to_one(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters, '~', '~');
    [err_1vR_pca_unscaled, ~, ~, ~, ~] = svm_one_to_rest(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters, '~', '~');
    errors_1v1(1,i) = err_1v1_scaled;
    errors_1vR(1,i) = err_1vR_scaled;
    errors_1v1_PCA(1,i) = err_1v1_pca_unscaled;
    errors_1vR_PCA(1,i) = err_1vR_pca_unscaled;
end
%% Plot results
figure('position', [0 0 1280 800]);
hold on;
plot(split_arr/10, errors_1v1, 'linewidth', 3)
plot(split_arr/10, errors_1vR, 'linewidth', 3)
plot(split_arr/10, errors_1v1_PCA, 'linewidth', 3)
plot(split_arr/10, errors_1vR_PCA, 'linewidth', 3)
title('Performance of Different Data Partitions','interpreter', 'latex')
xlabel('Proportion of Training Data');
ylabel('SVM error');
grid;
leg = legend('1v1 Scaled', '1vR Scaled', '1v1 PCA', '1vR PCA','Location','northeast');
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize', 32);

saveas(gcf,'data_partitioning_plot2.png')
    