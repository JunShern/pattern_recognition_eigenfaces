clear;

split_arr = 5:1:9;
errors = zeros(1,size(split_arr,2));
for i = 1:size(split_arr,2)
    split = split_arr(i);
    % Use raw image
    [training_data, test_data, l_train, l_test] = generate_partitioned_with_labels(split);

    % Scaled data
    N = size(training_data,2);
    raw_concat = horzcat(training_data, test_data);
    raw_scaled = zscore(raw_concat, 0, 2);
    training_scaled = raw_scaled(:, 1:N);
    test_scaled = raw_scaled(:, N+1:size(raw_scaled,2));
    
    kernel_parameters = sprintf('-t 0 -b 1');
    [err_1vR_scaled, ~, ~, ~, ~] = svm_one_to_rest(l_train, l_test, training_scaled, test_scaled, kernel_parameters, '~', '~');
    errors(1,i) = err_1vR_scaled;
end
%% Plot results
figure('position', [0 0 1280 800]);
plot(split_arr/10, errors, 'linewidth', 3)
title('Performance of Data Partitions','interpreter', 'latex')
xlabel('Proportion of Training Data')
ylabel('SVM error')

set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize',32);

saveas(gcf,'data_partitioning_plot.png')
    