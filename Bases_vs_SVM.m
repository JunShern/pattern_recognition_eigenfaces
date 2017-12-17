clear;

% Use raw image
split = 9;
[training_data, test_data, l_train, l_test] = generate_partitioned_by_class(split);

% PCA data
M = size(training_data,2);
n = 1;       % Counter for loop
for i = 30:10:M
    % load atapca.mat;
    % M = 50;
    [pca_training_data, pca_test_data] = get_pca(training_data, test_data, i);

    % Scaled PCA data
    N_pca = size(pca_training_data,2);
    pca_concat = horzcat(pca_training_data, pca_test_data);
    pca_scaled_1to1 = zscore(pca_concat, 0, 2);
    pca_training_scaled_1to1 = pca_scaled_1to1(:, 1:N_pca);
    pca_test_scaled_1to1 = pca_scaled_1to1(:, N_pca+1:size(pca_scaled_1to1,2));

    pca_scaled_1toR = zscore(pca_concat, 0, 1);
    pca_training_scaled_1toR = pca_scaled_1toR(:, 1:N_pca);
    pca_test_scaled_1toR =  pca_scaled_1toR(:, N_pca+1:size(pca_scaled_1toR,2));


    kernel_parameters = sprintf('-t 0 -b 1');

    [err_1v1_pca_unscaled, train_time_1v1_pca_unscaled, ~, ~, ~] = onev1_noconf(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters);
    [err_1v1_pca_scaled, train_time_1v1_pca_scaled, ~, ~, ~] = onev1_noconf(l_train, l_test, pca_training_scaled_1to1, pca_test_scaled_1to1, kernel_parameters);
    %[err_1vR_pca_unscaled, train_time_1vR_pca_unscaled, ~, ~, ~] = onevR_noconf(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters);
    [err_1vR_pca_scaled, train_time_1vR_pca_scaled, ~, ~, ~] = onevR_noconf(l_train, l_test, pca_training_scaled_1toR, pca_test_scaled_1toR, kernel_parameters);

    errors_1v1_pca_unscaled(1,n) = err_1v1_pca_unscaled;
    errors_1v1_pca_scaled(1,n) = err_1v1_pca_scaled;
    %errors_1vR_pca_unscaled(1,n) = err_1vR_pca_unscaled;
    errors_1vR_pca_scaled(1,n) = err_1vR_pca_scaled;

    time_pca_1v1_un(1,n) = train_time_1v1_pca_unscaled;
    time_pca_1v1_sc(1,n) = train_time_1v1_pca_scaled;
    %time_pca_1vR_un(1,n) = train_time_1vR_pca_unscaled;
    time_pca_1vR_sc(1,n) = train_time_1vR_pca_scaled;

    n = n + 1;

end

%% Plotting 

figure('position', [0 0 1280 800]);
hold on;
plot(10:10:M, errors_1v1_pca_unscaled, 'linewidth', 3)
plot(10:10:M, errors_1v1_pca_scaled, 'linewidth', 3)
%plot(10:10:M, errors_1vR_pca_unscaled, 'linewidth', 3)
plot(10:10:M, errors_1vR_pca_scaled, 'linewidth', 3)
title('Error in Multi-class SVM Methods with Varying M','interpreter', 'latex')
xlabel('Number of Bases M');
ylabel('SVM error');
grid;
leg = legend('1v1 PCA', '1v1 PCA Scaled', '1vR PCA', '1vR PCA Scaled','Location','southwest');
set(findall(gcf,'type','axes'),'fontsize', 28);
set(findall(gcf,'type','text'),'fontSize', 28);
set(leg,'FontSize', 24);
saveas(gcf,'Bases_vs_SVM.png')


%% More Plotting

%% Plotting 

figure('position', [0 0 1280 800]);
hold on;
plot(30:10:M, time_pca_1v1_un, 'linewidth', 3)
plot(30:10:M, time_pca_1v1_sc, 'linewidth', 3)
%plot(10:10:M, time_pca_1vR_un, 'linewidth', 3)
%plot(30:10:M, time_pca_1vR_sc, 'linewidth', 3)
title('Training Times in Multi-class SVM Methods with Varying M','interpreter', 'latex')
xlabel('Number of Bases M');
ylabel('Time (s)');
grid;
leg = legend('1v1 PCA', '1v1 PCA Scaled', '1vR PCA', '1vR PCA Scaled','Location','southeast');
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize', 32);
set(leg,'FontSize', 26);
saveas(gcf,'Bases_vs_Timings2.png')