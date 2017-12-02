clear;

% Use raw image
split = 9;
%[training_data, test_data, l_train, l_test] = generate_partitioned_with_labels(9);
load pca.mat;

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
pca_test_scaled_1toR =  pca_scaled_1toR(:, N_pca+1:size(pca_scaled_1toR,2));

%% ONE-TO-ONE
kernel_parameters = '-t 0';

% Basic
[err_1v1_unscaled, err_1v1_unscaled_train_time, err_1v1_unscaled_test_time]  = svm_one_to_one(l_train, l_test, training_data, test_data, kernel_parameters, 'Raw Data Unscaled Confusion Matrix (1v1)', 'confmat_err_1v1_unscaled');

% Scaled
[err_1v1_scaled, err_1v1_scaled_train_time, err_1v1_scaled_test_time] = svm_one_to_one(l_train, l_test, training_scaled, test_scaled, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1v1)', 'confmat_err_1v1_scaled');

% PCA unscaled
[err_1v1_pca_unscaled, err_1v1_pca_unscaled_train_time, err_1v1_pca_unscaled_test_time] = svm_one_to_one(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters, 'PCA Data Unscaled Confusion Matrix (1v1)', 'confmat_err_1v1_pca_unscaled');

% PCA scaled
[err_1v1_pca_scaled, err_1v1_pca_scaled_train_time, err_1v1_pca_scaled_test_time] = svm_one_to_one(l_train, l_test, pca_training_scaled_1to1, pca_test_scaled_1to1, kernel_parameters, 'PCA Data Scaled Confusion Matrix (1v1)', 'confmat_err_1v1_pca_scaled');

%% ONE-TO-REST

%kernel_parameters = sprintf('-t 0 -b 1 -c %i -g %i', cost, gamma);
kernel_parameters = sprintf('-t 0 -b 1');

[err_1vR_unscaled, err_1vR_unscaled_train_time, err_1vR_unscaled_test_time] = svm_one_to_rest(l_train, l_test, training_data, test_data, kernel_parameters, 'Raw Data Unscaled Confusion Matrix (1vR)', 'confmat_err_1vR_unscaled');

[err_1vR_scaled, err_1vR_scaled_train_time, err_1vR_scaled_test_time] = svm_one_to_rest(l_train, l_test, training_scaled, test_scaled, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1vR)', 'confmat_err_1vR_scaled');

[err_1vR_pca_unscaled, err_1vR_pca_unscaled_train_time, err_1vR_pca_unscaled_test_time] = svm_one_to_rest(l_train, l_test, pca_training_data, pca_test_data, kernel_parameters, 'PCA Data Unscaled Confusion Matrix (1vR)', 'confmat_err_1vR_pca_unscaled');

[err_1vR_pca_scaled, err_1vR_pca_scaled_train_time, err_1vR_pca_scaled_test_time] = svm_one_to_rest(l_train, l_test, pca_training_scaled_1toR, pca_test_scaled_1toR, kernel_parameters, 'PCA Data Scaled Confusion Matrix (1vR)', 'confmat_err_1vR_pca_scaled');


%% 
[err_1v1_scaled, err_1v1_scaled_train_time, err_1v1_scaled_test_time, svm_struct_arr, predicted_labels] = svm_one_to_one(l_train, l_test, training_scaled, test_scaled, kernel_parameters, 'Raw Data Scaled Confusion Matrix (1v1)', 'confmat_err_1v1_scaled');
guesses = [l_test == predicted_labels'; l_test; predicted_labels'];
% Correct
[r, c, v] = find(guesses(1,:)==1);
correct_index = c(1);
correct_class = guesses(2,correct_index);
% Incorrect
[r_, c_, v_] = find(guesses(1,:)~=1);
incorrect_index = c_(1);
incorrect_class = guesses(2,incorrect_index);
%correct_guesses = guesses(:,guesses(1,:)==1);
%incorrect_guesses = guesses(:,guesses(1,:)~=1);
%correct_class = correct_guesses(2,1);
%incorrect_class = incorrect_guesses(2,1);

% Show the support vectors of the model which classified correct face
for i = 1:size(svm_struct_arr,1)
    if ismember(correct_class, svm_struct_arr(i).Label(1))
        % Show the correctly classified face
        show_face(test_data(:,correct_index));
        fig = gcf;
        fig.PaperPositionMode = 'auto';
        print('correct_face','-dpng','-r0');

        plot_support_vectors(svm_struct_arr(i).SVs, 'correct_face_sv_');
        break
    end
end
% l_test(correct_index) = 30; guesses(:, correct_index) = [1, 30, 30]

% Show the support vectors of the model which classified incorrect face
for i = 1:size(svm_struct_arr,1)
    if ismember(incorrect_class, svm_struct_arr(i).Label(1))
        % Show the correctly classified face
        show_face(test_data(:,incorrect_index));
        fig = gcf;
        fig.PaperPositionMode = 'auto';
        print('incorrect_face','-dpng','-r0');

        plot_support_vectors(svm_struct_arr(i).SVs, 'incorrect_face_sv_');
        break
    end
end
% l_test(incorrect_index) = 40; guesses(:, incorrect_index) = [0, 40, 35]