function [error] = svm_one_to_one(l_train, l_test, training_data, test_data, kernel_parameters)
% Prepare data
train_concat = vertcat(l_train,training_data);
% test_concat = vertcat(l_test,test_data);

numClasses = size(unique(horzcat(l_train,l_test)), 2);
allPairs = combnk(unique(horzcat(l_train,l_test)), 2);
votes = zeros(size(l_test,2), numClasses*numClasses);
for n=1:size(allPairs,1)
    i = allPairs(n,1);
    j = allPairs(n,2);

    % Pick out columns for labels i and j
    train_concat_sel = train_concat(:, (l_train==i | l_train==j));
    labels_sel_train = train_concat_sel(1,:);
    data_sel_train = train_concat_sel(2:end,:);

    % Train
    svm_struct_train = svmtrain(labels_sel_train', data_sel_train', kernel_parameters);

    % Test
    [predicted_labels_sel] = svmpredict(l_test', test_data', svm_struct_train);
    votes(:,(i-1)*numClasses + j) = predicted_labels_sel;
end

% Remove all zero columns from votes matrix
votes( :, ~any(votes,1) ) = [];
predicted_labels = mode(votes, 2);

error = sum((l_test ~= predicted_labels'))/size(l_test,2);

% Plot confusion matrix
conMat = confusionmat(l_test,predicted_labels');
imagesc(conMat);
colorbar;

end