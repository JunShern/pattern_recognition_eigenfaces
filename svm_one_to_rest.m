function [error] = svm_one_to_rest(l_train, l_test, training_data, test_data, kernel_parameters)

numClasses = size(unique(horzcat(l_train,l_test)), 2);

confidences = zeros(size(l_test,2), numClasses);
for n=1:numClasses
    % Rename labels to be {n} or 0 ("the rest")
    binary_labels_train = ones(1,size(l_train,2));
    binary_labels_train(l_train~=n) = -1;
    binary_labels_test = ones(1,size(l_test,2));
    binary_labels_test(l_test~=n) = -1;

    % Train
    svm_struct_train = svmtrain(binary_labels_train', training_data', kernel_parameters);

    % Test
    [predicted_labels_binary, accuracy, dec_val] = svmpredict(binary_labels_test', test_data', svm_struct_train, '-b 1');
    confidences(:,n) = predicted_labels_binary .* dec_val(:,2);
end

% Take the maximum confidence column as our prediction
[max_value, predicted_labels] = max(confidences, [], 2);

error = sum((l_test ~= predicted_labels'))/size(l_test,2);

% Plot confusion matrix
conMat = confusionmat(l_test,predicted_labels');
imagesc(conMat);
colorbar;

end