function [error, train_time, test_time, svm_struct_train, predicted_labels] = svm_one_to_rest(l_train, l_test, training_data, test_data, kernel_parameters, conmat_title, conmat_filename)

numClasses = size(unique(horzcat(l_train,l_test)), 2);

confidences = zeros(size(l_test,2), numClasses);
train_time = 0;
test_time = 0;
for n=1:numClasses
    % Rename labels to be {n} or 0 ("the rest")
    binary_labels_train = ones(1,size(l_train,2));
    binary_labels_train(l_train~=n) = -1;
    binary_labels_test = ones(1,size(l_test,2));
    binary_labels_test(l_test~=n) = -1;

    % Train
    tic;
    svm_struct_train = svmtrain(binary_labels_train', training_data', kernel_parameters);
    train_time = train_time + toc;

    % Test
    tic;
    [predicted_labels_binary, accuracy, dec_val] = svmpredict(binary_labels_test', test_data', svm_struct_train, '-b 1');
    test_time = test_time + toc;
    
    confidences(:,n) = predicted_labels_binary .* dec_val(:,2);
end

% Take the maximum confidence column as our prediction
[max_value, predicted_labels] = max(confidences, [], 2);

error = sum((l_test ~= predicted_labels'))/size(l_test,2);

% Plot confusion matrix
%plot_confusion(l_test, predicted_labels', conmat_title, conmat_filename);

end