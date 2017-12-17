% Script for the Nearest Neighbour (NN) approximation

clear;

% Load the necessary variables from the previous tasks
%load partitioned_data;
load pca.mat;
load atapca.mat;

N = size(training_data,2);
P = size(test_data,2);

% Initialization of different variables to be used later
all_guesses = zeros(P, 3, N);
success = zeros(N, 1);
failure = zeros(N, 1);
timings = zeros(N, 1);

for M = 1:N
    % M is number of eigenvalues/vectors to use
    % Project each face onto each eigenvector, each row is a face
    faces_training_sel = faces_training(:, 1:M);
    faces_test_sel = faces_test(:, 1:M);
    
    % Initialize for speed
    guesses = zeros(P, 3);
    tic;
    for H = 1:P
        % For each face in testing set.
        l = l_test(H);
        test_face = faces_test_sel(H, :);
        
        % Calculate errors of test face to each training face projection
        error = faces_training_sel - test_face(ones(1,N), :);
        error_mag = sum(error .^2, 2);
        % Find the index of the minimum value
        [minimum, index] = min(error_mag);
        % Look up the index from l_train
        guess = l_train(index);
        
        % Check
        if l == guess
            guesses(H, :) = [1, guess, l];
        else
            guesses(H, :) = [0, guess, l];
        end
    
    end
    
     % Keep
    timings(M) = toc;
    all_guesses(:, :, M) = guesses;
    correct = sum(guesses(:, 1));
    incorrect = P - correct;
    success(M) = correct;
    failure(M) = incorrect;
end

%% ------------------------Plotting for the Report-----------------------

% Success and Failure cases plot
figure('position', [0 0 1280 800]);
hold on;
plot(success, 'LineWidth', 4);
plot(failure, 'LineWidth', 4);
hold off;
title('Success \& Failure Guesses vs. Number of Bases', 'interpreter', 'latex');
xlabel('Number of Bases');
ylabel('Number of Guesses');
grid;
leg = legend('Correct Guesses', 'Incorrect Guesses','Location','northeast');
% Format data, need to make letters big to see well in Latex
set(leg,'FontSize', 25);
set(findall(gcf,'type','axes'),'fontsize', 30);
set(findall(gcf,'type','text'),'fontSize', 30);
% Save
fig = gcf;
fig.PaperPositionMode = 'auto';
print('NN_success_failures','-dpng','-r0');

% Recognition Accuracy plot
figure('position', [0 0 1280 800]);
percentage = 100 * (success ./ size(test_data, 2));
plot(percentage, 'LineWidth', 4);
title('Recognition Accuracy vs. Number of Bases', 'interpreter', 'latex');
xlabel('Number of Bases');
ylabel('Accuracy Percentage');
grid;
% Format data, need to make letters big to see well in Latex
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize', 32);
% Save
fig = gcf;
fig.PaperPositionMode = 'auto';
print('NN_Recognition_Accuracy','-dpng','-r0');

% Timing plot
figure('position', [0 0 1280 800]);
plot(timings, 'LineWidth', 4);
title('Time Taken vs. Number of Bases', 'interpreter', 'latex');
xlabel('Number of Bases');
ylabel('Time (s)');
grid;
% Format data, need to make letters big to see well in Latex
set(findall(gcf,'type','axes'),'fontsize', 32);
set(findall(gcf,'type','text'),'fontSize', 32);
% Save
fig = gcf;
fig.PaperPositionMode = 'auto';
print('NN_timings','-dpng','-r0');

% Trying out another way of doing the confusion matrix
% guesses = all_guesses(:, :, 156);
% figure('position', [0 0 800 800]);
% plotconfusion(guesses(:, 1), l_test')

%% Confusion Matrix with labels and the predictions

% for i = 10:70:length(training_data) 
% guesses = all_guesses(:, :, i);
% num_bases = num2str(i); 
% title = strcat('NN PCA with  ',' ', num_bases, ' ', 'Bases');
% filename = strcat('NN_confusemat', num_bases);
% plot_confusion(l_test, guesses(:,2)', title, filename);
% end

% Plot Confusion Matrix for M = 10;
guesses = all_guesses(:, :, 10);
plot_confusion(l_test, guesses(:,2)', 'Confusion Matrix M = 10', 'NN_confusemat_10');

% Plot Confusion Matrix for M = 468;
guesses = all_guesses(:, :, 468);
plot_confusion(l_test, guesses(:,2)', 'Confusion Matrix M = 468', 'NN_confusemat_468');