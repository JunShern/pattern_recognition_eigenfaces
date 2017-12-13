clear;

% Load the pca.mat file
load pca.mat;

N = size(training_data,2);
P = size(test_data,2);

% Want to perform image reconstruction using the PCA bases learnt
% Create appropriate S2_eig_val (for ATA) that is correct from a_ni = A * u_i
% Need to normalize

eigvec2_adj = A * eigvec2;
eigvec2_adj = normc(eigvec2_adj);       % Normalize colums of matrix

% Now you need to project each face onto an eigenvector
% Each face will be represented by each row
faces_training = A' * eigvec2_adj;      % w_n = [a_n1, a_n2,...,a_nM]'
faces_test = B' * eigvec2_adj;

% Initializing variables to be used later
avgtraining_error = zeros(N, 1);    
avgtest_error = zeros(P, 1);            
total = size(eigvec2_adj, 2);       % Get how many eigvecs there are for a for loop

% Loop to represent each face onto each eigenface
for n=1:total
    % Select the particular eigenvector for each loop iteration
    eigvec2_sel = eigvec2_adj(:, 1:n);

    % Project each face onto each eigenvector, each row is a face
    faces_training_sel = faces_training(:, 1:n);
    faces_test_sel = faces_test(:, 1:n);

    % Reconstruct each face.
    faces_reconstructed_training = repmat(average_face, 1, N) + eigvec2_sel * faces_training_sel';
    faces_reconstructed_test = repmat(average_face, 1, P) + eigvec2_sel * faces_test_sel';
    
    % Calculation of different errors
    face_error_training = training_data - faces_reconstructed_training;
    face_error_test = test_data - faces_reconstructed_test;
    
    % Magnitude of each column
    mag_face_error_training = sqrt(sum(face_error_training .^2, 1));
    mag_face_error_test = sqrt(sum(face_error_test .^2, 1));
    
    % Assign and store
    avgtraining_error(n) = mean(mag_face_error_training);
    avgtest_error(n) = mean(mag_face_error_test);
end

save('atapca.mat','eigval2','eigvec2_adj', 'faces_training', ...
    'faces_test', 'avgtraining_error', 'avgtest_error');

%------------------------Plotting for the Report-----------------------

ATA_eigval2 = abs(sum(eigval2, 2));
eig_size = size(ATA_eigval2, 1);
eigs_unused = zeros(eig_size, 1);

for i=1:eig_size - 1
        eigs_unused(i) = sum(ATA_eigval2(i + 1:eig_size));
end

% Plotting the different errors in the question
figure
hold on;
plot(avgtraining_error, 'LineWidth', 3);
plot(avgtest_error, 'LineWidth', 3);
plot(sqrt(eigs_unused), 'LineWidth', 3);
hold off;
title('Average Reconstruction Error vs. Number of Bases', 'interpreter', 'latex');
xlabel('Number of Bases');
ylabel('Error');
grid;
leg = legend('Training Error', 'Test Error', 'Theorectical Error','Location','northeast');
% Format data, need to make letters big to see well in Latex
set(leg,'FontSize', 12);
set(findall(gcf,'type','axes'),'fontsize', 12);
set(findall(gcf,'type','text'),'fontSize', 12);
% Save
fig = gcf;
fig.PaperPositionMode = 'auto';
print('Error_vs_Bases_Graph','-dpng','-r0');

% Need to plot to reconstruct image
% M is the number of eigenvalues/vector to use
%----------------- M = 10 -----------------
M = 10;
eigvec2_sel = eigvec2_adj(1:M,:);

% Project each face onto each eigenvector, each row is a face
faces_training_sel = faces_training(1:M,:);
faces_test_sel = faces_test(1:M,:);

% Reconstruct each face
faces_reconstructed_training = repmat(average_face, 1, N) + eigvec2_sel * faces_training_sel';
faces_reconstructed_test = repmat(average_face, 1, P) + eigvec2_sel * faces_test_sel';

% Faces and reconstructions
face1 = training_data(:, 111);
face2 = training_data(:, 222);
face3 = test_data(:, 33);

reface1 = faces_reconstructed_training(:, 111);
reface2 = faces_reconstructed_training(:, 222);
reface3 = faces_reconstructed_test(:, 33);

show_face(face1);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('face1','-dpng','-r0');

show_face(face2);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('face2','-dpng','-r0');

show_face(face3);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('face3','-dpng','-r0');

show_face(reface1);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface1','-dpng','-r0');

show_face(reface2);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface2','-dpng','-r0');

show_face(reface3);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface3','-dpng','-r0');

%----------------- M = 25 -----------------
M = 75;
eigvec2_sel = eigvec2_adj(1:M,:);

% Project each face onto each eigenvector, each row is a face
faces_training_sel = faces_training(1:M,:);
faces_test_sel = faces_test(1:M,:);

% Reconstruct each face
faces_reconstructed_training = repmat(average_face, 1, N) + eigvec2_sel * faces_training_sel';
faces_reconstructed_test = repmat(average_face, 1, P) + eigvec2_sel * faces_test_sel';

reface1 = faces_reconstructed_training(:, 111);
reface2 = faces_reconstructed_training(:, 222);
reface3 = faces_reconstructed_test(:, 33);

show_face(reface1);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface12','-dpng','-r0');

show_face(reface2);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface22','-dpng','-r0');

show_face(reface3);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface32','-dpng','-r0');

%----------------- M = 50 -----------------
M = 150;
eigvec2_sel = eigvec2_adj(1:M,:);

% Project each face onto each eigenvector, each row is a face
faces_training_sel = faces_training(1:M,:);
faces_test_sel = faces_test(1:M,:);

% Reconstruct each face
faces_reconstructed_training = repmat(average_face, 1, N) + eigvec2_sel * faces_training_sel';
faces_reconstructed_test = repmat(average_face, 1, P) + eigvec2_sel * faces_test_sel';

reface1 = faces_reconstructed_training(:, 111);
reface2 = faces_reconstructed_training(:, 222);
reface3 = faces_reconstructed_test(:, 33);

show_face(reface1);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface13','-dpng','-r0');

show_face(reface2);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface23','-dpng','-r0');

show_face(reface3);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('reface33','-dpng','-r0');