% function [faces_training, faces_test] = get_pca(training_data, test_data)
function [faces_training, faces_test] = get_pca(training_data, test_data, M)

% First need to find the average face
N = size(training_data,2);
P = size(test_data,2);
average_face = sum(training_data,2) ./ N;           %Performing element-by-element division

% Remove average face from training set
A = training_data - average_face(:, ones(1,N));     %Remove all face columns at once
B = test_data - average_face(:, ones(1,P));

%% Covariance matrix using S = (1/N)AA(T)
% % Computing covariance matrix; For a) using S = (1/N)AA(T)
% S = (1/N) * A * A';
% 
% % Calculate the eigenvectors
% [eigvec, eigval] = eig(S);
% 
% % Find the number of non-zero eigenvalues 
% nonzero = nnz(round(eigval, 10));
% disp(['Rank of AAT is ' num2str(rank(S))]);
% disp(['Number of nonzero elements in AAT is ' num2str(nonzero)]);

%% Covariance matrix using S = (1/N)A(T)A
S2 = (1/N) * (A' * A);
[eigvec2, eigval2] = eig(S2);
% nonzero2 = nnz(round(eigval2, 10));
% disp(['Rank of ATA is ' num2str(rank(S2))]);
% disp(['Number of nonzero elements in ATA is ' num2str(nonzero2)]);

% Want to perform image reconstruction using the PCA bases learnt
% Create appropriate S2_eig_val (for ATA) that is correct from a_ni = A * u_i
% Need to normalize

eigvec2_adj = A * eigvec2;
eigvec2_adj = normc(eigvec2_adj);       % Normalize colums of matrix

% Now you need to project each face onto an eigenvector
% Each face will be represented by each row
faces_training = (A' * eigvec2_adj)';      % w_n = [a_n1, a_n2,...,a_nM]'
faces_test = (B' * eigvec2_adj)';

%% Choose M best
faces_training = faces_training(1:M,:);
faces_test= faces_test(1:M,:);

end