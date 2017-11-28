clear;

% Load the face.mat file
filename = 'face.mat';
[training_data, test_data] = generate_partitioned(filename);
disp(['Training length is ' num2str(size(training_data, 2)) ...
      '; Test length is ' num2str(size(test_data, 2)) '.']);

% First need to find the average face
N = size(training_data,2);
P = size(test_data,2);
average_face = sum(training_data,2) ./ N;           %Performing element-by-element division

% Show and save mean face
show_face(average_face);
fig = gcf;
fig.PaperPositionMode = 'auto';
print('mean_face','-dpng','-r0');

% Remove average face from training set
A = training_data - average_face(:, ones(1,N));     %Remove all face columns at once
B = test_data - average_face(:, ones(1,P));

% Computing covariance matrix; For a) using S = (1/N)AA(T)
S = (1/N) * A * A';

% Calculate the eigenvectors
[eigvec, eigval] = eig(S);

% Find the number of non-zero eigenvalues 
nonzero = nnz(round(eigval, 10));
disp(['Rank of AAT is ' num2str(rank(S))]);
disp(['Number of nonzero elements in AAT is ' num2str(nonzero)]);

%-------------------Part b)---------------------

% Repeat same steps but now have S = (1/N)A(T)A
S2 = (1/N) * A' * A;
[eigvec2, eigval2] = eig(S2);
nonzero2 = nnz(round(eigval2, 10));
disp(['Rank of ATA is ' num2str(rank(S2))]);
disp(['Number of nonzero elements in ATA is ' num2str(nonzero2)]);

% Save the variables into a file called pca.mat so that they are more
% easily accesible in other MATLAB scripts

save('pca.mat','training_data','test_data', 'A', 'B', 'average_face', ...
    'eigval', 'eigvec', 'eigval2', 'eigvec2', 'S', 'S2');

%------------Timings for each method for discussion in report-----------
AAT = @() eig((1/N) * A * A');
ATA = @() eig((1/N) * A' * A);
time_AAT = timeit(AAT, 2)
time_ATA = timeit(ATA, 2)

%--------------------Plotting for the Report--------------------

close all;

% Plot for AAT eigenvalues
AAT_eigval = sum(eigval, 2);
figure('position', [0 0 1280 800]);
plot(1:length(AAT_eigval), abs(AAT_eigval), 'linewidth', 5);
title('Eigenvalues of $$S = AA^T$$', 'interpreter', 'latex');
xlabel('Number of evals');
ylabel('Value');
grid;
% Format data, need to make letters big to see well in Latex
set(findall(gcf,'type','axes'),'fontsize', 40);
set(findall(gcf,'type','text'),'fontSize', 40);
% Save
fig = gcf;
fig.PaperPositionMode = 'auto';
print('AAT_eigvals','-dpng','-r0');

% Plot for ATA eigenvalues
ATA_eigval2 = sum(eigval2, 2);
figure('position', [0 0 1280 800]);
plot(1:length(ATA_eigval2), abs(ATA_eigval2), 'linewidth', 5);
title('Eigenvalues of $$S = A^TA$$', 'interpreter', 'latex');
xlabel('Number of evals');
ylabel('Value');
grid;
% Format data, need to make letters big to see well in Latex
set(findall(gcf,'type','axes'),'fontsize', 40);
set(findall(gcf,'type','text'),'fontSize', 40);
% Save
fig = gcf;
fig.PaperPositionMode = 'auto';
print('ATA_eigvals','-dpng','-r0');

% Show and save 3 eigenfaces, S=AAT.
for i = 1:3
    show_face(eigvec(:, i));
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    name = ['AAT_eigface', num2str(i)];
    print(name,'-dpng','-r0');
end

% Create appropriate S2_eig_val (for ATA) that is correct from a_ni = A * u_i
% Need to normalize

eigvec2_adj = A * eigvec2;
eigvec2_adj = normc(eigvec2_adj);       % Normalize colums of matrix

% Show and save 4 eigenfaces, S=ATA.
for i = 4:7
    show_face(eigvec2_adj(:, i - 3));
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    name = ['ATA_eigface', num2str(i)];
    print(name,'-dpng','-r0');
end
