function [] = plot_support_vectors(support_vectors, filename_base)
% Plotting support vectors
for i=1:size(support_vectors,1)
    figure;
    show_face(full(support_vectors(i,:)));
    % Save images
    fig = gcf;
    fig.PaperPositionMode = 'auto';
    print([filename_base, num2str(i)],'-dpng','-r0');
end
end
