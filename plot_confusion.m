function [] = plot_confusion(y, y_hat, title_text, filename)

figure('position', [0 0 800 800]);

confuse_mat = confusionmat(y, y_hat, 'order', [1:size(y,2)]);
imagesc(confuse_mat);

h = colorbar;
title(title_text, 'interpreter', 'latex');
xlabel('Predicted Class');
ylabel('Actual Class');
% Format data, need to make letters big to see well in Latex
set(findall(gcf,'type','axes'),'fontsize', 26);

fig = gcf;
fig.PaperPositionMode = 'auto';
print(filename,'-dpng','-r0');

end