function [ ] = show_face( img_data )
%show_face converts vector of image to grayscale image and then displays

img = vec2mat(img_data, 56);
G = mat2gray(img);
imshow(G', 'Border', 'tight', 'InitialMagnification', 500);

end

